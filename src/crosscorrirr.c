#if defined HAVE_CONFIG_H
# include "config.h"
#endif	/* HAVE_CONFIG_H */
#include <stddef.h>
#include <math.h>
#include <mkl.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>
#include <immintrin.h>
#include "nifty.h"

typedef float alf_t __attribute__((aligned(16)));
typedef double ald_t __attribute__((aligned(16)));

#if 0
static double
ccf_gauss(int k, unsigned int res, cots_ts_t *xt, cots_ts_t *yt, cots_pf_t *xp, cots_pf_t *yp, size_t nx, size_t ny)
{
	double rho = 0.d;
	double quo = 0.d;

	for (size_t i = 0U; i < nx; i++) {
		for (size_t j = 0U; j < ny; j++) {
			double bk = labs(ts_diff(xt[i], yt[j]) - k);
			quo += _krnl_gauss(bk);
		}
	}

	for (size_t i = 0U; i < nx; i++) {
		for (size_t j = 0U; j < ny; j++) {
			double bk = labs(ts_diff(xt[i], yt[j]) - k);
			xp[i] * yp[j] * _krnl_gauss(bk, );
		}
	}
	return rho;
}
#endif

static __attribute__((unused)) int
_stats_f(float mean[static 1U], float std[static 1U], alf_t s[], size_t ns)
{
#define VSL_SS_TASK	(VSL_SS_MEAN | VSL_SS_2R_MOM)
	VSLSSTaskPtr task;
	MKL_INT ndim = 1;
	MKL_INT dim1 = ns;
	MKL_INT stor = VSL_SS_MATRIX_STORAGE_ROWS;
	int rc = 0;

	rc += vslsSSNewTask(&task, &ndim, &dim1, &stor, s, 0, 0);
	rc += vslsSSEditTask(task, VSL_SS_ED_MEAN, mean);
	rc += vslsSSEditTask(task, VSL_SS_ED_2R_MOM, std);
	rc += vslsSSCompute(task, VSL_SS_TASK, VSL_SS_METHOD_FAST);
	rc += vslSSDeleteTask(&task);
	return rc;
#undef VSL_SS_TASK
}

static __attribute__((unused)) int
_quasi_stats_f(float mean[static 1U], float std[static 1U], alf_t s[], size_t ns)
{
	float min = s[0U], max = s[0U];

	for (size_t i = 1U; i < ns; i++) {
		if (s[i] > max) {
			max = s[i];
		} else if (s[i] < min) {
			min = s[i];
		}
	}

	*std = (max - min);
	*mean = ((s[0U] + s[ns - 1U]) / 2.f + (max + min) / 2.f) / 2.f;
	return 0;
}

static int
_norm_f(alf_t s[], size_t ns, int(*statf)(float*, float*, alf_t[], size_t))
{
	float mu, sigma;
	register __m128 mmu;
	register __m128 msd;

	if (UNLIKELY(statf(&mu, &sigma, s, ns) < 0)) {
		return -1;
	}
	/* go for (s + -mu) * 1/sigma */
	mu = -mu;
	mmu = _mm_load1_ps(&mu);
	sigma = 1.f / sigma;
	msd = _mm_load1_ps(&sigma);
	for (size_t i = 0U; i + 3U < ns; i += 4U) {
		register __m128 ms;
		ms = _mm_load_ps(s + i);
		ms = _mm_add_ps(ms, mmu);
		ms = _mm_mul_ps(ms, msd);
		_mm_store_ps(s + i, ms);
	}
	/* do the rest by hand */
	switch (ns % 4U) {
	case 3U:
		s[ns - 3U] += mu;
		s[ns - 3U] *= sigma;
	case 2U:
		s[ns - 2U] += mu;
		s[ns - 2U] *= sigma;
	case 1U:
		s[ns - 1U] += mu;
		s[ns - 1U] *= sigma;
	case 0U:
	default:
		break;
	}
	return 0;
}


static double
_mean_tdiff(ald_t t1[], size_t n1, ald_t t2[], size_t n2)
{
/* calculate average time differences
 * min(t1[i], t2[j]) - min(t1[i - 1], t2[j - 1]) */
	double sum = 0.0;
	double prev;
	size_t i, j;

	if (*t1 <= *t2) {
		i = 1U;
		j = 0U;
		prev = *t1;
	} else {
		i = 0U;
		j = 1U;
		prev = *t2;
	}
	while (i < n1 && j < n2) {
		if (t1[i] <= t2[j]) {
			sum += t1[i] - prev;
			prev = t1[i++];
		} else {
			sum += t2[j] - prev;
			prev = t2[j++];
		}
	}
	/* remainder of t1 */
	while (i < n1) {
		sum += t1[i] - prev;
		prev = t1[i++];
	}
	/* remainder of t2 */
	while (j < n2) {
		sum += t2[j] - prev;
		prev = t2[j++];
	}
	return sum / (double)(n1 + n2 - 1U);
}

static double
_krnl_bjoernstad_falck(double ktij)
{
#define KRNL_WIDTH	(0.25)
	/* -1/(2 h^2)  h being the kernel width */
	static const double _xf = -8.;
	/* 1/sqrt(2PI h) h being the kernel width */
	static const double _vf = 0.7978845608028654;
	return _vf * exp(_xf * ktij * ktij);
#undef KRNL_WIDTH
}

static double
xcf(int lag, ald_t t1[], alf_t y1[], size_t n1, ald_t t2[], alf_t y2[], size_t n2)
{
	double nsum = 0.f;
	double dsum = 0.f;
	/* we combine edelson-krolik rectangle with gauss kernel */
	size_t strt = 0U;
	size_t strk = 0U;

	for (size_t i = 0U; i < n1; i++) {
		const double kti = (double)lag + t1[i];

		/* find start of the interesting window */
		for (; strt < n2 && t2[strt] < kti - 1.1; strt++);
		for (strk = strk < strt ? strt : strk;
		     strk < n2 && t2[strk] < kti + 1.1; strk++);

		for (size_t j = strt; j < strk; j++) {
			double K = _krnl_bjoernstad_falck(lag - (t2[j] - t1[i]));
			dsum += K;
			nsum += y1[i] * y2[j] * K;
		}
	}
	return nsum / dsum;
}

static double
crosscorrirr(ald_t t1[], alf_t y1[], size_t n1, ald_t t2[], alf_t y2[], size_t n2, int nlags)
{
	double tau;
	int best_lag = nlags + 1;

	(void)_norm_f(y1, n1, _quasi_stats_f);
	(void)_norm_f(y2, n2, _quasi_stats_f);

	tau = _mean_tdiff(t1, n1, t2, n2);
	with (register double rtau = 1. / tau) {
		cblas_dscal(n1, rtau, t1, 1U);
		cblas_dscal(n2, rtau, t2, 1U);
	}

	with (double bestx = -INFINITY) {
		for (int k = -nlags; k <= nlags; k++) {
			double x = xcf(k, t1, y1, n1, t2, y2, n2);
			if (UNLIKELY(x > bestx)) {
				bestx = x;
				best_lag = k;
			}
		}
	}
	return (double)best_lag * tau;
}


#if defined STANDALONE
#include "crosscorrirr.yucc"

int
main(int argc, char *argv[])
{
//#	include "crosscorrirr_test_01.c"
#	include "crosscorrirr_test_02.c"
	double x;

	x = crosscorrirr(tx, px, countof(tx), ty, py, countof(py), 80);
	printf("lag\t%g\n", x);
	return 0;
}
#endif	/* STANDALONE */

/* cots-delay.c ends here */
