#if defined HAVE_CONFIG_H
# include "config.h"
#endif	/* HAVE_CONFIG_H */
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <mkl.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>
#include <immintrin.h>
#include "nifty.h"

#define NSECS	(1000000000)

typedef float alf_t __attribute__((aligned(16)));
typedef double ald_t __attribute__((aligned(16)));


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
#include <string.h>
#include <stdio.h>
#if defined HAVE_DFP754_H
# include <dfp754.h>
#elif defined HAVE_DFP_STDLIB_H
# include <dfp/stdlib.h>
#endif	/* HAVE_DFP754_H */
#include "hash.h"

typedef long unsigned int tv_t;
typedef _Decimal32 px_t;
#define strtopx		strtod32

typedef struct {
	size_t n;
	tv_t *t;
	px_t *p;
} book_t;

static size_t zsrc;
static size_t nsrc;
static const char **src;
static struct {
	book_t bid;
	book_t ask;
} *book;
static tv_t metr;

static book_t
make_book(void)
{
#define MAX_TICKS	(4096U)
	book_t res = {
		.n = 0U,
		.t = calloc(MAX_TICKS, sizeof(*res.t)),
		.p = calloc(MAX_TICKS, sizeof(*res.p)),
	};
	return res;
}

static book_t
book_slid(book_t b)
{
	size_t nun = (b.n + 1U) / 2U;

	b.n -= nun;
	memcpy(b.t, b.t + nun, b.n * sizeof(*b.t));
	memcpy(b.p, b.p + nun, b.n * sizeof(*b.p));
	return b;
}

static inline bool
book_full_p(book_t b)
{
	return b.n >= MAX_TICKS;
}

static void
push(const char *line, size_t UNUSED(llen))
{
	static struct {
		hx_t h;
		size_t i;
	} *hxs;
	size_t k;
	px_t b;
	px_t a;
	char *on;

	/* time value up first */
	with (long unsigned int s, x) {
		if (line[20U] != '\t') {
			return;
		} else if (!(s = strtoul(line, &on, 10))) {
			return;
		} else if (*on++ != '.') {
			return;
		} else if ((x = strtoul(on, &on, 10), *on != '\t')) {
			return;
		}
		metr = s * NSECS + x;
	}

	/* now comes the ECN */
	with (const char *ecn = ++on) {
		size_t hk;
		hx_t hx;

		if (UNLIKELY((on = strchr(ecn, '\t')) == NULL)) {
			return;
		} else if (UNLIKELY(!(hx = hash(ecn, on - ecn)))) {
			/* fuck */
			return;
		} else if (UNLIKELY(!zsrc)) {
			zsrc = 16U;
			hxs = calloc(zsrc, sizeof(*hxs));
			src = calloc(zsrc, sizeof(*src));
			book = calloc(zsrc, sizeof(*book));
		}
	ass:
		/* assign slot */
		hk = hx % zsrc;
		if (UNLIKELY(!hxs[hk].h)) {
			/* empty, phew */
			hxs[hk] = (typeof(*hxs)){hx, k = nsrc++};
			src[k] = strndup(ecn, on - ecn);
			book[k].bid = make_book();
			book[k].ask = make_book();
		} else if (LIKELY(hxs[hk].h == hx)) {
			/* all's well */
			k = hxs[hk].i;
		} else if (UNLIKELY((hk = (hk + 1U) % zsrc, !hxs[hk].h))) {
			/* lucky again, innit? */
			hxs[hk] = (typeof(*hxs)){hx, k = nsrc++};
			src[k] = strndup(ecn, on - ecn);
			book[k].bid = make_book();
			book[k].ask = make_book();
		} else if (LIKELY(hxs[hk].h == hx)) {
			/* okey doke */
			k = hxs[hk].i;
		} else if (LIKELY(zsrc *= 2U)) {
			/* resize */
			typeof(hxs) nuh = calloc(zsrc, sizeof(*hxs));
			typeof(src) nuv = calloc(zsrc, sizeof(*src));
			typeof(book) nub = calloc(zsrc, sizeof(*book));

			for (size_t i = 0U; i < zsrc / 2U; i++) {
				const hx_t oh = hxs[i].h;
				size_t j = oh % zsrc;

				if (LIKELY(!oh)) {
					continue;
				} else if (!nuh[j].h) {
					;
				} else if ((j = (j + 1U) % zsrc, !nuh[j].h)) {
					;
				} else {
					/* what now? */
					abort();
				}
				nuh[j] = hxs[i];
			}
			free(hxs);
			hxs = nuh;

			memcpy(nuv, src, nsrc * sizeof(*nuv));
			memcpy(nub, book, nsrc * sizeof(*nub));
			free(src);
			free(book);
			src = nuv;
			book = nub;
			goto ass;
		}
	}

	if (*++on != '\t' && (b = strtopx(on, &on))) {
#define BOOK	(book[k].bid)
		if (UNLIKELY(book_full_p(BOOK))) {
			BOOK = book_slid(BOOK);
		}
		BOOK.t[BOOK.n] = metr;
		BOOK.p[BOOK.n] = b;
		BOOK.n++;
#undef BOOK
	}
	if (*++on != '\t' && (a = strtopx(on, &on))) {
#define BOOK	(book[k].ask)
		if (UNLIKELY(book_full_p(BOOK))) {
			BOOK = book_slid(BOOK);
		}
		BOOK.t[BOOK.n] = metr;
		BOOK.p[BOOK.n] = a;
		BOOK.n++;
#undef BOOK
	}
	return;
}

static void
prep_book(ald_t t[], alf_t p[], book_t b, tv_t tref)
{
	for (size_t i = 0U; i < b.n; i++) {
		t[i] = (double)(b.t[i] - tref) / (double)NSECS;
		p[i] = (float)b.p[i];
	}
	return;
}

static void
skim(void)
{
#define EDG_TICKS	(3U * MAX_TICKS / 4U)
#define LOW_TICKS	(2U * MAX_TICKS / 4U)
	static ald_t t1[MAX_TICKS];
	static alf_t p1[MAX_TICKS];
	static ald_t t2[MAX_TICKS];
	static alf_t p2[MAX_TICKS];

	for (size_t i = 0U; i < nsrc; i++) {
		bool iinitp = false;
		size_t n1;
		tv_t tref;

		if ((n1 = book[i].bid.n) != EDG_TICKS) {
			continue;
		}
		for (size_t j = 0U; j < nsrc; j++) {
			double lag;
			size_t n2;

			if (i == j) {
				continue;
			} else if (j < i && book[j].bid.n == EDG_TICKS) {
				continue;
			} else if ((n2 = book[j].bid.n) < LOW_TICKS) {
				continue;
			} else if (!iinitp) {
				tref = book[i].bid.t[0U];
				prep_book(t1, p1, book[i].bid, tref);
				iinitp = true;
			}
			/* yep, do a i,j lead/lag run */
			prep_book(t2, p2, book[j].bid, tref);

			lag = crosscorrirr(t1, p1, n1, t2, p2, n2, 64);
			printf("%lu.%09lu\tBID\t%s\t%s\t%g\n",
			       metr / NSECS, metr % NSECS,
			       src[i], src[j], lag);
		}
	}

	for (size_t i = 0U; i < nsrc; i++) {
		bool iinitp = false;
		size_t n1;
		tv_t tref;

		if ((n1 = book[i].ask.n) != EDG_TICKS) {
			continue;
		}
		for (size_t j = 0U; j < nsrc; j++) {
			double lag;
			size_t n2;

			if (i == j) {
				continue;
			} else if (j < i && book[j].ask.n == EDG_TICKS) {
				continue;
			} else if ((n2 = book[j].ask.n) < LOW_TICKS) {
				continue;
			} else if (!iinitp) {
				tref = book[i].ask.t[0U];
				prep_book(t1, p1, book[i].ask, tref);
				iinitp = true;
			}
			/* yep, do a i,j lead/lag run */
			prep_book(t2, p2, book[j].ask, tref);

			lag = crosscorrirr(t1, p1, n1, t2, p2, n2, 64);
			printf("%lu.%09lu\tASK\t%s\t%s\t%g\n",
			       metr / NSECS, metr % NSECS,
			       src[i], src[j], lag);
		}
	}
	return;
}
#endif	/* STANDALONE */


#if defined STANDALONE
#include "crosscorrirr.yucc"

int
main(int argc, char *argv[])
{
	static yuck_t argi[1U];
	char *line = NULL;
	size_t llen = 0U;
	int rc = 0;

	if (yuck_parse(argi, argc, argv) < 0) {
		rc = 1;
		goto out;
	}

	for (ssize_t nrd; (nrd = getline(&line, &llen, stdin)) > 0;) {
		push(line, nrd);
		skim();
	}
	free(line);
out:
	yuck_free(argi);
	return rc;
}
#endif	/* STANDALONE */

/* crosscorrirr.c ends here */
