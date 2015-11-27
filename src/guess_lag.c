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

#define widthof(x, y)	(sizeof(x) / sizeof(y))
#define widthof(x, y)	(sizeof(x) / sizeof(y))

#if 0
/* AVX-512 */
#define __mX		__m512
#define __mXd		__m512d
#define _mmX_broadcast_ss(x)	_mm512_broadcastss_ps(_mm_load1_ps(x))
#define _mmX_broadcast_sd(x)	_mm512_broadcastsd_pd(_mm_load1_pd(x))
#define _mmX_load_ps	_mm512_load_ps
#define _mmX_load_pd	_mm512_load_pd
#define _mmX_store_ps	_mm512_store_ps
#define _mmX_store_pd	_mm512_store_pd
#define _mmX_add_ps	_mm512_add_ps
#define _mmX_add_pd	_mm512_add_pd
#define _mmX_mul_ps	_mm512_mul_ps
#define _mmX_mul_pd	_mm512_mul_pd
#elif 1
#define __mX		__m256
#define __mXd		__m256d
#define _mmX_broadcast_ss(x)	_mm256_broadcast_ss(x)
#define _mmX_broadcast_sd(x)	_mm256_broadcast_sd(x)
#define _mmX_load_ps	_mm256_load_ps
#define _mmX_load_pd	_mm256_load_pd
#define _mmX_store_ps	_mm256_store_ps
#define _mmX_store_pd	_mm256_store_pd
#define _mmX_add_ps	_mm256_add_ps
#define _mmX_add_pd	_mm256_add_pd
#define _mmX_mul_ps	_mm256_mul_ps
#define _mmX_mul_pd	_mm256_mul_pd
#else
/* plain old SSE */
#define __mX		__m128
#define __mXd		__m128d
#define _mmX_broadcast_ss(x)	_mm_load1_ps(x)
#define _mmX_broadcast_sd(x)	_mm_load1_pd(x)
#define _mmX_load_ps	_mm_load_ps
#define _mmX_load_pd	_mm_load_pd
#define _mmX_store_ps	_mm_store_ps
#define _mmX_store_pd	_mm_store_pd
#define _mmX_add_ps	_mm_add_ps
#define _mmX_add_pd	_mm_add_pd
#define _mmX_mul_ps	_mm_mul_ps
#define _mmX_mul_pd	_mm_mul_pd
#endif

typedef float alf_t __attribute__((aligned(sizeof(__mX))));
typedef double ald_t __attribute__((aligned(sizeof(__mXd))));


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
/* calculate (s - mean(s)) / std(s) */
	float mu, sigma;
	register __mX mmu;
	register __mX msd;

	if (UNLIKELY(statf(&mu, &sigma, s, ns) < 0)) {
		return -1;
	}

	/* go for (s + -mu) * 1/sigma */
	mu = -mu;
	mmu = _mmX_broadcast_ss(&mu);
	sigma = 1.f / sigma;
	msd = _mmX_broadcast_ss(&sigma);
#define WIDTH	(widthof(mmu, mu))
	for (size_t i = 0U; i + WIDTH - 1U < ns; i += WIDTH) {
		register __mX ms;
		ms = _mmX_load_ps(s + i);
		ms = _mmX_add_ps(ms, mmu);
		ms = _mmX_mul_ps(ms, msd);
		_mmX_store_ps(s + i, ms);
	}
#undef WIDTH
	return 0;
}

static int
_dscal(ald_t s[], size_t ns, double f)
{
/* calculate f * s */
	register __mXd mf = _mmX_broadcast_sd(&f);

#define WIDTH	(widthof(mf, f))
	for (size_t i = 0U; i + WIDTH - 1U < ns; i += WIDTH) {
		register __mXd ms;
		ms = _mmX_load_pd(s + i);
		ms = _mmX_mul_pd(ms, mf);
		_mmX_store_pd(s + i, ms);
	}
#undef WIDTH
	return 0;
}


static long int
_gcd_l(long int x, long int y)
{
/* euklid's gcd */
	while (y) {
		long int tmp = y;
		y = x % y;
		x = tmp;
	}
	return x;
}

static double
_gcd_d(double x, double y)
{
/* euklid's gcd for quotients */
	/* separate X into XI + 1/XD and Y into YI + 1/YD
	 * then use long int gcd */
	double xi = trunc(x);
	double yi = trunc(y);
	long int xd = 1;
	long int yd = 1;
	long int xn;
	long int yn;
	long int dd;
	long int dn;

	x -= xi;
	y -= yi;

	xn = (long int)xi;
	yn = (long int)yi;

	/* as a proxy for XD we use trunc(XD) */
	if (x > __DBL_EPSILON__) {
		xd = (long int)(1. / x);
		xn *= xd;
		xn++;
	}
	if (y > __DBL_EPSILON__) {
		yd = (long int)(1. / y);
		yn *= yd;
		yn++;
	}
	/* yay, we can finally calc the lcm denominator */
	dd = (xd * yd) / _gcd_l(xd, yd);
	/* now fiddle with the numerators */
	xn *= dd / xd;
	yn *= dd / yd;

	/* finally gcd the numerators and we're done */
	dn = _gcd_l(xn, yn);
	return (double)dn / (double)dd;
}

static double
_mean_tdiff(const ald_t t1[], size_t n1, const ald_t t2[], size_t n2)
{
/* calculate average time differences
 * gcd(mean(t1[i] - t1[i-1]), mean(t2[j], t2[j - 1])) */
	double tau1, tau2;
	double sum;

	sum = 0.;
	for (size_t i = 1U; i < n1; i++) {
		sum += t1[i] - t1[i - 1];
	}
	tau1 = sum / (double)(n1 - 1U);

	sum = 0.;
	for (size_t j = 1U; j < n2; j++) {
		sum += t2[j] - t2[j - 1];
	}
	tau2 = sum / (double)(n2 - 1U);

	return _gcd_d(tau1, tau2);
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
	int best_lag[4U] = {nlags + 1, nlags + 1, nlags + 1, nlags + 1};
	int least_best;

	(void)_norm_f(y1, n1, _quasi_stats_f);
	(void)_norm_f(y2, n2, _quasi_stats_f);

	tau = _mean_tdiff(t1, n1, t2, n2);
	with (register double rtau = 1. / tau) {
		_dscal(t1, n1, rtau);
		_dscal(t2, n2, rtau);
	}

	with (double bestx[4U] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}) {
		for (int k = -nlags; k <= nlags; k++) {
			double x = xcf(k, t1, y1, n1, t2, y2, n2);

			if (x < 1e-5) {
				continue;
			} else if (UNLIKELY(x > bestx[3U])) {
				bestx[0U] = bestx[1U];
				best_lag[0U] = best_lag[1U];

				bestx[1U] = bestx[2U];
				best_lag[1U] = best_lag[2U];

				bestx[2U] = bestx[3U];
				best_lag[2U] = best_lag[3U];

				bestx[3U] = x;
				best_lag[3U] = k;
			} else if (UNLIKELY(x > bestx[2U])) {
				bestx[0U] = bestx[1U];
				best_lag[0U] = best_lag[1U];

				bestx[1U] = bestx[2U];
				best_lag[1U] = best_lag[2U];

				bestx[2U] = x;
				best_lag[2U] = k;
			} else if (UNLIKELY(x > bestx[1U])) {
				bestx[0U] = bestx[1U];
				best_lag[0U] = best_lag[1U];

				bestx[1U] = x;
				best_lag[1U] = k;
			} else if (UNLIKELY(x > bestx[0U])) {
				bestx[0U] = x;
				best_lag[0U] = k;
			}
		}
	}
	least_best = nlags + 1;
	if (best_lag[0U] < least_best) {
		least_best = best_lag[0U];
	}
	if (best_lag[1U] < least_best) {
		least_best = best_lag[1U];
	}
	if (best_lag[2U] < least_best) {
		least_best = best_lag[2U];
	}
	if (best_lag[3U] < least_best) {
		least_best = best_lag[3U];
	}
	return least_best < nlags + 1 ? (double)least_best * tau : NAN;
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
#define MAX_TICKS	(256U)
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
		t[i] = (double)((long)b.t[i] - (long)tref) / (double)NSECS;
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
				printf("PREP %s\n", src[i]);
				prep_book(t1, p1, book[i].bid, tref);
				iinitp = true;
			}
			/* yep, do a i,j lead/lag run */
				printf("PREP %s\n", src[j]);
			prep_book(t2, p2, book[j].bid, tref);

			lag = crosscorrirr(t1, p1, n1, t2, p2, n2, 20);
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

			lag = crosscorrirr(t1, p1, n1, t2, p2, n2, 20);
			printf("%lu.%09lu\tASK\t%s\t%s\t%g\n",
			       metr / NSECS, metr % NSECS,
			       src[i], src[j], lag);
		}
	}
	return;
}
#endif	/* STANDALONE */


#if defined STANDALONE
#include "guess_lag.yucc"

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
