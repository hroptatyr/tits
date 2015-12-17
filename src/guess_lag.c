#if defined HAVE_CONFIG_H
# include "config.h"
#endif	/* HAVE_CONFIG_H */
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined HAVE_DFP754_H
# include <dfp754.h>
#elif defined HAVE_DFP_STDLIB_H
# include <dfp/stdlib.h>
#endif	/* HAVE_DFP754_H */
#include "hash.h"
#include "nifty.h"
#include "xcor.h"

#define NSECS	(1000000000)

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
prep_book(double *restrict t, double *restrict p, book_t b, tv_t tref)
{
	for (size_t i = 0U; i < b.n; i++) {
		t[i] = (double)((long)b.t[i] - (long)tref) / (double)NSECS;
		p[i] = (double)b.p[i];
	}
	return;
}

static void
skim(void)
{
#define NLAGS		(512U)
#define EDG_TICKS	(3U * MAX_TICKS / 4U)
#define LOW_TICKS	(2U * MAX_TICKS / 4U)
	static double t1[MAX_TICKS];
	static double p1[MAX_TICKS];
	static double t2[MAX_TICKS];
	static double p2[MAX_TICKS];

	for (size_t i = 0U; i < nsrc; i++) {
		size_t n1;
		tv_t tref;

		if ((n1 = book[i].bid.n) != EDG_TICKS) {
			continue;
		}
		for (size_t j = 0U; j < nsrc; j++) {
			double lags[2U * NLAGS + 1U];
			size_t n2;

			if (i == j) {
				continue;
			} else if (j < i && book[j].bid.n == EDG_TICKS) {
				continue;
			} else if ((n2 = book[j].bid.n) < LOW_TICKS) {
				continue;
			}
			/* yep, do a i,j lead/lag run */
			tref = book[i].bid.t[0U];
			prep_book(t1, p1, book[i].bid, tref);
			prep_book(t2, p2, book[j].bid, tref);

			cots_dxcor(
				lags,
				(dts_t){n1, t1, p1}, (dts_t){n2, t2, p2},
				NLAGS, 0.001);

			printf("%lu.%09lu\tBID\t%s\t%s\n",
			       metr / NSECS, metr % NSECS, src[i], src[j]);
			for (size_t k = 0U; k < countof(lags); k++) {
				printf("\t%d\t%g\n",
				       (int)k - (int)NLAGS, lags[k]);
			}
		}
	}
#if 0
	for (size_t i = 0U; i < nsrc; i++) {
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
			}
			/* yep, do a i,j lead/lag run */
			tref = book[i].ask.t[0U];
			prep_book(t1, p1, book[i].ask, tref);
			prep_book(t2, p2, book[j].ask, tref);

			lag = crosscorrirr(t1, p1, n1, t2, p2, n2, 2000, 0.001);
			printf("%lu.%09lu\tASK\t%s\t%s\t%g\n",
			       metr / NSECS, metr % NSECS,
			       src[i], src[j], lag);
		}
	}
#endif
	return;
}


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

/* crosscorrirr.c ends here */
