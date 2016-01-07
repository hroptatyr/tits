/*** guess_lag.h -- guess lag between time series
 *
 * Copyright (C) 2014-2015 Sebastian Freundt
 *
 * Author:  Sebastian Freundt <freundt@ga-group.nl>
 *
 * This file is part of tits.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the author nor the names of any contributors
 *    may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ***/
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
static union {
	struct {
		book_t bid;
		book_t ask;
	};
	book_t quo[2U];
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
	size_t nun = b.n - MAX_TICKS / 2U;

	b.n = MAX_TICKS / 2U;
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
	for (size_t i = 1U; i < b.n; i++) {
		t[i] = (double)((long)b.t[i] - (long)tref) / (double)NSECS;
		p[i] = (double)(b.p[i] - b.p[i - 1U]);
	}
	return;
}

static void
skim(bool best_lag_p)
{
#define NLAGS		(256U)
#define EDG_TICKS	(3U * MAX_TICKS / 4U + 1U)
#define LOW_TICKS	(2U * MAX_TICKS / 4U + 1U)
#define BID		0U
#define ASK		1U
	static const char *quostr[2U] = {"BID", "ASK"};
	static double t1[MAX_TICKS];
	static double p1[MAX_TICKS];
	static double t2[MAX_TICKS];
	static double p2[MAX_TICKS];
	const double tau = 0.01;
	size_t s = BID;

do_it:
	for (size_t i = 0U; i < nsrc; i++) {
		bool booki_prepped_p = false;
		size_t n1;
		tv_t tref;

		if ((n1 = book[i].quo[s].n) != EDG_TICKS) {
			continue;
		}
		for (size_t j = 0U; j < nsrc; j++) {
			double lags[2U * NLAGS + 1U];
			size_t n2;

			if (i == j) {
				continue;
			} else if ((n2 = book[j].quo[s].n) < LOW_TICKS) {
				continue;
			} else if (!booki_prepped_p) {
				tref = book[i].quo[s].t[0U];
				prep_book(t1, p1, book[i].quo[s], tref);
				booki_prepped_p = true;
			}
			/* yep, do a i,j lead/lag run */
			prep_book(t2, p2, book[j].quo[s], tref);

			n1--;
			n2--;
			tits_dxcor(
				lags,
				(dts_t){n1, t1, p1}, (dts_t){n2, t2, p2},
				NLAGS, tau);

			if (best_lag_p) {
				size_t bestl = 0U;
				double bestx = lags[0U];
				double lt;

				for (size_t k = 1U; k < countof(lags); k++) {
					if (lags[k] > bestx) {
						bestx = lags[k];
						bestl = k;
					}
				}
				if (UNLIKELY(isnan(bestx))) {
					lt = NAN;
				} else {
					lt = ((int)bestl - (int)NLAGS) * tau;
				}
				printf("%lu.%09lu\t%s\t%s\t%s\t%g\t%g\n",
				       metr / NSECS, metr % NSECS,
				       quostr[s], src[i], src[j], lt, bestx);
			} else {
				/* just print them all */
				printf("%lu.%09lu\t%s\t%s\t%s\t%g\n",
				       metr / NSECS, metr % NSECS,
				       quostr[s], src[i], src[j], tau);
				for (size_t k = 0U; k < countof(lags); k++) {
					int lag = (int)k - (int)NLAGS;
					double lt = (double)lag * tau;
					printf("\t%g\t%g\n", lt, lags[k]);
				}
			}
		}
	}

	/* crop the guys we've just done so we won't do them again
	 * just because another tick has been added to the counter-series */
	for (size_t i = 0U; i < nsrc; i++) {
		if (book[i].quo[s].n == EDG_TICKS) {
			book[i].quo[s] = book_slid(book[i].quo[s]);
		}
	}

	if (s++ < ASK) {
		goto do_it;
	}
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
		skim(argi->best_flag);
	}
	free(line);
out:
	yuck_free(argi);
	return rc;
}

/* guess_lag.c ends here */
