/*** norm.c -- destructive normalisation
 *
 * Copyright (C) 2015-2016 Sebastian Freundt
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
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>
#include "norm.h"
#include "nifty.h"

#define widthof(x)	(sizeof(x) / sizeof(double))
#if !defined _
# define _(x)		(x)
#endif	/* !_ */

#if !defined __mXd
#if 0
/* AVX-512 */
#define __mXd		__m512d
#define __mXs		__m512
#define _mmX_set1_pd(x)	_mm512_set1_pd(x)
#define _mmX_set1_ps(x)	_mm512_set1_ps(x)
#define _mmX_load_pd	_mm512_load_pd
#define _mmX_load_ps	_mm512_load_ps
#define _mmX_store_pd	_mm512_store_pd
#define _mmX_store_ps	_mm512_store_ps
#elif 1
#define __mXd		__m256d
#define __mXs		__m256
#define _mmX_set1_pd(x)	_mm256_set1_pd(x)
#define _mmX_set1_ps(x)	_mm256_set1_ps(x)
#define _mmX_load_pd	_mm256_load_pd
#define _mmX_load_ps	_mm256_load_ps
#define _mmX_store_pd	_mm256_store_pd
#define _mmX_store_ps	_mm256_store_ps
#else
/* plain old SSE */
#define __mXd		__m128d
#define __mXs		__m128
#define _mmX_set1_pd(x)	_mm_set1_pd(x)
#define _mmX_set1_ps(x)	_mm_set1_ps(x)
#define _mmX_load_pd	_mm_load_pd
#define _mmX_load_ps	_mm_load_ps
#define _mmX_store_pd	_mm_store_pd
#define _mmX_store_ps	_mm_store_ps
#endif
#endif	/* !__mXd */


static int
_stats_d(
	double *restrict mean, double *restrict svar,
	const double *src, size_t nsrc)
{
/* always assume we can at least overread SRC by widthof(__mXd)!! */
	__mXd m1 = {};
	__mXd m2 = {};

	/* always good to initialise things */
	*mean = 0;
	*svar = 0;

	if (UNLIKELY(!(nsrc / widthof(__mXd)))) {
		goto rest;
	}
	for (size_t i = 0U, n = nsrc / widthof(__mXd); i < n; i++) {
		register __mXd x = _mmX_load_pd(src + i * widthof(__mXd));
		register __mXd dlt = x - m1;

		m1 += dlt / _mmX_set1_pd((double)(i + 1U));
		m2 += dlt * (x - m1);
	}
	/* combine all the m1 and m2 moments */
	with (double sm1[widthof(__mXd)], sm2[widthof(__mXd)]) {
		_mmX_store_pd(sm1, m1);
		_mmX_store_pd(sm2, m2);

		/* add up the means */
		for (size_t i = 0U; i < widthof(__mXd); i++) {
			*mean += sm1[i];
		}
		*mean /= (double)widthof(__mXd);

		/* start adding the variances */
		/* make use of the fact that the mu's are still in sm1 */
		for (size_t i = 0U; i < widthof(__mXd); i++) {
			for (size_t j = i + 1U; j < widthof(__mXd); j++) {
				*svar += -2 * sm1[i] * sm1[j];
			}
		}

		/* mtmp <- (k-1) * m1 * m1 */
		with (__mXd mtmp = m1 * m1) {
			mtmp *= _mmX_set1_pd((double)(widthof(__mXd) - 1U));
			_mmX_store_pd(sm1, mtmp);
		}
		/* and add them, sm1 is garbage from here on */
		for (size_t i = 0U; i < widthof(__mXd); i++) {
			*svar += sm1[i];
		}

		/* SUM <- n/k SUM */
		*svar *= (double)(nsrc / widthof(__mXd));
		*svar /= (double)widthof(__mXd);

		/* now add on top the actual m2's */
		for (size_t i = 0U; i < widthof(__mXd); i++) {
			*svar += sm2[i];
		}
	}
rest:
	for (size_t i = nsrc / widthof(__mXd) * widthof(__mXd); i < nsrc; i++) {
		double numean = *mean + (src[i] - *mean) / (double)(i + 1U);

		*svar += (src[i] - *mean) * (src[i] - numean);
		*mean = numean;
	}

	/* normalise */
	*svar /= (double)(nsrc - 1U);
	return 0;
}


/* public API */
int
tits_dnorm(double *restrict tgtsrc, size_t nsrc)
{
/* calculate X <- (X - mean(X)) / sdev(X)  for X == tgtsrc */
	double mean, svar, sdev;
	register __mXd mmu;
	register __mXd msd;

	if (UNLIKELY(_stats_d(&mean, &svar, tgtsrc, nsrc) < 0)) {
		return -1;
	}

	/* go for (s + -mu) * 1/sigma */
	mean = -mean;
	mmu = _mmX_set1_pd(mean);
	sdev = _(1.) / sqrt(svar);
	msd = _mmX_set1_pd(sdev);
	for (size_t i = 0U, n = nsrc / widthof(__mXd); i < n; i++) {
		register __mXd ms;
		ms = _mmX_load_pd(tgtsrc + i * widthof(__mXd));
		ms += mmu;
		ms *= msd;
		_mmX_store_pd(tgtsrc + i * widthof(__mXd), ms);
	}
	for (size_t i = nsrc / widthof(__mXd) * widthof(__mXd); i < nsrc; i++) {
		tgtsrc[i] += mean;
		tgtsrc[i] *= sdev;
	}
	return 0;
}

#if !defined double
# define double		float

# define _stats_d	_stats_s

# define tits_dnorm	tits_snorm

/* libm */
# define sqrt		sqrtf

/* intrins */
# undef __mXd
# undef _mmX_set1_pd
# undef _mmX_load_pd
# undef _mmX_store_pd
# undef _mmX_add_pd
# undef _mmX_sub_pd
# undef _mmX_mul_pd
# undef _mmX_div_pd
# define __mXd		__mXs
# define _mmX_set1_pd	_mmX_set1_ps
# define _mmX_load_pd	_mmX_load_ps
# define _mmX_store_pd	_mmX_store_ps
# define _mmX_add_pd	_mmX_add_ps
# define _mmX_sub_pd	_mmX_sub_ps
# define _mmX_mul_pd	_mmX_mul_ps
# define _mmX_div_pd	_mmX_div_ps

# undef _
# define _(x)		(x ## f)

/* now go through it again for single precision */
# include __FILE__
#endif	/* !double */

/* norm.c ends here */
