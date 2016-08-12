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
#elif 1
#define __mXd		__m256d
#define __mXs		__m256
#define _mmX_set1_pd(x)	_mm256_set1_pd(x)
#define _mmX_set1_ps(x)	_mm256_set1_ps(x)
#else
/* plain old SSE */
#define __mXd		__m128d
#define __mXs		__m128
#define _mmX_set1_pd(x)	_mm_set1_pd(x)
#define _mmX_set1_ps(x)	_mm_set1_ps(x)
#endif
#endif	/* !__mXd */


static int
_stats_d(
	double *restrict mean, double *restrict svar,
	const double *src, size_t nsrc)
{
/* always assume we can at least overread SRC by widthof(__mXd)!! */
	const __mXd *Xsrc = (const __mXd*)src;
	__mXd m1 = {};
	__mXd m2 = {};

	/* always good to initialise things */
	*mean = 0;
	*svar = 0;

	if (UNLIKELY(!(nsrc / widthof(__mXd)))) {
		goto rest;
	}
	for (size_t i = 0U, n = nsrc / widthof(__mXd); i < n; i++) {
		register __mXd x = Xsrc[i];
		register __mXd dlt = x - m1;

		m1 += dlt / _mmX_set1_pd((double)(i + 1U));
		m2 += dlt * (x - m1);
	}
	/* combine all the m1 and m2 moments
	 * combining goes as follows:
	 * m_0,(1+...+k) = m_0,1 + ... + m_0,k  (assumed to be k*m0)
	 * m_1,(1+...+k) = (m_1,1 + ... + m_1,k) / k
	 * m_2,(1+2) = m_2,1 + ... + m_2,k +
	 *             n/k * ((k-1)*(m_1,1^2 + ... + m_1,k^2)
	 *                    -2 m_1,i m_1,j  (for j > i)) */
	/* add up the means */
	for (size_t i = 0U; i < widthof(__mXd); i++) {
		*mean += m1[i];
	}
	*mean /= (double)widthof(__mXd);

	/* start adding the variances, -2 x_i x_j  for j>i */
	for (size_t i = 0U; i < widthof(__mXd); i++) {
		for (size_t j = i + 1U; j < widthof(__mXd); j++) {
			*svar -= 2 * m1[i] * m1[j];
		}
	}

	/* mtmp <- (k-1) * m1 * m1 */
	with (__mXd mtmp = m1 * m1) {
		mtmp *= _mmX_set1_pd((double)(widthof(__mXd) - 1U));
		/* and add them, sm1 is garbage from here on */
		for (size_t i = 0U; i < widthof(__mXd); i++) {
			*svar += mtmp[i];
		}
	}

	/* SUM <- n/k SUM */
	*svar *= (double)(nsrc / widthof(__mXd));
	*svar /= (double)widthof(__mXd);

	/* now add on top the actual m2's */
	for (size_t i = 0U; i < widthof(__mXd); i++) {
		*svar += m2[i];
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
	with (__mXd *restrict Xts = (__mXd*)tgtsrc) {
		for (size_t i = 0U, n = nsrc / widthof(__mXd); i < n; i++) {
			register __mXd ms = Xts[i];
			ms += mmu;
			ms *= msd;
			Xts[i] = ms;
		}
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
# define __mXd		__mXs
# define _mmX_set1_pd	_mmX_set1_ps

# undef _
# define _(x)		(x ## f)

/* now go through it again for single precision */
# include __FILE__
#endif	/* !double */

/* norm.c ends here */
