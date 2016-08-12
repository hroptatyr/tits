/*** xcor.c -- cross correlation for irregular timeseries
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
#include "xcor.h"
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

typedef struct {
	size_t n;
	const double *t __attribute__((aligned(sizeof(__mXd))));
	const double *y __attribute__((aligned(sizeof(__mXd))));
} aldts_t;


static double
_meandiff_d(const double *s, size_t ns)
{
	double sum = 0.;

	if (UNLIKELY(ns-- <= 1U)) {
		return NAN;
	}

	for (size_t i = 0U; i < ns; i++) {
		sum += s[i] - s[i + 1U];
	}
	return -sum / (double)ns;
}

static int
_dscal(double *restrict s, size_t ns, double f)
{
/* calculate f * s */
	register __mXd mf = _mmX_set1_pd(f);

	with (__mXd *restrict ms = (__mXd*)s) {
		for (size_t i = 0U, n = ns / widthof(__mXd); i < n; i++) {
			ms[i] *= mf;
		}
	}
	/* and the glorious rest */
	for (size_t i = ns / widthof(__mXd) * widthof(__mXd); i < ns; i++) {
		s[i] *= f;
	}
	return 0;
}


/* kernel closure */
#define KRNL	_krnl_bjoernstad_falck
#define CLO	paste(KRNL, _clo)
#define SET	paste(KRNL, _set)

static struct {
	double width;
	/* scale exponent */
	double _xf;
	/* total scale */
	double _vf;
} CLO;

#include <stdio.h>
static void
SET(double tau)
{
	/* kernel width */
	const double h = _(0.25) * tau;

	CLO.width = h;
	/* -1/(2 h^2) (exponent scaling) */
	CLO._xf = _(-1.) / (_(2.) * h * h);
	/* 1/sqrt(2PI h) (scaling) */
	CLO._vf = _(1.) / sqrt(_(2.) * M_PI * h);
	return;
}

static double
KRNL(double ktij)
{
	return CLO._vf * exp(CLO._xf * ktij * ktij);
}
#undef CLO
#undef SET
#undef KRNL

static double
dxcf(int lag, aldts_t ts1, aldts_t ts2)
{
#define KRNL	_krnl_bjoernstad_falck
#define CLO	paste(KRNL, _clo)
#define SET	paste(KRNL, _set)
	const double thresh = CLO.width * _(5.);
	double nsum = _(0.);
	double dsum = _(0.);
	/* we combine edelson-krolik rectangle with gauss kernel */
	size_t strt = 0U;
	size_t strk = 0U;

	for (size_t i = 0U; i < ts1.n; i++) {
		const double kti = (double)lag + ts1.t[i];

		/* find start of the interesting window */
		for (; strt < ts2.n && ts2.t[strt] < kti - thresh; strt++);
		for (strk = strk < strt ? strt : strk;
		     strk < ts2.n && ts2.t[strk] < kti + thresh; strk++);

		for (size_t j = strt; j < strk; j++) {
			double K = KRNL(lag - (ts2.t[j] - ts1.t[i]));
			dsum += K;
			nsum += ts1.y[i] * ts2.y[j] * K;
		}
	}
	return nsum / dsum;
}


/* public API */
int
tits_dxcor(double *restrict tgt, dts_t ts1, dts_t ts2, int nlags, double tau)
{
	size_t n1, n2;
	double *t1, *t2;
	double *y1, *y2;

	/* shrink to numbers of elements divisible by the width */
	if (UNLIKELY(!(n1 = ts1.n - ts1.n % widthof(__mXd)))) {
		/* can't have no elements can we? */
		return -1;
	} else if (UNLIKELY(!(n2 = ts2.n - ts2.n % widthof(__mXd)))) {
		/* can't have no elements can we? */
		return -1;
	}

	/* snarf input */
	t1 = _mm_malloc(n1 * sizeof(*t1), sizeof(__mXd));
	y1 = _mm_malloc(n1 * sizeof(*y1), sizeof(__mXd));
	t2 = _mm_malloc(n2 * sizeof(*t2), sizeof(__mXd));
	y2 = _mm_malloc(n2 * sizeof(*y2), sizeof(__mXd));
	memcpy(t1, ts1.t, n1 * sizeof(*t1));
	memcpy(y1, ts1.y, n1 * sizeof(*y1));
	memcpy(t2, ts2.t, n2 * sizeof(*t2));
	memcpy(y2, ts2.y, n2 * sizeof(*y2));

	tits_dnorm(y1, n1);
	tits_dnorm(y2, n2);

	with (double tmd1, tmd2) {
		register double rtau = _(1.) / tau;

		tmd1 = _meandiff_d(t1, n1);
		tmd2 = _meandiff_d(t2, n2);
		/* scale by user tau */
		_dscal(t1, n1, rtau);
		_dscal(t2, n2, rtau);
		/* and set kernel width accordingly */
		SET((tmd1 < tmd2 ? tmd1 : tmd2) * rtau);
	}

	for (int k = -nlags, i = 0; k <= nlags; k++, i++) {
		tgt[i] = dxcf(k, (aldts_t){n1, t1, y1}, (aldts_t){n2, t2, y2});
	}

	_mm_free(t1);
	_mm_free(t2);
	_mm_free(y1);
	_mm_free(y2);
#undef CLO
#undef SET
#undef KRNL
	return 0;
}

#if !defined double
# define double		float
# define dts_t		sts_t
# define aldts_t	alsts_t

# define _dscal		_sscal
# define tits_dnorm	tits_snorm
# define _meandiff_d	_meandiff_s

# define dxcf		sxcf
# define _krnl_bjoernstad_falck	_krnl_bjoernstad_falck_s
# define tits_dxcor	tits_sxcor

/* libm */
# define sqrt		sqrtf
# define exp		expf

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

/* xcor.c ends here */
