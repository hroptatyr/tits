/*** sacf.c -- sample auto-correlation for irregular timeseries
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
#include "sacf.h"
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
#define _mmX_broadcast_sd(x)	_mm512_broadcastsd_pd(_mm_load1_pd(x))
#define _mmX_broadcast_ss(x)	_mm512_broadcastss_ps(_mm_load1_ps(x))
#define _mmX_load_pd	_mm512_load_pd
#define _mmX_load_ps	_mm512_load_ps
#define _mmX_store_pd	_mm512_store_pd
#define _mmX_store_ps	_mm512_store_ps
#define _mmX_add_pd	_mm512_add_pd
#define _mmX_add_ps	_mm512_add_ps
#define _mmX_mul_pd	_mm512_mul_pd
#define _mmX_mul_ps	_mm512_mul_ps
#elif 1
#define __mXd		__m256d
#define __mXs		__m256
#define _mmX_broadcast_sd(x)	_mm256_broadcast_sd(x)
#define _mmX_broadcast_ss(x)	_mm256_broadcast_ss(x)
#define _mmX_load_pd	_mm256_load_pd
#define _mmX_load_ps	_mm256_load_ps
#define _mmX_store_pd	_mm256_store_pd
#define _mmX_store_ps	_mm256_store_ps
#define _mmX_add_pd	_mm256_add_pd
#define _mmX_add_ps	_mm256_add_ps
#define _mmX_mul_pd	_mm256_mul_pd
#define _mmX_mul_ps	_mm256_mul_ps
#else
/* plain old SSE */
#define __mXd		__m128d
#define __mXs		__m128
#define _mmX_broadcast_sd(x)	_mm_load1_pd(x)
#define _mmX_broadcast_ss(x)	_mm_load1_ps(x)
#define _mmX_load_pd	_mm_load_pd
#define _mmX_load_ps	_mm_load_ps
#define _mmX_store_pd	_mm_store_pd
#define _mmX_store_ps	_mm_store_ps
#define _mmX_add_pd	_mm_add_pd
#define _mmX_add_ps	_mm_add_ps
#define _mmX_mul_pd	_mm_mul_pd
#define _mmX_mul_ps	_mm_mul_ps
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
	register __mXd mf = _mmX_broadcast_sd(&f);

	for (size_t i = 0U; i + widthof(__mXd) - 1U < ns; i += widthof(__mXd)) {
		register __mXd ms;
		ms = _mmX_load_pd(s + i);
		ms = _mmX_mul_pd(ms, mf);
		_mmX_store_pd(s + i, ms);
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
dacf(size_t lag, aldts_t ts)
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

	for (size_t i = 0U; i < ts.n; i++) {
		const double kti = (double)lag + ts.t[i];

		/* find start of the interesting window */
		for (; strt < ts.n && ts.t[strt] < kti - thresh; strt++);
		for (strk = strk < strt ? strt : strk;
		     strk < ts.n && ts.t[strk] < kti + thresh; strk++);

		for (size_t j = strt; j < strk; j++) {
			double K = KRNL(lag - (ts.t[j] - ts.t[i]));
			dsum += K;
			nsum += ts.y[i] * ts.y[j] * K;
		}
	}
	return nsum / dsum;
}


/* public API */
int
tits_dsacf(double *restrict tgt, dts_t ts, size_t nlags, double tau)
{
	size_t n;
	double *t;
	double *y;

	/* shrink to numbers of elements divisible by the width */
	if (UNLIKELY(!(n = ts.n - ts.n % widthof(__mXd)))) {
		/* can't have no elements can we? */
		return -1;
	}

	/* snarf input */
	t = _mm_malloc(n * sizeof(*t), sizeof(__mXd));
	y = _mm_malloc(n * sizeof(*y), sizeof(__mXd));
	memcpy(t, ts.t, n * sizeof(*t));
	memcpy(y, ts.y, n * sizeof(*y));

	tits_dnorm(y, n);

	with (double tmd) {
		register double rtau = _(1.) / tau;

		tmd = _meandiff_d(t, n);
		/* scale by user tau */
		_dscal(t, n, rtau);
		/* and set kernel width accordingly */
		SET(tmd * rtau);
	}

	for (size_t k = 0U; k < nlags; k++) {
		tgt[k] = dacf(k + 1U, (aldts_t){n, t, y});
	}

	_mm_free(t);
	_mm_free(y);
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

# define dacf		sacf
# define _krnl_bjoernstad_falck	_krnl_bjoernstad_falck_s
# define tits_dsacf	tits_ssacf

/* libm */
# define sqrt		sqrtf
# define exp		expf

/* intrins */
# undef __mXd
# undef _mmX_broadcast_sd
# undef _mmX_load_pd
# undef _mmX_store_pd
# undef _mmX_add_pd
# undef _mmX_mul_pd
# define __mXd		__mXs
# define _mmX_broadcast_sd	_mmX_broadcast_ss
# define _mmX_load_pd	_mmX_load_ps
# define _mmX_store_pd	_mmX_store_ps
# define _mmX_add_pd	_mmX_add_ps
# define _mmX_mul_pd	_mmX_mul_ps

# undef _
# define _(x)		(x ## f)

/* now go through it again for single precision */
# include __FILE__
#endif	/* !double */

/* sacf.c ends here */
