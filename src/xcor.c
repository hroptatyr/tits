/*** xcor.c -- cross correlation for irregular timeseries
 *
 * Copyright (C) 2015-2016 Sebastian Freundt
 *
 * Author:  Sebastian Freundt <freundt@ga-group.nl>
 *
 * This file is part of cotse.
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
#include <mkl.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>
#include <immintrin.h>
#include "xcor.h"
#include "nifty.h"

#define widthof(x)	(sizeof(x) / sizeof(double))

#if 0
/* AVX-512 */
#define __mXd		__m512d
#define _mmX_broadcast_sd(x)	_mm512_broadcastsd_pd(_mm_load1_pd(x))
#define _mmX_load_pd	_mm512_load_pd
#define _mmX_store_pd	_mm512_store_pd
#define _mmX_add_pd	_mm512_add_pd
#define _mmX_mul_pd	_mm512_mul_pd
#elif 1
#define __mXd		__m256d
#define _mmX_broadcast_sd(x)	_mm256_broadcast_sd(x)
#define _mmX_load_pd	_mm256_load_pd
#define _mmX_store_pd	_mm256_store_pd
#define _mmX_add_pd	_mm256_add_pd
#define _mmX_mul_pd	_mm256_mul_pd
#else
/* plain old SSE */
#define __mXd		__m128d
#define _mmX_broadcast_sd(x)	_mm_load1_pd(x)
#define _mmX_load_pd	_mm_load_pd
#define _mmX_store_pd	_mm_store_pd
#define _mmX_add_pd	_mm_add_pd
#define _mmX_mul_pd	_mm_mul_pd
#endif

typedef double ald_t __attribute__((aligned(sizeof(__mXd))));

typedef struct {
	size_t n;
	const ald_t *t;
	const ald_t *y;
} aldts_t;


static double
_meandiff_d(const ald_t s[], size_t ns)
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

static __attribute__((unused)) int
_stats_d(
	double mean[static 1U], double std[static 1U],
	const ald_t s[], size_t ns)
{
	double sum;

	sum = 0.;
	for (size_t i = 0U; i < ns; i++) {
		sum += s[i];
	}
	*mean = sum / (double)ns;

	sum = 0.;
	for (size_t i = 0U; i < ns; i++) {
		double tmp = (s[i] - *mean);
		sum += tmp * tmp;
	}
	*std = sqrt(sum / (double)ns);
	return 0;
}

static __attribute__((unused)) int
_quasi_stats_d(
	double mean[static 1U], double std[static 1U],
	const ald_t s[], size_t ns)
{
	double min = s[0U], max = s[0U];

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
_norm_d(ald_t s[], size_t ns,
	int(*statf)(double*, double*, const ald_t[], size_t))
{
/* calculate (s - mean(s)) / std(s) */
	double mu, sigma;
	register __mXd mmu;
	register __mXd msd;

	if (UNLIKELY(statf(&mu, &sigma, s, ns) < 0)) {
		return -1;
	}

	/* go for (s + -mu) * 1/sigma */
	mu = -mu;
	mmu = _mmX_broadcast_sd(&mu);
	sigma = 1. / sigma;
	msd = _mmX_broadcast_sd(&sigma);
	for (size_t i = 0U; i + widthof(__mXd) - 1U < ns; i += widthof(__mXd)) {
		register __mXd ms;
		ms = _mmX_load_pd(s + i);
		ms = _mmX_add_pd(ms, mmu);
		ms = _mmX_mul_pd(ms, msd);
		_mmX_store_pd(s + i, ms);
	}
	return 0;
}

static int
_dscal(ald_t s[], size_t ns, double f)
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
	const double h = 0.25 * tau;

	CLO.width = h;
	/* -1/(2 h^2) (exponent scaling) */
	CLO._xf = -1. / (2 * h * h);
	/* 1/sqrt(2PI h) (scaling) */
	CLO._vf = 1. / sqrt(2. * M_PI * h);
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
	const double thresh = _krnl_bjoernstad_falck_clo.width * 5.;
	double nsum = 0.f;
	double dsum = 0.f;
	/* we combine edelson-krolik rectangle with gauss kernel */
	size_t strt = 0U;
	size_t strk = 0U;

	for (size_t i = 0U; i < ts1.n; i++) {
		const double kti = (double)lag + ts1.t[i];

		/* find start of the interesting window */
		for (; strt < ts2.n && ts2.t[strt] < kti; strt++);
		for (strk = strk < strt ? strt : strk;
		     strk < ts2.n && ts2.t[strk] < kti + thresh; strk++);

		for (size_t j = strt; j < strk; j++) {
			double K = _krnl_bjoernstad_falck(
				lag - (ts2.t[j] - ts1.t[i]));
			dsum += K;
			nsum += ts1.y[i] * ts2.y[j] * K;
		}
	}
	return nsum / dsum;
}


/* public API */
int
cots_dxcor(double *restrict tgt, dts_t ts1, dts_t ts2, int nlags, double tau)
{
	size_t n1, n2;
	ald_t *t1, *t2;
	ald_t *y1, *y2;

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

	(void)_norm_d(y1, n1, _stats_d);
	(void)_norm_d(y2, n2, _stats_d);

	with (double tmd1, tmd2) {
		register double rtau = 1. / tau;

		tmd1 = _meandiff_d(t1, n1);
		tmd2 = _meandiff_d(t2, n2);
		/* scale by user tau */
		_dscal(t1, n1, rtau);
		_dscal(t2, n2, rtau);
		/* and set kernel width accordingly */
		_krnl_bjoernstad_falck_set((tmd1 < tmd2 ? tmd1 : tmd2) * rtau);
	}

	for (int k = -nlags, i = 0; k <= nlags; k++, i++) {
		tgt[i] = dxcf(k, (aldts_t){n1, t1, y1}, (aldts_t){n2, t2, y2});
	}

	_mm_free(t1);
	_mm_free(t2);
	_mm_free(y1);
	_mm_free(y2);
	return 0;
}

/* xcor.c ends here */
