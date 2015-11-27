/*** crosscorrirr.c -- cross correlation for irregular timeseries
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
#include <mex.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <mkl.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>
#include <immintrin.h>
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
	const double *t;
	const double *y;
} una_ts_t;

typedef struct {
	size_t n;
	const ald_t *t;
	const ald_t *y;
} ats_t;


static __attribute__((unused)) int
_stats_d(double mean[static 1U], double std[static 1U], ald_t s[], size_t ns)
{
#define VSL_SS_TASK	(VSL_SS_MEAN | VSL_SS_2R_MOM)
	VSLSSTaskPtr task;
	MKL_INT ndim = 1;
	MKL_INT dim1 = ns;
	MKL_INT stor = VSL_SS_MATRIX_STORAGE_ROWS;
	int rc = 0;

	rc += vsldSSNewTask(&task, &ndim, &dim1, &stor, s, 0, 0);
	rc += vsldSSEditTask(task, VSL_SS_ED_MEAN, mean);
	rc += vsldSSEditTask(task, VSL_SS_ED_2R_MOM, std);
	rc += vsldSSCompute(task, VSL_SS_TASK, VSL_SS_METHOD_FAST);
	rc += vslSSDeleteTask(&task);
	return rc;
#undef VSL_SS_TASK
}

static __attribute__((unused)) int
_quasi_stats_d(double mean[static 1U], double std[static 1U], ald_t s[], size_t ns)
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
_norm_d(ald_t s[], size_t ns, int(*statf)(double*, double*, ald_t[], size_t))
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
xcf(int lag, ats_t ts1, ats_t ts2)
{
	double nsum = 0.f;
	double dsum = 0.f;
	/* we combine edelson-krolik rectangle with gauss kernel */
	size_t strt = 0U;
	size_t strk = 0U;

	for (size_t i = 0U; i < ts1.n; i++) {
		const double kti = (double)lag + ts1.t[i];

		/* find start of the interesting window */
		for (; strt < ts2.n && ts2.t[strt] < kti - 1.1; strt++);
		for (strk = strk < strt ? strt : strk;
		     strk < ts2.n && ts2.t[strk] < kti + 1.1; strk++);

		for (size_t j = strt; j < strk; j++) {
			double K = _krnl_bjoernstad_falck(
				lag - (ts2.t[j] - ts1.t[i]));
			dsum += K;
			nsum += ts1.y[i] * ts2.y[j] * K;
		}
	}
	return nsum / dsum;
}

static int
cots_xcor(
	double *restrict tgt,
	una_ts_t ts1, una_ts_t ts2,
	int nlags, double tau)
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

	with (register double rtau = 1. / tau) {
		_dscal(t1, n1, rtau);
		_dscal(t2, n2, rtau);
	}

	for (int k = -nlags, i = 0; k <= nlags; k++, i++) {
		tgt[i] = xcf(k, (ats_t){n1, t1, y1}, (ats_t){n2, t2, y2});
	}

	_mm_free(t1);
	_mm_free(t2);
	_mm_free(y1);
	_mm_free(y2);
	return 0;
}


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double tau = -0.;
	int nlags = 20;
	double *tgt;
	size_t n1, n2;

	if (nrhs < 5) {
		mexErrMsgTxt("\
not enough input arguments, see `help crosscorrirr_'");
		return;
	} else if (nlhs > 2) {
		mexErrMsgTxt("\
too many output arguments, see `help crosscorrirr_'");
		return;
	} else if (!mxIsDouble(prhs[0U]) || !mxIsDouble(prhs[1U]) ||
		   !mxIsDouble(prhs[2U]) || !mxIsDouble(prhs[3U])) {
		mexErrMsgTxt("sampled times and values must be doubles");
		return;
	}
	if (!mxIsNumeric(prhs[4U])) {
		mexErrMsgTxt("tau must be a number");
		return;
	} else if ((tau = mxGetScalar(prhs[4U])) <= 0.) {
		mexErrMsgTxt("tau must be positive");
		return;
	}

	if (nrhs > 5) {
		if (!mxIsNumeric(prhs[5U])) {
			mexErrMsgTxt("number of lags parameter must be a number");
			return;
		} else if ((nlags = (int)mxGetScalar(prhs[5U])) < 0) {
			mexErrMsgTxt("number of lags must be non-negative");
			return;
		}
	}

	/* get the inputs */
	if ((n1 = mxGetM(prhs[0U])) != 1U && mxGetN(prhs[0U]) != 1U ||
	    n1 == 1U && !(n1 = mxGetN(prhs[0U]))) {
		mexErrMsgTxt("sample times (t1) must be a vector");
		return;
	}
	if ((n2 = mxGetM(prhs[2U])) != 1U && mxGetN(prhs[2U]) != 1U ||
	    n2 == 1U && !(n2 = mxGetN(prhs[2U]))) {
		mexErrMsgTxt("sample times (t2) must be a vector");
		return;
	}
	with (size_t tmp) {
		if ((tmp = mxGetM(prhs[1U])) != 1U && mxGetN(prhs[1U]) != 1U ||
		    tmp == 1U && !(tmp = mxGetN(prhs[1U]))) {
			mexErrMsgTxt("samples (y1) must be a vector");
			return;
		} else if (tmp != n1) {
			mexErrMsgTxt("\
samples (y1) and sample times (t1) must have same dimension");
			return;
		}

		if ((tmp = mxGetM(prhs[3U])) != 1U && mxGetN(prhs[3U]) != 1U ||
		    tmp == 1U && !(tmp = mxGetN(prhs[3U]))) {
			mexErrMsgTxt("samples (y2) must be a vector");
			return;
		} else if (tmp != n2) {
			mexErrMsgTxt("\
samples (y2) and sample times (t2) must have same dimension");
			return;
		}
	}
	/* initialise output */
	*plhs = mxCreateDoubleMatrix(1, 2 * nlags + 1, mxREAL);
	tgt = mxGetPr(*plhs);

	/* beef */
	with (const double *t1 = mxGetPr(prhs[0U]), *y1 = mxGetPr(prhs[1U]),
	      *t2 = mxGetPr(prhs[2U]), *y2 = mxGetPr(prhs[3U])) {
		cots_xcor(
			tgt,
			(una_ts_t){n1, t1, y1}, (una_ts_t){n2, t2, y2},
			nlags, tau);
	}

	if (nlhs > 1) {
		double *lags;

		plhs[1U] = mxCreateDoubleMatrix(1, 2 * nlags + 1, mxREAL);
		lags = mxGetPr(plhs[1U]);
		for (int k = -nlags, i = 0; k <= nlags; k++, i++) {
			lags[i] = (double)k * tau;
		}
	}
	return;
}

/* crosscorrirr.c ends here */
