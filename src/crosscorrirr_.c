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
#include <stdbool.h>
#include <math.h>
#include <mkl.h>
#include <mkl_vml.h>
#include <mkl_vsl.h>
#include <immintrin.h>
#include "nifty.h"

typedef double ald_t __attribute__((aligned(16)));


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
	double mu, sigma;
	register __m128d mmu;
	register __m128d msd;

	if (UNLIKELY(statf(&mu, &sigma, s, ns) < 0)) {
		return -1;
	}
	/* go for (s + -mu) * 1/sigma */
	mu = -mu;
	mmu = _mm_load1_pd(&mu);
	sigma = 1.f / sigma;
	msd = _mm_load1_pd(&sigma);
	for (size_t i = 0U; i + 1U < ns; i += 2U) {
		register __m128d ms;
		ms = _mm_load_pd(s + i);
		ms = _mm_add_pd(ms, mmu);
		ms = _mm_mul_pd(ms, msd);
		_mm_store_pd(s + i, ms);
	}
	/* do the rest by hand */
	switch (ns % 4U) {
	case 1U:
		s[0U] += mu;
		s[0U] *= sigma;
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
xcf(int lag, const ald_t t1[], const ald_t y1[], size_t n1, const ald_t t2[], const ald_t y2[], size_t n2)
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


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double tau;
	int nlags = 20;
	size_t n1, n2;
	ald_t *t1, *t2;
	ald_t *y1, *y2;
	double *tgt;
	double *lags = NULL;

	if (nrhs < 4 || !nlhs ||
	    !mxIsDouble(prhs[0U]) || !mxIsDouble(prhs[1U]) ||
	    !mxIsDouble(prhs[2U]) || !mxIsDouble(prhs[3U])) {
		mexErrMsgTxt("invalid usage, see `help crosscorrirr_'");
		return;
	}

	if (nrhs > 4) {
		if (!mxIsNumeric(prhs[4U])) {
			mexErrMsgTxt("number of lags parameter must be a number");
			return;
		} else if ((nlags = (int)mxGetScalar(prhs[4U])) < 0) {
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
		} else if (tmp != n1) {
			mexErrMsgTxt("\
samples (y2) and sample times (t2) must have same dimension");
			return;
		}
	}
	/* initialise output */
	*plhs = mxCreateDoubleMatrix(1, 2 * nlags + 1, mxREAL);
	tgt = mxGetPr(*plhs);
	if (nlhs > 1) {
		plhs[1U] = mxCreateDoubleMatrix(1, 2 * nlags + 1, mxREAL);
		lags = mxGetPr(plhs[1U]);
	}

	/* snarf input */
	t1 = mxGetPr(prhs[0U]);
	t2 = mxGetPr(prhs[2U]);
	y1 = mxGetPr(prhs[1U]);
	y2 = mxGetPr(prhs[3U]);

	(void)_norm_d(y1, n1, _stats_d);
	(void)_norm_d(y2, n2, _stats_d);

	tau = _mean_tdiff(t1, n1, t2, n2);
	with (register double rtau = 1. / tau) {
		cblas_dscal(n1, rtau, t1, 1U);
		cblas_dscal(n2, rtau, t2, 1U);
	}
	if (nlhs > 2) {
		plhs[2U] = mxCreateDoubleScalar(tau);
	}

	for (int k = -nlags, i = 0; k <= nlags; k++, i++) {
		tgt[i] = xcf(k, t1, y1, n1, t2, y2, n2);
		if (lags != NULL) {
			lags[i] = (double)k;
		}
	}
	return;
}

/* crosscorrirr.c ends here */
