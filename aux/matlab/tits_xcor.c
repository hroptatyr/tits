/*** tits_xcor.c -- cross correlation for irregular timeseries
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
#include <mex.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "xcor.h"
#include "nifty.h"


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
		tits_dxcor(
			tgt,
			(dts_t){n1, t1, y1}, (dts_t){n2, t2, y2},
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

/* tits_xcor.c ends here */
