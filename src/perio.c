/*** perio.c -- periodograms for irregular timeseries
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
#include <math.h>
#include "perio.h"
#include "nifty.h"

/* oversampling frequency */
#define OVERSAMP	4.


/* public API */
int
tits_dperio(double *restrict tgt, size_t nw, dts_t ts)
{
	for (size_t j = 0U; j < nw; j++) {
		const double wj = 2. * M_PI * (double)(j + 1U);
		double xc = 0.;
		double xs = 0.;
		double cc = 0.;
		double ss = 0.;
		double cs = 0.;

		for (size_t i = 0U; i < ts.n; i++) {
			double theta = wj * ts.t[i];
			double c = cos(theta);
			double s = sin(theta);

			xc += ts.y[i] * c;
			xs += ts.y[i] * s;
			cc += c * c;
			ss += s * s;
			cs += c * s;
		}

		with (double tau = atan2(2 * cs, cc - ss) / 2.) {
			double c = cos(tau);
			double s = sin(tau);
			double ct = c * xc + s * xs;
			double st = c * xs - s * xc;
			double cct = c * c, sst = s * s, cst = c * s;

			tgt[j] = 0.;
			tgt[j] += ((ct * ct) / (cct * cc + 2. * cst * cs + sst * ss));
			tgt[j] += ((st * st) / (cct * ss - 2. * cst * cs + sst * cc));
			tgt[j] *= 0.5;
		}
	}
	return 0;
}

/* perio.c ends here */
