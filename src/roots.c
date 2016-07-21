/*** roots.c -- numerical roots of polynomials
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
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "roots.h"
#include "nifty.h"

static inline __attribute__((const, pure)) double
max_d(double x, double y)
{
	return x > y ? x : y;
}

static int
_horner_reduce_d(double *restrict p, size_t n, double at)
{
/* use horner method to reduce P by one degree, factoring out (x - at) */
	double tmp[n + 1U];

	tmp[n] = 0;
	for (ssize_t i = n; i > 0U; i--) {
		tmp[i - 1U] = p[i] + tmp[i] * at;
	}
	memcpy(p, tmp, (n + 1U) * sizeof(*tmp));
	return 0;
}

static double
_horner_eval_d(const double *p, size_t n, double at)
{
/* use horner method to reduce P by one degree, factoring out (x - at) */
	double tmp = 0;

	for (ssize_t i = n; i > 0U; i--) {
		tmp = p[i] + tmp * at;
	}
	return p[0U] + tmp * at;
}

static double
_laguerre_d(const double *p, size_t n, double x)
{
/* find one root of polynomial P of degree N, guess to be at X */
	size_t iter = 32U;
	/* first deriv */
	double p1[n];
	/* second deriv */
	double p2[n];

	/* precalc first and second deriv */
	for (size_t i = 0U; i < n; i++) {
		p1[i] = p[i + 1U] * (i + 1);
	}
	for (size_t i = 0U; i < n - 1U; i++) {
		p2[i] = p1[i + 1U] * (i + 1);
	}

	while (iter--) {
		double y = _horner_eval_d(p, n, x);
		double G;
		double H;
		double r;
		double a;

		if (fabs(y) < DBL_EPSILON) {
			/* that's good enough */
			break;
		}
		y = 1 / y;
		G = _horner_eval_d(p1, n - 1U, x) * y;
		H = G * G - _horner_eval_d(p2, n - 2U, x) * y;
		r = sqrt((H * n - G * G) * (n - 1));
		a = max_d(G + r, G - r);
		a = 1 / a * n;
		if (fabs(a) < DBL_EPSILON) {
			/* good enough */
			break;
		}
		x -= a;
	}
	return x;
}


int
tits_droots(double *restrict r, const double *p, size_t n)
{
	double q[n + 1U];
	double guess = 0;
	size_t nr = 0;

	if (UNLIKELY(!p[n])) {
		/* as if, they can call us with a proper degree */
		return -1;
	} else if (UNLIKELY(n == 0U)) {
		return -1;
	} else if (UNLIKELY(n == 1U)) {
		goto linear;
	}

	memcpy(q, p, (n + 1U) * sizeof(*p));
	do {
		/* find the root in the reduced poly */
		guess = _laguerre_d(q, n, guess);

#if 0
		/* find the root in the original poly */
		if (UNLIKELY(isnan(guess = _laguerre_d(p, orign, guess)))) {
			return nr;
		}
#endif
		/* reduce q */
		_horner_reduce_d(q, n, guess);

		/* assign */
		r[nr++] = guess;
	} while (--n > 1U);

linear:
	r[nr++] = -q[0U] / q[1U];
	return nr;
}

#if !defined double
# define double		float

# define max_d		max_f
# define sqrt		sqrtf
# define _horner_eval_d	_horner_eval_s
# define _horner_reduce_d	_horner_reduce_s
# define _laguerre_d	_laguerre_s
# define tits_droots	tits_sroots

/* intrins */
# undef DBL_EPSILON
# define DBL_EPSILON	FLT_EPSILON

# undef _
# define _(x)		(x ## f)

/* now go through it again for single precision */
# include __FILE__
#endif	/* !double */

/* roots.c ends here */
