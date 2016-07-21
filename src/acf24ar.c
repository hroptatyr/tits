/*** acf24ar.c -- ACF <-> AR
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
#include <float.h>
#include "acf24ar.h"
#include "nifty.h"

static int
_levinson_d(double *restrict ar, const double *acf, size_t mo)
{
/* following ITU-T G.729 */
	double E;

	ar[0U] = 1;
	E = acf[0U];
	for (size_t i = 1U; i <= mo && E > DBL_EPSILON; i++) {
		double an[mo + 1U];
		double k = 0;

		for (size_t j = 0; j < i; j++) {
			k += ar[j] * acf[i - j];
		}
		an[i] = k = -k / E;

		for (size_t j = 1U; j < i; j++) {
			an[j] = ar[j] + k * ar[i - j];
		}
		/* ar coeffs for next round */
		memcpy(ar + 1U, an + 1U, mo * sizeof(*an));

		E *= 1 - k * k;
	}
	return (E > DBL_EPSILON) - 1;
}


int
tits_dacf2ar(double *restrict ar, const double *acf, size_t mo)
{
	return _levinson_d(ar, acf, mo);
}

#if !defined double
# define double		float

# define _levinson_d	_levinson_s
# define tits_dacf2ar	tits_sacf2ar

/* intrins */
# undef DBL_EPSILON
# define DBL_EPSILON	FLT_EPSILON

# undef _
# define _(x)		(x ## f)

/* now go through it again for single precision */
# include __FILE__
#endif	/* !double */

/* acf24ar.c ends here */
