#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <complex.h>
#include "roots.h"
#include "nifty.h"


int main(void)
{
	complex double r[3U] = {NAN, NAN, NAN};
	double p[] = {-140, -36, 1, 1};/* == (x+4+2i)(x+4-2i)(x-7) */
	const int ref[] = {
		-4000, -1999, -4000, 1999, 7000, 0,
	};
	int rc = 0;

	rc = tits_cdroots(r, p, countof(p) - 1U) != 3;
	for (size_t i = 0U; i < countof(r); i++) {
		rc |= (int)trunc(creal(r[i]) * 1000) != ref[2U * i + 0U];
		rc |= (int)trunc(cimag(r[i]) * 1000) != ref[2U * i + 1U];
	}
	if (rc) {
		for (size_t i = 0U; i < countof(r); i++) {
			printf(" %f + %fi", creal(r[i]), cimag(r[i]));
		}
		putchar('\n');
	}
	return rc;
}
