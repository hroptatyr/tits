#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <complex.h>
#include "roots.h"
#include "nifty.h"


int main(void)
{
	complex float r[5U] = {NAN, NAN, NAN, NAN, NAN};
	/* (x+4+2i)(x+4-2i)(x-3+i)(x-3-i)(x-4) */
	float p[] = {-800, 360, 32, -26, -2, 1};
	const int ref[] = {
		3999, 0, 3000, -999, 3000, 999, -4000, -1999, -4000, 1999,
	};
	int rc = 0;

	rc = tits_csroots(r, p, countof(p) - 1U) != 5;
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
