#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "roots.h"
#include "nifty.h"


int main(void)
{
	float r[5U] = {NAN, NAN, NAN, NAN, NAN};
	/* (x+4+2i)(x+4-2i)(x-3+i)(x-3-i)(x-4) */
	float p[] = {-800, 360, 32, -26, -2, 1};
	const int ref[] = {
		3999, -4000, 2000, 3000, 999,
	};
	int rc = 0;

	rc = tits_sroots(r, p, countof(p) - 1U) != 1U;
	for (size_t i = 0U; i < countof(r); i++) {
		rc |= (int)trunc(r[i] * 1000) != ref[i];
	}
	if (rc) {
		printf(" %f", r[0U]);
		for (size_t i = 1U; i < countof(r); i += 2U) {
			printf(" %f + %fi", r[i + 0U], r[i + 1U]);
			printf(" %f - %fi", r[i + 0U], r[i + 1U]);
		}
		putchar('\n');
	}
	return rc;
}
