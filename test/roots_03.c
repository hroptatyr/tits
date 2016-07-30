#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "roots.h"
#include "nifty.h"


int main(void)
{
	double r[3U] = {NAN, NAN, NAN};
	double p[] = {-140, -36, 1, 1};/* == (x+4+2i)(x+4-2i)(x-7) */
	const int ref[] = {
		7000, -4000, 1999,
	};
	int rc = 0;

	rc = tits_droots(r, p, countof(p) - 1U) != 1;
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
