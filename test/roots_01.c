#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "roots.h"
#include "nifty.h"


int main(void)
{
	double r[3U] = {NAN, NAN, NAN};
	double p[] = {140, -13, -8, 1};/* == (x+4)(x-5)(x-7) */
	const int ref[] = {
		5000, 7000, -4000,
	};
	int rc = 0;

	rc = tits_droots(r, p, countof(p) - 1U) != 3;
	for (size_t i = 0U; i < countof(r); i++) {
		int cmp = (int)trunc(r[i] * 1000);
		rc |= cmp < ref[i] - 1 || cmp > ref[i] + 1;
	}
	if (rc) {
		for (size_t i = 0U; i < countof(r); i++) {
			printf(" %f", r[i]);
		}
		putchar('\n');
	}
	return rc;
}
