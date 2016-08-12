#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "norm.h"
#include "nifty.h"


int main(void)
{
	double r[] = {-3, -2, -1, 0, 0, 1, 2, 3};
	const int ref[] = {-1500, -1000, -500, 0, 0, 500, 1000, 1500};
	int rc = 0;

	rc = tits_dnorm(r, countof(r)) < 0;
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
