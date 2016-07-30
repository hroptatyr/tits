#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include "roots.h"
#include "nifty.h"


int main(void)
{
	float r[3U] = {NAN, NAN, NAN};
	float p[] = {140, -13, -8, 1};/* == (x+4)(x-5)(x-7) */
	const int ref[] = {
		5000, 6999, -4000,
	};
	int rc = 0;

	rc = tits_sroots(r, p, countof(p) - 1U) != 3;
	for (size_t i = 0U; i < countof(r); i++) {
		rc |= (int)trunc(r[i] * 1000) != ref[i];
	}
	if (rc) {
		for (size_t i = 0U; i < countof(r); i++) {
			printf(" %f", r[i]);
		}
		putchar('\n');
	}
	return rc;
}
