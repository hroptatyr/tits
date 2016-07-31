#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "acf24ar.h"
#include "nifty.h"


int
main(int argc, char *argv[])
{
	static double acf[] = {
		0.8, 0.4, 0.2,
	};
	double ar[countof(acf)];
	const int ref[] = {
		-2000, 2000, -1000,
	};
	int rc = 0;

	rc = tits_dacf2ar(ar, acf, countof(acf)) < 0;
	for (size_t i = 0U; i < countof(acf); i++) {
		rc |= (int)trunc(ar[i] * 1000) != ref[i];
	}
	if (rc) {
		for (size_t i = 0U; i < countof(acf); i++) {
			printf(" %f", ar[i]);
		}
		putchar('\n');
	}
	return rc;
}
