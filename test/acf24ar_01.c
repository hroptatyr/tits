#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include "acf24ar.h"
#include "nifty.h"


int
main(int argc, char *argv[])
{
	static double acf[] = {
		0.0075,
		-0.3214,
		-0.0445,
		0.0641,
		-0.0006,
		-0.0299,
		-0.0027,
		0.0750,
		-0.0579,
		-0.0706,
	};
	double ar[countof(acf)];
	const int ref[] = {
		5, 335, 55, 52, 38, 11, 38, -54, 70, 30,
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
