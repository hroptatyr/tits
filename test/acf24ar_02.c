#include <stddef.h>
#include <math.h>
#include "acf24ar.h"
#include "nifty.h"


int
main(int argc, char *argv[])
{
	static float acf[] = {
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
	float ar[countof(acf)];
	const int ref[] = {
		5, 335, 55, 52, 38, 11, 38, -54, 70, 30,
	};
	int rc = 0;

	rc = tits_sacf2ar(ar, acf, countof(acf)) < 0;
	for (size_t i = 0U; i < countof(ar); i++) {
		rc |= (int)truncf(ar[i] * 1000) != ref[i];
	}
	return rc;
}
