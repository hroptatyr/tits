#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "sacf.h"
#include "nifty.h"

#define randn()		 gsl_ran_ugaussian(r)

static void
filter(float *restrict io, size_t nio, const float lambda[], size_t nlambda)
{
	for (size_t i = 0U; i < nlambda; i++) {
		for (size_t j = 0U; j < nio - i; j++) {
			io[j] += io[j + i] * lambda[i];
		}
	}
	return;
}

static __attribute__((pure)) int
betweenp(float x, float lower, float upper)
{
	return x < lower || x > upper;
}


int
main(int argc, char *argv[])
{
	float t[10000U];
	float v[countof(t)];
	float lags[10U];
	const gsl_rng_type * T;
	gsl_rng * r;

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	for (size_t i = 0; i < countof(t); i++) {
		t[i] = (float)i;
		v[i] = (float)randn();
	}
	gsl_rng_free(r);

	/* filter */
	filter(v, countof(v), (float[]){1., -0.2, -0.4, 0.1}, 4U);

	tits_ssacf(lags, (sts_t){countof(t), t, v}, countof(lags), 1.);

	printf("lag 0\t1.000000\n");
	for (size_t i = 0U; i < countof(lags); i++) {
		printf("lag %zu\t%f\n", i + 1U, lags[i]);
	}

	return betweenp(lags[0U], -0.3, -0.1) ||
		betweenp(lags[1U], -0.5, -0.3);
}
