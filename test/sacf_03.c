#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "sacf.h"
#include "nifty.h"

#define randn()		 gsl_ran_ugaussian(r)
#define rand()		 gsl_rng_uniform(r)

static void
filter(double *restrict io, size_t nio, const double lambda[], size_t nlambda)
{
	for (size_t i = 0U; i < nlambda; i++) {
		for (size_t j = 0U; j < nio - i; j++) {
			io[j] += io[j + i] * lambda[i];
		}
	}
	return;
}


int
main(int argc, char *argv[])
{
/* irregular */
	double t[10000U];
	double v[countof(t)];
	double lags[10U];
	const gsl_rng_type * T;
	gsl_rng * r;
	size_t nt = 0U;

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	for (size_t i = 0; i < countof(t); i++) {
		t[i] = (double)i;
		v[i] = randn();
	}

	/* filter */
	filter(v, countof(v), (double[]){1., 0., 0.4}, 3U);

	/* kick some elements from t */
	for (size_t i = 0U; i < countof(t); i++) {
		if (rand() < 0.80) {
			t[nt] = t[i];
			v[nt] = v[i];
			nt++;
		}
	}

	gsl_rng_free(r);

	/* oversampled, original data had a 1.0 step in t */
	tits_dsacf(lags, (dts_t){nt, t, v}, countof(lags), 0.5);

	printf("lag 0\t1.000000\t/%zu samples\n", nt);
	for (size_t i = 0U; i < countof(lags); i++) {
		printf("lag %zu\t%f\n", i + 1U, lags[i]);
	}

	return lags[3U/*2nd second*/] < 0.3 || lags[3U/*2nd second*/] > 0.5;
}
