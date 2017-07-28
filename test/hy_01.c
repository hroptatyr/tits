#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "hy.h"
#include "nifty.h"

#define randn()		 gsl_ran_ugaussian(r)

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
	double t[10000U];
	double v[countof(t)];
	double s[countof(t) + 1U];
	double w[countof(s)];
	double lags[21U];
	const gsl_rng_type * T;
	gsl_rng * r;
	size_t ns = 0U;
	size_t nt = 0U;

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	for (size_t i = 0; i < countof(t); i++) {
		t[i] = (double)i;
		v[i] = randn();
	}
	w[0U] = 0.;
	memcpy(s, t, sizeof(t));
	memcpy(w + 1, v, sizeof(v));

	/* kick some elements from t */
	for (size_t i = 0U; i < countof(t); i++) {
		if ((double)rand() / (double)RAND_MAX < 0.80) {
			t[nt] = t[i];
			v[nt] = v[i];
			nt++;
		}
	}

	/* kick some elements from s */
	for (size_t i = 0U; i < countof(s); i++) {
		if ((double)rand() / (double)RAND_MAX < 0.50) {
			s[ns] = s[i];
			w[ns] = w[i];
			ns++;
		}
	}

	gsl_rng_free(r);

	printf("got %zu ~ %zu\n", nt, ns);
	tits_dhy(lags, (dts_t){nt, t, v}, (dts_t){ns, s, w}, 10, 0.5);

	for (size_t i = 0U; i < countof(lags); i++) {
		printf("lag %zd\t%f\n", 10 - (ssize_t)i, lags[i]);
	}

	return 0;
}
