#ifndef CHUDNOVSKY_GMP
#define CHUDNOVSKY_GMP

void chudnovsky_algorithm_gmp(int, int, mpf_t, int, int);

void chudnovsky_iteration_gmp(mpf_t, int, mpf_t, mpf_t, mpf_t, mpf_t);

void init_dep_a_gmp(mpf_t, int);

#endif

