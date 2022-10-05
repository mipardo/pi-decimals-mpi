#ifndef MPI_OPERATIONS_MPFR
#define MPI_OPERATIONS_MPFR

void add_mpfr(void *, void *, int *, MPI_Datatype *);
void mul_mpfr(void *, void *, int *, MPI_Datatype *);
int pack_mpfr(void *, mpfr_t);
void unpack_mpfr(void *, mpfr_t);

#endif
