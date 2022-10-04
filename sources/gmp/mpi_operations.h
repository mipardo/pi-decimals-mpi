#ifndef MPI_OPERATIONS_GMP
#define MPI_OPERATIONS_GMP

void add_gmp(void *, void *, int *, MPI_Datatype *);
void mul_gmp(void *, void *, int *, MPI_Datatype *);
int pack_gmp(void *, mpf_t);
void unpack_gmp(void *, mpf_t);

#endif

