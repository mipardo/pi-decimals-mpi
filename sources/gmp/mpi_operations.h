#ifndef GMP_MPI_OPERATIONS
#define GMP_MPI_OPERATIONS

void gmp_add(void *, void *, int *, MPI_Datatype *);
void gmp_mul(void *, void *, int *, MPI_Datatype *);
int  gmp_pack(void *, mpf_t);
void gmp_unpack(void *, mpf_t);

#endif

