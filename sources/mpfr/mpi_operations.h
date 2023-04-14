#ifndef MPFR_MPI_OPERATIONS
#define MPFR_MPI_OPERATIONS

void mpfr_mpi_add(void *, void *, int *, MPI_Datatype *);
void mpfr_mpi_mul(void *, void *, int *, MPI_Datatype *);
int  mpfr_mpi_pack(void *, mpfr_t);
void mpfr_mpi_unpack(void *, mpfr_t);

#endif
