#ifndef MPI_OPERATIONS_GMP
#define MPI_OPERATIONS_GMP

void add(void *, void *, int *, MPI_Datatype *);
void mul(void *, void *, int *, MPI_Datatype *);
int pack(void *, mpf_t);
void unpack(void *, mpf_t);

#endif

