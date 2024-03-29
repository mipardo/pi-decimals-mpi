#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpfr.h>
#include "mpi.h"

/*
 * Pack mpf_t type
 * IMPORTANT: mpf_t data should have been previously initialized
 */
int mpfr_mpi_pack(void * buffer, mpfr_t data){
    int position, packet_size, d_elements;
    d_elements = (int) ceil((float) data -> _mpfr_prec / (float) GMP_NUMB_BITS);
    packet_size = 8 + sizeof(mpfr_exp_t) + (d_elements * sizeof(mp_limb_t));
    position = 0;
    MPI_Pack(&data -> _mpfr_prec, 1, MPI_INT, buffer, packet_size, &position, MPI_COMM_WORLD);
    MPI_Pack(&data -> _mpfr_sign, 1, MPI_INT, buffer, packet_size, &position, MPI_COMM_WORLD);
    MPI_Pack(&data -> _mpfr_exp, sizeof(mpfr_exp_t), MPI_BYTE, buffer, packet_size, &position, MPI_COMM_WORLD);
    MPI_Pack( data -> _mpfr_d, d_elements * sizeof(mp_limb_t), MPI_BYTE, buffer, packet_size, &position, MPI_COMM_WORLD);
    return position;
}

/*
 * Unpack mpf_t type
 * IMPORTANT: mpf_t data should have been previously initialized
 */
void mpfr_mpi_unpack(void * buffer, mpfr_t data){
    int position, packet_size, d_elements;
    d_elements = (int) ceil((float) data -> _mpfr_prec / (float) GMP_NUMB_BITS);
    packet_size = 8 + sizeof(mpfr_exp_t) + (d_elements * sizeof(mp_limb_t));
    position = 0;
    MPI_Unpack(buffer, packet_size, &position, &data -> _mpfr_prec, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buffer, packet_size, &position, &data -> _mpfr_sign, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buffer, packet_size, &position, &data -> _mpfr_exp, sizeof(mpfr_exp_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, packet_size, &position,  data -> _mpfr_d, d_elements * sizeof(mp_limb_t) , MPI_BYTE, MPI_COMM_WORLD);
}


/*
 * Operation defined for MPI
 * Adds mpf_t types
 */
void mpfr_mpi_add(void * invec, void * inoutvec, int *len, MPI_Datatype *dtype){
    mpfr_t a, b;
    mpfr_inits(a, b, NULL);
    mpfr_mpi_unpack(invec, a);
    mpfr_mpi_unpack(inoutvec, b);
    mpfr_add(b, b, a, MPFR_RNDN);
    mpfr_mpi_pack(inoutvec, b);
    mpfr_clears(a, b, NULL);
}

/*
 * Operation defined for MPI
 * Multiply mpf_t types
 */
void mpfr_mpi_mul(void * invec, void * inoutvec, int *len, MPI_Datatype *dtype){
    mpfr_t a, b;
    mpfr_inits(a, b, NULL);    
    mpfr_mpi_unpack(invec, a);
    mpfr_mpi_unpack(inoutvec, b);
    mpfr_mul(b, b, a, MPFR_RNDN);
    mpfr_mpi_pack(inoutvec, b);
    mpfr_clears(a, b, NULL);
}



