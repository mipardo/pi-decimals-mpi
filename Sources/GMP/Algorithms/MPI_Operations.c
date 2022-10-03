#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include "mpi.h"

/*
 * Pack mpf_t type
 * IMPORTANT: mpf_t data should have been previously initialized
 */
int pack(void * buffer, mpf_t data){
    int position, packet_size;
    packet_size = 8 + sizeof(mp_exp_t) + ((data -> _mp_prec + 1) * sizeof(mp_limb_t));
    position = 0;
    MPI_Pack(&data -> _mp_size, 1, MPI_INT, buffer, packet_size, &position, MPI_COMM_WORLD);
    MPI_Pack(&data -> _mp_prec, 1, MPI_INT, buffer, packet_size, &position, MPI_COMM_WORLD);
    MPI_Pack(&data -> _mp_exp, sizeof(mp_exp_t), MPI_BYTE, buffer, packet_size, &position, MPI_COMM_WORLD);
    MPI_Pack( data -> _mp_d,  (data -> _mp_prec + 1) * sizeof(mp_limb_t), MPI_BYTE, buffer, packet_size, &position, MPI_COMM_WORLD);
    return position;
}

/*
 * Unpack mpf_t type
 * IMPORTANT: mpf_t data should have been previously initialized
 */
void unpack(void * buffer, mpf_t data){
    int position, packet_size;
    packet_size = 8 + sizeof(mp_exp_t) + ((data -> _mp_prec + 1) * sizeof(mp_limb_t));
    position = 0;
    MPI_Unpack(buffer, packet_size, &position, &data -> _mp_size, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buffer, packet_size, &position, &data -> _mp_prec, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buffer, packet_size, &position, &data -> _mp_exp, sizeof(mp_exp_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, packet_size, &position,  data -> _mp_d, (data -> _mp_prec + 1) * sizeof(mp_limb_t) , MPI_BYTE, MPI_COMM_WORLD);
}

/*
 * Operation defined for MPI
 * Adds mpf_t types
 */
void add(void * invec, void * inoutvec, int *len, MPI_Datatype *dtype){
    mpf_t a, b;
    mpf_inits(a, b, NULL);
    unpack(invec, a);
    unpack(inoutvec, b);
    mpf_add(b, b, a);
    pack(inoutvec, b);
    mpf_clears(a, b, NULL);
}

/*
 * Operation defined for MPI
 * Multiply mpf_t types
 */
void mul(void * invec, void * inoutvec, int *len, MPI_Datatype *dtype){
    mpf_t a, b;
    mpf_inits(a, b, NULL);    
    unpack(invec, a);
    unpack(inoutvec, b);
    mpf_mul(b, b, a);
    pack(inoutvec, b);
    mpf_clears(a, b, NULL);
}



