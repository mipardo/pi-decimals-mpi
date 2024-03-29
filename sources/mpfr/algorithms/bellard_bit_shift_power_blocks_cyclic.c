#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>
#include <omp.h>
#include <math.h>
#include "mpi.h"
#include "../mpi_operations.h"


/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Bellard formula implementation                                                   *
 * This version allows computing Pi using processes and threads in hybrid way.      *
 *                                                                                  *
 ************************************************************************************
 * Bellard formula:                                                                 *
 *                 (-1)^n     32     1      256     64       4       4       1      *
 * 2^6 * pi = SUM( ------ [- ---- - ---- + ----- - ----- - ----- - ----- + -----])  *
 *                 1024^n    4n+1   4n+3   10n+1   10n+3   10n+5   10n+7   10n+9    *
 *                                                                                  *
 * Formula quotients are coded as:                                                  *
 *             32          1           256          64                              *
 *        a = ----,   b = ----,   c = -----,   d = -----,                           *
 *            4n+1        4n+3        10n+1        10n+3                            *
 *                                                                                  *
 *              4            4            1         (-1)^n                          *
 *        e = -----,   f = -----,   g = -----,   m = -----,                         *
 *            10n+5        10n+7        10n+9        2^10n                          *
 *                                                                                  *
 ************************************************************************************
 * Bellard formula dependencies:                                                    *
 *                           1            1                                         *
 *              dep_m(n) = ------ = -----------------                               *
 *                         1024^n   1024^(n-1) * 1024                               *
 *                                                                                  *
 *              dep_a(n) = 4n  = dep_a(n-1) + 4                                     *
 *                                                                                  *
 *              dep_b(n) = 10n = dep_a(n-1) + 10                                    *
 *                                                                                  *
 ************************************************************************************/


/*
 * An iteration of Bellard formula
 */
void mpfr_bellard_iteration(mpfr_t pi, int n, mpfr_t m, mpfr_t a, mpfr_t b, mpfr_t c, mpfr_t d, 
                    mpfr_t e, mpfr_t f, mpfr_t g, mpfr_t aux, int dep_a, int dep_b){
    mpfr_set_ui(a, 32, MPFR_RNDN);              // a = ( 32 / ( 4n + 1))
    mpfr_set_ui(b, 1, MPFR_RNDN);               // b = (  1 / ( 4n + 3))
    mpfr_set_ui(c, 256, MPFR_RNDN);             // c = (256 / (10n + 1))
    mpfr_set_ui(d, 64, MPFR_RNDN);              // d = ( 64 / (10n + 3))
    mpfr_set_ui(e, 4, MPFR_RNDN);               // e = (  4 / (10n + 5))
    mpfr_set_ui(f, 4, MPFR_RNDN);               // f = (  4 / (10n + 7))
    mpfr_set_ui(g, 1, MPFR_RNDN);               // g = (  1 / (10n + 9))
    mpfr_set_ui(aux, 0, MPFR_RNDN);             // aux = (- a - b + c - d - e - f + g)  

    mpfr_div_ui(a, a, dep_a + 1, MPFR_RNDN);    // a = ( 32 / ( 4n + 1))
    mpfr_div_ui(b, b, dep_a + 3, MPFR_RNDN);    // b = (  1 / ( 4n + 3))

    mpfr_div_ui(c, c, dep_b + 1, MPFR_RNDN);    // c = (256 / (10n + 1))
    mpfr_div_ui(d, d, dep_b + 3, MPFR_RNDN);    // d = ( 64 / (10n + 3))
    mpfr_div_ui(e, e, dep_b + 5, MPFR_RNDN);    // e = (  4 / (10n + 5))
    mpfr_div_ui(f, f, dep_b + 7, MPFR_RNDN);    // f = (  4 / (10n + 7))
    mpfr_div_ui(g, g, dep_b + 9, MPFR_RNDN);    // g = (  1 / (10n + 9))

    // aux = (- a - b + c - d - e - f + g)   
    mpfr_neg(a, a, MPFR_RNDN);
    mpfr_sub(aux, a, b, MPFR_RNDN);
    mpfr_sub(c, c, d, MPFR_RNDN);
    mpfr_sub(c, c, e, MPFR_RNDN);
    mpfr_sub(c, c, f, MPFR_RNDN);
    mpfr_add(c, c, g, MPFR_RNDN);
    mpfr_add(aux, aux, c, MPFR_RNDN);

    // aux = m * aux
    mpfr_mul(aux, aux, m, MPFR_RNDN);   

    mpfr_add(pi, pi, aux, MPFR_RNDN); 
}

void mpfr_bellard_bit_shift_power_blocks_cyclic_algorithm(int num_procs, int proc_id, mpfr_t pi, 
                                int num_iterations, int num_threads, int precision_bits){
    int block_size, block_start, block_end, position, packet_size, d_elements;
    mpfr_t local_proc_pi, ONE;

    block_size = (num_iterations + num_procs - 1) / num_procs;
    block_start = proc_id * block_size;
    block_end = block_start + block_size;
    if (block_end > num_iterations) block_end = num_iterations;

    mpfr_inits2(precision_bits, ONE, local_proc_pi, NULL);
    mpfr_set_ui(local_proc_pi, 0, MPFR_RNDN);
    mpfr_set_ui(ONE, 1, MPFR_RNDN); 

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, dep_a, dep_b, jump_dep_a, jump_dep_b, next_i;
        mpfr_t local_thread_pi, dep_m, a, b, c, d, e, f, g, aux;

        thread_id = omp_get_thread_num();

        mpfr_init2(local_thread_pi, precision_bits);               // private thread pi
        mpfr_set_ui(local_thread_pi, 0, MPFR_RNDN);
        dep_a = (block_start + thread_id) * 4;
        dep_b = (block_start + thread_id) * 10;
        jump_dep_a = 4 * num_threads;
        jump_dep_b = 10 * num_threads;
        mpfr_init2(dep_m, precision_bits);
        mpfr_mul_2exp(dep_m, ONE, 10 * (block_start + thread_id), MPFR_RNDN);
        mpfr_pow_si(dep_m, dep_m, -1, MPFR_RNDN);
        if((thread_id + block_start) % 2 != 0) mpfr_neg(dep_m, dep_m, MPFR_RNDN);                 
        mpfr_inits2(precision_bits, a, b, c, d, e, f, g, aux, NULL);

        //First Phase -> Working on a local variable
        for(i = block_start + thread_id; i < block_end; i+=num_threads){
            mpfr_bellard_iteration(local_thread_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
            // Update dependencies for next iteration:
            next_i = i + num_threads;
            mpfr_mul_2exp(dep_m, ONE, 10 * next_i, MPFR_RNDN);
            mpfr_pow_si(dep_m, dep_m, -1, MPFR_RNDN);
            if (next_i % 2 != 0) mpfr_neg(dep_m, dep_m, MPFR_RNDN);
            dep_a += jump_dep_a;
            dep_b += jump_dep_b;  
        }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpfr_add(local_proc_pi, local_proc_pi, local_thread_pi, MPFR_RNDN);

        //Clear thread memory
        mpfr_free_cache();
        mpfr_clears(local_thread_pi, dep_m, a, b, c, d, e, f, g, aux, NULL);   
    }

    //Create user defined operation
    MPI_Op add_op;
    MPI_Op_create((MPI_User_function *)mpfr_mpi_add, 0, &add_op);

    //Set buffers for cumunications and position for pack and unpack information
    d_elements = (int) ceil((float) local_proc_pi -> _mpfr_prec / (float) GMP_NUMB_BITS);
    packet_size = 8 + sizeof(mpfr_exp_t) + (d_elements * sizeof(mp_limb_t));
    char recbuffer[packet_size];
    char sendbuffer[packet_size];

    //Pack local_proc_pi in sendbuffuer
    position = mpfr_mpi_pack(sendbuffer, local_proc_pi);

    //Reduce piLocal
    MPI_Reduce(sendbuffer, recbuffer, position, MPI_PACKED, add_op, 0, MPI_COMM_WORLD);

    //Unpack recbuffer in global Pi and do the last operation
    if (proc_id == 0){
        mpfr_mpi_unpack(recbuffer, pi);
        mpfr_div_ui(pi, pi, 64, MPFR_RNDN);
    }

    //Clear memory
    MPI_Op_free(&add_op);
    mpfr_clears(local_proc_pi, ONE, NULL);       

}

