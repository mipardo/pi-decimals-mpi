#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
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
 *                 2^10n     4n+1   4n+3   10n+1   10n+3   10n+5   10n+7   10n+9    *
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
void bellard_iteration_gmp(mpf_t pi, int n, mpf_t m, mpf_t a, mpf_t b, mpf_t c, mpf_t d, 
                    mpf_t e, mpf_t f, mpf_t g, mpf_t aux, int dep_a, int dep_b){
    mpf_set_ui(a, 32);              // a = ( 32 / ( 4n + 1))
    mpf_set_ui(b, 1);               // b = (  1 / ( 4n + 3))
    mpf_set_ui(c, 256);             // c = (256 / (10n + 1))
    mpf_set_ui(d, 64);              // d = ( 64 / (10n + 3))
    mpf_set_ui(e, 4);               // e = (  4 / (10n + 5))
    mpf_set_ui(f, 4);               // f = (  4 / (10n + 7))
    mpf_set_ui(g, 1);               // g = (  1 / (10n + 9))
    mpf_set_ui(aux, 0);             // aux = (- a - b + c - d - e - f + g)  

    mpf_div_ui(a, a, dep_a + 1);    // a = ( 32 / ( 4n + 1))
    mpf_div_ui(b, b, dep_a + 3);    // b = (  1 / ( 4n + 3))

    mpf_div_ui(c, c, dep_b + 1);    // c = (256 / (10n + 1))
    mpf_div_ui(d, d, dep_b + 3);    // d = ( 64 / (10n + 3))
    mpf_div_ui(e, e, dep_b + 5);    // e = (  4 / (10n + 5))
    mpf_div_ui(f, f, dep_b + 7);    // f = (  4 / (10n + 7))
    mpf_div_ui(g, g, dep_b + 9);    // g = (  1 / (10n + 9))

    // aux = (- a - b + c - d - e - f + g)   
    mpf_neg(a, a);
    mpf_sub(aux, a, b);
    mpf_sub(c, c, d);
    mpf_sub(c, c, e);
    mpf_sub(c, c, f);
    mpf_add(c, c, g);
    mpf_add(aux, aux, c);

    // aux = m * aux
    mpf_mul(aux, aux, m);   

    mpf_add(pi, pi, aux); 
}


void bellard_blocks_and_cyclic_algorithm_gmp(int num_procs, int proc_id, mpf_t pi, int num_iterations, int num_threads){
    int block_size, block_start, block_end, position, packet_size;
    mpf_t local_proc_pi, ONE;

    block_size = (num_iterations + num_procs - 1) / num_procs;
    block_start = proc_id * block_size;
    block_end = block_start + block_size;
    if (block_end > num_iterations) block_end = num_iterations;

    mpf_init_set_ui(local_proc_pi, 0);
    mpf_init_set_ui(ONE, 1);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, dep_a, dep_b, jump_dep_a, jump_dep_b, next_i;
        mpf_t local_thread_pi, dep_m, a, b, c, d, e, f, g, aux;

        thread_id = omp_get_thread_num();
        mpf_init_set_ui(local_thread_pi, 0);       // private thread pi
        dep_a = (block_start + thread_id) * 4;
        dep_b = (block_start + thread_id) * 10;
        jump_dep_a = 4 * num_threads;
        jump_dep_b = 10 * num_threads;
        mpf_init(dep_m);
        mpf_mul_2exp(dep_m, ONE, 10 * (block_start + thread_id));
        mpf_div(dep_m, ONE, dep_m);
        if((thread_id + block_start) % 2 != 0) mpf_neg(dep_m, dep_m);
        mpf_inits(a, b, c, d, e, f, g, aux, NULL);

        //First Phase -> Working on a local variable
        #pragma omp parallel for
            for(i = block_start + thread_id; i < block_end; i += num_threads){
                bellard_iteration_gmp(local_thread_pi, i, dep_m, a, b, c, d, e, f, g, aux, dep_a, dep_b);
                // Update dependencies for next iteration:
                next_i = i + num_threads;
                mpf_mul_2exp(dep_m, ONE, 10 * next_i);
                mpf_div(dep_m, ONE, dep_m);
                if (next_i % 2 != 0) mpf_neg(dep_m, dep_m);
                dep_a += jump_dep_a;
                dep_b += jump_dep_b;
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpf_add(local_proc_pi, local_proc_pi, local_thread_pi);

        //Clear memory
        mpf_clears(local_thread_pi, dep_m, a, b, c, d, e, f, g, aux, NULL);
    }

    //Create user defined operation
    MPI_Op add_op;
    MPI_Op_create((MPI_User_function *)add_gmp, 0, &add_op);

    //Set buffers for cumunications and position for pack and unpack information
    packet_size = 8 + sizeof(mp_exp_t) + ((local_proc_pi -> _mp_prec + 1) * sizeof(mp_limb_t));
    char recbuffer[packet_size];
    char sendbuffer[packet_size];

    //Pack local_proc_pi in sendbuffuer
    position = pack_gmp(sendbuffer, local_proc_pi);

    //Reduce piLocal
    MPI_Reduce(sendbuffer, recbuffer, position, MPI_PACKED, add_op, 0, MPI_COMM_WORLD);

    //Unpack recbuffer in global Pi and do the last operation
    if (proc_id == 0){
        unpack_gmp(recbuffer, pi);
        mpf_div_ui(pi, pi, 64);
    }

    //Clear memory
    MPI_Op_free(&add_op);
    mpf_clears(local_proc_pi, ONE, NULL);       
}

