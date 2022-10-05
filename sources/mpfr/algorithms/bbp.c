#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>
#include <omp.h>
#include <math.h>
#include "mpi.h"
#include "../mpi_operations.h"

#define QUOTIENT 0.0625

/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Last version of Bailey Borwein Plouffe formula implementation                    *
 * This version allows computing Pi using processes and threads in hybrid way.      *
 *                                                                                  *
 ************************************************************************************
 * Bailey Borwein Plouffe formula:                                                  *
 *                      1        4          2        1       1                      *
 *    pi = SUMMATORY( ------ [ ------  - ------ - ------ - ------]),  n >=0         *
 *                     16^n    8n + 1    8n + 4   8n + 5   8n + 6                   *
 *                                                                                  *
 * Formula quotients are coded as:                                                  *
 *              4                 2                 1                 1             *
 *   quot_a = ------,  quot_b = ------,  quot_c = ------,  quot_d = ------,         *
 *            8n + 1            8n + 4            8n + 5            8n + 6          *
 *                                                                                  *
 *              1                                                                   *
 *   quot_m = -----                                                                 *
 *             16^n                                                                 *
 *                                                                                  *
 ************************************************************************************
 * Bailey Borwein Plouffe formula dependencies:                                     *
 *                                                                                  *
 *                        1            1                                            *
 *           dep_m(n) = ----- = ---------------                                     *
 *                       16^n   dep_m(n-1) * 16                                     *
 *                                                                                  *
 ************************************************************************************/

 /*
 * An iteration of Bailey Borwein Plouffe formula
 */
void bbp_iteration_mpfr(mpfr_t pi, int n, mpfr_t dep_m, mpfr_t quot_a, mpfr_t quot_b, mpfr_t quot_c, mpfr_t quot_d, mpfr_t aux){
    mpfr_set_ui(quot_a, 4, MPFR_RNDN);              // quot_a = ( 4 / (8n + 1))
    mpfr_set_ui(quot_b, 2, MPFR_RNDN);              // quot_b = (-2 / (8n + 4))
    mpfr_set_ui(quot_c, 1, MPFR_RNDN);              // quot_c = (-1 / (8n + 5))
    mpfr_set_ui(quot_d, 1, MPFR_RNDN);              // quot_d = (-1 / (8n + 6))
    mpfr_set_ui(aux, 0, MPFR_RNDN);                 // aux = a + b + c + d  

    int i = n << 3;                     // i = 8n
    mpfr_div_ui(quot_a, quot_a, i | 1, MPFR_RNDN);  // 4 / (8n + 1)
    mpfr_div_ui(quot_b, quot_b, i | 4, MPFR_RNDN);  // 2 / (8n + 4)
    mpfr_div_ui(quot_c, quot_c, i | 5, MPFR_RNDN);  // 1 / (8n + 5)
    mpfr_div_ui(quot_d, quot_d, i | 6, MPFR_RNDN);  // 1 / (8n + 6)

    // aux = (a - b - c - d)   
    mpfr_sub(aux, quot_a, quot_b, MPFR_RNDN);
    mpfr_sub(aux, aux, quot_c, MPFR_RNDN);
    mpfr_sub(aux, aux, quot_d, MPFR_RNDN);

    // aux = m * aux 
    mpfr_mul(aux, aux, dep_m, MPFR_RNDN);   
    
    mpfr_add(pi, pi, aux, MPFR_RNDN);  
}

/*
 * Parallel Pi number calculation using the BBP algorithm
 * The number of iterations is divided by blocks, 
 * so each process calculates a part of pi using threads. 
 * Each process will also divide the iterations in blocks
 * among the threads to calculate its part.  
 * Finally, a collective reduction operation will be performed
 * using a user defined function in OperationsMPI. 
 */
void bbp_algorithm_mpfr(int num_procs, int proc_id, mpfr_t pi, int num_iterations, int num_threads, int precision_bits){
    int block_size, block_start, block_end, position, packet_size, d_elements;
    mpfr_t local_proc_pi, quotient;

    block_size = (num_iterations + num_procs - 1) / num_procs;
    block_start = proc_id * block_size;
    block_end = block_start + block_size;
    if (block_end > num_iterations) block_end = num_iterations;

    mpfr_inits2(precision_bits, local_proc_pi, quotient, NULL);
    mpfr_set_d(quotient, QUOTIENT, MPFR_RNDN);
    mpfr_set_ui(local_proc_pi, 0, MPFR_RNDN);


    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, thread_block_size, thread_block_start, thread_block_end;
        mpfr_t local_thread_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux;

        thread_id = omp_get_thread_num();
        thread_block_size = (block_size + num_threads - 1) / num_threads;
        thread_block_start = (thread_id * thread_block_size) + block_start;
        thread_block_end = thread_block_start + thread_block_size;
        if (thread_block_end > block_end) thread_block_end = block_end;
        
        mpfr_init2(local_thread_pi, precision_bits);               // private thread pi
        mpfr_set_ui(local_thread_pi, 0, MPFR_RNDN);
        mpfr_init2(dep_m, precision_bits);
        mpfr_pow_ui(dep_m, quotient, thread_block_start, MPFR_RNDN);    // m = (1/16)^n                  
        mpfr_inits2(precision_bits, quot_a, quot_b, quot_c, quot_d, aux, NULL);
        

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = thread_block_start; i < thread_block_end; i++){
                bbp_iteration_mpfr(local_thread_pi, i, dep_m, quot_a, quot_b, quot_c, quot_d, aux);
                // Update dependencies:  
                mpfr_mul(dep_m, dep_m, quotient, MPFR_RNDN);
            }

        //Second Phase -> Accumulate the result in the global variable
        #pragma omp critical
        mpfr_add(local_proc_pi, local_proc_pi, local_thread_pi, MPFR_RNDN);

        //Clear thread memory
        mpfr_free_cache();
        mpfr_clears(local_thread_pi, dep_m, quot_a, quot_b, quot_c, quot_d, aux, NULL);   
    }

    //Create user defined operation
    MPI_Op add_op;
    MPI_Op_create((MPI_User_function *)add_mpfr, 0, &add_op);

    //Set buffers for cumunications and position for pack and unpack information
    d_elements = (int) ceil((float) local_proc_pi -> _mpfr_prec / (float) GMP_NUMB_BITS);
    packet_size = 8 + sizeof(mpfr_exp_t) + (d_elements * sizeof(mp_limb_t));
    char recbuffer[packet_size];
    char sendbuffer[packet_size];

    //Pack local_proc_pi in sendbuffuer
    position = pack_mpfr(sendbuffer, local_proc_pi);

    //Reduce piLocal
    MPI_Reduce(sendbuffer, recbuffer, position, MPI_PACKED, add_op, 0, MPI_COMM_WORLD);

    //Unpack recbuffer in global Pi and do the last operation
    if (proc_id == 0){
        unpack_mpfr(recbuffer, pi);
    }

    //Clear memory
    MPI_Op_free(&add_op);
    mpfr_clears(local_proc_pi, quotient, NULL);       

}

