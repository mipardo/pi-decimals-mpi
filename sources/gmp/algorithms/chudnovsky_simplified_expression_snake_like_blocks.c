#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "mpi.h"
#include "../mpi_operations.h"
#include "chudnovsky_simplified_expression_blocks_blocks.h"

#define A 13591409
#define B 545140134
#define C 640320
#define D 426880
#define E 10005

/************************************************************************************
 * Miguel Pardo Navarro. 17/07/2021                                                 *
 * Chudnovsky formula implementation                                                *
 * This version does not computes all the factorials                                *
 * This version allows computing Pi using processes and threads in hybrid way.      *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula:                                                              *
 *     426880 sqrt(10005)                 (6n)! (545140134n + 13591409)             *
 *    --------------------  = SUMMATORY( ----------------------------- ),  n >=0    *
 *            pi                            (n!)^3 (3n)! (-640320)^3n               *
 *                                                                                  *
 * Some operands of the formula are coded as:                                       *
 *      dep_a_dividend = (6n)!                                                      *
 *      dep_a_divisor  = (n!)^3 (3n)!                                               *
 *      e              = 426880 sqrt(10005)                                         *
 *                                                                                  *
 ************************************************************************************
 * Chudnovsky formula dependencies:                                                 *
 *                     (6n)!         (12n + 10)(12n + 6)(12n + 2)                   *
 *      dep_a(n) = --------------- = ---------------------------- * dep_a(n-1)      *
 *                 ((n!)^3 (3n)!)              (n + 1)^3                            *
 *                                                                                  *
 *      dep_b(n) = (-640320)^3n = (-640320)^3(n-1) * (-640320)^3)                   *
 *                                                                                  *
 *      dep_c(n) = (545140134n + 13591409) = dep_c(n - 1) + 545140134               *
 *                                                                                  *
 ************************************************************************************/


void gmp_chudnovsky_simplified_expression_snake_like_blocks_phase(mpf_t local_proc_pi, mpf_t c, int num_threads, int block_size, int block_start, int block_end){
        int thread_id, i, thread_block_size, thread_block_start, thread_block_end, factor_a;
        mpf_t local_thread_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux;

        thread_id = omp_get_thread_num();
        thread_block_size = (block_size + num_threads - 1) / num_threads;
        thread_block_start = (thread_id * thread_block_size) + block_start;
        thread_block_end = thread_block_start + thread_block_size;
        if (thread_block_end > block_end) thread_block_end = block_end;
        
        mpf_init_set_ui(local_thread_pi, 0);    // private thread pi
        mpf_inits(dep_a, dep_b, dep_a_dividend, dep_a_divisor, aux, NULL);
        gmp_init_dep_a(dep_a, thread_block_start);
        mpf_pow_ui(dep_b, c, thread_block_start);
        mpf_init_set_ui(dep_c, B);
        mpf_mul_ui(dep_c, dep_c, thread_block_start);
        mpf_add_ui(dep_c, dep_c, A);
        factor_a = 12 * thread_block_start;

        //First Phase -> Working on a local variable        
        for(i = thread_block_start; i < thread_block_end; i++){
            gmp_chudnovsky_iteration(local_thread_pi, i, dep_a, dep_b, dep_c, aux);
            //Update dep_a:
            mpf_set_ui(dep_a_dividend, factor_a + 10);
            mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6);
            mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2);
            mpf_mul(dep_a_dividend, dep_a_dividend, dep_a);

            mpf_set_ui(dep_a_divisor, i + 1);
            mpf_pow_ui(dep_a_divisor, dep_a_divisor, 3);
            mpf_div(dep_a, dep_a_dividend, dep_a_divisor);
            factor_a += 12; 

            //Update dep_b:
            mpf_mul(dep_b, dep_b, c);

            //Update dep_c:
            mpf_add_ui(dep_c, dep_c, B);
        }

        //Second Phase -> Accumulate the result in the global variable 
        #pragma omp critical
        mpf_add(local_proc_pi, local_proc_pi, local_thread_pi);

        //Clear thread memory
        mpf_clears(local_thread_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux, NULL);   
}


void gmp_chudnovsky_simplified_expression_snake_like_blocks_algorithm(int num_procs, int proc_id, mpf_t pi, int num_iterations, int num_threads){
    int packet_size, position, block_size, first_block_start, first_block_end, second_block_start, second_block_end; 
    mpf_t local_proc_pi, e, c;  

    block_size = (num_iterations + (num_procs * 2) - 1) / (num_procs * 2);
    first_block_start = proc_id * block_size;
    first_block_end = first_block_start + block_size;
    second_block_start = (proc_id + num_procs) * block_size;
    second_block_end = second_block_start + block_size;
    if (second_block_end > num_iterations) second_block_end = num_iterations;

    mpf_init_set_ui(local_proc_pi, 0);   
    mpf_init_set_ui(e, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    // Compute the first block of iterations and then the second
    # pragma omp parallel
    {
        gmp_chudnovsky_simplified_expression_snake_like_blocks_phase(local_proc_pi, c, num_threads, block_size, first_block_start, first_block_end);
        gmp_chudnovsky_simplified_expression_snake_like_blocks_phase(local_proc_pi, c, num_threads, block_size, second_block_start, second_block_end);
    } 
    
    //Create user defined operation
    MPI_Op add_op;
    MPI_Op_create((MPI_User_function *)gmp_add, 0, &add_op);

    //Set buffers for cumunications and position for pack and unpack information 
    packet_size = 8 + sizeof(mp_exp_t) + ((local_proc_pi -> _mp_prec + 1) * sizeof(mp_limb_t));
    char recbuffer[packet_size];
    char sendbuffer[packet_size];

    //Pack local_proc_pi in sendbuffuer
    position = gmp_pack(sendbuffer, local_proc_pi);

    //Reduce local_proc_pi
    MPI_Reduce(sendbuffer, recbuffer, position, MPI_PACKED, add_op, 0, MPI_COMM_WORLD);

    //Unpack recbuffer in global Pi and do the last operations to get Pi
    if (proc_id == 0){
        gmp_unpack(recbuffer, pi);
        mpf_sqrt(e, e);
        mpf_mul_ui(e, e, D);
        mpf_div(pi, e, pi); 
    }    

    //Clear process memory
    MPI_Op_free(&add_op);
    mpf_clears(local_proc_pi, e, c, NULL);
}



