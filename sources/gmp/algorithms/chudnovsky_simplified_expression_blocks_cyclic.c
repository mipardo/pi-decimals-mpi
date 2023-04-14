#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "mpi.h"
#include "../mpi_operations.h"

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


/*
 * An iteration of Chudnovsky formula
 */
void gmp_chudnovsky_iteration(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, mpf_t dep_c, mpf_t aux){
    mpf_mul(aux, dep_a, dep_c);
    mpf_div(aux, aux, dep_b);
    
    mpf_add(pi, pi, aux);
}


/*
 * This method is used by ParallelChudnovskyAlgorithm procs
 * for computing the first value of dep_a
 */
void gmp_init_dep_a(mpf_t dep_a, int block_start){
    mpz_t factorial_n, dividend, divisor;
    mpf_t float_dividend, float_divisor;
    mpz_inits(factorial_n, dividend, divisor, NULL);
    mpf_inits(float_dividend, float_divisor, NULL);

    mpz_fac_ui(factorial_n, block_start);
    mpz_fac_ui(divisor, 3 * block_start);
    mpz_fac_ui(dividend, 6 * block_start);

    mpz_pow_ui(factorial_n, factorial_n, 3);
    mpz_mul(divisor, divisor, factorial_n);

    mpf_set_z(float_dividend, dividend);
    mpf_set_z(float_divisor, divisor);

    mpf_div(dep_a, float_dividend, float_divisor);

    mpz_clears(factorial_n, dividend, divisor, NULL);
    mpf_clears(float_dividend, float_divisor, NULL);
}


void gmp_compute_dep_a(mpf_t dep_a, int iteration){
    int i, factor_a;
    mpf_t dividend, divisor, result;

    mpf_inits(result, dividend, divisor, NULL);
    mpf_set_ui(dep_a, 1);

    for (i = 1; i <= iteration; i++) {
        factor_a = 12 * (i - 1);
        mpf_set_ui(dividend, factor_a + 10);
        mpf_mul_ui(dividend, dividend, factor_a + 6);
        mpf_mul_ui(dividend, dividend, factor_a + 2);

        mpf_set_ui(divisor, i);
        mpf_pow_ui(divisor, divisor, 3);
        mpf_div(result, dividend, divisor);
        mpf_mul(dep_a, dep_a, result);
    }    

    mpf_clears(result, dividend, divisor, NULL);
}

void gmp_compute_portion_of_dep_a(mpf_t dep_a, int next_i, int current_i){
    int i, factor_a;
    mpf_t result, dividend, divisor;

    mpf_inits(result, dividend, divisor, NULL);

    for (i = current_i + 1; i <= next_i; i++){
        factor_a = 12 * (i - 1);
        mpf_set_ui(dividend, factor_a + 10);
        mpf_mul_ui(dividend, dividend, factor_a + 6);
        mpf_mul_ui(dividend, dividend, factor_a + 2);

        mpf_set_ui(divisor, i);
        mpf_pow_ui(divisor, divisor, 3);
        mpf_div(result, dividend, divisor);

        mpf_mul(dep_a, dep_a, result);
    } 

    mpf_clears(result, dividend, divisor, NULL);
}


void gmp_chudnovsky_simplified_expression_blocks_cyclic_algorithm(int num_procs, int proc_id, mpf_t pi, int num_iterations, int num_threads){
    int packet_size, position, block_size, block_start, block_end; 
    mpf_t local_proc_pi, e, c, jump;  

    block_size = (num_iterations + num_procs - 1) / num_procs;
    block_start = proc_id * block_size;
    block_end = block_start + block_size;
    if (block_end > num_iterations) block_end = num_iterations;

    mpf_init_set_ui(local_proc_pi, 0);   
    mpf_init_set_ui(e, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);
    mpf_inits(jump, NULL);
    mpf_pow_ui(jump, c, num_threads);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, j;
        mpf_t local_thread_pi, dep_a, dep_b, dep_c, aux;

        thread_id = omp_get_thread_num();
       
        mpf_init_set_ui(local_thread_pi, 0);    // private thread pi
        mpf_inits(dep_a, dep_b, aux, NULL);
        gmp_compute_dep_a(dep_a, block_start + thread_id);
        mpf_pow_ui(dep_b, c, block_start + thread_id);
        mpf_init_set_ui(dep_c, B);
        mpf_mul_ui(dep_c, dep_c, block_start + thread_id);
        mpf_add_ui(dep_c, dep_c, A);

        //First Phase -> Working on a local variable        
        for(i = block_start + thread_id; i < block_end; i += num_threads){
            gmp_chudnovsky_iteration(local_thread_pi, i, dep_a, dep_b, dep_c, aux);

            //Update dep_a:
            gmp_compute_portion_of_dep_a(dep_a, i + num_threads, i);

            //Update dep_b:
            mpf_mul(dep_b, dep_b, jump);

            //Update dep_c:
            mpf_add_ui(dep_c, dep_c, B * num_threads);
        }

        //Second Phase -> Accumulate the result in the global variable 
        #pragma omp critical
        mpf_add(local_proc_pi, local_proc_pi, local_thread_pi);

        //Clear thread memory
        mpf_clears(local_thread_pi, dep_a, dep_b, dep_c, aux, NULL);   
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
    mpf_clears(local_proc_pi, e, c, jump, NULL);
}

