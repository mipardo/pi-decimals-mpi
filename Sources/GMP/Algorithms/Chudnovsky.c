#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <omp.h>
#include "mpi.h"
#include "MPI_Operations.h"

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
void chudnovsky_iteration_gmp(mpf_t pi, int n, mpf_t dep_a, mpf_t dep_b, mpf_t dep_c, mpf_t aux){
    mpf_mul(aux, dep_a, dep_c);
    mpf_div(aux, aux, dep_b);
    
    mpf_add(pi, pi, aux);
}

/*
 * This method provides an optimal distribution for each thread of any proc
 * based on the Chudnovsky iterations analysis.
 * IMPORTANT: The number of threads used MUST be the same in every process
 * IMPORTANT: (num_procs * num_threads) % 4 == 0 OR (num_procs * num_threads) == 2
 * It returns an array of three integers:
 *   distribution[0] -> block size
 *   distribution[1] -> block start
 *   distribution[2] -> block end 
 */
int * get_distribution(int num_procs, int proc_id, int num_threads, int thread_id, int num_iterations){
    int * distribution, i, block_size, block_start, block_end, my_row, my_column, row, column;
    FILE * ratios_file;
    float working_ratios[160][41], my_working_ratio;

    //Open the working_ratios file 
    ratios_file = fopen("Resources/working_ratios.txt", "r");
    if(ratios_file == NULL){
        printf("working_ratios.txt not found \n");
        MPI_Finalize();
        exit(-1);
    } 

    //Load the working_ratios matrix 
    row = 0;
    while (fscanf(ratios_file, "%f", &working_ratios[row][0]) == 1){
        for (column = 1; column < 41; column++){
            fscanf(ratios_file, "%f", &working_ratios[row][column]);
        }
        row++;
    }

    distribution = malloc(sizeof(int) * 3);
    if(num_threads * num_procs == 1){
        distribution[0] = num_iterations;
        distribution[1] = 0;
        distribution[2] = num_iterations;
        return distribution; 
    }

    my_row = (num_threads * proc_id) + thread_id;
    my_column = (num_procs * num_threads) / 4;
    my_working_ratio = working_ratios[my_row][my_column];

    block_size = my_working_ratio * num_iterations / 100;
    block_start = 0;
    for(i = 0; i < my_row; i ++){
        block_start += working_ratios[i][my_column] * num_iterations / 100;
    }
    block_end = block_start + block_size;

    if (thread_id == (num_threads - 1) && proc_id == (num_procs - 1)){ 
        //If Last thread from last process:
        block_end = num_iterations;
        block_size = block_end - block_start;
    }

    distribution[0] = block_size;
    distribution[1] = block_start;    
    distribution[2] = block_end;    

    return distribution;
}


/*
 * This method is used by ParallelChudnovskyAlgorithm procs
 * for computing the first value of dep_a
 */
void init_dep_a(mpf_t dep_a, int block_start){
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

/*
 * Parallel Pi number calculation using the Chudnovsky algorithm
 * The number of iterations is divided by blocks 
 * so each process calculates a part of pi with multiple threads (or just one thread). 
 * Each process will also divide the iterations in blocks
 * among the threads to calculate its part.  
 * Finally, a collective reduction operation will be performed 
 * using a user defined function in OperationsMPI. 
 */
void chudnovsky_algorithm_gmp(int num_procs, int proc_id, mpf_t pi, 
                                    int num_iterations, int num_threads){
    int packet_size, position; 
    mpf_t local_proc_pi, e, c;  

    mpf_init_set_ui(local_proc_pi, 0);   
    mpf_init_set_ui(e, E);
    mpf_init_set_ui(c, C);
    mpf_neg(c, c);
    mpf_pow_ui(c, c, 3);

    //Set the number of threads 
    omp_set_num_threads(num_threads);

    #pragma omp parallel 
    {
        int thread_id, i, thread_block_size, thread_block_start, thread_block_end, factor_a;
        int *distribution;
        mpf_t local_thread_pi, dep_a, dep_a_dividend, dep_a_divisor, dep_b, dep_c, aux;

        thread_id = omp_get_thread_num();
        distribution = get_distribution(num_procs, proc_id, num_threads, thread_id, num_iterations);
        thread_block_size = distribution[0];
        thread_block_start = distribution[1];
        thread_block_end = distribution[2];

        mpf_init_set_ui(local_thread_pi, 0);    // private thread pi
        mpf_inits(dep_a, dep_b, dep_a_dividend, dep_a_divisor, aux, NULL);
        init_dep_a(dep_a, thread_block_start);
        mpf_pow_ui(dep_b, c, thread_block_start);
        mpf_init_set_ui(dep_c, B);
        mpf_mul_ui(dep_c, dep_c, thread_block_start);
        mpf_add_ui(dep_c, dep_c, A);
        factor_a = 12 * thread_block_start;

        //First Phase -> Working on a local variable        
        #pragma omp parallel for 
            for(i = thread_block_start; i < thread_block_end; i++){
                chudnovsky_iteration_gmp(local_thread_pi, i, dep_a, dep_b, dep_c, aux);
                //Update dep_a:
                mpf_set_ui(dep_a_dividend, factor_a + 10);
                mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 6);
                mpf_mul_ui(dep_a_dividend, dep_a_dividend, factor_a + 2);
                mpf_mul(dep_a_dividend, dep_a_dividend, dep_a);

                mpf_set_ui(dep_a_divisor, i + 1);
                mpf_pow_ui(dep_a_divisor, dep_a_divisor ,3);
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
    
    //Create user defined operation
    MPI_Op add_op;
    MPI_Op_create((MPI_User_function *)add, 0, &add_op);

    //Set buffers for cumunications and position for pack and unpack information 
    packet_size = 8 + sizeof(mp_exp_t) + ((local_proc_pi -> _mp_prec + 1) * sizeof(mp_limb_t));
    char recbuffer[packet_size];
    char sendbuffer[packet_size];

    //Pack local_proc_pi in sendbuffuer
    position = pack(sendbuffer, local_proc_pi);

    //Reduce local_proc_pi
    MPI_Reduce(sendbuffer, recbuffer, position, MPI_PACKED, add_op, 0, MPI_COMM_WORLD);

    //Unpack recbuffer in global Pi and do the last operations to get Pi
    if (proc_id == 0){
        unpack(recbuffer, pi);
        mpf_sqrt(e, e);
        mpf_mul_ui(e, e, D);
        mpf_div(pi, e, pi); 
    }    

    //Clear process memory
    MPI_Op_free(&add_op);
    mpf_clears(local_proc_pi, e, c, NULL);
}

