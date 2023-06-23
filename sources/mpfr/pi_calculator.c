#include <stdio.h>
#include <stdlib.h>
#include <mpfr.h>
#include <time.h>
#include <stdbool.h>
#include "mpi.h"
#include "algorithms/bbp_blocks_blocks.h"
#include "algorithms/bellard_bit_shift_power_blocks_cyclic.h"
#include "algorithms/chudnovsky_simplified_expression_blocks_blocks.h"
#include "check_decimals.h"
#include "../common/printer.h"


double gettimeofday();


void mpfr_calculate_pi(int num_procs, int proc_id, int algorithm, int precision, int num_threads, bool print_in_csv_format){
    double execution_time;
    struct timeval t1, t2;
    int num_iterations, decimals_computed, precision_bits; 
    mpfr_t pi;    
    char *algorithm_tag;

    //Get init time 
    if(proc_id == 0){
        gettimeofday(&t1, NULL);
    }

    //Set gmp float precision (in bits) and init pi
    precision_bits = precision * 8;
    mpfr_set_default_prec(precision_bits); 
    if (proc_id == 0){
        mpfr_init_set_ui(pi, 0, MPFR_RNDN);
    }

    switch (algorithm)
    {
    case 0:
        num_iterations = precision * 0.84;
        check_errors(num_procs, precision, num_iterations, num_threads, proc_id);
        algorithm_tag = "MPFR-BBP-BLC-BLC";
        mpfr_bbp_blocks_blocks_algorithm(num_procs, proc_id, pi, num_iterations, num_threads, precision_bits);
        break;

    case 1:
        num_iterations = precision / 3;
        check_errors(num_procs, precision, num_iterations, num_threads, proc_id);
        algorithm_tag = "MPFR-BEL-BSP-BLC-CYC";
        mpfr_bellard_bit_shift_power_blocks_cyclic_algorithm(num_procs, proc_id, pi, num_iterations, num_threads, precision_bits);
        break;

    case 2:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        check_errors(num_procs, precision, num_iterations, num_threads, proc_id);
        algorithm_tag = "MPFR-CHD-SME-BLC-BLC";
        mpfr_chudnovsky_simplified_expression_blocks_blocks_algorithm(num_procs, proc_id, pi, num_iterations, num_threads, precision_bits);
        break;

    default:
        if (proc_id == 0){
            printf("  Algorithm number selected not availabe, try with another number. \n");
            printf("\n");
        } 
        MPI_Finalize();
        exit(-1);
        break;
    }

    //Get time, check decimals, free pi and print the results
    if (proc_id == 0) {  
        gettimeofday(&t2, NULL);
        execution_time = ((t2.tv_sec - t1.tv_sec) * 1000000u +  t2.tv_usec - t1.tv_usec)/1.e6; 
        decimals_computed = mpfr_check_decimals(pi);
        if (print_in_csv_format) { print_results_csv("MPFR", algorithm_tag, precision, num_iterations, num_procs, num_threads, decimals_computed, execution_time); } 
        else { print_results("MPFR", algorithm_tag, precision, num_iterations, num_procs, num_threads, decimals_computed, execution_time); }
        mpfr_clear(pi);
    }

}

