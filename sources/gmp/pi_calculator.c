#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>
#include <stdbool.h>
#include "mpi.h"
#include "algorithms/bbp_blocks_and_cyclic.h"
#include "algorithms/bellard_blocks_and_cyclic.h"
#include "algorithms/chudnovsky_blocks_and_blocks.h"
#include "algorithms/chudnovsky_snake_like_and_blocks.h"
#include "algorithms/chudnovsky_non_uniform_and_blocks.h"
#include "check_decimals.h"
#include "../common/printer.h"


double gettimeofday();


void calculate_pi_gmp(int num_procs, int proc_id, int algorithm, int precision, int num_threads, bool print_in_csv_format){
    double execution_time;
    struct timeval t1, t2;
    int num_iterations, decimals_computed; 
    mpf_t pi;
    char *algorithm_tag;


    //Get init time 
    if(proc_id == 0){
        gettimeofday(&t1, NULL);
    }

    //Set gmp float precision (in bits) and init pi
    mpf_set_default_prec(precision * 8); 
    if (proc_id == 0){
        mpf_init_set_ui(pi, 0);
    }
    
    
    switch (algorithm)
    {
    case 0:
        num_iterations = precision * 0.84;
        check_errors(num_procs, precision, num_iterations, num_threads, proc_id);
        algorithm_tag = "GMP-BBP-BLC-CYC";
        bbp_blocks_and_cyclic_algorithm_gmp(num_procs, proc_id, pi, num_iterations, num_threads);
        break;

    case 1:
        num_iterations = precision / 3;
        check_errors(num_procs, precision, num_iterations, num_threads, proc_id);
        algorithm_tag = "GMP-BEL-BLC-CYC";
        bellard_blocks_and_cyclic_algorithm_gmp(num_procs, proc_id, pi, num_iterations, num_threads);
        break;

    case 2:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        check_errors(num_procs, precision, num_iterations, num_threads, proc_id);
        algorithm_tag = "GMP-CHD-BLC-BLC-SME";
        chudnovsky_blocks_and_blocks_algorithm_gmp(num_procs, proc_id, pi, num_iterations, num_threads);
        break;
    
    case 3:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        check_errors(num_procs, precision, num_iterations, num_threads, proc_id);
        algorithm_tag = "GMP-CHD-SNK-BLC-SME";
        chudnovsky_snake_like_and_blocks_algorithm_gmp(num_procs, proc_id, pi, num_iterations, num_threads);
        break;

    case 4:
        num_iterations = (precision + 14 - 1) / 14;  //Division por exceso
        check_errors(num_procs, precision, num_iterations, num_threads, proc_id);
        algorithm_tag = "GMP-CHD-CHT-BLC-SME";
        chudnovsky_non_uniform_and_blocks_algorithm_gmp(num_procs, proc_id, pi, num_iterations, num_threads);
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
        decimals_computed = check_decimals_gmp(pi);
        if (print_in_csv_format) { 
            print_results_csv("GMP", algorithm_tag, precision, num_iterations, num_procs, num_threads, decimals_computed, execution_time); 
        } else { 
            print_results("GMP", algorithm_tag, precision, num_iterations, num_procs, num_threads, decimals_computed, execution_time); 
        }
        mpf_clear(pi);
    }

}



