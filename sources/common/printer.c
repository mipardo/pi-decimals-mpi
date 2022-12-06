#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"


void print_title(){
    FILE * file;
    file = fopen("resources/pi_decimals_title.txt", "r");
    if(file == NULL){
        printf("pi_decimals_title.txt not found \n");
        exit(-1);
    } 

    char character;
    int i = 0;
    while((character = fgetc(file)) != EOF){
        printf("%c", character);
    }
    
    fclose(file);
}

void check_errors(int num_procs, int precision, int num_iterations, int num_threads, int proc_id){
    if (precision <= 0){
        if(proc_id == 0) printf("  Precision should be greater than cero. \n\n");
        MPI_Finalize();
        exit(-1);
    } 
    if (num_iterations < (num_threads * num_procs)){
        if(proc_id == 0){
            printf("  The number of iterations required for the computation is too small to be solved with %d threads and %d procesess. \n", num_threads, num_procs);
            printf("  Try using a greater precision or lower threads/processes number. \n\n");
        }
        MPI_Finalize();
        exit(-1);
    }
}

void print_results(char *library, char *algorithm_tag, int precision, int num_iterations, int num_procs, int num_threads, int decimals_computed, double execution_time) {
    printf("  Library used: %s \n", library);
    printf("  Algorithm: %s \n", algorithm_tag);
    printf("  Precision used: %d \n", precision);
    printf("  Number of iterations: %d \n", num_iterations);
    printf("  Number of processes: %d\n", num_procs);
    printf("  Number of threads (per process): %d\n", num_threads);
    if (decimals_computed >= precision) { printf("  Correct decimals: %d \n", decimals_computed); } 
    else { printf("  Something went wrong. The execution just achieved %d decimals \n", decimals_computed); }
    printf("  Execution time: %f seconds \n", execution_time);
    printf("\n");
}

void print_results_csv(char *library, char *algorithm_tag, int precision, int num_iterations, int num_procs, int num_threads, int decimals_computed, double execution_time) {
    printf("MPI;");
    printf("%s;", library);
    printf("%s;", algorithm_tag);
    printf("%d;", precision);
    printf("%d;", num_iterations);
    printf("%d;", num_procs);
    printf("%d;", num_threads);
    printf("%d;", decimals_computed);
    printf("%f;\n", execution_time);
}
