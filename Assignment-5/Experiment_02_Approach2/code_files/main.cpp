#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {
    // ==========================================
    // CONFIGURATION 3 PARAMETERS (Change for 1 & 2)
    // Config 1: 250x100 | Config 2: 500x200 | Config 3: 1000x400
    // ==========================================
    NX = 1000; 
    NY = 400;
    Maxiter = 10;
    
    // Fixed at 14 Million for Scalability Test
    NUM_Points = 14000000; 

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // Allocate memory
    double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *)calloc(NUM_Points, sizeof(Points));

    if (points == NULL || mesh_value == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    // Initialize particles exactly ONCE outside the loop
    initializepoints(points);

    printf("--- HPC Assignment 05 : Experiment 02 ---\n");
    printf("Grid: %d x %d | Particles: %d | Iters: %d\n", NX, NY, NUM_Points, Maxiter);
    printf("--------------------------------------------------\n");

    // Array of threads to test (1 is included to get the baseline for Speedup)
    int threads_to_test[] = {1, 2, 4, 8, 16};
    int num_tests = 5;

    for (int t = 0; t < num_tests; t++) {
        int num_threads = threads_to_test[t];
        omp_set_num_threads(num_threads);
        
        printf("\n=> Execution with %d Threads\n", num_threads);
        
        double total_interp = 0.0, total_mover = 0.0, total_iter = 0.0;

        for (int iter = 0; iter < Maxiter; iter++) {
            // 1. Interpolation Phase
            double start_interp = omp_get_wtime();
            interpolation(mesh_value, points);
            double end_interp = omp_get_wtime();

            // 2. Mover Phase (Parallel Immediate Replacement)
            mover_parallel_immediate(points, dx, dy);
            double end_mover = omp_get_wtime();

            // Accumulate timings
            total_interp += (end_interp - start_interp);
            total_mover += (end_mover - end_interp);
            total_iter += (end_mover - start_interp);
        }

        printf("Interpolation Time : %lf seconds\n", total_interp);
        printf("Mover Time         : %lf seconds\n", total_mover);
        printf("Total Runtime      : %lf seconds\n", total_iter);
    }

    free(mesh_value);
    free(points);

    return 0;
}
