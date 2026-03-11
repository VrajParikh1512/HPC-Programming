#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv)
{

    // ==========================================
    // EXPERIMENT 03 CONFIGURATION PARAMETERS
    // ==========================================
    // Set NX and NY for the specific configuration:
    // Config 1: NX = 250,  NY = 100
    // Config 2: NX = 500,  NY = 200
    // Config 3: NX = 1000, NY = 400
    NX = 1000;
    NY = 400;
    Maxiter = 10;

    // Change NUM_Points to 100, 10000, 1000000, 100000000, 1000000000 for the 5 runs
    NUM_Points = 10;

    // Number of threads
    omp_set_num_threads(4);
    // ==========================================

    // Grid cell calculations
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // Allocate memory for grid and Points (Excluded from timing)
    double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *)calloc(NUM_Points, sizeof(Points));

    // NOTE: For Experiments 02 and 03, you must initialize particles exactly ONCE.
    // To do that, uncomment the line below, and comment out the one INSIDE the loop.
    // initializepoints(points);

    printf("--- HPC Assignment 04 : Experiment 03 ---\n");
    printf("Grid: %d x %d | Particles: %d | Iterations: %d\n", NX, NY, NUM_Points, Maxiter);
    printf("--------------------------------------------------\n");
    printf("Iter\tInterp(s)\tMover(s)\tTotal(s)\n");

    double total_time_all_iters = 0.0;

    for (int iter = 0; iter < Maxiter; iter++)
    {

        // 1. Initialize particles INSIDE the loop for Experiment 01
        initializepoints(points);

        // 2. Interpolation Phase Timing
        double start_interp = omp_get_wtime();
        interpolation(mesh_value, points);
        double end_interp = omp_get_wtime();

        // 3. Mover Phase Timing
        // Switch to mover_serial(points, dx, dy) for baseline serial timing
        // mover_serial(points, dx, dy);
        // double t3 = omp_get_wtime();

        // Calculate interpolation duration and accumulate it
        double interp_time = end_interp - start_interp;

        // Calculate durations
        // double interp_time = end_interp - start_interp;
        // double mover_time = t3 - end_interp;
        // double iter_total = t3 - start_interp;

        // Accumulate it
        // total_time_all_iters += iter_total;

        // Print iteration data in the required table format
        printf("Iter %d: %lf seconds\n", iter + 1, interp_time);
        // printf("%d\t%.4f\t\t%.4f\t\t%.4f\n", iter + 1, interp_time, mover_time, iter_total);

        // (The mover operations are omitted/commented out here because
        // Experiment 01 strictly analyzes interpolation scaling)
    }

    printf("--------------------------------------------------\n");
    printf("Total Execution Time (Over %d Iters) = %lf seconds\n", Maxiter, total_time_all_iters);

    // Save the final mesh output
    save_mesh(mesh_value);

    // Free memory to prevent memory leaks
    free(mesh_value);
    free(points);

    return 0;
}
