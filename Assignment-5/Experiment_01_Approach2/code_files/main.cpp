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
    // CONFIGURATION 1 PARAMETERS
    // Change these for Config 2 (500x200) and Config 3 (1000x400)
    // ==========================================
    NX = 250; 
    NY = 100;
    Maxiter = 10;
    
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // The particle counts to test: 10^2, 10^4, 10^6, 10^8, 10^9
    // Note: 10^9 requires ~16GB of RAM. If your Lab PC crashes, test up to 10^8 locally.
    int particle_counts[] = {100, 10000, 1000000, 100000000, 1000000000};
    int num_tests = 5;

    printf("--- HPC Assignment 05 : Experiment 01 ---\n");
    printf("Grid: %d x %d | Iterations: %d\n", NX, NY, Maxiter);
    printf("--------------------------------------------------\n");

    for (int t = 0; t < num_tests; t++) {
        NUM_Points = particle_counts[t];
        
        // Allocate memory
        double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points *)calloc(NUM_Points, sizeof(Points));

        if (points == NULL || mesh_value == NULL) {
            printf("Memory allocation failed for %d particles.\n", NUM_Points);
            continue; // Skip if out of memory
        }

        // 1. Initialize particles exactly ONCE outside the loop
        initializepoints(points);

        double total_time_all_iters = 0.0;

        // 2. Execute simulation loop
        for (int iter = 0; iter < Maxiter; iter++) {
            double start_time = omp_get_wtime();

            // Interpolation Phase
            interpolation(mesh_value, points);

            // Mover Phase (Using the Serial Immediate Replacement approach)
            //mover_serial_deferred(points, dx, dy);
            mover_serial_immediate(points, dx, dy);

            double end_time = omp_get_wtime();
            total_time_all_iters += (end_time - start_time);
        }

        printf("Particles: %10d | Total Exec Time (10 Iters): %lf seconds\n", NUM_Points, total_time_all_iters);

        // Free memory before the next particle count iteration
        free(mesh_value);
        free(points);
    }

    return 0;
}
