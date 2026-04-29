#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
#ifdef _OPENMP
    #include <omp.h>
#else
    #include <time.h>
    // Fallback timer for serial compilation (no OpenMP)
    double omp_get_wtime() {
        return (double)clock() / CLOCKS_PER_SEC;
    }
#endif
 
#include "init.h"
#include "utils.h"
 
// Global simulation parameters (definitions; declared extern in init.h)
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;
 
int main(int argc, char **argv) {
 
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
 
    // ── Open binary input file ────────────────────────────────────────────
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening input file: %s\n", argv[1]);
        return 1;
    }
 
    // ── Read global parameters ────────────────────────────────────────────
    fread(&NX,         sizeof(int), 1, file);
    fread(&NY,         sizeof(int), 1, file);
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter,    sizeof(int), 1, file);
 
    // Grid nodes = cells + 1 in each dimension
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx     = 1.0 / NX;
    dy     = 1.0 / NY;
 
    // ── Allocate mesh and particle arrays ─────────────────────────────────
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points     = (Points *) calloc(NUM_Points,       sizeof(Points));
 
    // ── Per-phase accumulated timers ─────────────────────────────────────
    double total_int_time    = 0.0;
    double total_norm_time   = 0.0;
    double total_move_time   = 0.0;
    double total_denorm_time = 0.0;
 
    // Read initial particle positions (single read before the time loop)
    read_points(file, points);
 
    // ── Main time-stepping loop ───────────────────────────────────────────
    for (int iter = 0; iter < Maxiter; iter++) {
 
        double t0 = omp_get_wtime();
        interpolation(mesh_value, points);
        double t1 = omp_get_wtime();
 
        normalization(mesh_value);
        double t2 = omp_get_wtime();
 
        mover(mesh_value, points);
        double t3 = omp_get_wtime();
 
        denormalization(mesh_value);
        double t4 = omp_get_wtime();
 
        total_int_time    += (t1 - t0);
        total_norm_time   += (t2 - t1);
        total_move_time   += (t3 - t2);
        total_denorm_time += (t4 - t3);
    }
 
    // ── Save output and print timing summary ──────────────────────────────
    save_mesh(mesh_value);
 
    const double total_alg_time = total_int_time  + total_norm_time +
                                  total_move_time + total_denorm_time;
 
    printf("Total Interpolation Time  = %lf seconds\n", total_int_time);
    printf("Total Normalization Time  = %lf seconds\n", total_norm_time);
    printf("Total Mover Time          = %lf seconds\n", total_move_time);
    printf("Total Denormalization Time= %lf seconds\n", total_denorm_time);
    printf("Total Algorithm Time      = %lf seconds\n", total_alg_time);
    printf("Total Number of Voids     = %lld\n",        void_count(points));
 
    // Machine-readable CSV line for the benchmark script
    printf("CSV,%lf,%lf,%lf\n", total_alg_time, total_int_time, total_move_time);
 
    // ── Free resources ────────────────────────────────────────────────────
    free(mesh_value);
    free(points);
    fclose(file);
 
    return 0;
}
