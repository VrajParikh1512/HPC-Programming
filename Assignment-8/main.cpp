#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "init.h"
#include "utils.h"

/* Global variables (declared extern in init.h / utils.h) */
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    /* ── MPI Initialization ──────────────────────────────────────────── */
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* ── Read binary input file (all ranks read; avoids broadcast overhead) ── */
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        if (rank == 0) printf("Error opening input file\n");
        MPI_Finalize();
        return 1;
    }

    fread(&NX,         sizeof(int), 1, file);
    fread(&NY,         sizeof(int), 1, file);
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter,    sizeof(int), 1, file);

    /* Grid has NX+1 nodes in x, NY+1 nodes in y */
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx     = 1.0 / NX;
    dy     = 1.0 / NY;

    if (rank == 0) {
        printf("Grid: %d x %d  |  Particles: %d  |  Iterations: %d  |  MPI ranks: %d\n",
               NX, NY, NUM_Points, Maxiter, nprocs);
    }

    /* ── Allocate memory ─────────────────────────────────────────────── */
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points     = (Points *) calloc(NUM_Points,       sizeof(Points));

    /* ── Load first iteration's particle positions ───────────────────── */
    read_points(file, points);

    /* ── Timing accumulators (use MPI_Wtime for wall-clock accuracy) ─── */
    double total_int_time   = 0.0;
    double total_norm_time  = 0.0;
    double total_move_time  = 0.0;
    double total_denorm_time= 0.0;

    /* ── Main iteration loop ─────────────────────────────────────────── */
    for (int iter = 0; iter < Maxiter; iter++) {

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        interpolation(mesh_value, points);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        normalization(mesh_value);

        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();

        mover(mesh_value, points);

        MPI_Barrier(MPI_COMM_WORLD);
        double t3 = MPI_Wtime();

        denormalization(mesh_value);

        MPI_Barrier(MPI_COMM_WORLD);
        double t4 = MPI_Wtime();

        total_int_time    += t1 - t0;
        total_norm_time   += t2 - t1;
        total_move_time   += t3 - t2;
        total_denorm_time += t4 - t3;
    }

    /* ── Output (rank 0 only) ────────────────────────────────────────── */
    if (rank == 0) {
        save_mesh(mesh_value);

        printf("Total Interpolation Time  = %lf seconds\n", total_int_time);
        printf("Total Normalization Time  = %lf seconds\n", total_norm_time);
        printf("Total Mover Time          = %lf seconds\n", total_move_time);
        printf("Total Denormalization Time= %lf seconds\n", total_denorm_time);
        printf("Total Algorithm Time      = %lf seconds\n",
               total_int_time + total_norm_time + total_move_time + total_denorm_time);
        printf("Total Number of Voids     = %lld\n", void_count(points));
    }

    /* ── Cleanup ─────────────────────────────────────────────────────── */
    free(mesh_value);
    free(points);
    fclose(file);

    MPI_Finalize();
    return 0;
}