#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    FILE *file = NULL;
    if (rank == 0) {
        file = fopen(argv[1], "rb");
        if (!file) { perror("input file"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fread(&NX, sizeof(int), 1, file);
        fread(&NY, sizeof(int), 1, file);
        fread(&NUM_Points, sizeof(int), 1, file);
        fread(&Maxiter, sizeof(int), 1, file);
    }
    MPI_Bcast(&NX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_Points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    GRID_X = NX + 1; GRID_Y = NY + 1;
    dx = 1.0 / NX; dy = 1.0 / NY;

    double *mesh_value = (double*) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points*) malloc(NUM_Points * sizeof(Points));
    double *F = (double*) malloc(NUM_Points * sizeof(double));
    int *active = (int*) malloc(NUM_Points * sizeof(int));

    if (rank == 0) {
        read_initial_points(file, points);
        fclose(file);
    }
    MPI_Bcast(points, NUM_Points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < NUM_Points; i++) active[i] = 1;

    double interp_time = 0.0, norm_time = 0.0, reverse_time = 0.0, mover_time = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {
        double t1 = MPI_Wtime();
        interpolation(mesh_value, points, active);
        double t2 = MPI_Wtime(); interp_time += (t2 - t1);

        t1 = MPI_Wtime();
        double min_val, max_val;
        find_min_max(mesh_value, &min_val, &max_val);
        normalize_mesh(mesh_value, min_val, max_val);
        t2 = MPI_Wtime(); norm_time += (t2 - t1);

        t1 = MPI_Wtime();
        reverse_interpolation(mesh_value, points, F, active);
        t2 = MPI_Wtime(); reverse_time += (t2 - t1);

        t1 = MPI_Wtime();
        mover(points, F, active);
        t2 = MPI_Wtime(); mover_time += (t2 - t1);

        t1 = MPI_Wtime();
        denormalize_mesh(mesh_value, min_val, max_val);
        t2 = MPI_Wtime(); norm_time += (t2 - t1);

        #pragma omp parallel for
        for (int i = 0; i < GRID_X * GRID_Y; i++) mesh_value[i] = 0.0;
    }

    double total_time = interp_time + norm_time + reverse_time + mover_time;

    if (rank == 0) {
        // Final interpolation to save correct mesh
        interpolation(mesh_value, points, active);
        save_mesh(mesh_value);

        // Print to console
        printf("Total: %f, Interp: %f, Norm: %f, Reverse: %f, Mover: %f\n",
               total_time, interp_time, norm_time, reverse_time, mover_time);

        // Append to CSV file
        /*FILE *csv = fopen("timing_data.csv", "a");
        if (csv) {
            fprintf(csv, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f\n",
                    size, omp_get_max_threads(), NX, NY, NUM_Points,
                    total_time, interp_time, norm_time, reverse_time, mover_time);
            fclose(csv);
        }*/
    }

    free(mesh_value); free(points); free(F); free(active);
    MPI_Finalize();
    return 0;
}
