#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"
#include "init.h"

extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points, Maxiter;
extern double dx, dy;

// ----------------------------------------------------------------------
void find_min_max(double *mesh, double *min_val, double *max_val) {
    double local_min = 1e100, local_max = -1e100;
    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        if (mesh[i] < local_min) local_min = mesh[i];
        if (mesh[i] > local_max) local_max = mesh[i];
    }
    MPI_Allreduce(&local_min, min_val, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, max_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

// ----------------------------------------------------------------------
void normalize_mesh(double *mesh, double min_val, double max_val) {
    double range = max_val - min_val;
    if (range < 1e-12) range = 1.0;
    #pragma omp parallel for
    for (int i = 0; i < GRID_X * GRID_Y; i++)
        mesh[i] = -1.0 + 2.0 * (mesh[i] - min_val) / range;
}

// ----------------------------------------------------------------------
void denormalize_mesh(double *mesh, double min_val, double max_val) {
    double range = max_val - min_val;
    if (range < 1e-12) range = 1.0;
    #pragma omp parallel for
    for (int i = 0; i < GRID_X * GRID_Y; i++)
        mesh[i] = min_val + (mesh[i] + 1.0) * range * 0.5;
}

// ----------------------------------------------------------------------
void interpolation(double *mesh_value, Points *points, int *active) {
    int num_threads = omp_get_max_threads();
    double **private_mesh = (double**) malloc(num_threads * sizeof(double*));
    for (int t = 0; t < num_threads; t++)
        private_mesh[t] = (double*) calloc(GRID_X * GRID_Y, sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *local_mesh = private_mesh[tid];
        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {
            if (!active[p]) continue;
            double x = points[p].x, y = points[p].y;
            if (x >= 1.0) x = 1.0 - 1e-12;
            if (y >= 1.0) y = 1.0 - 1e-12;
            int i = (int)(x / dx);
            int j = (int)(y / dy);
            double lx = x - i * dx, ly = y - j * dy;
            double w00 = (dx - lx) * (dy - ly);
            double w10 = lx * (dy - ly);
            double w01 = (dx - lx) * ly;
            double w11 = lx * ly;
            int idx00 = j * GRID_X + i;
            int idx10 = j * GRID_X + (i+1);
            int idx01 = (j+1) * GRID_X + i;
            int idx11 = (j+1) * GRID_X + (i+1);
            local_mesh[idx00] += w00;
            local_mesh[idx10] += w10;
            local_mesh[idx01] += w01;
            local_mesh[idx11] += w11;
        }
    }

    #pragma omp parallel for
    for (int idx = 0; idx < GRID_X * GRID_Y; idx++) {
        double sum = 0.0;
        for (int t = 0; t < num_threads; t++)
            sum += private_mesh[t][idx];
        mesh_value[idx] = sum;
    }

    for (int t = 0; t < num_threads; t++) free(private_mesh[t]);
    free(private_mesh);
}

// ----------------------------------------------------------------------
void reverse_interpolation(double *mesh_value, Points *points, double *F, int *active) {
    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++) {
        if (!active[p]) { F[p] = 0.0; continue; }
        double x = points[p].x, y = points[p].y;
        if (x >= 1.0) x = 1.0 - 1e-12;
        if (y >= 1.0) y = 1.0 - 1e-12;
        int i = (int)(x / dx), j = (int)(y / dy);
        double lx = x - i * dx, ly = y - j * dy;
        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx * ly;
        int idx00 = j * GRID_X + i;
        int idx10 = j * GRID_X + (i+1);
        int idx01 = (j+1) * GRID_X + i;
        int idx11 = (j+1) * GRID_X + (i+1);
        F[p] = w00 * mesh_value[idx00] + w10 * mesh_value[idx10] +
               w01 * mesh_value[idx01] + w11 * mesh_value[idx11];
    }
}

// ----------------------------------------------------------------------
void mover(Points *points, double *F, int *active) {
    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++) {
        if (!active[p]) continue;
        double new_x = points[p].x + F[p] * dx;
        double new_y = points[p].y + F[p] * dy;
        if (new_x >= 0.0 && new_x <= 1.0 && new_y >= 0.0 && new_y <= 1.0) {
            points[p].x = new_x;
            points[p].y = new_y;
        } else {
            active[p] = 0;
        }
    }
}

// ----------------------------------------------------------------------
void read_initial_points(FILE *file, Points *points) {
    for (int i = 0; i < NUM_Points; i++) {
        fread(&points[i].x, sizeof(double), 1, file);
        fread(&points[i].y, sizeof(double), 1, file);
    }
}

// ----------------------------------------------------------------------
void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) { perror("Mesh.out"); exit(1); }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++)
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        fprintf(fd, "\n");
    }
    fclose(fd);
}
