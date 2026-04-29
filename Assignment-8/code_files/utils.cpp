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

/* ======================================================================
 * find_min_max
 * Computes global min/max of the mesh across all MPI ranks.
 * The MPI_Allreduce ensures every rank gets the same global values.
 * ====================================================================== */
void find_min_max(double *mesh, double *min_val, double *max_val) {
    double local_min =  1e100;
    double local_max = -1e100;
    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        if (mesh[i] < local_min) local_min = mesh[i];
        if (mesh[i] > local_max) local_max = mesh[i];
    }
    /* After interpolation+Allreduce every rank holds the same full mesh,
     * so local_min/max are already global — but we keep the Allreduce for
     * safety when this function is called in other contexts.              */
    MPI_Allreduce(&local_min, min_val, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, max_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

/* ======================================================================
 * normalize_mesh  — maps mesh values to [-1, 1]
 * ====================================================================== */
void normalize_mesh(double *mesh, double min_val, double max_val) {
    double range = max_val - min_val;
    if (range < 1e-12) range = 1.0;
    #pragma omp parallel for
    for (int i = 0; i < GRID_X * GRID_Y; i++)
        mesh[i] = -1.0 + 2.0 * (mesh[i] - min_val) / range;
}

/* ======================================================================
 * denormalize_mesh  — inverse of normalize_mesh
 * NOTE: this is provided for completeness but is NOT called in the main
 * loop because the mesh is discarded (memset to 0) at the start of each
 * new iteration anyway.  Calling denormalize before that reset was a
 * source of unnecessary computation in the original code.
 * ====================================================================== */
void denormalize_mesh(double *mesh, double min_val, double max_val) {
    double range = max_val - min_val;
    if (range < 1e-12) range = 1.0;
    #pragma omp parallel for
    for (int i = 0; i < GRID_X * GRID_Y; i++)
        mesh[i] = min_val + (mesh[i] + 1.0) * range * 0.5;
}

/* ======================================================================
 * interpolation_partial
 * Forward scatter: particles [p_start, p_end) -> mesh.
 *
 * Uses per-thread private meshes to avoid atomic operations, then
 * reduces them into mesh_value.  The caller is responsible for
 * zeroing mesh_value before calling and for summing partial meshes
 * from all MPI ranks via MPI_Allreduce afterwards.
 * ====================================================================== */
void interpolation_partial(double *mesh_value, Points *points,
                            int *active, int p_start, int p_end) {
    int num_threads = omp_get_max_threads();
    int mesh_size   = GRID_X * GRID_Y;

    /* Allocate one private mesh per OpenMP thread */
    double **private_mesh = (double **) malloc(num_threads * sizeof(double *));
    for (int t = 0; t < num_threads; t++)
        private_mesh[t] = (double *) calloc(mesh_size, sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *local_mesh = private_mesh[tid];

        #pragma omp for schedule(dynamic, 512)
        for (int p = p_start; p < p_end; p++) {
            if (!active[p]) continue;

            double x = points[p].x;
            double y = points[p].y;

            /* Clamp to avoid out-of-bounds index at the upper edge */
            if (x >= 1.0) x = 1.0 - 1e-12;
            if (y >= 1.0) y = 1.0 - 1e-12;

            int i = (int)(x / dx);
            int j = (int)(y / dy);

            double lx = x - i * dx;
            double ly = y - j * dy;

            /* Bilinear weights — sum = dx*dy (correct for PIC with fi=1) */
            double w00 = (dx - lx) * (dy - ly);
            double w10 =       lx  * (dy - ly);
            double w01 = (dx - lx) *       ly;
            double w11 =       lx  *       ly;

            local_mesh[j       * GRID_X + i    ] += w00;
            local_mesh[j       * GRID_X + (i+1)] += w10;
            local_mesh[(j+1)   * GRID_X + i    ] += w01;
            local_mesh[(j+1)   * GRID_X + (i+1)] += w11;
        }
    }

    /* Reduce all private meshes into mesh_value */
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < mesh_size; idx++) {
        double sum = 0.0;
        for (int t = 0; t < num_threads; t++)
            sum += private_mesh[t][idx];
        mesh_value[idx] += sum;   /* += so caller can pre-initialise if needed */
    }

    for (int t = 0; t < num_threads; t++) free(private_mesh[t]);
    free(private_mesh);
}

/* ======================================================================
 * reverse_interpolation_partial
 * Gather: mesh -> F values for particles [p_start, p_end).
 * Read-only on mesh_value, so no race conditions; fully OpenMP-parallel.
 * ====================================================================== */
void reverse_interpolation_partial(double *mesh_value, Points *points,
                                   double *F, int *active,
                                   int p_start, int p_end) {
    #pragma omp parallel for schedule(dynamic, 512)
    for (int p = p_start; p < p_end; p++) {
        if (!active[p]) { F[p] = 0.0; continue; }

        double x = points[p].x;
        double y = points[p].y;

        if (x >= 1.0) x = 1.0 - 1e-12;
        if (y >= 1.0) y = 1.0 - 1e-12;

        int i = (int)(x / dx);
        int j = (int)(y / dy);

        double lx = x - i * dx;
        double ly = y - j * dy;

        double w00 = (dx - lx) * (dy - ly);
        double w10 =       lx  * (dy - ly);
        double w01 = (dx - lx) *       ly;
        double w11 =       lx  *       ly;

        F[p] = w00 * mesh_value[j       * GRID_X + i    ]
             + w10 * mesh_value[j       * GRID_X + (i+1)]
             + w01 * mesh_value[(j+1)   * GRID_X + i    ]
             + w11 * mesh_value[(j+1)   * GRID_X + (i+1)];
    }
}

/* ======================================================================
 * mover_partial
 * Update positions of particles [p_start, p_end) using field F.
 * Particles that leave [0,1]x[0,1] are deactivated.
 * ====================================================================== */
void mover_partial(Points *points, double *F, int *active,
                   int p_start, int p_end) {
    #pragma omp parallel for schedule(dynamic, 512)
    for (int p = p_start; p < p_end; p++) {
        if (!active[p]) continue;

        double new_x = points[p].x + F[p] * dx;
        double new_y = points[p].y + F[p] * dy;

        if (new_x >= 0.0 && new_x <= 1.0 &&
            new_y >= 0.0 && new_y <= 1.0) {
            points[p].x = new_x;
            points[p].y = new_y;
        } else {
            active[p] = 0;
        }
    }
}

/* ======================================================================
 * Legacy wrappers — operate on the full particle array.
 * Used by rank 0 for the final save-interpolation.
 * ====================================================================== */
void interpolation(double *mesh_value, Points *points, int *active) {
    interpolation_partial(mesh_value, points, active, 0, NUM_Points);
}

void reverse_interpolation(double *mesh_value, Points *points,
                           double *F, int *active) {
    reverse_interpolation_partial(mesh_value, points, F, active,
                                  0, NUM_Points);
}

void mover(Points *points, double *F, int *active) {
    mover_partial(points, F, active, 0, NUM_Points);
}

/* ======================================================================
 * I/O helpers
 * ====================================================================== */
void read_initial_points(FILE *file, Points *points) {
    for (int i = 0; i < NUM_Points; i++) {
        fread(&points[i].x, sizeof(double), 1, file);
        fread(&points[i].y, sizeof(double), 1, file);
    }
}

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
