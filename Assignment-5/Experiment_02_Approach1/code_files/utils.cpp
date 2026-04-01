#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

// ---------------------------------------------------------
// Cloud-In-Cell (CIC) Bilinear Interpolation
// ---------------------------------------------------------
void interpolation(double *mesh_value, Points *points) {
    // Reset mesh before interpolation
    for(int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = 0.0;
    }

    for (int i = 0; i < NUM_Points; i++) {
        // Find logical coordinates
        double lx = points[i].x * NX;
        double ly = points[i].y * NY;

        int i_idx = (int)lx;
        int j_idx = (int)ly;

        // Ensure we don't go out of bounds on the upper edge
        if (i_idx >= NX) i_idx = NX - 1;
        if (j_idx >= NY) j_idx = NY - 1;

        double hx = lx - i_idx;
        double hy = ly - j_idx;

        // Cloud-In-Cell weight distribution (assuming charge/mass = 1.0)
        double w00 = (1.0 - hx) * (1.0 - hy);
        double w10 = hx * (1.0 - hy);
        double w01 = (1.0 - hx) * hy;
        double w11 = hx * hy;

        // Accumulate to mesh
        mesh_value[i_idx + j_idx * GRID_X] += w00;
        mesh_value[(i_idx + 1) + j_idx * GRID_X] += w10;
        mesh_value[i_idx + (j_idx + 1) * GRID_X] += w01;
        mesh_value[(i_idx + 1) + (j_idx + 1) * GRID_X] += w11;
    }
}

// ---------------------------------------------------------
// APPROACH 1: Deferred Insertion (Serial)
// ---------------------------------------------------------
void mover_serial_deferred(Points *points, double dx, double dy) {
    int deleted_count = 0;
    int *voids = (int *)malloc(NUM_Points * sizeof(int));

    for (int i = 0; i < NUM_Points; i++) {
        double rx = ((double)rand() / RAND_MAX * 2.0 * dx) - dx;
        double ry = ((double)rand() / RAND_MAX * 2.0 * dy) - dy;

        points[i].x += rx;
        points[i].y += ry;

        // Check bounds and collect void indices
        if (points[i].x < 0.0 || points[i].x > 1.0 || points[i].y < 0.0 || points[i].y > 1.0) {
            voids[deleted_count++] = i;
        }
    }

    // Insert new particles at the void locations
    for (int k = 0; k < deleted_count; k++) {
        int idx = voids[k];
        points[idx].x = (double)rand() / RAND_MAX;
        points[idx].y = (double)rand() / RAND_MAX;
    }

    free(voids);
}

// ---------------------------------------------------------
// APPROACH 2: Immediate Replacement (Serial)
// ---------------------------------------------------------
void mover_serial_immediate(Points *points, double dx, double dy) {
    for (int i = 0; i < NUM_Points; i++) {
        double rx = ((double)rand() / RAND_MAX * 2.0 * dx) - dx;
        double ry = ((double)rand() / RAND_MAX * 2.0 * dy) - dy;

        double nx = points[i].x + rx;
        double ny = points[i].y + ry;

        if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0) {
            // Immediately replace
            points[i].x = (double)rand() / RAND_MAX;
            points[i].y = (double)rand() / RAND_MAX;
        } else {
            points[i].x = nx;
            points[i].y = ny;
        }
    }
}

// ---------------------------------------------------------
// APPROACH 1: Deferred Insertion (OpenMP Parallel)
// ---------------------------------------------------------
void mover_parallel_deferred(Points *points, double dx, double dy) {
    int total_deleted = 0;
    int *global_voids = (int *)malloc(NUM_Points * sizeof(int));

    #pragma omp parallel
    {
        // Thread-safe random seed
        unsigned int seed = 12345 ^ omp_get_thread_num();
        
        // Thread-local array to avoid atomic locks on every deletion
        int *local_voids = (int *)malloc(NUM_Points * sizeof(int));
        int local_count = 0;

        #pragma omp for
        for (int i = 0; i < NUM_Points; i++) {
            double rx = ((double)rand_r(&seed) / RAND_MAX * 2.0 * dx) - dx;
            double ry = ((double)rand_r(&seed) / RAND_MAX * 2.0 * dy) - dy;

            points[i].x += rx;
            points[i].y += ry;

            if (points[i].x < 0.0 || points[i].x > 1.0 || points[i].y < 0.0 || points[i].y > 1.0) {
                local_voids[local_count++] = i;
            }
        }

        // Synchronize and write local voids to the global void array
        int start_idx;
        #pragma omp atomic capture
        {
            start_idx = total_deleted;
            total_deleted += local_count;
        }

        for (int k = 0; k < local_count; k++) {
            global_voids[start_idx + k] = local_voids[k];
        }
        free(local_voids);

        // Ensure all threads finish collecting voids before insertion phase
        #pragma omp barrier

        // Parallel insertion phase
        #pragma omp for
        for (int k = 0; k < total_deleted; k++) {
            int idx = global_voids[k];
            points[idx].x = (double)rand_r(&seed) / RAND_MAX;
            points[idx].y = (double)rand_r(&seed) / RAND_MAX;
        }
    }

    free(global_voids);
}

// ---------------------------------------------------------
// APPROACH 2: Immediate Replacement (OpenMP Parallel)
// ---------------------------------------------------------
void mover_parallel_immediate(Points *points, double dx, double dy) {
    #pragma omp parallel
    {
        unsigned int seed = 12345 ^ omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < NUM_Points; i++) {
            double rx = ((double)rand_r(&seed) / RAND_MAX * 2.0 * dx) - dx;
            double ry = ((double)rand_r(&seed) / RAND_MAX * 2.0 * dy) - dy;

            double nx = points[i].x + rx;
            double ny = points[i].y + ry;

            if (nx < 0.0 || nx > 1.0 || ny < 0.0 || ny > 1.0) {
                points[i].x = (double)rand_r(&seed) / RAND_MAX;
                points[i].y = (double)rand_r(&seed) / RAND_MAX;
            } else {
                points[i].x = nx;
                points[i].y = ny;
            }
        }
    }
}

// ---------------------------------------------------------
// Helper: Save Mesh
// ---------------------------------------------------------
void save_mesh(double *mesh_value) {
    FILE *fp = fopen("Mesh.out", "wb");
    if (fp) {
        fwrite(mesh_value, sizeof(double), GRID_X * GRID_Y, fp);
        fclose(fp);
    }
}
