#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;

void interpolation(double *mesh_value, Points *points) {
    int num_cells = GRID_X * GRID_Y;
    memset(mesh_value, 0, num_cells * sizeof(double));

    double x_scale = (double)NX;
    double y_scale = (double)NY;
    double cell_area = dx * dy;

    int num_threads = omp_get_max_threads();
    
    double *thread_local_grids = (double*)calloc(num_threads * num_cells, sizeof(double));

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        double *my_grid = &thread_local_grids[tid * num_cells];

        #pragma omp for schedule(static)
        for (int p = 0; p < NUM_Points; p++) {
            if (points->is_void[p]) continue; 

            double x_pos = points->x[p];
            double y_pos = points->y[p];

            int i_cell = (int)(x_pos * x_scale);
            int j_cell = (int)(y_pos * y_scale);

            i_cell = (i_cell >= NX) ? NX - 1 : ((i_cell < 0) ? 0 : i_cell);
            j_cell = (j_cell >= NY) ? NY - 1 : ((j_cell < 0) ? 0 : j_cell);

            double x_frac = (x_pos * x_scale) - i_cell;
            double y_frac = (y_pos * y_scale) - j_cell;

            double x_comp = 1.0 - x_frac;
            double y_comp = 1.0 - y_frac;

            double w00 = x_comp * y_comp * cell_area;
            double w10 = x_frac * y_comp * cell_area;
            double w01 = x_comp * y_frac * cell_area;
            double w11 = x_frac * y_frac * cell_area;

            int base_idx = j_cell * GRID_X + i_cell;

            my_grid[base_idx] += w00;
            my_grid[base_idx + 1] += w10;
            my_grid[base_idx + GRID_X] += w01;
            my_grid[base_idx + GRID_X + 1] += w11;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < num_cells; c++) {
        double sum = 0.0;
        for (int th = 0; th < num_threads; th++) {
            sum += thread_local_grids[th * num_cells + c];
        }
        mesh_value[c] += sum;
    }

    free(thread_local_grids);
}

void normalization(double *mesh_value) {
    int num_cells = GRID_X * GRID_Y;
    min_val = mesh_value[0];
    max_val = mesh_value[0];
    
    #pragma omp parallel
    {
        double thread_min = mesh_value[0];
        double thread_max = mesh_value[0];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < num_cells; i++) {
            if (mesh_value[i] < thread_min) thread_min = mesh_value[i];
            if (mesh_value[i] > thread_max) thread_max = mesh_value[i];
        }
        
        #pragma omp critical
        {
            if (thread_min < min_val) min_val = thread_min;
            if (thread_max > max_val) max_val = thread_max;
        }
    }

    double range = max_val - min_val;
    if (range == 0.0) range = 1.0; 

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_cells; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - min_val) / range - 1.0;
    }
}

void mover(double *mesh_value, Points *points) {
    double x_scale = (double)NX;
    double y_scale = (double)NY;
    double cell_area = dx * dy;

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < NUM_Points; p++) {
        if (points->is_void[p]) continue;

        double x_pos = points->x[p];
        double y_pos = points->y[p];

        int i_cell = (int)(x_pos * x_scale);
        int j_cell = (int)(y_pos * y_scale);

        i_cell = (i_cell >= NX) ? NX - 1 : ((i_cell < 0) ? 0 : i_cell);
        j_cell = (j_cell >= NY) ? NY - 1 : ((j_cell < 0) ? 0 : j_cell);

        double x_frac = (x_pos * x_scale) - i_cell;
        double y_frac = (y_pos * y_scale) - j_cell;

        double x_comp = 1.0 - x_frac;
        double y_comp = 1.0 - y_frac;

        double w00 = x_comp * y_comp * cell_area;
        double w10 = x_frac * y_comp * cell_area;
        double w01 = x_comp * y_frac * cell_area;
        double w11 = x_frac * y_frac * cell_area;

        int base_idx = j_cell * GRID_X + i_cell;

        double interpolated_force = w00 * mesh_value[base_idx] +
                                    w10 * mesh_value[base_idx + 1] +
                                    w01 * mesh_value[base_idx + GRID_X] +
                                    w11 * mesh_value[base_idx + GRID_X + 1];

        points->x[p] += interpolated_force * dx;
        points->y[p] += interpolated_force * dy;

        if (points->x[p] < 0.0 || points->x[p] > 1.0 || 
            points->y[p] < 0.0 || points->y[p] > 1.0) {
            points->is_void[p] = true;
        }
    }
}

void denormalization(double *mesh_value) {
    int num_cells = GRID_X * GRID_Y;
    double range = max_val - min_val;
    if (range == 0.0) range = 1.0;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_cells; i++) {
        mesh_value[i] = (mesh_value[i] + 1.0) * range / 2.0 + min_val;
    }
}

long long int void_count(Points *points) {
    long long int void_counter = 0;
    #pragma omp parallel for reduction(+:void_counter)
    for (int p = 0; p < NUM_Points; p++) {
        void_counter += (int)points->is_void[p];
    }
    return void_counter;
}

void save_mesh(double *mesh_value) {
    FILE *output_file = fopen("Mesh.out", "w");
    if (!output_file) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }
    for (int r = 0; r < GRID_Y; r++) {
        for (int c = 0; c < GRID_X; c++) {
            fprintf(output_file, "%lf ", mesh_value[r * GRID_X + c]);
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);
}
