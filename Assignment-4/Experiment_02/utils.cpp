#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Optimized Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {
    // Reset mesh values for the current iteration
    memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

    //Strength Reduction
    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;

    for (int i = 0; i < NUM_Points; i++) {
        double px = points[i].x;
        double py = points[i].y;

        // Calculate grid indices
        int j = (int)(px * inv_dx);
        int k = (int)(py * inv_dy);

        // Safety check to ensure particles are within the 1x1 domain
        if (j >= 0 && j < NX && k >= 0 && k < NY) {
            double x_weight = (px - (j * dx)) * inv_dx;
            double y_weight = (py - (k * dy)) * inv_dy;

            // Bilinear Interpolation logic
            mesh_value[k * GRID_X + j]         += (1.0 - x_weight) * (1.0 - y_weight);
            mesh_value[k * GRID_X + (j + 1)]     += x_weight * (1.0 - y_weight);
            mesh_value[(k + 1) * GRID_X + j]     += (1.0 - x_weight) * y_weight;
            mesh_value[(k + 1) * GRID_X + (j + 1)] += x_weight * y_weight;
        }
    }
}

// Stochastic Mover (Serial Code) 
void mover_serial(Points *points, double deltaX, double deltaY) {
    for (int i = 0; i < NUM_Points; i++) {
        double new_x, new_y;
        int valid = 0;
        
        while (!valid) {
            // Generate random displacements within +/- deltaX and +/- deltaY 
            double rx = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            double ry = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            
            new_x = points[i].x + (rx * deltaX);
            new_y = points[i].y + (ry * deltaY);
            
            // Check if updated position remains inside the 1x1 grid 
            if (new_x >= 0.0 && new_x <= 1.0 && new_y >= 0.0 && new_y <= 1.0) {
                valid = 1;
            }
        }
        points[i].x = new_x;
        points[i].y = new_y;
    }
}

// Stochastic Mover (Parallel Code) 
void mover_parallel(Points *points, double deltaX, double deltaY) {
    // Parallelize Mover using basic OpenMP directives
    #pragma omp parallel for
    for (int i = 0; i < NUM_Points; i++) {
        // Use thread-safe random seeding for parallel performance
        unsigned int seed = (unsigned int)(omp_get_thread_num() + i);
        double new_x, new_y;
        int valid = 0;
        
        while (!valid) {
            double rx = ((double)rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
            double ry = ((double)rand_r(&seed) / RAND_MAX) * 2.0 - 1.0;
            
            new_x = points[i].x + (rx * deltaX);
            new_y = points[i].y + (ry * deltaY);
            
            if (new_x >= 0.0 && new_x <= 1.0 && new_y >= 0.0 && new_y <= 1.0) {
                valid = 1;
            }
        }
        points[i].x = new_x;
        points[i].y = new_y;
    }
}

// Write mesh to file
void save_mesh(double *mesh_value) {

    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}
