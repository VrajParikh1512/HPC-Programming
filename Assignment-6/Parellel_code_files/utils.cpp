#include <omp.h>
#include "utils.h"
#include <vector>
#include <iostream>
#include <cstdio>
#include <algorithm>

void interpolation(double *mesh_value, Points *points)
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_area = inv_dx * inv_dy;

    const int total_nodes = GRID_X * GRID_Y;
    const int nthreads = omp_get_max_threads();

    // Private mesh per thread: nthreads x total_nodes
    std::vector<double> local_mesh((size_t)nthreads * total_nodes, 0.0);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double *local = local_mesh.data() + (size_t)tid * total_nodes;

        #pragma omp single
        {
            std::cout << "Number of threads = " << omp_get_num_threads() << std::endl;
        }

        #pragma omp for schedule(static)
        for (int p = 0; p < NUM_Points; p++)
        {
            double x = points[p].x;
            double y = points[p].y;

            int i = (int)(x * inv_dx);
            int j = (int)(y * inv_dy);

            if (i >= NX) i = NX - 1;
            if (j >= NY) j = NY - 1;
            if (i < 0) i = 0;
            if (j < 0) j = 0;

            double Xi = i * dx;
            double Yj = j * dy;

            double lx = x - Xi;
            double ly = y - Yj;

            double dx_minus_lx = dx - lx;
            double dy_minus_ly = dy - ly;

            double w00 = dx_minus_lx * dy_minus_ly * inv_area;
            double w10 = lx          * dy_minus_ly * inv_area;
            double w01 = dx_minus_lx * ly          * inv_area;
            double w11 = lx          * ly          * inv_area;

            int base = j * GRID_X + i;

            local[base]              += w00;
            local[base + 1]          += w10;
            local[base + GRID_X]     += w01;
            local[base + GRID_X + 1] += w11;
        }
    }

    const double inv_N = 1.0 / (double)NUM_Points;

    // Parallel merge + normalization
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < total_nodes; k++)
    {
        double sum = 0.0;
        for (int t = 0; t < nthreads; t++)
        {
            sum += local_mesh[(size_t)t * total_nodes + k];
        }
        mesh_value[k] = sum * inv_N;
    }
}

void save_mesh(double *mesh_value)
{
    FILE *fp_bin = fopen("mesh_output.bin", "wb");
    if (!fp_bin) {
        perror("Error: Unable to create mesh_output.bin");
        return;
    }

    fwrite(&GRID_X, sizeof(int), 1, fp_bin);
    fwrite(&GRID_Y, sizeof(int), 1, fp_bin);
    fwrite(mesh_value, sizeof(double), GRID_X * GRID_Y, fp_bin);
    fclose(fp_bin);

    FILE *fp_csv = fopen("mesh_output.csv", "w");
    if (!fp_csv) {
        perror("Error: Unable to create mesh_output.csv");
        return;
    }

    for (int j = 0; j < GRID_Y; j++) {
        for (int i = 0; i < GRID_X; i++) {
            fprintf(fp_csv, "%.6f", mesh_value[j * GRID_X + i]);
            if (i < GRID_X - 1) fprintf(fp_csv, ",");
        }
        fprintf(fp_csv, "\n");
    }

    fclose(fp_csv);

    printf("Mesh saved to 'mesh_output.bin' and 'mesh_output.csv' (%d x %d nodes).\n",
           GRID_X, GRID_Y);
}