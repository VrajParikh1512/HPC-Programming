#include "utils.h"
#include <stdlib.h>
//#include <immintrin.h> 
#include <algorithm>   
#include <string.h>   

void interpolation(double *mesh_value, Points *points)
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_area = inv_dx * inv_dy;   // 1 / (dx * dy)

    for (int p = 0; p < NUM_Points; p++)
    {
        double x = points[p].x;
        double y = points[p].y;

        // Cell index (fast)
        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;

        // Lower-left node
        double Xi = i * dx;
        double Yj = j * dy;

        // Local distances
        double lx = x - Xi;
        double ly = y - Yj;

        double dx_minus_lx = dx - lx;
        double dy_minus_ly = dy - ly;

        // Normalized bilinear weights
        double w00 = dx_minus_lx * dy_minus_ly * inv_area;
        double w10 = lx          * dy_minus_ly * inv_area;
        double w01 = dx_minus_lx * ly          * inv_area;
        double w11 = lx          * ly          * inv_area;

        int base = j * GRID_X + i;

        mesh_value[base]                 += w00;
        mesh_value[base + 1]             += w10;
        mesh_value[base + GRID_X]        += w01;
        mesh_value[base + GRID_X + 1]    += w11;
    }
}

void save_mesh(double *mesh_value)
{
    // --- Binary output ---
    FILE *fp_bin = fopen("mesh_output.bin", "wb");
    if (!fp_bin) {
        perror("Error: Unable to create mesh_output.bin");
        return;
    }
 
    // Write grid dimensions so the file is self-describing
    fwrite(&GRID_X, sizeof(int), 1, fp_bin);
    fwrite(&GRID_Y, sizeof(int), 1, fp_bin);
 
    // Write the flat mesh array (GRID_Y rows x GRID_X cols)
    fwrite(mesh_value, sizeof(double), GRID_X * GRID_Y, fp_bin);
    fclose(fp_bin);
 
    // --- Human-readable CSV output ---
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