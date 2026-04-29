#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

// PIC operations
void interpolation(double *mesh_value, Points *points, int *active);
void save_mesh(double *mesh_value);

// Helper functions for normalization and reverse interpolation
void find_min_max(double *mesh, double *min_val, double *max_val);
void normalize_mesh(double *mesh, double min_val, double max_val);
void denormalize_mesh(double *mesh, double min_val, double max_val);
void reverse_interpolation(double *mesh_value, Points *points, double *F, int *active);
void mover(Points *points, double *F, int *active);

// I/O helper
void read_initial_points(FILE *file, Points *points);

#endif
