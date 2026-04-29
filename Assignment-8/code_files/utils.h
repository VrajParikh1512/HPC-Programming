#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include "init.h"

/* -----------------------------------------------------------------------
 * Partial (rank-local) versions — operate on particles [p_start, p_end).
 * These are the functions called from main.cpp for MPI+OpenMP execution.
 * ----------------------------------------------------------------------- */
void interpolation_partial(double *mesh_value, Points *points,
                            int *active, int p_start, int p_end);

void reverse_interpolation_partial(double *mesh_value, Points *points,
                                   double *F, int *active,
                                   int p_start, int p_end);

void mover_partial(Points *points, double *F, int *active,
                   int p_start, int p_end);

/* -----------------------------------------------------------------------
 * Full-range wrappers (operate on all NUM_Points particles).
 * Provided for backward compatibility and the final rank-0 save step.
 * ----------------------------------------------------------------------- */
void interpolation(double *mesh_value, Points *points, int *active);
void reverse_interpolation(double *mesh_value, Points *points,
                           double *F, int *active);
void mover(Points *points, double *F, int *active);

/* -----------------------------------------------------------------------
 * Mesh helpers
 * ----------------------------------------------------------------------- */
void find_min_max(double *mesh, double *min_val, double *max_val);
void normalize_mesh(double *mesh, double min_val, double max_val);
void denormalize_mesh(double *mesh, double min_val, double max_val);

/* -----------------------------------------------------------------------
 * I/O
 * ----------------------------------------------------------------------- */
void read_initial_points(FILE *file, Points *points);
void save_mesh(double *mesh_value);

#endif /* UTILS_H */
