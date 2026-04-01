#ifndef UTILS_H
#define UTILS_H

#include "init.h"

// Interpolation
void interpolation(double *mesh_value, Points *points);

// Serial Movers
void mover_serial_deferred(Points *points, double dx, double dy);
void mover_serial_immediate(Points *points, double dx, double dy);

// Parallel Movers
void mover_parallel_deferred(Points *points, double dx, double dy);
void mover_parallel_immediate(Points *points, double dx, double dy);

// Output
void save_mesh(double *mesh_value);

#endif
