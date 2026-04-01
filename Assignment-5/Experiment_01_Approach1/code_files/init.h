#ifndef INIT_H
#define INIT_H

// Particle data structure
typedef struct {
    double x;
    double y;
} Points;

// Global variables declaration
extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points, Maxiter;
extern double dx, dy;

// Function declarations
void initializepoints(Points *points);

#endif
