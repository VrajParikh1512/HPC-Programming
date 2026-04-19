#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"

// Random particle initialization (optional)
void initializepoints(Points *points) {
    points->x = (double *) malloc(NUM_Points * sizeof(double));
    points->y = (double *) malloc(NUM_Points * sizeof(double));
    points->is_void = (bool *) malloc(NUM_Points * sizeof(bool));
    
    for (int i = 0; i < NUM_Points; i++) {
        points->x[i] = (double) rand() / RAND_MAX;
        points->y[i] = (double) rand() / RAND_MAX;
        points->is_void[i] = false;
    }
}

// Read particle positions from binary file
void read_points(FILE *file, Points *points) {
    points->x = (double *) malloc(NUM_Points * sizeof(double));
    points->y = (double *) malloc(NUM_Points * sizeof(double));
    points->is_void = (bool *) malloc(NUM_Points * sizeof(bool));
    
    for (int i = 0; i < NUM_Points; i++) {
        fread(&points->x[i], sizeof(double), 1, file);
        fread(&points->y[i], sizeof(double), 1, file);
        points->is_void[i] = false;   
    }
}