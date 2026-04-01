#include <stdlib.h>
#include <time.h>
#include "init.h"

void initializepoints(Points *points) {
    // Seed the random number generator
    srand(time(NULL));

    // Initialize all particles randomly inside the 1x1 domain
    for (int i = 0; i < NUM_Points; i++) {
        points[i].x = (double)rand() / RAND_MAX;
        points[i].y = (double)rand() / RAND_MAX;
    }
}
