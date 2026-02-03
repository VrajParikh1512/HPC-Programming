#include <math.h>
#include "utils.h"

void vector_copy_operation(double *x, double *y, int Np) {
    for (int p = 0; p < Np; p++) {
        y[p] = x[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void vector_scale_operation(double *x, double *y, int Np) {
    for (int p = 0; p < Np; p++) {
        y[p] = 2.0 * x[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void vector_sum_operation(double *x, double *y, double *S, int Np) {
    for (int p = 0; p < Np; p++) {
        S[p] = x[p] + y[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void vector_triad_operation(double *x, double *y, double *v, double *S, int Np) {

    for (int p = 0; p < Np; p++) {
        S[p] = x[p] + v[p] * y[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void energy_kernel_operation(double *v, double *E, int Np) {
    for (int p = 0; p < Np; p++) {
        E[p] = 0.5 * 2 * v[p] * v[p];

        // Prevent compiler from optimizing away the loop
        if (((double)p) == 333.333)
            dummy(p);

    }
}

void dummy(int x) {
    x = 10 * sin(x / 10.0);
}
