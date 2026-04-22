/*
 * utils.cpp  —  Parallel PIC Interpolation Pipeline
 * Uses MPI (distributed across processes) + OpenMP (threaded within each process)
 *
 * Pipeline per iteration:
 *   interpolation()   : scatter particles → grid  (particle-to-mesh)
 *   normalization()   : scale grid values to [-1, 1]
 *   mover()           : grid → particles (mesh-to-particle), update positions
 *   denormalization() : restore grid to original value range
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>
#include "utils.h"

/* Global min/max saved during normalization for later denormalization */
double min_val, max_val;

/* =========================================================
 * INTERPOLATION  (Particle → Mesh)
 *
 * For each particle p at (x, y) with f_i = 1:
 *   1. Find cell indices  i = floor(x/dx),  j = floor(y/dy)
 *   2. Local offsets      lx = x - i*dx,    ly = y - j*dy
 *   3. Bilinear weights:
 *        w(i,  j  ) = (dx-lx)*(dy-ly)
 *        w(i+1,j  ) =    ly  *(dx-lx)
 *        w(i,  j+1) =   lx  *(dy-ly)
 *        w(i+1,j+1) =   lx  *   ly
 *   4. Accumulate  F[node] += w * f_i
 *
 * MPI strategy  : Partition particles across MPI ranks.
 *                 Each rank accumulates into a local partial grid,
 *                 then MPI_Allreduce sums all partial grids.
 * OpenMP strategy: Within each rank, threads own private partial grids
 *                  (avoids atomic overhead on hot grid cells),
 *                  then reduce into the rank-local grid.
 * ========================================================= */
void interpolation(double *mesh_value, Points *points) {

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int total_nodes = GRID_X * GRID_Y;

    /* Zero out the global mesh before accumulation */
    memset(mesh_value, 0, total_nodes * sizeof(double));

    /* Determine this rank's slice of particles */
    int chunk   = NUM_Points / nprocs;
    int rem     = NUM_Points % nprocs;
    int p_start = rank * chunk + (rank < rem ? rank : rem);
    int p_end   = p_start + chunk + (rank < rem ? 1 : 0);

    int nthreads;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    /* Each thread gets its own private partial grid to avoid race conditions */
    double *local_grid = (double *) calloc((long long)nthreads * total_nodes,
                                            sizeof(double));
    if (!local_grid) {
        fprintf(stderr, "Rank %d: calloc failed in interpolation\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    #pragma omp parallel for schedule(static)
    for (int p = p_start; p < p_end; p++) {

        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        /* Skip out-of-domain particles */
        if (x < 0.0 || x > 1.0 || y < 0.0 || y > 1.0) {
            points[p].is_void = true;
            continue;
        }

        /* Cell indices — clamp to avoid out-of-bounds on boundary */
        int ci = (int)(x / dx);
        int cj = (int)(y / dy);
        if (ci >= NX) ci = NX - 1;
        if (cj >= NY) cj = NY - 1;

        /* Grid-point coordinates of the cell's bottom-left corner */
        double Xi = ci * dx;
        double Yj = cj * dy;

        /* Local offsets inside cell */
        double lx = x - Xi;
        double ly = y - Yj;

        /* Bilinear weights */
        double w00 = (dx - lx) * (dy - ly);   /* (ci,   cj  ) */
        double w10 = ly         * (dx - lx);   /* (ci,   cj+1) — note index order below */
        double w01 = lx         * (dy - ly);   /* (ci+1, cj  ) */
        double w11 = lx         * ly;          /* (ci+1, cj+1) */

        /* f_i = 1 for all particles (as per assignment spec) */

        int tid = omp_get_thread_num();
        double *tgrid = local_grid + (long long)tid * total_nodes;

        /*
         * Grid layout: row-major, row = j (y direction), col = i (x direction)
         * Index = j * GRID_X + i
         */
        tgrid[ cj      * GRID_X + ci     ] += w00;
        tgrid[(cj + 1) * GRID_X + ci     ] += w10;
        tgrid[ cj      * GRID_X + (ci+1) ] += w01;
        tgrid[(cj + 1) * GRID_X + (ci+1) ] += w11;
    }

    /* Reduce thread-private grids into rank-local mesh */
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < total_nodes; n++) {
        double sum = 0.0;
        for (int t = 0; t < nthreads; t++) {
            sum += local_grid[(long long)t * total_nodes + n];
        }
        mesh_value[n] = sum;
    }

    free(local_grid);

    /* MPI global reduction: sum partial grids from all ranks */
    MPI_Allreduce(MPI_IN_PLACE, mesh_value, total_nodes,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/* =========================================================
 * NORMALIZATION  — scale mesh to [-1, 1]
 * Saves min_val and max_val for denormalization.
 * ========================================================= */
void normalization(double *mesh_value) {

    int total_nodes = GRID_X * GRID_Y;

    double local_min =  DBL_MAX;
    double local_max = -DBL_MAX;

    /* OpenMP parallel min/max reduction */
    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max) schedule(static)
    for (int n = 0; n < total_nodes; n++) {
        if (mesh_value[n] < local_min) local_min = mesh_value[n];
        if (mesh_value[n] > local_max) local_max = mesh_value[n];
    }

    /* MPI global min/max (all ranks hold full mesh after Allreduce, but kept for generality) */
    MPI_Allreduce(&local_min, &min_val, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &max_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double range = max_val - min_val;
    if (range < 1e-15) range = 1.0; /* Avoid division by zero */

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < total_nodes; n++) {
        mesh_value[n] = 2.0 * (mesh_value[n] - min_val) / range - 1.0;
    }
}

/* =========================================================
 * MOVER  (Mesh → Particle, reverse interpolation)
 *
 * For each particle p at (x, y):
 *   1. Find its cell and compute same bilinear weights as interpolation.
 *   2. Read back the weighted-average field value from the 4 surrounding
 *      (normalized) grid nodes  →  field_p
 *   3. Update particle position:
 *        x_new = x + field_p * dx
 *        y_new = y + field_p * dy
 *   4. Mark particle as void if it leaves [0,1]×[0,1].
 *
 * MPI: same particle partitioning as interpolation.
 * OpenMP: embarrassingly parallel over particles (no shared writes).
 * ========================================================= */
void mover(double *mesh_value, Points *points) {

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int chunk   = NUM_Points / nprocs;
    int rem     = NUM_Points % nprocs;
    int p_start = rank * chunk + (rank < rem ? rank : rem);
    int p_end   = p_start + chunk + (rank < rem ? 1 : 0);

    #pragma omp parallel for schedule(static)
    for (int p = p_start; p < p_end; p++) {

        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int ci = (int)(x / dx);
        int cj = (int)(y / dy);
        if (ci >= NX) ci = NX - 1;
        if (cj >= NY) cj = NY - 1;

        double Xi = ci * dx;
        double Yj = cj * dy;
        double lx = x - Xi;
        double ly = y - Yj;

        double w00 = (dx - lx) * (dy - ly);
        double w10 = ly         * (dx - lx);
        double w01 = lx         * (dy - ly);
        double w11 = lx         * ly;

        /* Weighted field value at particle location */
        double field_p =
            w00 * mesh_value[ cj      * GRID_X + ci     ] +
            w10 * mesh_value[(cj + 1) * GRID_X + ci     ] +
            w01 * mesh_value[ cj      * GRID_X + (ci+1) ] +
            w11 * mesh_value[(cj + 1) * GRID_X + (ci+1) ];

        /* Normalize weight sum (= dx*dy for interior cells) */
        double wsum = w00 + w10 + w01 + w11;
        if (wsum > 1e-15) field_p /= wsum;

        /* Update position */
        double x_new = x + field_p * dx;
        double y_new = y + field_p * dy;

        /* Mark void if outside domain */
        if (x_new < 0.0 || x_new > 1.0 || y_new < 0.0 || y_new > 1.0) {
            points[p].is_void = true;
        } else {
            points[p].x = x_new;
            points[p].y = y_new;
        }
    }

    /*
     * Synchronize updated particle positions across all MPI ranks.
     * Each rank updated its own slice; broadcast all slices so every rank
     * has the full, up-to-date particle array for the next iteration.
     */
    int total_doubles = NUM_Points * 2; /* x and y interleaved */

    /* Pack x,y into a contiguous buffer, allreduce with MAX trick:
     * Only the rank that owns a particle sets it; others leave 0.
     * Instead, use MPI_Allgatherv for correctness. */

    /* Build send/recv counts */
    int *scounts = (int *) malloc(nprocs * sizeof(int));
    int *displs  = (int *) malloc(nprocs * sizeof(int));
    for (int r = 0; r < nprocs; r++) {
        int rc = NUM_Points / nprocs;
        int rr = NUM_Points % nprocs;
        int rs = r * rc + (r < rr ? r : rr);
        int re = rs + rc + (r < rr ? 1 : 0);
        scounts[r] = (re - rs) * 2; /* 2 doubles per particle */
        displs[r]  = rs * 2;
    }

    /* Pack local particles into send buffer */
    int local_count = (p_end - p_start) * 2;
    double *sendbuf = (double *) malloc(local_count * sizeof(double));
    double *recvbuf = (double *) malloc(total_doubles * sizeof(double));

    for (int p = p_start; p < p_end; p++) {
        sendbuf[(p - p_start) * 2    ] = points[p].x;
        sendbuf[(p - p_start) * 2 + 1] = points[p].y;
    }

    MPI_Allgatherv(sendbuf, local_count, MPI_DOUBLE,
                   recvbuf, scounts, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

    /* Unpack back into points array */
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < NUM_Points; p++) {
        points[p].x = recvbuf[p * 2    ];
        points[p].y = recvbuf[p * 2 + 1];
    }

    /* Synchronize is_void flags */
    int *void_buf = (int *) malloc(NUM_Points * sizeof(int));
    int *void_local = (int *) calloc(NUM_Points, sizeof(int));
    for (int p = p_start; p < p_end; p++) {
        void_local[p] = (int) points[p].is_void;
    }
    MPI_Allreduce(void_local, void_buf, NUM_Points, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    for (int p = 0; p < NUM_Points; p++) {
        points[p].is_void = (bool) void_buf[p];
    }

    free(scounts); free(displs);
    free(sendbuf); free(recvbuf);
    free(void_buf); free(void_local);
}

/* =========================================================
 * DENORMALIZATION  — restore mesh to original value range
 * ========================================================= */
void denormalization(double *mesh_value) {

    int total_nodes = GRID_X * GRID_Y;
    double range = max_val - min_val;
    if (range < 1e-15) range = 1.0;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < total_nodes; n++) {
        mesh_value[n] = (mesh_value[n] + 1.0) * 0.5 * range + min_val;
    }
}

/* =========================================================
 * VOID COUNT  — count inactive particles
 * ========================================================= */
long long int void_count(Points *points) {

    long long int voids = 0;
    #pragma omp parallel for reduction(+:voids) schedule(static)
    for (int i = 0; i < NUM_Points; i++) {
        voids += (long long int) points[i].is_void;
    }
    return voids;
}

/* =========================================================
 * SAVE MESH  — write structured grid to Mesh.out
 * Only rank 0 writes (all ranks hold identical mesh).
 * ========================================================= */
void save_mesh(double *mesh_value) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) return;

    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int j = 0; j < GRID_Y; j++) {
        for (int i = 0; i < GRID_X; i++) {
            fprintf(fd, "%lf ", mesh_value[j * GRID_X + i]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}