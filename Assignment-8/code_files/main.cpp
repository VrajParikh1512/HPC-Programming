#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Read header on rank 0 and broadcast to all ranks                    */
    /* ------------------------------------------------------------------ */
    FILE *file = NULL;
    if (rank == 0) {
        file = fopen(argv[1], "rb");
        if (!file) { perror("input file"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fread(&NX,         sizeof(int), 1, file);
        fread(&NY,         sizeof(int), 1, file);
        fread(&NUM_Points, sizeof(int), 1, file);
        fread(&Maxiter,    sizeof(int), 1, file);
    }
    MPI_Bcast(&NX,         1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NY,         1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_Points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Maxiter,    1, MPI_INT, 0, MPI_COMM_WORLD);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    /* ------------------------------------------------------------------ */
    /* Allocate ALL particles on ALL ranks (needed for Bcast & gather)     */
    /* ------------------------------------------------------------------ */
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points     = (Points *)  malloc(NUM_Points * sizeof(Points));
    double *F          = (double *)  malloc(NUM_Points * sizeof(double));
    int    *active     = (int *)     malloc(NUM_Points * sizeof(int));

    /* Rank 0 reads all points, then broadcasts to every rank */
    if (rank == 0) {
        read_initial_points(file, points);
        fclose(file);
    }
    MPI_Bcast(points, NUM_Points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < NUM_Points; i++) active[i] = 1;

    /* ------------------------------------------------------------------ */
    /* Particle decomposition: each rank owns a contiguous slice           */
    /* ------------------------------------------------------------------ */
    int base  = NUM_Points / size;
    int rem   = NUM_Points % size;
    /* rank r owns particles [p_start, p_end) */
    int p_start = rank * base + (rank < rem ? rank       : rem);
    int p_end   = p_start + base + (rank < rem ? 1        : 0);

    /* Build MPI_Scatterv / Gatherv displacement arrays                    */
    int *counts_2  = (int *) malloc(size * sizeof(int)); /* doubles per rank  */
    int *displs_2  = (int *) malloc(size * sizeof(int));
    int *counts_1i = (int *) malloc(size * sizeof(int)); /* ints   per rank   */
    int *displs_1i = (int *) malloc(size * sizeof(int));

    for (int r = 0; r < size; r++) {
        int rs  = r * base + (r < rem ? r    : rem);
        int rc  = base     + (r < rem ? 1    : 0);
        counts_2[r]  = rc * 2;   displs_2[r]  = rs * 2;
        counts_1i[r] = rc;       displs_1i[r] = rs;
    }

    /* mesh_global: reduction buffer for MPI Allreduce                     */
    /* mesh_save:   holds the pre-normalisation mesh from the last iter    */
    /*              — this is what Test_Mesh.out / Mesh.out must contain.  */
    double *mesh_global = (double *) malloc(GRID_X * GRID_Y * sizeof(double));
    double *mesh_save   = (double *) malloc(GRID_X * GRID_Y * sizeof(double));

    double interp_time = 0.0, norm_time  = 0.0,
           reverse_time = 0.0, mover_time = 0.0;

    /* ================================================================== */
    for (int iter = 0; iter < Maxiter; iter++) {

        /* --- 1. Forward interpolation: particles -> mesh --------------- */
        /*        Each rank accumulates only its own slice of particles.   */
        /*        Results are then summed across all ranks via Allreduce.  */
        double t1 = MPI_Wtime();
        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));
        interpolation_partial(mesh_value, points, active, p_start, p_end);

        /* Reduce partial meshes from all ranks into every rank            */
        MPI_Allreduce(mesh_value, mesh_global, GRID_X * GRID_Y,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        memcpy(mesh_value, mesh_global, GRID_X * GRID_Y * sizeof(double));
        double t2 = MPI_Wtime();
        interp_time += (t2 - t1);

        /* FIX: snapshot the raw (pre-normalisation) mesh from this iter.  */
        /* The output file must contain exactly this mesh from the final   */
        /* iteration — NOT a re-interpolation with post-mover positions.   */
        memcpy(mesh_save, mesh_value, GRID_X * GRID_Y * sizeof(double));

        /* --- 2. Normalize mesh ---------------------------------------- */
        t1 = MPI_Wtime();
        double min_val, max_val;
        find_min_max(mesh_value, &min_val, &max_val);
        normalize_mesh(mesh_value, min_val, max_val);
        t2 = MPI_Wtime();
        norm_time += (t2 - t1);

        /* --- 3. Reverse interpolation: mesh -> F (each rank, own slice) */
        t1 = MPI_Wtime();
        reverse_interpolation_partial(mesh_value, points, F, active,
                                      p_start, p_end);
        t2 = MPI_Wtime();
        reverse_time += (t2 - t1);

        /* --- 4. Mover: update particle positions (each rank, own slice) */
        t1 = MPI_Wtime();
        mover_partial(points, F, active, p_start, p_end);
        t2 = MPI_Wtime();
        mover_time += (t2 - t1);

        /* Synchronise updated particle positions and active flags         */
        /* across all ranks so every rank has the complete updated state.  */
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       points,  counts_2,  displs_2,  MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       active,  counts_1i, displs_1i, MPI_INT,    MPI_COMM_WORLD);
    }
    /* ================================================================== */

    double total_time = interp_time + norm_time + reverse_time + mover_time;

    /* --- Save mesh from last forward interpolation (rank 0 only) ------ */
    /* mesh_save holds the raw mesh accumulated at the START of the last  */
    /* iteration before any normalisation — this matches the reference.   */
    if (rank == 0) {
        save_mesh(mesh_save);

        printf("Total: %f, Interp: %f, Norm: %f, Reverse: %f, Mover: %f\n",
               total_time, interp_time, norm_time, reverse_time, mover_time);

        /* Uncomment to log timing data
        FILE *csv = fopen("timing_data.csv", "a");
        if (csv) {
            fprintf(csv, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f\n",
                    size, omp_get_max_threads(), NX, NY, NUM_Points,
                    total_time, interp_time, norm_time, reverse_time, mover_time);
            fclose(csv);
        }
        */
    }

    free(mesh_value); free(mesh_global); free(mesh_save);
    free(points);     free(F);
    free(active);
    free(counts_2);   free(displs_2);
    free(counts_1i);  free(displs_1i);
    MPI_Finalize();
    return 0;
}
