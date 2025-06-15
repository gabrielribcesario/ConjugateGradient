#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "linalg.h"
#include "mtxmarket.h"
#include "ConjugateGradient.h"
#include "PreconditionedCG.h"

int main(void) {
    const int pcd = 1; // 0 - Jacobi; 1 - SSOR
    const double tol = 1e-10;
    const int maxiter = 1000;
    const double w = 1.; // Relaxation factor

    if (maxiter <= 0 || w <= 0. || w >= 2. || tol <= 0.) { 
        fprintf(stderr, "Invalid param\n");
        return EXIT_FAILURE; 
    }

    int nrows, ncols;
    double *A = NULL; // Must be initialized to null or allocated immediately!
    if (!parse_mat("../data/bcsstk14.mtx", &nrows, &ncols, &A)) { 
        fprintf(stderr, "mtx failure\n");
        return EXIT_FAILURE;
    }
    if (nrows != ncols) {
        fprintf(stderr, "Non-square matrix\n");
        return EXIT_FAILURE;        
    }
    int n = nrows;
    double *b = malloc(n * sizeof(double));
    if (!b) { 
        fprintf(stderr, "b malloc failure\n");
        free(A);
        return EXIT_FAILURE; 
    }
    double *x = calloc(n, sizeof(double));
    if (!x) { 
        fprintf(stderr, "x malloc failure\n");
        free(A);
        free(b);
        return EXIT_FAILURE; 
    }
    double *sol = calloc(n, sizeof(double));
    if (!sol) { 
        fprintf(stderr, "sol calloc failure\n");
        free(A);
        free(b);
        free(x);
        return EXIT_FAILURE; 
    }
    for (int i = 0; i < n; ++i) { sol[i] = i % 2 == 0 ? -1. : 1.; }

    for (int i = 0; i < n; ++i) { b[i] = dot(n, A + i * n, sol); }

    struct timespec tic, toc;

    printf("Unpreconditioned:\n");
    // Time it
    clock_gettime(CLOCK_MONOTONIC, &tic);
    int iter = ConjugateGradient(n, maxiter, tol, A, b, x);
    clock_gettime(CLOCK_MONOTONIC, &toc);
    double elapsed = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) * 1.e-9;
    // ||Ax - b||
    mvmul(n, n, 1., A, x, 0., sol);
    vvadd(n, 1., b, -1., sol);
    double err = norm(n, sol);
    printf("|   Ran %d iterations in %.6f[s]\n", iter, elapsed);
    printf("|   Returned with error %#g\n", err);

    printf("Preconditioned:\n");
    // Time it
    clock_gettime(CLOCK_MONOTONIC, &tic);
    iter = PreconditionedCG(n, maxiter, tol, pcd, w, A, b, x);
    clock_gettime(CLOCK_MONOTONIC, &toc);
    elapsed = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) * 1.e-9;
    // ||Ax - b||
    mvmul(n, n, 1., A, x, 0., sol);
    vvadd(n, 1., b, -1., sol);
    err = norm(n, sol);
    printf("|   Ran %d iterations in %.6f[s]\n", iter, elapsed);
    printf("|   Returned with error %#g\n", err);

    free(A);
    free(b);
    free(x);
    free(sol);

    return EXIT_SUCCESS;
}