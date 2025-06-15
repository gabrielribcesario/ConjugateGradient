#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "linalg.h"
#include "ConjugateGradient.h"
#include "PreconditionedCG.h"

// Generates a m x n Hilbert matrix
void HilbertMatrix(int m, int n, double *A) {
    int i, j;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) { A[i * n + j] = 1. / (i + j + 1); }
    }
}

int main(void) {
    const int n = 40000;
    double tol = 1.E-5;
    int maxiter = 1000;

    double *A = malloc(n * n * sizeof(double));
    if (!A) { 
        fprintf(stderr, "A malloc failure\n");
        return EXIT_FAILURE; 
    }
    double *b = malloc(n * sizeof(double));
    if (!b) { 
        fprintf(stderr, "b malloc failure\n");
        free(A);
        return EXIT_FAILURE; 
    }
    double *x = calloc(n, sizeof(double));
    if (!x) { 
        fprintf(stderr, "x calloc failure\n");
        free(A);
        free(b);
        return EXIT_FAILURE; 
    }

    HilbertMatrix(n, n, A);
    // x[i] = 1
    for (int i = 0; i < n; ++i) { b[i] = reduce_sum(n, A + i * n); }

    struct timespec tic, toc;

    // Time it
    clock_gettime(CLOCK_MONOTONIC, &tic);
    int iter = ConjugateGradient(n, maxiter, tol, A, b, x);
    clock_gettime(CLOCK_MONOTONIC, &toc);
    double elapsed = (toc.tv_sec - tic.tv_sec) + (toc.tv_nsec - tic.tv_nsec) * 1.e-9;
    printf("Ran %d iterations in %.6f[s]\n", iter, elapsed);
    // ||Ax - b||
    mvmul(n, n, 1., A, x, -1., b);
    double err = norm(n, b);
    printf("Returned with error %#g\n", err);

    free(A);
    free(b);
    free(x);

    return EXIT_SUCCESS;
}