#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
// #include <mkl.h>
#include "linalg.h"

/*
    Testing the linear algebra subroutines and compiler link lines

    GCC - Slightly faster than clang (smaller binary too)
    oneAPI - Insanely large binary, about as faster as Clang and GCC

    gcc ../src/cblastest.c -o test -I../include -O3 -fopenmp -msse4.2 -lm 

    clang ../src/cblastest.c -o test -I../include -O3 -fopenmp -msse4.2 -lm

    icx ../src/cblastest.c -o test -I../include -O3 -mavx512f -L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -I"${MKLROOT}/include"
*/

int main(int argc, char **argv) {
    int m = 15000, n = 15000;
    int ntrials = 1000;

    double *A = malloc(sizeof(double) * m * n);
    if (!A) {
        printf("malloc failure\n");
        return EXIT_FAILURE;
    }
    double *x = malloc(sizeof(double) * m);
    if (!x) {
        free(A);
        printf("malloc failure\n");
        return EXIT_FAILURE;
    }
    double *y = calloc(m, sizeof(double));
    if (!y) {
        free(A);
        free(x);
        printf("malloc failure\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            A[i * n + j] = (double) i;
        }
        x[i] = (double) i;
    }
    
    struct timespec tic, toc;
    double elapsed;

    clock_gettime(CLOCK_MONOTONIC, &tic);
    for (int i = 0; i != ntrials; ++i) {
        mvmul(m, n, 1., A, x, 1., y);
    }
    clock_gettime(CLOCK_MONOTONIC, &toc);
    elapsed = (toc.tv_sec - tic.tv_sec) * 1.e3 + (toc.tv_nsec - tic.tv_nsec) * 1.e-6;
    printf("Parallel averaged %.6f[ms]\n", elapsed / ntrials);

    // vscale(m, 0., y);

    // clock_gettime(CLOCK_MONOTONIC, &tic);
    // for (int i = 0; i != ntrials; ++i) {
    //     cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1., A, n, x, 1, 1., y, 1);
    // }
    // clock_gettime(CLOCK_MONOTONIC, &toc);
    // elapsed = (toc.tv_sec - tic.tv_sec) * 1.e3 + (toc.tv_nsec - tic.tv_nsec) * 1.e-6;
    // printf("oneMKL averaged %.6f[ms]\n", elapsed / ntrials);

    free(A);
    free(x);
    free(y);
    return EXIT_SUCCESS;
}