#ifndef LINALG_H
#define LINALG_H

#include <omp.h>

#define VLEN 4

double dot(const double *a, const double *b, const int n) {
    int i, peel = n % VLEN; 
    register double c0 = 0.;
    if (peel) { // Loop peeling
        for (i = 0; i != peel; ++i) { c0 += *(a + i) * *(b + i); }
    }
    #pragma omp simd reduction(+:c0) aligned(a, b:VLEN)
    for (i = peel; i != n; ++i) { c0 += *(a + i) * *(b + i); }
    return c0;
}

void mvmul(const double *A, const double *b, double *c, const int m, const int n) {
    int i, j, peel = n % VLEN;
    register double c0;
    for (i = 0; i != m; ++i) {
        c0 = 0.0;
        if (peel) { // Loop peeling
            for (j = 0; j != peel; ++j) { c0 += *(A + j + i * m) * *(b + j); }
        }
        #pragma omp simd reduction(+:c0) aligned(A, b, c:VLEN) 
        for (j = peel; j != n; ++j) { c0 += *(A + j + i * m) * *(b + j); }
        *(c + i) = c0;
    }
}

#endif