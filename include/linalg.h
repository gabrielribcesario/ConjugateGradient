#ifndef LINALG_H
#define LINALG_H

#include <math.h>
#include <omp.h>

#define VLEN 4

/* <x, y> */
double dot(const int n, const double *x, const double *y) {
    int i, peel = n % VLEN; 
    register double sum = 0.;
    if (peel) { // Loop peeling
        for (i = 0; i != peel; ++i) { sum += x[i] * y[i]; }
    }
    #pragma omp simd reduction(+:sum) aligned(x, y:VLEN)
    for (i = peel; i != n; ++i) { sum += x[i] * y[i]; }
    return sum;
}

/* ||y|| */
double norm(const int n, const double *y) {
    int i, peel = n % VLEN; 
    register double sum = 0.;
    if (peel) {
        for (i = 0; i != peel; ++i) { sum += y[i] * y[i]; }
    }
    #pragma omp simd reduction(+:sum) aligned(y:VLEN)
    for (i = peel; i != n; ++i) { sum += y[i] * y[i]; }
    return sqrt(sum);
}

/* y = alpha*y */
void vscale(const int n, const double beta, double *y) {
    int i, peel = n % VLEN; 
    if (peel) {
        for (i = 0; i != peel; ++i) { y[i] *= beta; }
    }
    #pragma omp simd aligned(y:VLEN)
    for (i = peel; i != n; ++i) { y[i] *= beta; }
}

/* y = y + alpha*x */
void vvadd(const int n, const double alpha, const double *x, const double beta, double *y) {
    int i, peel = n % VLEN; 
    if (peel) {
        for (i = 0; i != peel; ++i) { y[i] = beta * y[i] + alpha * x[i]; }
    }
    #pragma omp simd aligned(x, y:VLEN)
    for (i = peel; i != n; ++i) { y[i] = beta * y[i] + alpha * x[i]; }
}

/* y = beta*y + alpha*Ax */
void mvmul(const int m, const int n, const double alpha, const double *A, const double *x, 
           const double beta, double *y) { // about as fast as oneMKL cblas_dgemv for larger matrices (icx -O3 -avx512f -qmkl=parallel)
    int i, j, peel = n % VLEN;
    register double sum;
    #pragma omp parallel for if(n > 999 && m * n > 9999) schedule(static) private(sum) // Fine-tune condition later
    for (i = 0; i != m; ++i) {
        sum = 0.0;
        if (peel) {
            for (j = 0; j != peel; ++j) { sum += A[j + i * n] * x[j]; }
        }
        #pragma omp simd reduction(+:sum) aligned(A, x:VLEN) 
        for (j = peel; j != n; ++j) { sum += A[j + i * n] * x[j]; }
        y[i] = beta * y[i] + alpha * sum;
    }
}

#endif