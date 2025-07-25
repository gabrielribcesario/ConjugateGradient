#ifndef PRECG_H
#define PRECG_H

#include "linalg.h"
           
/*
    A - Symmetric positive-semidefinite matrix, n x n;
    b - Target vecto, n x 1;
    x - Initial guess / solution vector, n x 1;
    tol - Error tolerance;
    maxiter - Maximum number of iterations.
*/          
int PreconditionedCG(const int n, const int maxiter, const double tol, 
                     const int preconditioner, const double w,
                     const double *A, const double *b, double *x);


/* Two-sided preconditioning */

/*
    Let D * x = y 
    Solve for x where D is diagonal
*/
void solve_jacobi(int n, const double *D, const double *y, double *x) {
    int i, peel = n % VLEN;
    if (peel) { 
        for (i = 0; i != peel; ++i) { x[i] = y[i] / D[i * (n + 1)]; } 
    }
    #pragma omp simd aligned(D, y, x:VLEN*8)
    for (i = peel; i != n; ++i) { x[i] = y[i] / D[i * (n + 1)]; }
}

/*
    Let 0 < w < 2
    Let u = (D / w + L)^T * x
    Let M = w / (2 - w) * (D / w + L) * D^-1
    Let M * u = y

    Solve for u and then solve for x where D is 
    diagonal and L is lower triangular
*/
void solve_ssor(const int n, const double w, const double *M, const double *y, double *x) {
    int i, j, peel;
    register double sum;
    // Solve M * u = y using foward substitution
    x[0] = y[0] * (2. - w);
    #pragma omp parallel for if(n > 999) schedule(static) private(sum, peel, j)
    for (i = 1; i != n; ++i) {
        peel = i % VLEN;
        sum = 0.;
        if (peel) { 
            for (j = 0; j != peel; ++j) { sum += x[j] * M[j + i * n] / M[j * (n + 1)]; }
        }
        #pragma omp simd reduction(+:sum) aligned(M, x:VLEN*8)
        for (j = peel; j != i; ++j) {
            sum += x[j] * M[j + i * n] / M[j * (n + 1)];
        }
        x[i] = y[i] * (2. - w) - sum * w;
    }
    // Solve (D / w + L)^T * x = u using backward substitution
    x[n - 1] *= w / M[n * n - 1];
    #pragma omp parallel for if(n > 999) schedule(static) private(sum, peel, j)
    for (i = n - 2; i != -1; --i) {
        peel = (n - i - 1) % VLEN;
        sum = 0.;
        if (peel) { 
            for (j = i + 1; j != peel + i + 1; ++j) { sum += x[j] * M[j + i * n]; }
        }
        #pragma omp simd reduction(+:sum) aligned(M, x:VLEN*8)
        for (j = peel + i + 1; j != n; ++j) { sum += x[j] * M[j + i * n]; }
        x[i] = (x[i] - sum) * w / M[i * (n + 1)];
    }
}

/*
void solve_ichol() {
    
}
*/

inline void solve_preconditioner(const int n, const int precond, const double w, 
                                 const double *M, const double *r, double *z) {
    if (precond == 0) { solve_jacobi(n, M, r, z); }
    else if (precond == 1) { solve_ssor(n, w, M, r, z); }
    // else if (precond == 2) { solve_ichol(M, r, z, w, n); }
}

#endif