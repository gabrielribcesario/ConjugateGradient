#ifndef PRECONDITIONING_H
#define PRECONDITIONING_H

#include <linalg.h>

/* Two-sided preconditioning */

/*
    Let D * x = y 
    Solve for x where D is diagonal
*/
void solve_jacobi(const double *D, const double *y, double *x, int n) {
    int i, peel = n % VLEN;
    if (peel) { 
        for (i = 0; i != peel; ++i) { *(x + i) = *(y + i) / *(D + i * (n + 1)); } 
    }
    #pragma omp simd aligned(D, y, x:VLEN)
    for (i = peel; i != n; ++i) { *(x + i) = *(y + i) / *(D + i * (n + 1)); }
}

/*
    Let 0 < w < 2
    Let u = (D / w + L)^T * x
    Let M = w / (2 - w) * (D / w + L) * D^-1
    Let M * u = y

    Solve for u and then solve for x where D is 
    diagonal and L is lower triangular
*/
void solve_ssor(const double *M, const double *y, double *x, const int n, const double w) {
    int i, j, peel;
    register double c0;
    // Solve M * u = y using foward substitution
    *x = *y * (2. - w);
    for (i = 0; i != n; ++i) {
        peel = i % VLEN;
        c0 = 0.;
        if (peel) { 
            for (j = 0; j != peel; ++j) { c0 += *(x + j) * *(M + j + i * n) / *(M + j * (n + 1)); }
        }
        #pragma omp reduction(+:c0) aligned(M, x:VLEN)
        for (j = peel; j != i; ++j) {
            c0 += *(x + j) * *(M + j + i * n) / *(M + j * (n + 1));
        }
        *(x + i) = *(y + i) * (2. - w) - c0 * w;
    }
    // Solve (D / w + L)^T * x = u using backward substitution
    *(x + n - 1) *= w / *(M + n * n - 1);
    for (i = n - 2; i != -1; --i) {
        peel = (i + 1) % VLEN;
        c0 = 0.;
        if (peel) { 
            for (j = i + 1; j != peel + i + 1; ++j) { c0 += *(x + j) * *(M + j + i * n) / *(M + j * (n + 1)); }
        }
        #pragma omp reduction(+:c0) aligned(M, x:VLEN)
        for (j = peel + i + 1; j != n; ++j) {
            c0 += *(x + j) * *(M + j + i * n) / *(M + j * (n + 1));
        }
        *(x + i) = (*(x + i) - c0) * w / *(M + i * (n + 1));
    }
}

/*
void solve_ichol() {
    
}
*/

inline void solve_preconditioner(const double *M, const double *r, double *z, const int n,
                                 const double w, const int precond) {
    if (precond == 0) { solve_jacobi(M, r, z, n); }
    else if (precond == 1) { solve_ssor(M, r, z, w, n); }
    // else if (precond == 2) { solve_ichol(M, r, z, w, n); }
}

#endif