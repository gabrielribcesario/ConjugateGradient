#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "linalg.h"
#include "ConjugateGradient.h"

// History is a matrix of maxiter x n dimensions
int HistoryCG(const int n, const int maxiter, const double tol, 
              const double *A, const double *b, double *x, double *history) {
    int i, j, k, peel = n % VLEN;
    // x is the null vector
    for (k = 0; k != n; ++k) { x[k] = 0.; } 
    k = 0;
    // k-th residual vector
    double *r_k = malloc(n * sizeof(double)); 
    if (!r_k) { 
        fprintf(stderr, "r_k malloc failure\n");
        return -1; 
    }
    // x_0 = 0 ==> r_0 = b;
    memcpy(r_k, b, n * sizeof(double)); 
    // ||r_k||^2 
    double rknorm_sqd = dot(n, r_k, r_k); // storing ||r_k||^2 instead of ||r_k|| improves precision
    // ||r_(k+1)||^2 
    double rkp1norm_sqd = 0.; 
    // If ||r_0|| < tol then x already is the solution.
    if (sqrt(rknorm_sqd) >= tol) {
        // k-th direction vector
        double *p_k = malloc(n * sizeof(double)); 
        if (!p_k) { 
            fprintf(stderr, "p_k malloc failure\n");
            free(r_k);
            return -11; 
        }
        // A * p_k
        double *Ap_k = calloc(n, sizeof(double)); 
        if (!Ap_k) { 
            fprintf(stderr, "Ap_k malloc failure\n");
            free(r_k);
            free(p_k);
            return -111; 
        }
        // p_0 = r_0
        memcpy(p_k, r_k, n * sizeof(double)); 
        // Iterates the steps
        register double sum;
        for (k = 0; k != maxiter; ++k) {
            double alpha_k = 0.0;
            // Ap_k = A * p_k and alpha_k = ||r_k||^2 / (p_k' * A * p_k)
            #pragma omp parallel for if(n > 999) schedule(static) private(j, sum) reduction(+:alpha_k)
            for (i = 0; i != n; ++i) {
                sum = 0.;
                if (peel) { 
                    for (j = 0; j != peel; ++j) { sum += A[j + i * n] * p_k[j]; }
                }
                #pragma omp simd reduction(+:sum) aligned(A, p_k:VLEN*8) 
                for (j = peel; j != n; ++j) { sum += A[j + i * n] * p_k[j]; }
                Ap_k[i] = sum;
                alpha_k += p_k[i] * sum; // p_k * A * p_k
            }
            alpha_k = rknorm_sqd / alpha_k; // ||r_k||^2 / (p_k' * A * p_k)
            // x_(k+1) = x_k + alpha_k*p_k; r_(k+1) = r_k - alpha_k*Ap_k; ||r_(k+1)||
            rkp1norm_sqd = 0.;
            if (peel) { 
                for (i = 0; i != peel; ++i) { 
                    x[i] += alpha_k * p_k[i];
                    history[i + k * n] = x[i];
                    r_k[i] -= alpha_k * Ap_k[i];
                    rkp1norm_sqd += r_k[i] * r_k[i];
                }
            }
            #pragma omp simd reduction(+:rkp1norm_sqd) aligned(x, r_k, Ap_k, p_k, history:VLEN*8) 
            for (i = peel; i != n; ++i) { 
                x[i] += alpha_k * p_k[i];
                history[i + k * n] = x[i];
                r_k[i] -= alpha_k * Ap_k[i];
                rkp1norm_sqd += r_k[i] * r_k[i];
            }
            // Returns the solution if ||r_(k+1)|| < tol
            if (sqrt(rkp1norm_sqd) < tol) {
                free(r_k);
                free(p_k);
                free(Ap_k);
                return k + 1;
            }
            // beta_k = ||r_(k+1)||^2 / ||r_k||^2
            double beta_k = rkp1norm_sqd / rknorm_sqd;
            rknorm_sqd = rkp1norm_sqd; // Update the residual
            // p_(k+1) = beta_k*p_k + r_(k+1)
            vvadd(n, 1., r_k, beta_k, p_k);
        }
        free(p_k);
        free(Ap_k);
    }
    free(r_k);
    return k;
}