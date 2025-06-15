#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "linalg.h"
#include "PreconditionedCG.h"

int PreconditionedCG(const int n, const int maxiter, const double tol, 
                     const int preconditioner, const double w,
                     const double *A, const double *b, double *x) {
    int i, j, k, peel = n % VLEN;
    // x is the null vector
    for (k = 0; k != n; ++k) { *(x + k) = 0.; } 
    k = 0;
    // k-th residual vector
    double *r_k = malloc(n * sizeof(double)); 
    if (!r_k) { 
        fprintf(stderr, "r_k malloc failure\n");
        return -1; 
    }
    // x_0 = 0 ==> r_0 = b;
    memcpy(r_k, b, n * sizeof(double)); 
    // ||r_k||
    double rknorm = norm(n, r_k); 
    // If ||r_0|| < tol then x already is the solution.
    if (rknorm >= tol) {
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
        // k-th z vector
        double *z_k = malloc(n * sizeof(double)); 
        if (!z_k) { 
            fprintf(stderr, "z_k malloc failure\n");
            free(r_k);
            free(p_k);
            free(Ap_k);
            return -1111; 
        }
        // Solve M*z_k = r_k
        solve_preconditioner(n, preconditioner, w, A, r_k, z_k);
        // p_0 = z_0
        memcpy(p_k, z_k, n * sizeof(double));
        // r_k' * z_k
        double rz_k = dot(n, r_k, z_k);
        // r_(k+1)' * z_(k+1)
        double rz_kp1 = 0.;
        // Iterates the steps
        register double sum;
        for (k = 0; k != maxiter; ++k) {
            double alpha_k = 0.0;
            // Ap_k = A * p_k, rz_k = r_k' * z_k and alpha_k = ||r_k||^2 / (p_k' * A * p_k)
            #pragma omp parallel for if(n > 999) schedule(static) private(j, sum) reduction(+:alpha_k)
            for (i = 0; i != n; ++i) {
                sum = 0.;
                if (peel) { 
                    for (j = 0; j != peel; ++j) { sum += A[j + i * n] * p_k[j]; }
                }
                #pragma omp simd reduction(+:sum) aligned(A, p_k:VLEN) 
                for (j = peel; j != n; ++j) { sum += A[j + i * n] * p_k[j]; }
                Ap_k[i] = sum;
                alpha_k += p_k[i] * sum; // p_k * A * p_k
            }
            alpha_k = rz_k / alpha_k; // ||r_k||^2 / (p_k' * A * p_k)
            // x_(k+1) = x_k + alpha_k*p_k, r_(k+1) = r_k - alpha_k*Ap_k and ||r_(k+1)||
            rknorm = 0.;
            if (peel) { 
                for (i = 0; i != peel; ++i) { 
                    x[i] += alpha_k * p_k[i];
                    r_k[i] -= alpha_k * Ap_k[i];
                    rknorm += r_k[i] * r_k[i];
                }
            }
            #pragma omp simd reduction(+:rknorm) aligned(x, r_k, Ap_k, p_k:VLEN) 
            for (i = peel; i != n; ++i) { 
                x[i] += alpha_k * p_k[i];
                r_k[i] -= alpha_k * Ap_k[i];
                rknorm += r_k[i] * r_k[i];
            }
            rknorm = sqrt(rknorm);
            // Returns the solution if ||r_(k+1)|| < tol
            if (rknorm < tol) {
                free(r_k);
                free(p_k);
                free(Ap_k);
                free(z_k);
                return k + 1;
            }
            // Solve M*z_(k+1) = r_(k+1)
            solve_preconditioner(n, preconditioner, w, A, r_k, z_k);
            rz_kp1 = dot(n, r_k, z_k);
            // beta_k = r_(k+1)' * z_(k+1) / (r_k' * z_k)
            double beta_k = rz_kp1 / rz_k;
            // Update rz_k
            rz_k = rz_kp1;
            // p_(k+1) = beta_k*p_k + z_(k+1)
            vvadd(n, 1., z_k, beta_k, p_k);
        }
        free(p_k);
        free(Ap_k);
        free(z_k);
    }
    free(r_k);
    return k;
}