#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "linalg.h"
#include "PreconditionedCG.h"

int PreconditionedCG(const int n, const int maxiter, const double tol, 
                     const int preconditioner, const double w,
                     const double *A, const double *b, double *x) {
    int k;
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
        // r_k' * z_k
        double rz_k = dot(n, r_k, z_k);
        // r_(k+1)' * z_(k+1)
        double rz_kp1 = 0.;
        // p_0 = z_0
        memcpy(p_k, z_k, n * sizeof(double)); 
        // Iterates the steps
        for (k = 0; k != maxiter; ++k) {
            // Ap_k = A * p_k
            mvmul(n, n, 1., A, p_k, 0., Ap_k);
            // alpha_k = r_k' * z_k / (p_k' * A * p_k)
            double alpha_k = rz_k / dot(n, p_k, Ap_k);
            // x_(k+1) = x_k + alpha_k*p_k
            vvadd(n, alpha_k, p_k, 1., x);
            // r_(k+1) = r_k - alpha_k*Ap_k
            vvadd(n, -alpha_k, Ap_k, 1., r_k);
            rknorm = norm(n, r_k);
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