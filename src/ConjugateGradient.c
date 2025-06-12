#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "linalg.h"
#include "ConjugateGradient.h"

int ConjugateGradient(const int n, const int maxiter, const double tol, 
                      const double *A, const double *b, double *x) {
    int k;
    // x is the null vector
    for (k = 0; k != n; ++k) { *(x + k) = 0.; } 
    // k-th residual vector
    double *r_k = malloc(n * sizeof(double)); 
    if (!r_k) { 
        fprintf(stderr, "r_k malloc failure\n");
        return EXIT_FAILURE; 
    }
    // x_0 = 0 ==> r_0 = b;
    memcpy(r_k, b, n * sizeof(double)); 
    // ||r_k||
    double rknorm = norm(n, r_k); 
    // ||r_(k+1)||
    double rkp1norm = 0.; 
    // If ||r_0|| < tol then x already is the solution.
    if (rknorm >= tol) {
        // k-th direction vector
        double *p_k = malloc(n * sizeof(double)); 
        if (!p_k) { 
            fprintf(stderr, "p_k malloc failure\n");
            free(r_k);
            return EXIT_FAILURE; 
        }
        // p_0 = r_0
        memcpy(p_k, r_k, n * sizeof(double)); 
        // A * p_k
        double *Ap_k = calloc(n, sizeof(double)); 
        if (!Ap_k) { 
            fprintf(stderr, "Ap_k malloc failure\n");
            free(r_k);
            free(p_k);
            return EXIT_FAILURE; 
        }
        // Iterates the steps
        printf("Running a maximum of %d iterations of the algorithm\n", maxiter);
        for (k = 0; k != maxiter; ++k) {
            // Ap_k = A * p_k
            mvmul(n, n, 1., A, p_k, 0., Ap_k);
            // alpha_k = ||r_k||^2 / (p_k' * A * p_k)
            double alpha_k = rknorm * rknorm / dot(n, p_k, Ap_k);
            // x_(k+1) = x_k + alpha_k*p_k
            vvadd(n, alpha_k, p_k, 1., x);
            // r_(k+1) = r_k - alpha_k*Ap_k
            vvadd(n, -alpha_k, Ap_k, 1., r_k);
            rkp1norm = norm(n, r_k);
            // Returns the solution if ||r_(k+1)|| < tol
            if (rkp1norm < tol) {
                printf("Converged with %g error in %d iterations\n", rkp1norm, k + 1);
                free(r_k);
                free(p_k);
                free(Ap_k);
                return EXIT_SUCCESS;
            }
            // beta_k = ||r_(k+1)||^2 / ||r_k||^2
            double beta_k = (rkp1norm * rkp1norm) / (rknorm * rknorm);
            rknorm = rkp1norm; // Update the residual
            // p_(k+1) = beta_k*p_k + r_(k+1)
            vvadd(n, 1., r_k, beta_k, p_k);
        }
        free(p_k);
        free(Ap_k);
        printf("Could not achieve convergence with error %g\n", rknorm);
        return EXIT_SUCCESS;
    }
    else {
        free(r_k);
        printf("Returning the trivial solution with error %g\n", rknorm);
        return EXIT_SUCCESS;
    }
}