#ifndef CONJGRAD_H
#define CONJGRAD_H

/*
    A - Symmetric positive-semidefinite matrix, n x n;
    b - Target vecto, n x 1;
    x - Initial guess / solution vector, n x 1;
    tol - Error tolerance;
    maxiter - Maximum number of iterations.
*/
int ConjugateGradient(const int n, const int maxiter, const double tol, 
                      const double *A, const double *b, double *x);

#endif