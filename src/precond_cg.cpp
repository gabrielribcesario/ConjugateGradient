#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <immintrin.h>
#include <string.h>

#include "preconditioners.h"

// Compile with:
// icpx -shared -o ./lib/precond_cg.so ./src/precond_cg.cpp -fPIC -O2 -mavx512f -Iinclude

extern "C" {
    
    /*
        A - Symmetric positive-semidefinite n x n matrix;
        b - Target vector of length n;
        x - Initial guess / solution vector of length n;
        tol - Error tolerance.
        max_iter - Maximum number of iterations.
        precond - Preconditioning matrix, one of the following:
            0 : jacobi;
            1 : ssor;
            2 : incomplete cholesky decomposition.
        n - Size of vector b.
    */
    void ConjugateGradient(const double* A, const double* b, double* x,
                           const double tol, const double w,
                           const unsigned int max_iter, const unsigned int precond,
                           const size_t n) {        
        if (precond > 2) {
            throw std::invalid_argument("The specified preconditioner does not exist.");
        }
        using namespace Preconditioner;
        
        // k-th residual vector.
        double* r_k = new double[n];
        // ||r_k||.
        double err = 0.;
        // Calculates r = b - A * x and ||r||^2.
        for (int i = 0; i < n; i++) {
            r_k[i] = b[i];
            int j = 0;
            for (; j < 8 * (n / 8); j += 8) {
                __m512d __A = _mm512_load_pd(A + j + i * n);
                __m512d __x = _mm512_load_pd(x + j);
                r_k[i] -= _mm512_reduce_add_pd(_mm512_mul_pd(__A, __x)); 
            }
            for (; j < n; j++) {
                r_k[i] -= A[j + i * n] * x[j];
            }
            err += r_k[i] * r_k[i];
        }
        err = sqrt(err);

        // If ||r|| <= tol then x already is the solution.
        if (err > tol) {
            // k-th z vector.
            double* z_k = new double[n];
            solve_preconditioner(A, r_k, z_k, w, n, precond);

            // k-th direction vector.
            double* p_k = new double[n];
            // p_0 = z_0
            memcpy(p_k, z_k, n * sizeof(double));
            // A * p_k
            double* Ap_k = new double[n];

            // Iterates the steps.
            for (int k = 0; k < max_iter; k++) {
                // (r_k' * z_k) / (p_k' * A * p_k)
                double alpha_k = 0.;
                // r_k' * z_k
                double rz_k = 0.;

                int i = 0;
                for (; i < n; i++) {
                    rz_k += r_k[i] * z_k[i];
                    Ap_k[i] = 0.;
                    int j = 0;
                    for (; j < 8 * (n / 8); j += 8) {
                        __m512d __A = _mm512_load_pd(A + j + i * n);
                        __m512d __p = _mm512_load_pd(p_k + j);
                        Ap_k[i] += _mm512_reduce_add_pd(_mm512_mul_pd(__A, __p)); 
                    }
                    for (; j < n; j++) {
                        Ap_k[i] += A[j + i * n] * p_k[j];
                    }
                    alpha_k += p_k[i] * Ap_k[i];
                }
                alpha_k = rz_k / alpha_k;
                __m512d __alpha = _mm512_set1_pd(alpha_k);

                err = 0.;
                i = 0;
                for (; i < 8 * (n / 8); i += 8) {
                    __m512d __Ap_k = _mm512_load_pd(Ap_k + i);
                    __m512d __r = _mm512_load_pd(r_k + i);
                    __r = _mm512_sub_pd(__r, _mm512_mul_pd(__alpha, __Ap_k));
                    _mm512_store_pd(r_k + i, __r);

                    err += _mm512_reduce_add_pd(_mm512_mul_pd(__r, __r));

                    __m512d __p = _mm512_load_pd(p_k + i);
                    __m512d __x = _mm512_load_pd(x + i);
                    __x = _mm512_add_pd(__x, _mm512_mul_pd(__alpha, __p));
                    _mm512_store_pd(x + i, __x);
                }
                for (; i < n; i++) {
                    r_k[i] -= alpha_k * Ap_k[i];
                    err += r_k[i] * r_k[i];
                    x[i] += alpha_k * p_k[i];
                }
                err = sqrt(err);

                // Returns the solution if ||r|| < tol
                if (err <= tol) {
                    break;
                }

                solve_preconditioner(A, r_k, z_k, w, n, precond);

                // ||z_(k + 1)||.
                double rz_kp1 = 0.;
                i = 0;
                for (; i < 8 * (n / 8); i += 8) {
                    __m512d __r = _mm512_load_pd(r_k + i);
                    __m512d __z = _mm512_load_pd(z_k + i);
                    rz_kp1 += _mm512_reduce_add_pd(_mm512_mul_pd(__r, __z));
                }
                for (; i < n; i++) {
                    rz_kp1 += r_k[i] * z_k[i];
                }

                // (r_kp1' * z_kp1) / (r_k' * z_k).
                double beta_k = rz_kp1 / rz_k;
                __m512d __beta = _mm512_set1_pd(beta_k);
                // Updates rz_k.
                rz_k = rz_kp1;

                // Updates the direction.
                i = 0;
                for (; i < 8 * (n / 8); i += 8) {
                    __m512d __p = _mm512_load_pd(p_k + i);
                    __m512d __z = _mm512_load_pd(z_k + i);
                    __p = _mm512_add_pd(__z, _mm512_mul_pd(__beta, __p));
                    _mm512_store_pd(p_k + i, __p);
                }
                for (; i < n; i++) {
                    p_k[i] = z_k[i] + beta_k * p_k[i];
                }
            }
            delete[] Ap_k;
            delete[] p_k;
            delete[] z_k;
        }
        delete[] r_k;
    }
}