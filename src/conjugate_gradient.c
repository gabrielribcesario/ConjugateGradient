#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

// Compile with:
// icx -shared -o ./lib/cg_c.so ./src/conjugate_gradient.c -fPIC -O2 -mavx512f

/*
    A - Symmetric positive-semidefinite n x n matrix;
    b - Target vector of length n;
    x - Initial guess / solution vector of length n;
    tol - Error tolerance.
    max_iter - Maximum number of iterations.
    n - Size of vector b.
*/
void ConjugateGradient(const double* A, const double* b, double* x,
                       const double tol, const unsigned int max_iter,
                       const size_t n) {
    // k-th residual vector.
    double* r_k = malloc(n * sizeof(double));
    // ||r_k||^2.
    double squaredRes_k = 0.;
    // Calculates r = b - A * x and ||r||^2.
    for (int i = 0; i < n; i++) {
        double r_ki = *(b + i);
        int j = 0;
        for (; j < 8 * (n / 8); j += 8) {
            __m512d __A = _mm512_load_pd(A + j + i * n);
            __m512d __x = _mm512_load_pd(x + j);
            r_ki -= _mm512_reduce_add_pd(_mm512_mul_pd(__A, __x)); 
        }
        for (; j < n; j++) {
            r_ki -= (*(A + j + i * n)) * (*(x + j));
        }
        *(r_k + i) = r_ki;
        squaredRes_k += r_ki * r_ki;
    }

    // If ||r|| <= tol then x already is the solution.
    if (sqrt(squaredRes_k) > tol) {
        // k-th direction vector.
        double* p_k = malloc(n * sizeof(double));
        // p_0 = r_0
        memcpy(p_k, r_k, n * sizeof(double));
        // A * p_k
        double* Ap_k = malloc(n * sizeof(double));

        // Iterates the steps.
        for (int k = 0; k < max_iter; k++) {
            int i;
            // (r' * r) / (p_k' * A * p_k)
            double alpha_k = 0.;
            for (i = 0; i < n; i++) {
                double Ap_ki = 0.;
                int j = 0;
                for (; j < 8 * (n / 8); j += 8) {
                    __m512d __A = _mm512_load_pd(A + j + i * n);
                    __m512d __p = _mm512_load_pd(p_k + j);
                    Ap_ki += _mm512_reduce_add_pd(_mm512_mul_pd(__A, __p)); 
                }
                // Data cleanup.
                for (; j < n; j++) {
                    Ap_ki += (*(A + j + i * n)) * (*(p_k + j));
                }
                // alpha_k += p_k' * A * p_k
                alpha_k += (*(p_k + i)) * Ap_ki;
                *(Ap_k + i) = Ap_ki;
            }
            alpha_k = squaredRes_k / alpha_k;
            __m512d __alpha = _mm512_set1_pd(alpha_k);

            // ||r_(k + 1)||^2.
            double squaredRes_kp1 = 0.;
            // Updates x and r and calculates the new value for ||r||^2.
            for (i = 0; i < 8 * (n / 8); i += 8) {
                __m512d __Ap_k = _mm512_load_pd(Ap_k + i);
                __m512d __r = _mm512_load_pd(r_k + i);
                __m512d __p = _mm512_load_pd(p_k + i);
                __m512d __x = _mm512_load_pd(x + i);

                __x = _mm512_add_pd(__x, _mm512_mul_pd(__alpha, __p));
                _mm512_store_pd(x + i, __x);

                __r = _mm512_sub_pd(__r, _mm512_mul_pd(__alpha, __Ap_k));
                _mm512_store_pd(r_k + i, __r);

                squaredRes_kp1 += _mm512_reduce_add_pd(_mm512_mul_pd(__r, __r));
            }
            // Data cleanup.
            for (; i < n; i++) {
                double r_ki = *(r_k + i);
                r_ki -= alpha_k * (*(Ap_k + i));
                *(r_k + i) = r_ki;
                squaredRes_kp1 += r_ki * r_ki;
                *(x + i) += alpha_k * (*(p_k + i));
            }

            // Returns the solution if ||r|| < tol
            if (sqrt(squaredRes_kp1) <= tol) {
                break;
            }

            // (r_kp1' * r_kp1) / (r_k' * r_k).
            double beta_k = squaredRes_kp1 / squaredRes_k;
            __m512d __beta = _mm512_set1_pd(beta_k);
            // Updates the direction.
            for (i = 0; i < 8 * (n / 8); i += 8) {
                __m512d __p = _mm512_load_pd(p_k + i);
                __m512d __r = _mm512_load_pd(r_k + i);

                __p = _mm512_add_pd(__r, _mm512_mul_pd(__beta, __p));
                _mm512_store_pd(p_k + i, __p);
            }
            for (; i < n; i++) {
                *(p_k + i) = (*(r_k + i)) + beta_k * (*(p_k + i));
            }
            // Updates the residual.
            squaredRes_k = squaredRes_kp1;
        }
        free(p_k);
        free(Ap_k);
    }
    free(r_k);
}