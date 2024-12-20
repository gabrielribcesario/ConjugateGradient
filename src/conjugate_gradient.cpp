#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <string.h>

// Compile with:
// icpx -shared -o ./lib/cg_cpp.so ./src/conjugate_gradient.cpp -fPIC -O2 -mavx512f

/*
    A - Symmetric positive-semidefinite n x n matrix;
    b - Target vector of length n;
    x - Initial guess / solution vector of length n;
    tol - Error tolerance.
    max_iter - Maximum number of iterations.
    n - Size of vector b.
*/
extern "C" {
    void ConjugateGradient(const double* A, const double* b, double* x,
                           const double tol, const unsigned int max_iter,
                           const size_t n) {
        // k-th residual vector.
        double* r_k = new double[n];
        // k-th squared residual norm.
        double squaredRes_k = 0.;

        // Calculates r = b - A * x and ||r_k||^2.
        for (int i = 0; i < n; i++) {
            double r_ki = b[i];
            int j = 0;
            for (; j < 8 * (n / 8); j += 8) {
                __m512d __A = _mm512_load_pd(A + j + i * n);
                __m512d __x = _mm512_load_pd(x + j);
                r_ki -= _mm512_reduce_add_pd(_mm512_mul_pd(__A, __x)); 
            }
            for (; j < n; j++) {
                r_ki -= A[j + i * n] * x[j];
            }
            r_k[i] = r_ki;
            squaredRes_k += r_ki * r_ki;
        }

        // If ||r_k|| <= tol then x already is the solution.
        if (sqrt(squaredRes_k) > tol) {
            // k-th direction vector.
            double* p_k = new double[n];
            // p_0 = r_0
            memcpy(p_k, r_k, n * sizeof(double));
            // A * p_k
            double* Ap_k = new double[n];

            // Iterates the steps.
            for (int k = 0; k < max_iter; k++) {
                // alpha_k = (r_k' * r_k) / (p_k' * A * p_k)
                double alpha_k = 0.;
                
                int i = 0;
                // Calculates alpha_k and A * p_k.
                for (; i < n; i++) {
                    double Ap_ki = 0.;
                    int j = 0;
                    for (; j < 8 * (n / 8); j += 8) {
                        __m512d __A = _mm512_load_pd(A + j + i * n);
                        __m512d __p = _mm512_load_pd(p_k + j);
                        Ap_ki += _mm512_reduce_add_pd(_mm512_mul_pd(__A, __p)); 
                    }
                    for (; j < n; j++) {
                        Ap_ki += A[j + i * n] * p_k[j];
                    }
                    Ap_k[i] = Ap_ki;
                    alpha_k += p_k[i] * Ap_ki;
                }
                alpha_k = squaredRes_k / alpha_k;
                __m512d __alpha = _mm512_set1_pd(alpha_k);

                // (k + 1)-th squared residual norm.
                double squaredRes_kp1 = 0.;
                // Updates x and r and calculates ||r_(k+1)||^2.
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
                for (; i < n; i++) {
                    double r_kp1 = r_k[i] - alpha_k * Ap_k[i];
                    squaredRes_kp1 += r_kp1 * r_kp1;
                    x[i] += alpha_k * p_k[i];
                    r_k[i] = r_kp1;
                }

                // Returns the solution if ||r_(k+1)|| < tol.
                if (sqrt(squaredRes_kp1) <= tol) {
                    break;
                }

                // beta_k = (r_kp1' * r_kp1) / (r_k' * r_k)
                double beta_k = squaredRes_kp1 / squaredRes_k;
                __m512d __beta = _mm512_set1_pd(beta_k);
                // Updates ||r_k||^2.
                squaredRes_k = squaredRes_kp1;

                // Updates the direction.
                for (i = 0; i < 8 * (n / 8); i += 8) {
                    __m512d __p = _mm512_load_pd(p_k + i);
                    __m512d __r = _mm512_load_pd(r_k + i);

                    __p = _mm512_add_pd(__r, _mm512_mul_pd(__beta, __p));
                    _mm512_store_pd(p_k + i, __p);
                }
                for (; i < n; i++) {
                    p_k[i] = r_k[i] + beta_k * p_k[i];
                }
            }
            delete[] p_k;
            delete[] Ap_k;
        }
        delete[] r_k;
    }
}