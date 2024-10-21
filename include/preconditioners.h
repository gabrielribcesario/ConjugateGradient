#ifndef PRECONDITIONERS_H
#define PRECONDITIONERS_H

#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string.h>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

// Two-sided preconditioning.
namespace Preconditioner {

    /*
      Solves a D * x = y linear system, where D is a n x n diagonal matrix.
      For this function D is a n-length 1D-array.
    */
    static inline void solve_jacobi(const double* D, const double* y, double* x, const size_t n) {
        for (int i = 0; i < n; i++) {
            x[i] = y[i] / D[i * (n + 1)];    
        }
    }

   /*
     Solves a M * x = y linear system, where M is a n x n SSOR preconditioning 
     matrix. For this function M is a n^2-length 1D-array. (const double) w is 
     a relaxation factor, where 0 < w < 2.
   */
    static inline void solve_ssor(const double* M, const double* y, double* x, 
                                  const double w, const size_t n) {
        // Let u = (D / w + L)^T * x
        // This block of code solves w / (2 - w) * (D / w + L) * D^-1 * u = y
        // using forward substitution.
        x[0] = y[0] * (2. - w);
        int i = 1;
        for (; i < n; i++) {
            double rowSum = 0.;
            for (int j = 0; j < i; j++) {
                rowSum += x[j] * M[j + i * n] / M[j * (n + 1)];
            }
            x[i] = y[i] * (2. - w) - rowSum * w;
        }
        
        // Solves (D / w + L)^T * x = y using backward substitution.
        x[n - 1] *= w / M[n * n - 1];
        i = n - 2;
        for (; i > -1; i--) {
            double rowSum = 0.;
            int j = i + 1;
            for (; j < n - 8; j += 8) {
                __m512d __M = _mm512_load_pd(M + j + i * n);
                __m512d __x = _mm512_load_pd(x + j);
                rowSum += _mm512_reduce_add_pd(_mm512_mul_pd(__M, __x));
            }
            for (; j < n; j++) {
                rowSum += M[j + i * n] * x[j];
            }
            x[i] = (x[i] - rowSum) * (w / M[i * (n + 1)]);
        }
    }

    /*
    static inline void solve_ichol() {
        
    }
    */

    static inline void solve_preconditioner(const double* M, const double* r, double* z, 
                                            const double w, const size_t n, 
                                            const unsigned int precond) {
        if (precond == 0) {
            solve_jacobi(M, r, z, n);
        }
        else if (precond == 1) {
            solve_ssor(M, r, z, w, n);
        }
    }

}

#ifdef __cplusplus
}
#endif

#endif