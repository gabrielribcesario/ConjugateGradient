import pandas as pd
import numpy as np
import scipy
import ctypes
import timeit

# Import the library.
cg_dll = ctypes.cdll.LoadLibrary("./ConjugateGradient.so")
# Import the function.
cg = cg_dll.ConjugateGradient
# Argument types.
cg.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double), # A
               np.ctypeslib.ndpointer(ctypes.c_double), # b
               np.ctypeslib.ndpointer(ctypes.c_double), # x
               ctypes.c_double,                         # tol
               ctypes.c_uint,                           # max_iter
               ctypes.c_size_t                          # n
               ]
# Return types.
cg.restypes = None

# timeit.timeit() number.
repeat = 10000
# Tolerance.
tol = 1.E-10
# Number of rows / columns.
size = 50
# Maximum number of iterations (>= n).
max_iter = max(size, 1000)

np.random.seed(42)
# A matrix (n x n, symmetric).
A = np.random.uniform(-10., 10., size=(size, size))
A = np.dot(A.T, A)
A_rav = A.ravel()
A_diag = np.diag(A)
A_lower = np.tril(A)
kappa_A = np.linalg.cond(A)
kappa_PA = np.linalg.cond(np.dot(np.diag(1. / A_diag), A))
print(f"A's condition number: {kappa_A}")
print(f"Preconditioned A's condition number: {kappa_PA}")
# Original solution.
x_sol = np.random.uniform(-1., 1., size=(size,))
# b vector.
b = np.dot(A, x_sol)

if __name__ == "__main__":
    # Time benchmarking.
    np.random.seed(0)
    t_cg = timeit.timeit("cg(A_rav, b, np.random.uniform(-1., 1., size=(size,)), tol, max_iter, size)",
                         "from __main__ import A_rav, b, np, tol, max_iter, size, cg", number=repeat) / repeat * 1000.
    # Norm of residuals calculation.
    x_cg = np.random.uniform(-1., 1., size=(size,))
    cg(A_rav, b, x_cg, tol, max_iter, size)
    r_cg = np.linalg.norm(b - np.dot(A, x_cg))
    print(f"CG average time [ms]: {t_cg}")
    print(f"CG norm of residuals: {r_cg}")

    # Time benchmarking.
    np.random.seed(0)
    t_np = timeit.timeit("np.linalg.solve(A, b)",
                         "from __main__ import A, b, np", number=repeat) / repeat * 1000.
    # Norm of residuals calculation.
    x_np = np.linalg.solve(A, b)
    r_np = np.linalg.norm(b - np.dot(A, x_np))
    print(f"NP average time [ms]: {t_np}")
    print(f"NP norm of residuals: {r_np}")

    # Time benchmarking.
    np.random.seed(0)
    t_sp = timeit.timeit("scipy.sparse.linalg.cg(A, b, np.random.uniform(-1., 1., size=(size,)))",
                         "from __main__ import A, b, scipy, np, size", number=repeat) / repeat * 1000.
    # Norm of residuals calculation.
    x_sp, status = scipy.sparse.linalg.cg(A, b, np.random.uniform(-1., 1., size=(size,)))
    r_sp = np.linalg.norm(b - np.dot(A, x_sp))
    print(f"SP average time [ms]: {t_sp}")
    print(f"SP norm of residuals: {r_sp}")

    # Export the results.
    results_df = pd.DataFrame({"Numpy": [t_np, r_np],
                               "MyCG": [t_cg, r_cg],
                               "Scipy": [t_sp, r_sp]
                               }, index=["avg dt [ms]", "norm of residuals"])
    results_df.to_csv("./benchmark_results.csv")