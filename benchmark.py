import pandas as pd
import numpy as np
import ctypes
import timeit

# Import the libraries.
cg_dll_c = ctypes.cdll.LoadLibrary("./lib/cg_c.so")
cg_dll_cpp = ctypes.cdll.LoadLibrary("./lib/cg_cpp.so")
# Import the functions.
cg_c = cg_dll_c.ConjugateGradient
cg_cpp = cg_dll_cpp.ConjugateGradient
# Argument types.
cg_c.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double), # A
                 np.ctypeslib.ndpointer(ctypes.c_double), # b
                 np.ctypeslib.ndpointer(ctypes.c_double), # x
                 ctypes.c_double,                         # tol
                 ctypes.c_uint,                           # max_iter
                 ctypes.c_size_t                          # n
                 ]
cg_cpp.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double), # A
                   np.ctypeslib.ndpointer(ctypes.c_double), # b
                   np.ctypeslib.ndpointer(ctypes.c_double), # x
                   ctypes.c_double,                         # tol
                   ctypes.c_uint,                           # max_iter
                   ctypes.c_size_t                          # n
                   ]
# Return types.
cg_c.restypes = None
cg_cpp.restypes = None

# timeit.timeit() number.
repeat = 100000
# Tolerance.
tol = 1.E-5
# Number of rows / columns.
size = 50
# Maximum number of iterations (>= n).
max_iter = max(size, 1000)

np.random.seed(42)
# A matrix (n x n, symmetric).
A = np.random.uniform(-10., 10., size=(size, size))
A = np.dot(A.T, A)
A_rav = A.ravel()
kappa_A = np.linalg.cond(A)
print(f"A's condition number: {kappa_A}")
# Original solution.
x_sol = np.random.uniform(-1., 1., size=(size,))
# b vector.
b = np.dot(A, x_sol)

if __name__ == "__main__":
    # CG (written in C)
    ## Time benchmarking.
    np.random.seed(0)
    t_cg_c = timeit.timeit("cg_c(A_rav, b, np.random.uniform(-1., 1., size=(size,)), tol, max_iter, size)",
                         "from __main__ import A_rav, b, np, tol, max_iter, size, cg_c", number=repeat) / repeat * 1000.
    ## Norm of residuals calculation.
    x_cg_c = np.random.uniform(-1., 1., size=(size,))
    cg_c(A_rav, b, x_cg_c, tol, max_iter, size)
    r_cg_c = np.linalg.norm(b - np.dot(A, x_cg_c))
    print(f"CG average time [ms]: {t_cg_c}")
    print(f"CG norm of residuals: {r_cg_c}")

    # CG (written in C++)
    ## Time benchmarking.
    np.random.seed(0)
    t_cg_cpp = timeit.timeit("cg_cpp(A_rav, b, np.random.uniform(-1., 1., size=(size,)), tol, max_iter, size)",
                             "from __main__ import A_rav, b, np, tol, max_iter, size, cg_cpp", number=repeat) / repeat * 1000.
    ## Norm of residuals calculation.
    x_cg_cpp = np.random.uniform(-1., 1., size=(size,))
    cg_cpp(A_rav, b, x_cg_cpp, tol, max_iter, size)
    r_cg_cpp = np.linalg.norm(b - np.dot(A, x_cg_cpp))
    print(f"CG++ average time [ms]: {t_cg_cpp}")
    print(f"CG++ norm of residuals: {r_cg_cpp}")

    # Numpy
    ## Time benchmarking.
    np.random.seed(0)
    t_np = timeit.timeit("np.linalg.solve(A, b)",
                         "from __main__ import A, b, np", number=repeat) / repeat * 1000.
    ## Norm of residuals calculation.
    x_np = np.linalg.solve(A, b)
    r_np = np.linalg.norm(b - np.dot(A, x_np))
    print(f"NP average time [ms]: {t_np}")
    print(f"NP norm of residuals: {r_np}")

    # Export the results.
    results_df = pd.DataFrame({"Numpy": [t_np, r_np],
                               "CG": [t_cg_c, r_cg_c],
                               "CG++": [t_cg_cpp, r_cg_cpp],
                               }, index=["avg dt [ms]", "norm of residuals"])
    results_df.to_csv("./benchmark_results.csv")