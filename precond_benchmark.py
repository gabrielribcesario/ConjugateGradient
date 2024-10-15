import pandas as pd
import numpy as np
import ctypes
import timeit

def jacobi(A, **kwargs):
    D_inv = np.diag(1. / np.diag(A))
    return D_inv

def ssor(A, w=1., **kwargs):
    D = np.diag(np.diag(A))
    L_inv = np.linalg.inv(D / w + np.tril(A, k=-1))
    return ((2. - w) / w) * np.linalg.multi_dot([L_inv.T, D, L_inv])

def tridiag(a, b):
    return np.diag(a) + np.diag(b, -1) + np.diag(b, 1)

# Import the libraries.
prec_cg_dll = ctypes.cdll.LoadLibrary("./lib/precond_cg.so")
unp_cg_dll = ctypes.cdll.LoadLibrary("./lib/cg_cpp.so")

# Import the functions.
prec_cg = prec_cg_dll.ConjugateGradient
unp_cg = unp_cg_dll.ConjugateGradient

# Argument types.
prec_cg.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double), # A
                    np.ctypeslib.ndpointer(ctypes.c_double), # b
                    np.ctypeslib.ndpointer(ctypes.c_double), # x
                    ctypes.c_double,                         # tol
                    ctypes.c_double,                         # w
                    ctypes.c_uint,                           # max_iter
                    ctypes.c_uint,                           # precond 
                    ctypes.c_size_t                          # n
                    ]
unp_cg.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double), # A
                   np.ctypeslib.ndpointer(ctypes.c_double), # b
                   np.ctypeslib.ndpointer(ctypes.c_double), # x
                   ctypes.c_double,                         # tol
                   ctypes.c_uint,                           # max_iter
                   ctypes.c_size_t                          # n
                   ]
# Return types.
prec_cg.restypes = None
unp_cg.restypes = None

# timeit.timeit() number.
repeat = 10
# Tolerance.
tol = 1.E-10
# Number of rows / columns.
size = 2000
# Maximum number of iterations (>= n).
max_iter = max(size, 1000)
# Preconditioner.
precond = b"ssor"
# SSOR relaxation factor.
# For whatever reason a w factor very close to 2 (w > 1.999...) increases the tridiagonal system's
# convergence rate, even though this also causes kappa(M^-1 * A) to increase.
w = 1.975

prec_func = {b"jacobi": jacobi, b"ssor": ssor}#b"incomplete_cholesky": incomplete_cholesky}
prec_enum = {b"jacobi": 0, b"ssor": 1}#b"incomplete_cholesky": 2}

np.random.seed(42)
# A matrix (n x n symmetric).
#A = np.random.uniform(-100., 100., size=(size, size))
#A = np.dot(A.T, A)
A = tridiag(np.full(shape=(size,), fill_value=5.), np.full(shape=(size - 1,), fill_value=-2.5))
A_rav = A.ravel()

print("-" * 80)
if size <= 2000:
    kargs = {"A": A, "w": w}

    kappa_A = np.linalg.cond(A)
    kappa_PA = np.linalg.cond(np.dot(prec_func[precond](**kargs), A))
    pos_def = np.all(np.linalg.eigvals(A) > 0)

    print(f"A is positive definite? {pos_def}")
    print(f"A's condition number: {kappa_A}")
    print(f"Preconditioned A's condition number: {kappa_PA}")
    print("-" * 80)

# Original solution.
x_sol = np.random.uniform(-1., 1., size=(size,))
# b vector.
b = np.dot(A, x_sol)

if __name__ == "__main__":
    # Preconditioned CG
    ## Time benchmarking.
    np.random.seed(0)
    t_prec_cg = timeit.timeit("prec_cg(A_rav, b, np.random.uniform(-1., 1., size=(size,)), tol, w, max_iter, prec_enum[precond], size)",
                              "from __main__ import A_rav, b, np, tol, w, max_iter, prec_enum, precond, size, prec_cg", number=repeat) / repeat * 1000.
    ## Norm of residuals calculation.
    x_prec_cg = np.random.uniform(-1., 1., size=(size,))
    prec_cg(A_rav, b, x_prec_cg, tol, w, max_iter, prec_enum[precond], size)
    r_prec_cg = np.linalg.norm(b - np.dot(A, x_prec_cg))
    print(f"P_CG average time [ms]: {t_prec_cg}")
    print(f"P_CG norm of residuals: {r_prec_cg}")
    print("-" * 80)

    # Unpreconditioned CG
    ## Time benchmarking.
    np.random.seed(0)
    t_unp_cg = timeit.timeit("unp_cg(A_rav, b, np.random.uniform(-1., 1., size=(size,)), tol, max_iter, size)",
                             "from __main__ import A_rav, b, np, tol, max_iter, size, unp_cg", number=repeat) / repeat * 1000.
    ## Norm of residuals calculation.
    x_unp_cg = np.random.uniform(-1., 1., size=(size,))
    unp_cg(A_rav, b, x_unp_cg, tol, max_iter, size)
    r_unp_cg = np.linalg.norm(b - np.dot(A, x_unp_cg))
    print(f"U_CG average time [ms]: {t_unp_cg}")
    print(f"U_CG norm of residuals: {r_unp_cg}")
    print("-" * 80)

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
    print("-" * 80)

    # Export the results.
    results_df = pd.DataFrame({"Numpy": [t_np, r_np],
                               "U_CG": [t_prec_cg, r_prec_cg],
                               "P_CG": [t_unp_cg, r_unp_cg],
                               }, index=["avg dt [ms]", "norm of residuals"])
    results_df.to_csv("./benchmark_results.csv")