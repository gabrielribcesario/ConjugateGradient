from memory_profiler import memory_usage
import pandas as pd
import numpy as np
import ctypes
import timeit

# Creates a tridiagonal matrix.
def tridiag(a, b):
    diag2 = np.diag(b, -1)
    return np.diag(a) + diag2 + diag2.T

# Benchmarks the Preconditioned CG.
def bench_pcg():
    np.random.seed(0)
    # Time
    t_pcg = timeit.timeit("prec_cg(A_rav, b, np.random.uniform(-1., 1., size=(size,)), tol, w, max_iter, prec_enum[precond], size)",
                          "from __main__ import A_rav, b, np, tol, w, max_iter, prec_enum, precond, size, prec_cg", number=repeat) / repeat * 1000.
    # Memory
    x_pcg = np.zeros((size,), dtype=np.float64)
    peak_pcg  = memory_usage((prec_cg, [A_rav, b, x_pcg, tol, w, max_iter, prec_enum[precond], size], {}), max_usage=True, max_iterations=1)
    # Residual
    r_pcg = np.linalg.norm(b - np.dot(A, x_pcg))
    return r_pcg, t_pcg, peak_pcg

# Benchmarks the Unpreconditioned CG.
def bench_ucg():
    np.random.seed(0)
    # Time
    t_ucg = timeit.timeit("unp_cg(A_rav, b, np.random.uniform(-1., 1., size=(size,)), tol, max_iter, size)",
                          "from __main__ import A_rav, b, np, tol, max_iter, size, unp_cg", number=repeat) / repeat * 1000.
    # Memory
    x_ucg = np.zeros((size,), dtype=np.float64)
    peak_ucg = memory_usage((unp_cg, [A_rav, b, x_ucg, tol, max_iter, size], {}), max_usage=True, max_iterations=1)
    # Residual
    r_ucg = np.linalg.norm(b - np.dot(A, x_ucg))
    return r_ucg, t_ucg, peak_ucg

# Benchmarks Numpy's linear system solver.
def bench_np():
    np.random.seed(0)
    # Time
    t_np = timeit.timeit("np.linalg.solve(A, b)",
                         "from __main__ import A, b, np", number=repeat) / repeat * 1000.
    # Memory
    peak_np, x_np = memory_usage((np.linalg.solve, [A, b], {}), max_usage=True, retval=True, max_iterations=1)
    # Residual
    r_np = np.linalg.norm(b - np.dot(A, x_np))
    return r_np, t_np, peak_np

def print_results(prefix, residual_norm, avg_time, peak_mem_usage):
    print(f"{prefix}'s norm of residuals: {residual_norm}")
    print(f"{prefix}'s average time [ms]: {avg_time}")
    print(f"{prefix}'s peak memory usage [KiB]: {peak_mem_usage}")

# Whether to benchmark the unpreconditioned version.
test_ucg = False

# timeit.timeit() number of repeats.
repeat = 100
# Tolerance.
tol = 1.E-10
# Number of rows / columns.
size = 10000
# Maximum number of iterations (>= n).
max_iter = max(size, 1000)
# Preconditioner.
precond = b"ssor"
prec_enum = {b"jacobi": 0, b"ssor": 1}#b"incomplete_cholesky": 2}=
# SSOR relaxation factor. For whatever reason, choosing a w factor very close to 2 (w > 1.999...) 
# increases the tridiagonal system's convergence rate, even though this also causes kappa(M^-1 * A) to 
# increase. This could be because I'm only implicitly calculating w / (2 - w) when solving for the residual.
w = 1.9999

np.random.seed(42)
A = tridiag(np.full(shape=(size,), fill_value=5.), np.full(shape=(size - 1,), fill_value=-2.5))
A_rav = A.ravel()

print("-" * 80)
if size <= 2000:
    kappa_A = np.linalg.cond(A)
    pos_def = np.all(np.linalg.eigvals(A) > 0)
    print(f"A is positive-definite? {pos_def}")
    print(f"A's condition number: {kappa_A}")
    print("-" * 80)

# Original solution.
x_sol = np.random.uniform(-1., 1., size=(size,))
# b vector.
b = np.dot(A, x_sol)

# Import the library and define the argument/return types for the preconditioned CG function.
prec_cg_dll = ctypes.cdll.LoadLibrary("./lib/precond_cg.so")
prec_cg = prec_cg_dll.ConjugateGradient
prec_cg.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double), # A
                    np.ctypeslib.ndpointer(ctypes.c_double), # b
                    np.ctypeslib.ndpointer(ctypes.c_double), # x
                    ctypes.c_double,                         # tol
                    ctypes.c_double,                         # w
                    ctypes.c_uint,                           # max_iter
                    ctypes.c_uint,                           # precond 
                    ctypes.c_size_t                          # n
                    ]
prec_cg.restypes = None

# Same as the above, but for the unpreconditioned version.
if test_ucg:
    unp_cg_dll = ctypes.cdll.LoadLibrary("./lib/cg_cpp.so")
    unp_cg = unp_cg_dll.ConjugateGradient
    unp_cg.argtypes = [np.ctypeslib.ndpointer(ctypes.c_double), # A
                       np.ctypeslib.ndpointer(ctypes.c_double), # b
                       np.ctypeslib.ndpointer(ctypes.c_double), # x
                       ctypes.c_double,                         # tol
                       ctypes.c_uint,                           # max_iter
                       ctypes.c_size_t                          # n
                       ]
    unp_cg.restypes = None

if __name__ == "__main__":
    # Preconditioned CG
    r_pcg, t_pcg, peak_pcg = bench_pcg()
    print_results("P_CG", r_pcg, t_pcg, peak_pcg)
    print("-" * 80)
    # Numpy
    r_np, t_np, peak_np = bench_np()
    print_results("NP", r_np, t_np, peak_np)
    print("-" * 80)
    # Export the results.
    results_df = pd.DataFrame({"Numpy": [t_np, r_np, peak_np],
                               "P_CG": [t_pcg, r_pcg, peak_pcg],
                               }, index=["avg_dt[ms]", "norm_of_residuals", "peak_memory_usage[KiB]"])
    # Unpreconditioned CG
    if test_ucg:
        r_ucg, t_ucg, peak_ucg = bench_ucg()
        print_results("U_CG", r_ucg, t_ucg, peak_ucg)
        print("-" * 80)
        results_df['U_CG'] = [t_pcg, r_pcg, peak_ucg]
    results_df.to_csv("./benchmark_results.csv")