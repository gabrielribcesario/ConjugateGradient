from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import ctypes

def HilbertMatrix(n):
    H = np.arange(1, n + 1).reshape(-1,1)
    return 1. / (H + H.T - 1)

libHistoryCG = ctypes.cdll.LoadLibrary("lib/libHistoryCG.so")
historyCG = libHistoryCG.HistoryCG
historyCG.argtypes = [ctypes.c_int,                            # n
                      ctypes.c_int,                            # maxiter
                      ctypes.c_double,                         # tol
                      np.ctypeslib.ndpointer(ctypes.c_double), # A
                      np.ctypeslib.ndpointer(ctypes.c_double), # b
                      np.ctypeslib.ndpointer(ctypes.c_double), # x
                      np.ctypeslib.ndpointer(ctypes.c_double), # history
                      ]
historyCG.restypes = ctypes.c_int

# High dimensionality and low number of iterations leads to poor PCA
# The idea here is to force a large number of iterations by using 
# a tight tolerance while also solving for a small linear system
tol = 1e-18 
n = 10  # size of the x vector
maxiter = 1000
history = np.zeros((maxiter * n), dtype="float64") # x vector history

# Set the linear system system (Ax=b)
rng = np.random.default_rng(42)
A = HilbertMatrix(n)
x = np.zeros(n, dtype="float64")
sol = rng.normal(0, 1., n)
b = A.dot(sol)
A = A.ravel()

# Solve linear system
niter = historyCG(n, maxiter, tol, A, b, x, history)
print(f"Ran CG algorithm with {niter}/{maxiter} iterations")
# Reshape and slice history
history = history.reshape(maxiter, n)[:niter]

# Compute loss history
A = A.reshape(n, n)
loss = np.array([np.linalg.norm(A.dot(x_i) - b) for x_i in history])

# PCA projection of the history
pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(2, svd_solver="full"))])
pca2d_hist = pipe.fit_transform(history) # fit the steps history
[x1, x2] = pca2d_hist.T
est_sol = pca2d_hist[loss.argmin()] # estimated solution obtained with CG after the PCA
true_sol = pipe.transform(sol.reshape(1,-1))[0] # ground truth after the PCA

# Contour limits
x1lim, x2lim = np.max(np.abs([x1.min(), x1.max()])) * 1.5, \
               np.max(np.abs([x2.min(), x2.max()])) * 1.5 

# Set up contour grid
nelem = 100 # grid size
u, v = np.linspace(est_sol[0] - x1lim, est_sol[0] + x1lim, nelem), \
       np.linspace(est_sol[1] - x2lim, est_sol[1] + x2lim, nelem)
U, V = np.meshgrid(u, v) # {U, V} = vector space grid (i.e. ℝ^n)
grid = pipe.inverse_transform(np.c_[U.ravel(), V.ravel()]) # certainly a lot of information loss but whatever
loss_contour = np.array([np.linalg.norm(A.dot(x_i) - b) for x_i in grid]).reshape(nelem, nelem)

# RMSE with the ground truth
rmse_original = np.linalg.norm(sol - history[np.argmin(loss)]) / sol.size
rmse_pca = np.linalg.norm(est_sol - true_sol) / est_sol.size
print(f"RMSE original: {rmse_original:.6g}")
print(f"RMSE PCA:      {rmse_pca:.6g}")

plt.contour(U, V, loss_contour, levels=20, cmap="coolwarm")
plt.scatter([true_sol[0]], [true_sol[1]], marker="x")
plt.plot(x1, x2, ls="--")
plt.scatter(x1, x2, s=10.)
plt.show()
