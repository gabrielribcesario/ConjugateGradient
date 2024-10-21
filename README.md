# ConjugateGradient
A C/C++ implementation of the conjugate gradient method.

I wrote this as a simple alternative to the direct solvers in some 
languages such as Python and LabVIEW.

This implementation uses AVX512 instructions for faster execution. The AVX 
instructions actually make it slower than your regular np.linalg.solve for smaller
systems, but you do get increasing returns with larger (and sparser) systems.
