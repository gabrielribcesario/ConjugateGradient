# ConjugateGradient
A C implementation of the conjugate gradient method.

I wrote this as a simple alternative to the direct solvers in some 
languages such as Python and LabVIEW.

This implementation uses AVX512 instructions for faster execution.
In fact, the AVX instructions actually make it slower for smaller systems, 
but you do get increasing returns with larger systems. But I don't think
you'd want to apply these methods to smaller systems anyway.