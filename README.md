# ConjugateGradient
A C++ implementation of the (preconditioned) conjugate gradient method.

I wrote this so I could have a simple, less memory intensive alternative to the direct 
solvers in some languages such as Python and LabVIEW. This implementation uses AVX512 
instructions for faster execution. The AVX instructions actually make it slower than your 
regular np.linalg.solve for smaller systems, but you do get increasing returns with larger 
and sparser systems. 

The benchmark was made with a relatively small system (10000 x 10000) since I only needed
to make sure that the code worked. I only benchmarked the preconditioned version since the 
unpreconditioned one takes a ridiculous amount of time to solve the system.

If you want to re-compile any of the .so files then you will need an Intel compiler.