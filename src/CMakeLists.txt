cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(ConjugateGradientMethod LANGUAGES C CXX)

find_package(OpenMP REQUIRED)

link_directories("${CMAKE_CURRENT_LIST_DIR}/../lib")
include_directories("${CMAKE_CURRENT_LIST_DIR}/../include")

# output dirs
file(MAKE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../build")

# libm
set(MATH_LIBRARY m)

# ---Build executables-----------------------------------------------------------
add_library(ConjugateGradient SHARED HighPerformanceCG.c)
target_link_libraries(ConjugateGradient OpenMP::OpenMP_CXX ${MATH_LIBRARY})
target_compile_options(ConjugateGradient PRIVATE "-shared;-fPIC;-O3;-fopenmp;-msse4.2;-lm")

add_library(PreconditionedCG SHARED HighPerformancePCG.c)
target_link_libraries(PreconditionedCG OpenMP::OpenMP_CXX ${MATH_LIBRARY})
target_compile_options(PreconditionedCG PRIVATE "-shared;-fPIC;-O3;-fopenmp;-msse4.2;-lm")

add_executable(hilbert hilbert.c)
target_link_libraries(hilbert ConjugateGradient PreconditionedCG)

add_executable(mtx mtx.c)
target_link_libraries(mtx ConjugateGradient PreconditionedCG)