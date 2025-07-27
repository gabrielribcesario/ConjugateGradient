cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(ConjugateGradient LANGUAGES C CXX)

if (NOT TARGET OpenMP::OpenMP_CXX)
    find_package(OpenMP REQUIRED)
endif()

link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# output dirs
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

# ---Build executables-----------------------------------------------------------
add_library(ConjugateGradient SHARED src/HighPerformanceCG.c)
target_link_libraries(ConjugateGradient OpenMP::OpenMP_CXX m)
target_compile_options(ConjugateGradient PRIVATE "-shared;-fPIC;-O3;-msse4.2;-lm;-fopenmp")

add_library(PreconditionedCG SHARED src/HighPerformancePCG.c)
target_link_libraries(PreconditionedCG OpenMP::OpenMP_CXX m)
target_compile_options(PreconditionedCG PRIVATE "-shared;-fPIC;-O3;-msse4.2;-lm;-fopenmp")

add_library(HistoryCG SHARED src/HistoryCG.c)
target_link_libraries(HistoryCG OpenMP::OpenMP_CXX m)
target_compile_options(HistoryCG PRIVATE "-shared;-fPIC;-O3;-msse4.2;-lm;-fopenmp")

add_executable(hilbert src/hilbert.c)
target_link_libraries(hilbert ConjugateGradient m)
target_compile_options(hilbert PRIVATE "-O3;-lm")

add_executable(mtx src/mtx.c)
target_link_libraries(mtx ConjugateGradient PreconditionedCG m)
target_compile_options(mtx PRIVATE "-O3;-lm")