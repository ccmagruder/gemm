#! /bin/bash
clang-tidy -p build \
    gemm/**/*.h \
    gemm/**/*.cpp \
    gemm/**/*.cuh \
    gemm/**/*.cu

