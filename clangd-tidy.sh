#! /bin/bash
clangd-tidy -p build \
    gemm/**/*.h \
    gemm/**/*.cpp \
    gemm/**/*.cuh \
    gemm/**/*.cu
