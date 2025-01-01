#! /bin/bash
# format the relivant files in place
clang-format $@ gemm/**/*.h gemm/**/*.cpp gemm/**/*.cuh gemm/**/*.cu
