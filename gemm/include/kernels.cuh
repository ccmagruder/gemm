#pragma once

#include <cuda_runtime.h>

void sgemm_naive(const int M,
                 const int N,
                 const int K,
                 const float* const A,
                 const float* const B,
                 float* const C,
                 const int V,
                 const int W);

void sgemm(const int M,
           const int N,
           const int K,
           const float* const A,
           const float* const B,
           float* const C,
           const int V,
           const int W,
           int tile_dim = 0);
