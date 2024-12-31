#pragma once

void cudaCheck();

void sgemm(const int M,
           const int N,
           const int K,
           const float* const a,
           const float* const b,
           float* const c,
           const int V,
           const int W);
