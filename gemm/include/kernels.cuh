#pragma once

void sgemm(const int M,
           const int N,
           const int K,
           const float* const a,
           const float* const b,
           float* c,
           const int V,
           const int W);
