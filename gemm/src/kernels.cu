#include <cassert>
#include "kernels.cuh"

__global__ void __sgemm(const int M,
                        const int N,
                        const int K,
                        const float* const a,
                        const float* const b,
                        float* c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N)
        return;

    int aid, bid, cid = i + j * M;
    c[cid] = 0.0;
    for (int k = 0; k < K; k++) {
        aid = i + M * k;
        bid = j * K + k;
        c[cid] += a[aid] * b[bid];
    }
}

void sgemm(const int M,
           const int N,
           const int K,
           const float* const a,
           const float* const b,
           float* c,
           const int V,
           const int W) {
    dim3 gridDim(M / V + (M % V != 0), N / W + (N % W != 0), 1);
    dim3 blockDim(V, W, 1);
    __sgemm<<<gridDim, blockDim>>>(M, N, K, a, b, c);
    cudaDeviceSynchronize();
}
