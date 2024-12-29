#include <cassert>
#include "kernels.cuh"

__global__ void __sgemm(const uint M,
                        const uint N,
                        const uint K,
                        const float* const a,
                        const float* const b,
                        float* c) {
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > M || j > N)
        return;

    uint aid, bid, cid = i + j * M;
    c[cid] = 0.0;
    for (uint k = 0; k < K; k++) {
        aid = i + M * k;
        bid = j * K + k;
        c[cid] += a[aid] * b[bid];
    }
}

void sgemm(const uint M,
           const uint N,
           const uint K,
           const float* const a,
           const float* const b,
           float* c,
           const uint W) {
    dim3 gridDim(M / W + (M % W != 0), N / W + (N % W != 0), 1);
    dim3 blockDim(W, W, 1);
    __sgemm<<<gridDim, blockDim>>>(M, N, K, a, b, c);
    cudaDeviceSynchronize();
}
