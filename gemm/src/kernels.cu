#include <cassert>
#include "kernels.cuh"

__global__ void __sgemm(const int k,
                        const float* const a,
                        const float* const b,
                        float* c) {
    int aid, bid, cid = blockIdx.x + blockIdx.y * gridDim.x;
    c[cid] = 0.0;
    for (int i = 0; i < k; i++) {
        aid = blockIdx.x + gridDim.x * i;
        bid = blockIdx.y * k + i;
        c[cid] += a[aid] * b[bid];
    }
}

void sgemm(const int m,
           const int n,
           const int k,
           const float* const a,
           const float* const b,
           float* c) {
    dim3 grid_size(m, n, 1);
    dim3 block_size(1, 1, 1);
    __sgemm<<<grid_size, block_size>>>(k, a, b, c);
    cudaDeviceSynchronize();
}
