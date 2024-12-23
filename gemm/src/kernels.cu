#include "kernels.cuh"

__global__ void __sgemm(
    const int k,
    const float* const a,
    const float* const b,
    float* c)
{
    int aid, bid, cid = blockIdx.x + blockIdx.y * blockDim.y;
    c[cid] = 0.0;
    for (int i = 0; i < k; i++) {
        aid = blockIdx.x + k * i;
        bid = blockIdx.y * blockDim.y + k;
        c[cid] += a[aid] * b[bid];
    }
}

void sgemm(
    const int m,
    const int n,
    const int k,
    const float* const a,
    const float* const b,
    float* c)
{
    __sgemm<<<m, n>>>(k, a, b, c);
}
