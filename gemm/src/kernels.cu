#include <stdexcept>

#include "kernels.cuh"

void cudaCheck() {
    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess) {
        char msg[100];
        sprintf(msg, "GPU kernel assert: %s:%d \"%s\"\n", __FILE__, __LINE__,
                cudaGetErrorString(code));
        throw std::runtime_error(msg);
    }
}

__global__ void __sgemm(const int M,
                        const int N,
                        const int K,
                        const float* const A,
                        const float* const B,
                        float* const C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N)
        return;

    const float* a = A + i;
    const float* b = B + j * K;
    float* const c = C + i + j * M;
    *c = 0.0;
    for (int k = 0; k < K; k++) {
        *c += *a * *b;
        a += M;
        b++;
    }
}

void sgemm(const int M,
           const int N,
           const int K,
           const float* const a,
           const float* const b,
           float* const c,
           const int V,
           const int W) {
    dim3 gridDim(M / V + (M % V != 0), N / W + (N % W != 0), 1);
    dim3 blockDim(V, W, 1);
    __sgemm<<<gridDim, blockDim>>>(M, N, K, a, b, c);
    cudaDeviceSynchronize();
    cudaCheck();
}
