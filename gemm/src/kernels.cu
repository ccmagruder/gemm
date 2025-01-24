#include "kernels.cuh"
#include "utils.cuh"

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

    const float* a = A + i;          // a = A[i, 0]
    const float* b = B + j * K;      // b = B[0, j]
    float* const c = C + i + j * M;  // c = C[i, j]
    float sum = 0.0;
    for (int k = 0; k < K; k++) {
        sum += *a * *b;  // C[i, j] += A[i, k] * B[k, j]
        a += M;          // A[i, k] -> A[i, k + 1]
        b++;             // B[k, j] -> B[k + 1, j]
    }
    *c = sum;
}

class KernelSettings {
   public:
    KernelSettings() { setMaxSharedMemory(__sgemm); }

   private:
    static KernelSettings _settings;
};

KernelSettings KernelSettings::_settings = KernelSettings();

void sgemm(const int M,
           const int N,
           const int K,
           const float* const A,
           const float* const B,
           float* const C,
           const int V,
           const int W) {
    dim3 gridDim(M / V + (M % V != 0),  // gridDim.x = CEIL_DIV(M, V)
                 N / W + (N % W != 0),  // gridDim.y = CEIL_DIV(N, W)
                 1);                    // gridDim.z = 1
    dim3 blockDim(V, W, 1);
    __sgemm<<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaDeviceSynchronize();
    cudaCheck(__FILE__, __LINE__);
}
