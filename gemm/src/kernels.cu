#include <stdexcept>

#include "kernels.cuh"
#include "utils.cuh"

__global__ void __sgemm_naive(const int M,
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

void sgemm_naive(const int M,
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
    __sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C);
    cudaDeviceSynchronize();
    cudaCheck(__FILE__, __LINE__);
}

__global__ void __sgemm(const int M,
                        const int N,
                        const int K,
                        const float* const A,
                        const float* const B,
                        float* const C,
                        const int tileDim) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    float* const c = C + i + j * M;  // c = C[i, j]

    extern __shared__ float smem[];
    float* sA = smem;                         // (blockDim.x, tileDim)
    float* sB = smem + blockDim.x * tileDim;  // (tileDim, blockDim.y)

    int gIdx;        // global memory index
    int sIdx;        // shard memory index
    int tIdx;        // tile index
    float lC = 0.0;  // local memory partial summation

    for (tIdx = 0; tIdx < K / tileDim + (K % tileDim != 0); tIdx++) {
        for (sIdx = tid; sIdx < blockDim.x * tileDim; sIdx += block_size) {
            gIdx = (tIdx * tileDim + sIdx / blockDim.x) * M +
                   blockIdx.x * blockDim.x + sIdx % blockDim.x;
            if (gIdx < M * K)
                sA[sIdx] = A[gIdx];
        }

        for (sIdx = tid; sIdx < tileDim * blockDim.y; sIdx += block_size) {
            gIdx = (blockIdx.y * blockDim.y + sIdx / tileDim) * K +
                   tIdx * tileDim + sIdx % tileDim;
            if (gIdx < K * N)
                sB[sIdx] = B[gIdx];
        }

        __syncthreads();

        if (i < M && j < N) {
            for (sIdx = 0; sIdx < tileDim && tIdx * tileDim + sIdx < K;
                 sIdx++) {
                lC += sA[sIdx * blockDim.x + threadIdx.x] *
                      sB[threadIdx.y * tileDim + sIdx];
            }
        }

        __syncthreads();
    }

    if (i < M && j < N) {
        *c = lC;
    }
}

void sgemm(const int M,
           const int N,
           const int K,
           const float* const A,
           const float* const B,
           float* const C,
           const int V,
           const int W,
           int tile_dim) {
    // The maximum memory for the RTX 4070 (Compute Capability 8.9) is 99KB
    const size_t smem_size_max = 99 * 1024;

    dim3 gridDim(M / V + (M % V != 0),  // gridDim.x = CEIL_DIV(M, V)
                 N / W + (N % W != 0),  // gridDim.y = CEIL_DIV(N, W)
                 1);                    // gridDim.z = 1
    dim3 blockDim(V, W, 1);

    // sizeof(float)*(blockDim.x*tile_dim+tile_dim*blockDim.y)<=smem_size_max
    // tile_dim <= smem_size_max / sizeof(float) / (blockDim.x + blockDim.y)
    if (!tile_dim)
        tile_dim = static_cast<int>(static_cast<float>(smem_size_max) /
                                    sizeof(float) / (blockDim.x + blockDim.y));

    // kernel __gemm below assumes tha tile_dim <= K, will segfault otherwise;
    // there are no performance benefits to tile_time > K regardless
    tile_dim = std::min(tile_dim, K);

    const size_t smem_size = (V + W) * tile_dim * sizeof(float);
    if (smem_size > smem_size_max) {
        throw std::runtime_error("smem_size > smem_size_max");
    }

    // The maximum memory for the RTX 4070 (Compute Capability 8.9) is 99KB;
    // however, the default cap is 48KB for hardware compatibility. To
    // override:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x
    __sgemm<<<gridDim, blockDim, smem_size>>>(M, N, K, A, B, C, tile_dim);

    cudaDeviceSynchronize();
    cudaCheck(__FILE__, __LINE__);
}

class KernelSettings {
   public:
    KernelSettings() { setMaxSharedMemory(__sgemm); }

   private:
    static KernelSettings _settings;
};

KernelSettings KernelSettings::_settings = KernelSettings();
