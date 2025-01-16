#include "kernels.cuh"
#include "utils.cuh"

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

    int si;
    int sj;
    float sum = 0.0;

    for (int tileIdx = 0; tileIdx < K / tileDim + (K % tileDim != 0);
         tileIdx++) {
        for (int idx = tid; idx < blockDim.x * tileDim; idx += block_size) {
            si = idx % blockDim.x;
            sj = idx / blockDim.x;
            if ((tileIdx * tileDim + sj) * M + blockIdx.x * blockDim.x + si <
                M * K)
                sA[idx] = A[(tileIdx * tileDim + sj) * M +
                            blockIdx.x * blockDim.x + si];
        }

        for (int idx = tid; idx < tileDim * blockDim.y; idx += block_size) {
            si = idx % tileDim;
            sj = idx / tileDim;
            if ((blockIdx.y * blockDim.y + sj) * K + tileIdx * tileDim + si <
                K * N) {
                sB[idx] = B[(blockIdx.y * blockDim.y + sj) * K +
                            tileIdx * tileDim + si];
            }
        }

        __syncthreads();

        if (i < M && j < N) {
            for (int k = 0; k < tileDim && tileIdx * tileDim + k < K; k++) {
                sum += sA[k * blockDim.x + threadIdx.x] *
                       sB[threadIdx.y * tileDim + k];
            }
        }

        __syncthreads();
    }

    if (i < M && j < N) {
        *c = sum;
    }
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
