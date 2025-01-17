#include "utils.cuh"

__global__ void __passthrough() {
    extern float* smem[];
}

void sharedMemoryAlloc(int size, bool max_smem) {
    float* ptr;
    cudaError_t code = cudaMalloc((void**)&ptr, sizeof(float));
    cudaCheck(code, __FILE__, __LINE__);
    if (max_smem)
        setMaxSharedMemory(__passthrough);
    __passthrough<<<1, 1, size>>>();
    cudaCheck(__FILE__, __LINE__);
    cudaFree(ptr);
}
