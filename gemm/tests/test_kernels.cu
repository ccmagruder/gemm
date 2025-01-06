#include "kernels.cuh"

__global__ void __passthrough() {
    extern float* smem[];
}

// TODO: move to kernels.cu
void setMaxSharedMemory(void (*kernel)(void)) {
    int device;
    int sharedMemoryPerBlockOptin;
    // TODO: Refactor checkCuda(code, FILE, LINE)
    // cudaError_t code;

    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sharedMemoryPerBlockOptin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sharedMemoryPerBlockOptin);
}

void sharedMemoryAlloc() {
    float* ptr;
    cudaError_t code = cudaMalloc((void**)&ptr, sizeof(float));
    setMaxSharedMemory(__passthrough);
    __passthrough<<<1, 1, 99 * 1024>>>();
    cudaCheck(__FILE__, __LINE__);
    cudaFree(ptr);
}
