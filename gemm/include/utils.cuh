#include <cuda_runtime.h>

template <typename... Args>
void setMaxSharedMemory(void (*kernel)(Args... args)) {
    int device;
    int sharedMemoryPerBlockOptin;

    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sharedMemoryPerBlockOptin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sharedMemoryPerBlockOptin);
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
}

void cudaCheck(const char* file, const int line);

void cudaCheck(cudaError_t code, const char* file, const int line);
