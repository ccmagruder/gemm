#pragma once

#include <cuda_runtime.h>

void cudaCheck(const char* file, const int line);
void cudaCheck(cudaError_t code, const char* file, const int line);

template <typename... Args>
void setMaxSharedMemory(void (*kernel)(Args... args)) {
    int device;
    int sharedMemoryPerBlockOptin;

    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sharedMemoryPerBlockOptin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sharedMemoryPerBlockOptin);
}

void sgemm(const int M,
           const int N,
           const int K,
           const float* const A,
           const float* const B,
           float* const C,
           const int V,
           const int W);
