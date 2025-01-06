#include "gtest/gtest.h"

#include <cuda_runtime.h>

void sharedMemoryAlloc();

TEST(tCuda, Device) {
    int count = 0, device = -1;
    cudaError_t code;

    code = cudaGetDeviceCount(&count);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(count, 1);

    code = cudaGetDevice(&device);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(device, 0);
}

TEST(tCuda, ComputeCapability) {
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    cudaError_t code;

    // For RTX 4070, Compute Capability == 8.9
    code = cudaDeviceGetAttribute(&computeCapabilityMajor,
                                  cudaDevAttrComputeCapabilityMajor, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(computeCapabilityMajor, 8);

    code = cudaDeviceGetAttribute(&computeCapabilityMinor,
                                  cudaDevAttrComputeCapabilityMinor, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(computeCapabilityMinor, 9);
}

TEST(tCuda, SharedMemoryPerBlock) {
    int sharedMemory = 0;
    cudaError_t code;

    // Max Shared Memory Per Block: 48 KB
    code = cudaDeviceGetAttribute(&sharedMemory,
                                  cudaDevAttrMaxSharedMemoryPerBlock, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(sharedMemory, 48 * 1024);

    // Max Shared Memory Per Block (Optin): 99 KB
    // RTX 4070 has twice the available shared memory if optin enabled per
    // kernel. The default max of 48 KB is widely compatible with NVIDIA GPUs.
    // The optin maximums are device-specific.
    code = cudaDeviceGetAttribute(&sharedMemory,
                                  cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(sharedMemory, 99 * 1024);

    // Reserved Shared Memory Per Block: 1 KB
    // Reserved memory for the SM (e.g. blockDim.x, etc.).
    // Avilable shared memory for kernels remains 48 KB and 99 KB, respectively.
    code = cudaDeviceGetAttribute(&sharedMemory,
                                  cudaDevAttrReservedSharedMemoryPerBlock, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(sharedMemory, 1 * 1024);

    // Max Shared Memory Per Multiprocessor: 100 KB
    code = cudaDeviceGetAttribute(
        &sharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(sharedMemory, 100 * 1024);
}

TEST(tCuda, SharedMemoryOptIn) {
    sharedMemoryAlloc();
}
