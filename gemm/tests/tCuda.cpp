#include "gtest/gtest.h"

#include <cuda_runtime.h>

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

TEST(tCuda, SharedMemoryPerBlock) {
    int sharedMemoryPerBlock = 0;
    cudaError_t code;

    // Max Shared Memory Per Block: 48 KB
    code = cudaDeviceGetAttribute(&sharedMemoryPerBlock,
                                  cudaDevAttrMaxSharedMemoryPerBlock, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(sharedMemoryPerBlock, 48 * 1024);

    // Max Shared Memory Per Block (Optin): 99 KB
    // RTX 4070 has twice the available shared memory if optin enabled per kernel.
    // The default max of 48 KB is widely compatible with NVIDIA GPUs. The optin
    // maximums are device-specific.
    code = cudaDeviceGetAttribute(&sharedMemoryPerBlock,
                                  cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(sharedMemoryPerBlock, 99 * 1024);

    // Reserved Shared Memory Per Block: 1 KB
    // Reserved memory for the SM (e.g. blockDim.x, etc.).
    // Avilable shared memory for kernel usage is 47 KB or 98 KB, respectively.
    code = cudaDeviceGetAttribute(&sharedMemoryPerBlock,
                                  cudaDevAttrReservedSharedMemoryPerBlock, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(sharedMemoryPerBlock, 1 * 1024);
}

TEST(tCuda, ComputeCapability) {
    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;

    // For RTX 4070, Compute Capability == 8.9
    cudaError_t code = cudaDeviceGetAttribute(
        &computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(computeCapabilityMajor, 8);

    code = cudaDeviceGetAttribute(&computeCapabilityMinor,
                                  cudaDevAttrComputeCapabilityMinor, 0);
    EXPECT_EQ(code, cudaSuccess);
    EXPECT_EQ(computeCapabilityMinor, 9);
}
