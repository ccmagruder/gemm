#include <memory>
#include <random>

#include "cuda_runtime.h"

#include "Matrix.h"

void Matrix::DevDeleter::operator()(float* devPtr) {
    if (devPtr) {
        cudaFree(devPtr);
        devPtr = nullptr;
    }
};

void Matrix::_devAlloc() const {
    cudaError_t cudaStat;
    if (!this->_dev_ptr) {
        float* ptr;
        cudaStat = cudaMalloc ((void**)&ptr, this->_m * this->_n * sizeof(float));
        this->_dev_ptr = std::unique_ptr<float, DevDeleter>(ptr);
    }
    assert(cudaStat == cudaSuccess);
}

void Matrix::toDevice() const {
    this->_devAlloc();
    cudaError_t cudaStat;
    cudaStat = cudaMemcpy(this->_dev_ptr.get(),
        this->_host_ptr.get(),
        this->_m * this->_n * sizeof(float),
        cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
}

void Matrix::toHost() {
    cudaError_t cudaStat;
    cudaStat = cudaMemcpy(this->_host_ptr.get(), this->_dev_ptr.get(),
        sizeof(float), cudaMemcpyDeviceToHost);
    assert(cudaStat == cudaSuccess);
}

std::unique_ptr<const Matrix> Matrix::normalIID(size_t m, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist;

    std::unique_ptr<Matrix> A = std::make_unique<Matrix>(m, n);
    float* ptr = A->get();
    for (size_t i=0; i<m*n; i++) {
        ptr[i] = dist(gen);
    }
    return A;
}
