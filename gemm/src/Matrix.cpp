#include <cassert>
#include <memory>
#include <random>

#include "cuda_runtime.h"

#include "Matrix.h"

std::unique_ptr<Matrix> Matrix::makeDevice(size_t m, size_t n) {
    std::unique_ptr<Matrix> A(new Matrix(m, n));
    A->_devAlloc();
    return A;
}

std::unique_ptr<Matrix> Matrix::makeHost(size_t m, size_t n) {
    return std::unique_ptr<Matrix>(new Matrix(m, n));
}

std::unique_ptr<const Matrix> Matrix::normalIID(size_t m, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist;

    std::unique_ptr<Matrix> A(new Matrix(m, n));
    float* ptr = A->getHostPtr();
    for (ptrdiff_t i=0; i<m*n; i++) {
        ptr[i] = dist(gen);
    }
    return A;
}

std::unique_ptr<Matrix> Matrix::fill(size_t m, size_t n, float value) {
    std::unique_ptr<Matrix> A(new Matrix(m, n));
    float* ptr = A->getHostPtr();
    for (ptrdiff_t i=0; i<m*n; i++) {
        ptr[i] = value;
    }
    
    return A;
}

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
        cudaStat = cudaMalloc ((void**)&ptr, this->m * this->n * sizeof(float));
        assert(cudaStat == cudaSuccess);
        this->_dev_ptr = std::unique_ptr<float, DevDeleter>(ptr);
    }
}

void Matrix::toDevice() const {
    this->_devAlloc();
    cudaError_t cudaStat;
    cudaStat = cudaMemcpy(this->_dev_ptr.get(),
        this->_host_ptr.get(),
        this->m * this->n * sizeof(float),
        cudaMemcpyHostToDevice);
    assert(cudaStat == cudaSuccess);
}

void Matrix::toHost() {
    cudaError_t cudaStat;
    cudaStat = cudaMemcpy(this->_host_ptr.get(),
        this->_dev_ptr.get(),
        this->m * this->n * sizeof(float),
        cudaMemcpyDeviceToHost);
    assert(cudaStat == cudaSuccess);
}

