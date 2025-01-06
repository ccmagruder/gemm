#include "Matrix.h"
#include <cuda_runtime.h>  // need to use angle brackets so doesn't get marked as 'unused-include'
#include <cassert>
#include <cstddef>
#include <memory>
#include <random>
#include "kernels.cuh"

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
    for (ptrdiff_t i = 0; i < m * n; i++) {
        ptr[i] = dist(gen);
    }
    return A;
}

std::unique_ptr<Matrix> Matrix::fill(size_t m, size_t n, float value) {
    std::unique_ptr<Matrix> A(new Matrix(m, n));
    float* ptr = A->getHostPtr();
    for (ptrdiff_t i = 0; i < m * n; i++) {
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
    if (!this->_dev_ptr) {
        float* ptr;
        cudaError_t code =
            cudaMalloc((void**)&ptr, this->m * this->n * sizeof(float));
        cudaCheck(code, __FILE__, __LINE__);
        this->_dev_ptr = std::unique_ptr<float, DevDeleter>(ptr);
    }
}

void Matrix::toDevice() const {
    this->_devAlloc();
    cudaError_t code =
        cudaMemcpy(this->_dev_ptr.get(), this->_host_ptr.get(),
                   this->m * this->n * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheck(code, __FILE__, __LINE__);
}

void Matrix::toHost() {
    cudaError_t code =
        cudaMemcpy(this->_host_ptr.get(), this->_dev_ptr.get(),
                   this->m * this->n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck(code, __FILE__, __LINE__);
}

float Matrix::lInfNorm(const Matrix& ref) const {
    assert(this->m == ref.m);
    assert(this->n == ref.n);

    const float* ptr = this->getHostPtr();
    float error = 0.0;
    for (ptrdiff_t i = 0; i < this->m * this->n; i++) {
        error = std::max(error, std::abs(ptr[i] - ref.getHostPtr()[i]));
    }
    return error;
}

std::unique_ptr<const Matrix> Matrix::copy() const {
    std::unique_ptr<Matrix> A = Matrix::fill(this->m, this->n, 0.0);
    for (ptrdiff_t i = 0; i < this->m * this->n; i++) {
        A->getHostPtr()[i] = this->getHostPtr()[i];
    }
    return A;
}
