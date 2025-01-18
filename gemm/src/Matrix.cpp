#include "Matrix.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <random>

Matrix::Matrix(size_t m, size_t n)
    : m(m), n(n), _host_ptr(new float[m * n]), _dev_ptr(nullptr) {}

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
