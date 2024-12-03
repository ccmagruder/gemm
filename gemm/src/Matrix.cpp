#include <memory>
#include <random>

#include "Matrix.h"

std::unique_ptr<Matrix> Matrix::iid(size_t m, size_t n) {
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
