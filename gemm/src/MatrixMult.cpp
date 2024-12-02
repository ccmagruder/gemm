#include "MatrixMult.h"

std::unique_ptr<const Matrix> MatrixMult::compute() const {
    std::unique_ptr<Matrix> C
        = std::make_unique<Matrix>(this->_A->m(), this->_B->n(), 0.0);
    const Matrix& A = *this->_A;
    float* ptr = C->get();
    for (size_t i=0; i<C->m(); i++) {
        for (size_t j=0; j<C->n(); j++) {
            for (size_t k=0; k<this->_A->n(); k++) {
                (*C)[i][j] += (*this->_A)[i][k] * (*this->_B)[k][j];
            }
        }
    }

    return C;
}
