# pragma once

#include "Matrix.h"

class MatrixMult {
 public:
    MatrixMult(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B)
      : _A(std::move(A)), _B(std::move(B)) {}

    std::shared_ptr<Matrix> compute();

    void _setup();
    void _run();
    void _teardown();

    std::shared_ptr<Matrix> get() { return this->_C; }

 protected:
    std::unique_ptr<const Matrix> _A;
    std::unique_ptr<const Matrix> _B;
    std::shared_ptr<Matrix> _C;
};
