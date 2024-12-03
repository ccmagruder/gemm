# pragma once

#include "Matrix.h"

class MatrixMult {
 public:
    MatrixMult(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B)
      : _A(std::move(A)), _B(std::move(B)) {}

    std::unique_ptr<const Matrix> compute () const;

 protected:
    std::unique_ptr<const Matrix> _A;
    std::unique_ptr<const Matrix> _B;
};
