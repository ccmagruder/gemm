# pragma once

#include <cassert>

#include "Matrix.h"

enum Algo { Naive, CuBlas, Mkl };

template<Algo T>
class Gemm {
 public:
    Gemm(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B)
      : _A(std::move(A)), _B(std::move(B)) {}

    std::shared_ptr<Matrix> compute() {
        this->_setup();
        this->_run();
        this->_teardown();
        return this->get();
    }

    void _setup();
    void _run();
    void _teardown();

    std::shared_ptr<Matrix> get() { return this->_C; }

 protected:
    std::unique_ptr<const Matrix> _A;
    std::unique_ptr<const Matrix> _B;
    std::shared_ptr<Matrix> _C;
};

