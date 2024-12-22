# pragma once

#include <cassert>

#include "Matrix.h"

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

    virtual void _setup() = 0;
    virtual void _run() = 0;
    virtual void _teardown() = 0;

    std::shared_ptr<Matrix> get() { return this->_C; }

 protected:
    std::unique_ptr<const Matrix> _A;
    std::unique_ptr<const Matrix> _B;
    std::shared_ptr<Matrix> _C;
};

class GemmNaive : public Gemm {
 public:
    using Gemm::Gemm;

    void _setup() override;
    void _run() override;
    void _teardown() override;
};

class GemmCuBlas : public Gemm {
 public:
    using Gemm::Gemm;

    void _setup() override;
    void _run() override;
    void _teardown() override;
};

class GemmMKL : public Gemm {
 public:
    using Gemm::Gemm;

    void _setup() override;
    void _run() override;
    void _teardown() override;
};
