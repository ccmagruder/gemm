# pragma once

#include <cassert>

#include "cublas_v2.h"

#include "Matrix.h"

class Gemm {
 public:
    Gemm(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B)
      : _A(std::move(A)), _B(std::move(B)) {}

    std::shared_ptr<Matrix> compute();

    virtual void _setup();
    virtual void _run();
    virtual void _teardown();

    std::shared_ptr<Matrix> get() { return this->_C; }

 protected:
    std::unique_ptr<const Matrix> _A;
    std::unique_ptr<const Matrix> _B;
    std::shared_ptr<Matrix> _C;
};

class GemmCuBlas : public Gemm {
 public:
    GemmCuBlas(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B);
    ~GemmCuBlas();

    void _setup() override;
    void _run() override;
    void _teardown() override;

 protected:
    cublasHandle_t handle;
};
