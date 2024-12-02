# pragma once

#include <cassert>

#include "Matrix.h"

#include "cublas_v2.h"

class MatrixMult {
 public:
    MatrixMult(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B)
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

class MatrixMultCuBLAS : public MatrixMult {
 public:
    MatrixMultCuBLAS(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B)
      : MatrixMult(std::move(A), std::move(B)) {}

    void _setup() override {
        this->_A->toDevice();
        this->_B->toDevice();
        this->_C = Matrix::makeDevice(this->_A->m(), this->_B->n());
    }

    void _run() override {
        cublasHandle_t handle;
        cublasStatus_t stat; 
        stat = cublasCreate(&handle);
        assert(stat==CUBLAS_STATUS_SUCCESS);

        float alpha=1;
        float beta=0;
        stat = cublasSgemm(handle,         // handle
            CUBLAS_OP_N,           // transa
            CUBLAS_OP_N,           // transb
            this->_C->m(),         // m
            this->_C->n(),         // n
            this->_A->n(),         // k
            &alpha,                // alpha
            this->_A->getDevPtr(), // A
            this->_A->m(),         // lda
            this->_B->getDevPtr(), // B
            this->_B->m(),         // ldb
            &beta,                 // beta
            this->_C->getDevPtr(), // C
            this->_C->m());        // ldc

        assert(stat==CUBLAS_STATUS_SUCCESS);
        cublasDestroy(handle);
    }

    void _teardown() override {
        this->_C->toHost();
    }
};
