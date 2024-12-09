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

    float* devPtrA;
    float* devPtrB;
    float* devPtrC;

    void _setup() override {
        this->_C = std::make_shared<Matrix>(this->_A->m(), this->_B->n(), 0.0);

        cudaError_t cudaStat;

        cudaStat = cudaMalloc ((void**)&devPtrA,
            this->_A->m() * this->_A->n() * sizeof(float));
        assert(cudaStat == cudaSuccess);
        cudaStat = cudaMalloc ((void**)&devPtrB,
            this->_B->m() * this->_B->n() * sizeof(float));
        assert(cudaStat == cudaSuccess);
        cudaStat = cudaMalloc (
            (void**)&devPtrC,
            this->_C->m() * this->_C->n() * sizeof(float)
        );
        assert(cudaStat == cudaSuccess);


        cudaStat = cudaMemcpy(this->devPtrA,
            this->_A->get(),
            this->_A->m() * this->_A->n() * sizeof(float),
            cudaMemcpyHostToDevice);
        assert(cudaStat == cudaSuccess);
        cudaStat = cudaMemcpy(this->devPtrB,
            this->_B->get(),
            this->_B->m() * this->_B->n() * sizeof(float),
            cudaMemcpyHostToDevice);
        assert(cudaStat == cudaSuccess);
    }

    void _run() override {
        cublasHandle_t handle;
        cublasStatus_t stat; 
        stat = cublasCreate(&handle);
        assert(stat==CUBLAS_STATUS_SUCCESS);

        float alpha=1;
        float beta=0;
        stat = cublasSgemm(handle,         // handle
                                         CUBLAS_OP_N,     // transa
                                         CUBLAS_OP_N,     // transb
                                         this->_C->m(),   // m
                                         this->_C->n(),   // n
                                         this->_A->n(),   // k
                                         &alpha,          // alpha
                                         devPtrA, // A
                                         this->_A->m(),   // lda
                                         devPtrB, // B
                                         this->_B->m(),   // ldb
                                         &beta,           // beta
                                         devPtrC, // C
                                         this->_C->m());  // ldc

        assert(stat==CUBLAS_STATUS_SUCCESS);
        cublasDestroy(handle);
    }

    void _teardown() override {
        cudaError_t cudaStat;
        cudaStat = cudaMemcpy(this->_C->get(), this->devPtrC,
            sizeof(float), cudaMemcpyDeviceToHost);
        assert(cudaStat == cudaSuccess);
        cudaFree (this->devPtrA);
        cudaFree (this->devPtrB);
        cudaFree (this->devPtrC);
    }
};
