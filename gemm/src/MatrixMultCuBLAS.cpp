#include "cublas_v2.h"

#include "MatrixMult.h"

MatrixMultCuBLAS::MatrixMultCuBLAS(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B)
        : MatrixMult(std::move(A), std::move(B)) {
    cublasStatus_t stat = cublasCreate(&handle);
    assert(stat==CUBLAS_STATUS_SUCCESS);
}

MatrixMultCuBLAS::~MatrixMultCuBLAS() {
    cublasDestroy(handle);
}

void MatrixMultCuBLAS::_setup() {
    this->_A->toDevice();
    this->_B->toDevice();
    this->_C = Matrix::makeDevice(this->_A->m(), this->_B->n());
}

void MatrixMultCuBLAS::_run() {
    float alpha=1;
    float beta=0;
    cublasStatus_t stat = cublasSgemm(handle,         // handle
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
}

void MatrixMultCuBLAS::_teardown() {
    this->_C->toHost();
}
