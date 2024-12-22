#include "cublas_v2.h"

#include "Gemm.h"

///////////////////// CuBLAS Handle Singleton /////////////////

class CuBlasHandle {
    CuBlasHandle() {
        cublasStatus_t stat = cublasCreate(&this->_handle);
        assert(stat==CUBLAS_STATUS_SUCCESS);
    }

 public:
    ~CuBlasHandle() { cublasDestroy(this->_handle); }

    static cublasHandle_t get() {
        return CuBlasHandle::instance->_handle;
    }

 private:
    cublasHandle_t _handle;
    static std::unique_ptr<CuBlasHandle> instance;
};

std::unique_ptr<CuBlasHandle> CuBlasHandle::instance(new CuBlasHandle);


//////////////////////// GemmCuBlas ///////////////////////////

template<>
void Gemm<CuBlas>::_setup() {
    this->_A->toDevice();
    this->_B->toDevice();
    this->_C = Matrix::makeDevice(this->_A->m, this->_B->n);
}

template<>
void Gemm<CuBlas>::_run() {
    float alpha=1;
    float beta=0;
    cublasStatus_t stat = cublasSgemm(
        CuBlasHandle::get(),   // handle
        CUBLAS_OP_N,           // transa
        CUBLAS_OP_N,           // transb
        this->_C->m,           // m
        this->_C->n,           // n
        this->_A->n,           // k
        &alpha,                // alpha
        this->_A->getDevPtr(), // A
        this->_A->m,           // lda
        this->_B->getDevPtr(), // B
        this->_B->m,           // ldb
        &beta,                 // beta
        this->_C->getDevPtr(), // C
        this->_C->m);          // ldc

    assert(stat==CUBLAS_STATUS_SUCCESS);
}

template<>
void Gemm<CuBlas>::_teardown() {
    this->_C->toHost();
}
