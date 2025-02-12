#include "cublas_v2.h"

#include "Gemm.h"

///////////////////// CuBLAS Handle Singleton /////////////////

class CuBlasHandle {
    CuBlasHandle() {
        cublasStatus_t stat = cublasCreate(&this->_handle);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("CuBlasHandle::CuBlasHandle()");
    }

   public:
    ~CuBlasHandle() { cublasDestroy(this->_handle); }

    static cublasHandle_t get() { return CuBlasHandle::instance->_handle; }

   private:
    cublasHandle_t _handle;
    static std::unique_ptr<CuBlasHandle> instance;
};

std::unique_ptr<CuBlasHandle> CuBlasHandle::instance(new CuBlasHandle);

//////////////////////// GemmCuBlas ///////////////////////////

void GemmCuBlas::_run() {
    float alpha = 1;
    float beta = 0;
    cublasStatus_t stat = cublasSgemm(CuBlasHandle::get(),    // handle
                                      CUBLAS_OP_N,            // transa
                                      CUBLAS_OP_N,            // transb
                                      this->_C->m,            // m
                                      this->_C->n,            // n
                                      this->_A->n,            // k
                                      &alpha,                 // alpha
                                      this->_A->getDevPtr(),  // A
                                      this->_A->m,            // lda
                                      this->_B->getDevPtr(),  // B
                                      this->_B->m,            // ldb
                                      &beta,                  // beta
                                      this->_C->getDevPtr(),  // C
                                      this->_C->m);           // ldc

    cudaDeviceSynchronize();
    if (stat != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("GemmCuBlas::_run()");
}
