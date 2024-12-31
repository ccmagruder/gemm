#include <mkl/mkl.h>

#include "Gemm.h"

void GemmMkl::_run() {
    cblas_sgemm(CblasColMajor,           // Layout
                CblasNoTrans,            // transa
                CblasNoTrans,            // transb
                this->_C->m,             // m
                this->_C->n,             // n
                this->_A->n,             // k
                1,                       // alpha
                this->_A->getHostPtr(),  // a
                this->_A->m,             // lda
                this->_B->getHostPtr(),  // b
                this->_B->m,             // ldb
                0,                       // beta
                this->_C->getHostPtr(),  // c
                this->_C->m);            // ldc
}
