#include <cuda.h>

#include "Gemm.h"
#include "kernels.cuh"

template<>
void Gemm<Cuda>::_setup() {
    this->_A->toDevice();
    this->_B->toDevice();
    this->_C = Matrix::makeDevice(this->_A->m, this->_B->n);
}

template<>
void Gemm<Cuda>::_run() {
    sgemm(this->_A->m, this->_B->n, this->_A->n, this->_A->getDevPtr(), this->_B->getDevPtr(), this->_C->getDevPtr());
}

template<>
void Gemm<Cuda>::_teardown() {
    this->_C->toHost();
}

