#include "Gemm.h"

template <>
void Gemm<Host>::_setup() {
    this->_C = Matrix::makeHost(this->_A->m, this->_B->n);
}

template <>
void Gemm<Host>::_teardown() {}

template <>
void Gemm<Device>::_setup() {
    this->_A->toDevice();
    this->_B->toDevice();
    this->_C = Matrix::makeDevice(this->_A->m, this->_B->n);
}

template <>
void Gemm<Device>::_teardown() {
    this->_C->toHost();
}
