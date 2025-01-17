#include "Matrix.h"
#include "utils.cuh"

void Matrix::DevDeleter::operator()(float* devPtr) {
    if (devPtr) {
        cudaFree(devPtr);
        devPtr = nullptr;
    }
}

void Matrix::_devAlloc() const {
    if (!this->_dev_ptr) {
        float* ptr;
        cudaError_t code =
            cudaMalloc((void**)&ptr, this->m * this->n * sizeof(float));
        cudaCheck(code, __FILE__, __LINE__);
        this->_dev_ptr = std::unique_ptr<float, DevDeleter>(ptr);
    }
}

void Matrix::toDevice() const {
    this->_devAlloc();
    cudaError_t code =
        cudaMemcpy(this->_dev_ptr.get(), this->_host_ptr.get(),
                   this->m * this->n * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheck(code, __FILE__, __LINE__);
}

void Matrix::toHost() {
    cudaError_t code =
        cudaMemcpy(this->_host_ptr.get(), this->_dev_ptr.get(),
                   this->m * this->n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck(code, __FILE__, __LINE__);
}
