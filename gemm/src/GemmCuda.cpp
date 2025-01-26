#include <cuda.h>

#include "Gemm.h"
#include "kernels.cuh"

void GemmCuda::__run_naive(const uint V, const uint W) {
    sgemm_naive(this->_A->m, this->_B->n, this->_A->n, this->_A->getDevPtr(),
                this->_B->getDevPtr(), this->_C->getDevPtr(), V, W);
}

void GemmCuda::__run(const uint V, const uint W) {
    sgemm(this->_A->m, this->_B->n, this->_A->n, this->_A->getDevPtr(),
          this->_B->getDevPtr(), this->_C->getDevPtr(), V, W);
}
