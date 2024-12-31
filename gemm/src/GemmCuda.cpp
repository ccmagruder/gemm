#include <cuda.h>

#include "Gemm.h"
#include "kernels.cuh"

void GemmCuda::_run() {
    sgemm(this->_A->m, this->_B->n, this->_A->n, this->_A->getDevPtr(),
          this->_B->getDevPtr(), this->_C->getDevPtr());
}
