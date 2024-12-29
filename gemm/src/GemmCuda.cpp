#include <cuda.h>

#include "Gemm.h"
#include "kernels.cuh"

void GemmCuda::__run(const uint W) {
    sgemm(this->_A->m, this->_B->n, this->_A->n, this->_A->getDevPtr(),
          this->_B->getDevPtr(), this->_C->getDevPtr(), W);
}
