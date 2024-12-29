#include <cstddef>
#include "Gemm.h"

void GemmNaive::_run() {
  for (size_t i = 0; i < this->_C->m; i++) {
    for (size_t j = 0; j < this->_C->n; j++) {
      (*this->_C)(i, j) = 0;
      for (size_t k = 0; k < this->_A->n; k++) {
        (*this->_C)(i, j) += (*this->_A)(i, k) * (*this->_B)(k, j);
      }
    }
  }
}
