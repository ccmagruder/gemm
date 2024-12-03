#pragma once

#include <memory>
#include <stdint.h>
#include <vector>

class Matrix {
 public:
    Matrix(size_t m, size_t n) : _m(m), _n(n) {
        this->_data = std::unique_ptr<float>(new float[m*n]);
    }

    float* operator[](size_t i) { return this->_data.get() + this->_n*i; }

 private:
    size_t _m;
    size_t _n;
    std::unique_ptr<float> _data;
};

