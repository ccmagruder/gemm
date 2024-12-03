#pragma once

#include <memory>

class Matrix {
 public:
    Matrix(size_t m, size_t n) : _m(m), _n(n), _data(new float[m*n]) {}

    Matrix(size_t m, size_t n, float value) : Matrix(m, n) {
        for (size_t i=0; i<m*n; i++) {
            this->_data[i] = value;
        }
    }

    float* operator[](size_t i) { return this->_data.get() + this->_n*i; }

    float* get() const { return this->_data.get(); }

    static std::unique_ptr<Matrix> iid(size_t m, size_t n);

 protected:
    size_t _m;
    size_t _n;
    std::unique_ptr<float[], std::default_delete<float[]>> _data;
};

