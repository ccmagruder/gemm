#pragma once

#include <memory>

class Matrix {
 public:
    Matrix(size_t m, size_t n)
            : _m(m), _n(n), _host_ptr(new float[m*n]), _dev_ptr(nullptr) {}

    // Factories
    static std::unique_ptr<Matrix> makeDevice(size_t m, size_t n);
    static std::unique_ptr<Matrix> makeHost(size_t m, size_t n);

    static std::unique_ptr<const Matrix> normalIID(size_t m, size_t n);
    static std::unique_ptr<const Matrix> fill(size_t m, size_t n, float value);

    const float* const operator[](size_t i) const {
        return this->_host_ptr.get() + this->_n*i;
    }
    float* operator[](size_t i) { return this->_host_ptr.get() + this->_n*i; }

    const float* const get() const { return this->_host_ptr.get(); }
    float* get() { return this->_host_ptr.get(); }

    const float* const getDevPtr() const { return this->_dev_ptr.get(); }
    float* getDevPtr() { return this->_dev_ptr.get(); }

    size_t m() const { return this->_m; }
    size_t n() const { return this->_n; }

    void toDevice() const;
    void toHost();

 protected:
    struct DevDeleter {
        void operator()(float* devPtr);
    };

    void _devAlloc() const;

    size_t _m;
    size_t _n;
    std::unique_ptr<float[], std::default_delete<float[]>> _host_ptr;
    mutable std::unique_ptr<float, DevDeleter> _dev_ptr;
};

