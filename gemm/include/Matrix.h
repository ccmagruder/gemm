#pragma once

#include <memory>

class Matrix {
 public:
    Matrix(size_t m, size_t n)
            : m(m), n(n), _host_ptr(new float[m*n]), _dev_ptr(nullptr) {}

    // Factories
    static std::unique_ptr<Matrix> makeDevice(size_t m, size_t n);
    static std::unique_ptr<Matrix> makeHost(size_t m, size_t n);

    static std::unique_ptr<const Matrix> normalIID(size_t m, size_t n);
    static std::unique_ptr<const Matrix> fill(size_t m, size_t n, float value);

    // Accessors
    // TODO: Convert to column-major indexing
    const float* const operator[](size_t i) const {
        return this->_host_ptr.get() + this->n*i;
    }
    // TODO: Convert to column-major indexing
    float* operator[](size_t i) { return this->_host_ptr.get() + this->n*i; }

    const float* const getHostPtr() const { return this->_host_ptr.get(); }
    float* getHostPtr() { return this->_host_ptr.get(); }

    const float* const getDevPtr() const { return this->_dev_ptr.get(); }
    float* getDevPtr() { return this->_dev_ptr.get(); }

    void toDevice() const;
    void toHost();

 public:
    const size_t m;
    const size_t n;
 protected:
    void _devAlloc() const;

    std::unique_ptr<float[], std::default_delete<float[]>> _host_ptr;
    struct DevDeleter {
        void operator()(float* devPtr);
    };
    mutable std::unique_ptr<float, DevDeleter> _dev_ptr;
};

