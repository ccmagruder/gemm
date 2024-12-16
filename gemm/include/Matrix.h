#pragma once

#include <memory>

class Matrix {
    Matrix(size_t m, size_t n)
            : m(m), n(n), _host_ptr(new float[m*n]), _dev_ptr(nullptr) {}

 public:
    Matrix() = delete;
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;

    // Factories
    static std::unique_ptr<Matrix> makeDevice(size_t m, size_t n);
    static std::unique_ptr<Matrix> makeHost(size_t m, size_t n);

    static std::unique_ptr<const Matrix> normalIID(size_t m, size_t n);
    static std::unique_ptr<const Matrix> fill(size_t m, size_t n, float value);

    // Accessors (Column Major)
    const float& operator()(size_t i, size_t j) const {
        return (this->_host_ptr.get() + this->m*j)[i];
    }
    float& operator()(size_t i, size_t j) {
        return (this->_host_ptr.get() + this->m*j)[i];
    }

    const float* const getHostPtr() const { return this->_host_ptr.get(); }
    float* getHostPtr() { return this->_host_ptr.get(); }

    const float* const getDevPtr() const { return this->_dev_ptr.get(); }
    float* getDevPtr() { return this->_dev_ptr.get(); }

    // Movers
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

