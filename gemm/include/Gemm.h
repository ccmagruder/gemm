# pragma once

#include <cassert>

#include "Matrix.h"

enum Algo { Naive, CuBlas, Mkl, Cuda };

template<Algo T>
class Gemm {
 public:
    Gemm(std::unique_ptr<const Matrix> A, std::unique_ptr<const Matrix> B)
      : _A(std::move(A)), _B(std::move(B)) {}

    std::shared_ptr<Matrix> compute() {
        this->_setup();
        this->_run();
        this->_teardown();
        return this->get();
    }

    void _setup() { static_assert(false); }
    void _run() { static_assert(false); }
    void _teardown() { static_assert(false); };

    std::shared_ptr<Matrix> get() { return this->_C; }

 protected:
    std::unique_ptr<const Matrix> _A;
    std::unique_ptr<const Matrix> _B;
    std::shared_ptr<Matrix> _C;
};

template<> void Gemm<Naive>::_setup();
template<> void Gemm<Naive>::_run();
template<> void Gemm<Naive>::_teardown();

template<> void Gemm<CuBlas>::_setup();
template<> void Gemm<CuBlas>::_run();
template<> void Gemm<CuBlas>::_teardown();

template<> void Gemm<Mkl>::_setup();
template<> void Gemm<Mkl>::_run();
template<> void Gemm<Mkl>::_teardown();

template<> void Gemm<Cuda>::_setup();
template<> void Gemm<Cuda>::_run();
template<> void Gemm<Cuda>::_teardown();

