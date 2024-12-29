#pragma once

#include <memory>

#include "Matrix.h"

enum Memory { Host, Device };

template <Memory T>
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
  virtual void _run() = 0;
  void _teardown() { static_assert(false); };

  std::shared_ptr<Matrix> get() { return this->_C; }

 protected:
  std::unique_ptr<const Matrix> _A;
  std::unique_ptr<const Matrix> _B;
  std::shared_ptr<Matrix> _C;
};

template <>
void Gemm<Host>::_setup();

template <>
void Gemm<Host>::_teardown();

template <>
void Gemm<Device>::_setup();

template <>
void Gemm<Device>::_teardown();
