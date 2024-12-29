#pragma once

#include <cassert>

#include "Gemm.hpp"

class GemmNaive : public Gemm<Host> {
 public:
  using Gemm<Host>::Gemm;
  void _run() override;
};

class GemmMkl : public Gemm<Host> {
 public:
  using Gemm<Host>::Gemm;
  void _run() override;
};

class GemmCuBlas : public Gemm<Device> {
 public:
  using Gemm<Device>::Gemm;
  void _run() override;
};

class GemmCuda : public Gemm<Device> {
 public:
  using Gemm<Device>::Gemm;
  void _run() override;
};
