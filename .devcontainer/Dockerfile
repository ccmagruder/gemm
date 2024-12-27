FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

RUN apt update -y && apt upgrade -y \
  && apt install -y --no-install-recommends \
    build-essential \
    clang \
    clang-format \
    clang-tidy \
    cmake \
    gcc \
    git \
    intel-mkl \
    libbenchmark-dev \
    libgtest-dev \
    # Dependency for intel-mkl
    libomp-dev \
  && rm -rf /var/lib/apt/lists/*
