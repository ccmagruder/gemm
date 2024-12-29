FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

RUN apt update -y && apt upgrade -y \
  && apt install -y --no-install-recommends \
    build-essential \
    clang \
    clangd \
    clang-format \
    cmake \
    gcc \
    git \
    intel-mkl \
    libbenchmark-dev \
    libgtest-dev \
    # Dependency for intel-mkl
    libomp-dev \
    # Dependency for clangd-tidy
    python3 \
    python3-pip \
    python3-venv \
  && rm -rf /var/lib/apt/lists/* \
  && python3 -m venv /.venv \
  # Clangd-tidy is documented at https://github.com/lljbash/clangd-tidy
  # It provides a similar interface to clang-tidy using clangd as the backend
  # enabling it to work with CUDA without throwing misleading errors
  && PATH=/.venv/bin:$PATH pip install clangd-tidy \
  && echo "source /.venv/bin/activate" >> ~/.bashrc 


