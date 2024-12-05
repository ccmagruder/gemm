FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

RUN apt update -y && apt upgrade -y \
  && apt install -y --no-install-recommends \
    build-essential \
    clang \
    cmake \
    gcc \
    git \
    libbenchmark-dev \
    libgtest-dev \
  && rm -rf /var/lib/apt/lists/*
