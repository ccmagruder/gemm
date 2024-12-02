cmake_minimum_required(VERSION 3.28)
project(gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(tests)

enable_testing()
