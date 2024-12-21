# gemm/CMakeLists.txt

cmake_minimum_required(VERSION 3.28)
project(gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

############# GEMM LIBRARY ##############

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_library(gemm STATIC
        ${CMAKE_SOURCE_DIR}/src/Matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmNaive.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmCuBlas.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmMKL.cpp)
else()
    add_library(gemm SHARED
        ${CMAKE_SOURCE_DIR}/src/Matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmNaive.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmCuBlas.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmMKL.cpp)
endif()

target_include_directories(gemm PUBLIC ${CMAKE_SOURCE_DIR}/include)

################ CUDA ###################

find_package(CUDAToolkit REQUIRED)

target_link_libraries(gemm CUDA::cudart CUDA::cublas)

################# MKL ###################

find_library(MKL mkl_rt)

target_link_libraries(gemm ${MKL})

################ TESTS ##################
add_subdirectory(tests)

enable_testing()
