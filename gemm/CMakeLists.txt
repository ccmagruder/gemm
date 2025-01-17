# gemm/CMakeLists.txt

cmake_minimum_required(VERSION 3.28)
project(gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

############# GEMM LIBRARY ##############

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_library(gemm STATIC
        ${CMAKE_SOURCE_DIR}/src/Matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/Matrix.cu
        ${CMAKE_SOURCE_DIR}/src/Gemm.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmNaive.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmCuBlas.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmMkl.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmCuda.cpp
        ${CMAKE_SOURCE_DIR}/src/kernels.cu
        ${CMAKE_SOURCE_DIR}/src/utils.cu
    )
else()
    add_library(gemm SHARED
        ${CMAKE_SOURCE_DIR}/src/Matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/Matrix.cu
        ${CMAKE_SOURCE_DIR}/src/Gemm.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmNaive.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmCuBlas.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmMkl.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmCuda.cpp
        ${CMAKE_SOURCE_DIR}/src/kernels.cu
        ${CMAKE_SOURCE_DIR}/src/utils.cu
    )
endif()

option(GDB "Debug Symbols")

if (GDB)
    target_compile_options(gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                            -g
                            -G
                           >)
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
