# gemm/CMakeLists.txt

cmake_minimum_required(VERSION 3.28)
project(gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

############### CUDA ####################

find_package(CUDAToolkit REQUIRED)

############# GEMM LIBRARY ##############

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_library(gemm STATIC
        ${CMAKE_SOURCE_DIR}/src/Matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmNaive.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmCuBlas.cpp)
else()
    add_library(gemm SHARED
        ${CMAKE_SOURCE_DIR}/src/Matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmNaive.cpp
        ${CMAKE_SOURCE_DIR}/src/GemmCuBlas.cpp)
endif()

target_include_directories(gemm PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(gemm CUDA::cudart CUDA::cublas)

################ TESTS ################
add_subdirectory(tests)

enable_testing()
