project(gemm)

find_package(GTest REQUIRED)
include(GoogleTest)

find_package(benchmark REQUIRED)

add_executable(tMatrix tMatrix.cpp)

target_link_libraries(tMatrix GTest::gtest GTest::gtest_main)

enable_testing()

add_test(NAME tMatrix
         WORKING_DIRECTORY tests
         COMMAND tMatrix)
