project(gemm)

# Unit Tests

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(tMatrix tMatrix.cpp)

target_link_libraries(tMatrix gemm GTest::gtest GTest::gtest_main)

enable_testing()

add_test(NAME tMatrix
         WORKING_DIRECTORY tests
         COMMAND tMatrix)

# Performance Tests

find_package(benchmark REQUIRED)

add_executable(bMatrix bMatrix.cpp)

target_link_libraries(bMatrix gemm benchmark::benchmark benchmark::benchmark_main)


