project(gemm)

# Unit Tests

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(tMatrix tMatrix.cpp)
add_executable(tMatrixMult tMatrixMult.cpp)

target_link_libraries(tMatrix gemm GTest::gtest GTest::gtest_main)
target_link_libraries(tMatrixMult gemm GTest::gtest GTest::gtest_main)

enable_testing()

add_test(NAME tMatrix
         WORKING_DIRECTORY tests
         COMMAND tMatrix)

add_test(NAME tMatrixMult
         WORKING_DIRECTORY tests
         COMMAND tMatrixMult)

# Performance Tests

find_package(benchmark REQUIRED)

add_executable(bMatrix bMatrix.cpp)
add_executable(bMatrixMult bMatrixMult.cpp)

target_link_libraries(bMatrix gemm benchmark::benchmark benchmark::benchmark_main)
target_link_libraries(bMatrixMult gemm benchmark::benchmark benchmark::benchmark_main)

