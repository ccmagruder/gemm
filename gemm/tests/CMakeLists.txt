project(gemm)

# Unit Tests

find_package(GTest REQUIRED)

add_executable(tCuda tCuda.cpp)
add_executable(tMatrix tMatrix.cpp)
add_executable(tGemm tGemm.cpp)

target_link_libraries(tCuda gemm GTest::gtest GTest::gtest_main)
target_link_libraries(tMatrix gemm GTest::gtest GTest::gtest_main)
target_link_libraries(tGemm gemm GTest::gtest GTest::gtest_main)

enable_testing()

add_test(NAME tCuda
         WORKING_DIRECTORY tests
         COMMAND tCuda)

add_test(NAME tMatrix
         WORKING_DIRECTORY tests
         COMMAND tMatrix)

add_test(NAME tGemm
         WORKING_DIRECTORY tests
         COMMAND tGemm)

# Performance Tests

find_package(benchmark REQUIRED)

add_executable(bGemm bGemm.cpp)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(bGemm PRIVATE "RELEASE=1")
endif()

add_executable(bGemmSweep bGemmSweep.cpp)

target_link_libraries(bGemm gemm benchmark::benchmark benchmark::benchmark_main)
target_link_libraries(bGemmSweep gemm benchmark::benchmark benchmark::benchmark_main)

