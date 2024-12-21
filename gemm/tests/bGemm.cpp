#include <benchmark/benchmark.h>

#include "Matrix.h"
#include "Gemm.h"

#if(RELEASE)
    const int cublas_n_max = 4096;
    const int mkl_n_max = 4096;
    const int naive_n_max = 1024;
#else
    const int cublas_n_max = 256;
    const int mkl_n_max = 256;
    const int naive_n_max = 256;
#endif

template<typename T>
class GemmFixture : public benchmark::Fixture {
 protected:
    static void RoundTrip(benchmark::State& state) {
        size_t n = state.range(0);
        T mult(Matrix::normalIID(n, n), Matrix::normalIID(n, n));
        for (auto _ : state) {
            mult.compute();
        }
    }

    static void Gemm(benchmark::State& state) {
        size_t n = state.range(0);
        T mult(Matrix::normalIID(n, n), Matrix::normalIID(n, n));
        mult._setup();
        for (auto _ : state) {
            mult._run();
        }
    }
};

BENCHMARK_TEMPLATE_DEFINE_F(GemmFixture, RoundTripCuBLAS, GemmCuBlas)(benchmark::State& state) {
    GemmFixture::RoundTrip(state);
}

BENCHMARK_REGISTER_F(GemmFixture, RoundTripCuBLAS)
    ->RangeMultiplier(2)->Range(64, cublas_n_max);

BENCHMARK_TEMPLATE_DEFINE_F(GemmFixture, GemmNaive, GemmNaive)(benchmark::State& state) {
    GemmFixture::Gemm(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(GemmFixture, GemmCuBLAS, GemmCuBlas)(benchmark::State& state) {
    GemmFixture::Gemm(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(GemmFixture, GemmMKL, GemmMKL)(benchmark::State& state) {
    GemmFixture::Gemm(state);
}

BENCHMARK_REGISTER_F(GemmFixture, GemmCuBLAS)
    ->RangeMultiplier(2)->Range(64, cublas_n_max);

BENCHMARK_REGISTER_F(GemmFixture, GemmNaive)
    ->RangeMultiplier(2)->Range(64, naive_n_max);

BENCHMARK_REGISTER_F(GemmFixture, GemmMKL)
    ->RangeMultiplier(2)->Range(64, mkl_n_max);

