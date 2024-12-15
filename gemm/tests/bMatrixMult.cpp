#include <benchmark/benchmark.h>

#include "Matrix.h"
#include "MatrixMult.h"

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

BENCHMARK_TEMPLATE_DEFINE_F(GemmFixture, RoundTripNaive, MatrixMult)(benchmark::State& state) {
    GemmFixture::RoundTrip(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(GemmFixture, RoundTripCuBLAS, MatrixMultCuBLAS)(benchmark::State& state) {
    GemmFixture::RoundTrip(state);
}

BENCHMARK_REGISTER_F(GemmFixture, RoundTripCuBLAS)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(GemmFixture, RoundTripNaive)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE_DEFINE_F(GemmFixture, GemmCuBLAS, MatrixMultCuBLAS)(benchmark::State& state) {
    GemmFixture::Gemm(state);
}

BENCHMARK_TEMPLATE_DEFINE_F(GemmFixture, GemmNaive, MatrixMult)(benchmark::State& state) {
    GemmFixture::Gemm(state);
}

BENCHMARK_REGISTER_F(GemmFixture, GemmCuBLAS)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(GemmFixture, GemmNaive)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);

