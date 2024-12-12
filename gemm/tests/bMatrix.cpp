#include <benchmark/benchmark.h>

#include "Matrix.h"

void bInitializeSquare(benchmark::State& state) {
    for (auto _ : state) {
        size_t n = state.range(0);
        Matrix A(n, n, 0);
    }
}

BENCHMARK(bInitializeSquare)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);


void bInitializeIID(benchmark::State& state) {

    for (auto _ : state) {
        size_t n = state.range(0);
        std::unique_ptr<const Matrix> A = Matrix::normalIID(n, n);
    }
}

BENCHMARK(bInitializeIID)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);
