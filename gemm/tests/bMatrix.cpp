#include <benchmark/benchmark.h>

#include "Matrix.h"

void bInitializeSquare(benchmark::State& state) {
    for (auto _ : state) {
        size_t n = state.range(0);
        Matrix A(n, n, 0);
        for (size_t i=0; i<state.range(0); i++) {
            for (size_t j=0; j<state.range(0); j++) {
                A[i][j] = 0;
            }
        }
    }
}

BENCHMARK(bInitializeSquare)
    ->RangeMultiplier(2)->Range(512, 2048)
    ->Unit(benchmark::kMillisecond);


void bInitializeIID(benchmark::State& state) {

    for (auto _ : state) {
        size_t n = state.range(0);
        std::unique_ptr<const Matrix> A = Matrix::iid(n, n);
    }
}

BENCHMARK(bInitializeIID)
    ->RangeMultiplier(2)->Range(512, 2048)
    ->Unit(benchmark::kMillisecond);
