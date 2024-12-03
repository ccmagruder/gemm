#include <benchmark/benchmark.h>

#include "Matrix.h"

void bInitializeSquare(benchmark::State& state) {
    for (auto _ : state) {
        Matrix A(state.range(0), state.range(0));
        for (size_t i=0; i<state.range(0); i++) {
            for (size_t j=0; j<state.range(0); j++) {
                A[i][j] = 0;
            }
        }
    }
}

BENCHMARK(bInitializeSquare)
    ->RangeMultiplier(2)->Range(1024, 8*1024)
    ->Unit(benchmark::kMillisecond);

