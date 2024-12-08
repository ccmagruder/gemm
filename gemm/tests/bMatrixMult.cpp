#include <benchmark/benchmark.h>

#include "Matrix.h"
#include "MatrixMult.h"

void bMatrixMult(benchmark::State& state) {
    size_t n = state.range(0);
    MatrixMult mult(Matrix::iid(n, n), Matrix::iid(n, n));
    for (auto _ : state) {
        mult.compute();
    }
}

BENCHMARK(bMatrixMult)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);


void bMatrixMultSetup(benchmark::State& state) {
    size_t n = state.range(0);
    MatrixMult mult(Matrix::iid(n, n), Matrix::iid(n, n));
    for (auto _ : state) {
        mult._setup();
    }
}

BENCHMARK(bMatrixMultSetup)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);


void bMatrixMultRun(benchmark::State& state) {
    size_t n = state.range(0);
    MatrixMult mult(Matrix::iid(n, n), Matrix::iid(n, n));
    mult._setup();
    for (auto _ : state) {
        mult._run();
    }
}

BENCHMARK(bMatrixMultRun)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);


void bMatrixMultTeardown(benchmark::State& state) {
    size_t n = state.range(0);
    MatrixMult mult(Matrix::iid(n, n), Matrix::iid(n, n));
    mult._setup();
    mult._run();
    for (auto _ : state) {
        mult._teardown();
    }
}

BENCHMARK(bMatrixMultTeardown)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);

