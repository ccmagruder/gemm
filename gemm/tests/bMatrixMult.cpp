#include <benchmark/benchmark.h>

#include "Matrix.h"
#include "MatrixMult.h"

void bMatrixMult(benchmark::State& state) {
    for (auto _ : state) {
        size_t n = state.range(0);
        MatrixMult mult(Matrix::iid(n, n), Matrix::iid(n, n));
        std::unique_ptr<const Matrix> C = mult.compute();
    }
}

BENCHMARK(bMatrixMult)
    ->RangeMultiplier(2)->Range(64, 256)
    ->Unit(benchmark::kMillisecond);
