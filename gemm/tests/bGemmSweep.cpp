#include <benchmark/benchmark.h>

#include "Gemm.h"
#include "Matrix.h"

static void bCudaBlockSize(benchmark::State& state) {
    const size_t n = 2048;
    GemmCuda mult(Matrix::normalIID(n, n), Matrix::normalIID(n, n));
    mult._setup();
    for (auto _ : state) {
        mult.__run(state.range(0));
    }
    mult._teardown();
}

BENCHMARK(bCudaBlockSize)->RangeMultiplier(2)->Range(1, 32);
