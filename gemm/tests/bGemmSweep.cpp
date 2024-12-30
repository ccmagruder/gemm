#include <benchmark/benchmark.h>

#include "Gemm.h"
#include "Matrix.h"

class GemmSweepFixture : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(GemmSweepFixture, bCudaBlockSize)(benchmark::State& state) {
    const size_t n = 2048;
    GemmCuda mult(Matrix::normalIID(n, n), Matrix::normalIID(n, n));
    mult._setup();
    for (auto _ : state) {
        mult.__run(state.range(0));
    }
    mult._teardown();
}

BENCHMARK_REGISTER_F(GemmSweepFixture, bCudaBlockSize)
    ->RangeMultiplier(2)
    ->Range(1, 32);
