#include <benchmark/benchmark.h>

#include "Gemm.h"
#include "Matrix.h"

class GemmSweepFixture : public benchmark::Fixture {
   public:
    void SetUp(::benchmark::State& state) {
        const size_t n = 2048;
        this->mult = std::make_unique<GemmCuda>(Matrix::normalIID(n, n),
                                                Matrix::normalIID(n, n));
        this->mult->_setup();
    }

    void TearDown(::benchmark::State& state) { this->mult->_teardown(); }

    std::unique_ptr<GemmCuda> mult;
};

BENCHMARK_DEFINE_F(GemmSweepFixture, bCudaBlockSize)(benchmark::State& state) {
    for (auto _ : state) {
        this->mult->__run(state.range(0), state.range(1));
    }
}

BENCHMARK_REGISTER_F(GemmSweepFixture, bCudaBlockSize)
    ->Args({1, 1})
    ->Args({1, 32})
    ->Args({1, 1024})
    ->Args({32, 1})
    ->Args({32, 32})
    ->Args({1024, 1});
