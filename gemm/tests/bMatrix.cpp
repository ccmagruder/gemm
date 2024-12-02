#include <benchmark/benchmark.h>

void bMalloc(benchmark::State& state) {
  for (auto _ : state) {
    float* ptr = reinterpret_cast<float*>(malloc(state.range(0)*sizeof(float)));
    for (int i=0; i<state.range(0); i++) ptr[i] = 0;
    free(ptr);
  }
}

BENCHMARK(bMalloc)
    ->RangeMultiplier(2)->Range(1024, 32*1024)
    ->Unit(benchmark::kMicrosecond);

