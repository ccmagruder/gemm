#pragma once

void sgemm(const uint M,
           const uint N,
           const uint K,
           const float* const a,
           const float* const b,
           float* c,
           const uint W);
