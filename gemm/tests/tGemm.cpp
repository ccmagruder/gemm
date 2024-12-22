#include "gtest/gtest.h"

#include "Matrix.h"
#include "Gemm.h"

template <typename T>
class tGemm : public testing::Test {};

using Types = ::testing::Types<GemmNaive, GemmCuBlas, GemmMKL>;
TYPED_TEST_SUITE(tGemm, Types);

TYPED_TEST(tGemm, Ones) {
    const size_t m = 64, k = 32, n = 16;
    TypeParam mult(Matrix::fill(m, k, 1.0), Matrix::fill(k, n, 1.0));
    std::unique_ptr<Matrix> ref = Matrix::fill(m, n, static_cast<float>(k));
    mult.compute();
    std::shared_ptr<Matrix> C = mult.get();
    EXPECT_FLOAT_EQ(C->lInfNorm(*ref), 0.0);
}

