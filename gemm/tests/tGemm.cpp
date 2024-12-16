#include "gtest/gtest.h"

#include "Matrix.h"
#include "Gemm.h"

template <typename T>
class tGemm : public testing::Test {};

using Types = ::testing::Types<GemmNaive, GemmCuBlas>;
TYPED_TEST_SUITE(tGemm, Types);

TYPED_TEST(tGemm, Ones) {
    TypeParam mult(Matrix::fill(64, 32, 1), Matrix::fill(32, 16, 1));
    mult.compute();
    std::shared_ptr<Matrix> C = mult.get();
    EXPECT_FLOAT_EQ((*C)(0,0), 32.0);
}

