#include "gtest/gtest.h"

#include "Matrix.h"
#include "MatrixMult.h"

template <typename T>
class tMatrixMult : public testing::Test {};

using Types = ::testing::Types<MatrixMult, MatrixMultCuBLAS>;
TYPED_TEST_SUITE(tMatrixMult, Types);

TYPED_TEST(tMatrixMult, Ones) {
    TypeParam mult(Matrix::fill(64, 32, 1), Matrix::fill(32, 16, 1));
    mult.compute();
    std::shared_ptr<Matrix> C = mult.get();
    EXPECT_FLOAT_EQ((*C)[0][0], 32.0);
}

