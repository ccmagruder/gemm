#include "gtest/gtest.h"

#include "Matrix.h"
#include "MatrixMult.h"

TEST(tMatrixMult, Ones) {
    std::unique_ptr<const Matrix> A = std::make_unique<const Matrix>(64, 32, 1);
    std::unique_ptr<const Matrix> B = std::make_unique<const Matrix>(32, 16, 1);
    MatrixMult mult(std::move(A), std::move(B));
    mult.compute();
    std::shared_ptr<Matrix> C = mult.get();
    ASSERT_FLOAT_EQ((*C)[0][0], 32.0);
}
