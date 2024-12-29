#include "gtest/gtest.h"

#include "Matrix.h"

TEST(tMatrix, toFromHost) {
  std::unique_ptr<Matrix> A = Matrix::makeHost(1, 2);
  (*A)(0, 0) = 1;
  (*A)(0, 1) = 3.14;
  A->toDevice();
  (*A)(0, 0) = 2;
  (*A)(0, 1) = 2;
  A->toHost();
  EXPECT_FLOAT_EQ((*A)(0, 0), 1);
  EXPECT_FLOAT_EQ((*A)(0, 1), 3.14);
}

TEST(tMatrix, lInfNorm) {
  std::unique_ptr<Matrix> A = Matrix::fill(2, 2, 0.0);
  std::unique_ptr<Matrix> B = Matrix::fill(2, 2, 0.0);
  (*A)(0, 0) = -1.0;
  (*B)(0, 0) = -1.1;
  EXPECT_NEAR(A->lInfNorm(*B), 0.1, 1e-5);
  (*B)(0, 0) = 1.1;
  EXPECT_NEAR(A->lInfNorm(*B), 2.1, 1e-5);
  (*A)(1, 1) = 3.14;
  EXPECT_NEAR(A->lInfNorm(*B), 3.14, 1e-5);
  (*B)(1, 1) = 3.14;
  EXPECT_NEAR(A->lInfNorm(*B), 2.1, 1e-5);
}
