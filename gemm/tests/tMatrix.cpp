#include "gtest/gtest.h"

#include "Matrix.h"

TEST(tMatrix, toFromHost) {
    std::unique_ptr<Matrix> A = Matrix::makeHost(1, 1);
    (*A)(0,0) = 1;
    A->toDevice();
    (*A)(0,0) = 2;
    A->toHost();
    EXPECT_EQ((*A)(0,0), 1);
}

