#include "gtest/gtest.h"

#include "Matrix.h"
#include "Gemm.h"

template <typename T>
class tGemm : public testing::Test {};

using Types = ::testing::Types<Gemm<Naive>, Gemm<CuBlas>, Gemm<Mkl>, Gemm<Cuda>>;
TYPED_TEST_SUITE(tGemm, Types);

TYPED_TEST(tGemm, Ones) {
    const size_t m = 64, k = 32, n = 16;
    TypeParam mult(Matrix::fill(m, k, 1.0), Matrix::fill(k, n, 1.0));
    std::unique_ptr<Matrix> ref = Matrix::fill(m, n, static_cast<float>(k));
    mult.compute();
    std::shared_ptr<Matrix> C = mult.get();
    EXPECT_FLOAT_EQ(C->lInfNorm(*ref), 0.0);
}

TEST(tGemm, GemmCudaDims) {
    const size_t m=2, k=3, n=3;
    std::unique_ptr<Matrix> A = Matrix::fill(m, k, 0.0);
    std::unique_ptr<Matrix> B = Matrix::fill(k, n, 0.0);
    std::unique_ptr<Matrix> C = Matrix::fill(m, n, 0.0);
    (*A)(1, 0) = 1.0;
    (*A)(0, 1) = -1.0;
    (*B)(0, 0) = -4.0;
    (*C)(1, 0) = -4.0;
    (*B)(1, 1) = 3.0;
    (*C)(0, 1) = -3.0;
    (*B)(0, 2) = 2.0;
    (*C)(1, 2) = 2.0;
    Gemm<Cuda> mult(std::move(A), std::move(B));
    mult.compute();
    EXPECT_FLOAT_EQ(mult.get()->lInfNorm(*C), 0.0);
}
