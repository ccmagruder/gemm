#include <cstddef>
#include <memory>
#include "Gemm.h"
#include "Matrix.h"
#include "gtest/gtest.h"

template <typename T>
class tGemm : public testing::Test {};

using Types = ::testing::Types<GemmNaive, GemmCuBlas, GemmMkl, GemmCuda>;
TYPED_TEST_SUITE(tGemm, Types);

TYPED_TEST(tGemm, Ones) {
    const size_t M = 64, K = 32, N = 16;
    TypeParam mult(Matrix::fill(M, K, 1.0), Matrix::fill(K, N, 1.0));
    std::unique_ptr<Matrix> ref = Matrix::fill(M, N, static_cast<float>(K));
    mult.compute();
    std::shared_ptr<Matrix> C = mult.get();
    EXPECT_FLOAT_EQ(C->lInfNorm(*ref), 0.0);
}

TEST(tGemmCuda, vsRef) {
    const size_t M = 65, K = 33, N = 17;
    std::unique_ptr<const Matrix> A = Matrix::normalIID(M, K);
    std::unique_ptr<const Matrix> B = Matrix::normalIID(K, N);

    GemmNaive multNaive(A->copy(), B->copy());
    multNaive.compute();
    std::shared_ptr<Matrix> ref = multNaive.get();

    GemmCuda multCuda(A->copy(), B->copy());
    multCuda.compute();
    std::shared_ptr<Matrix> C = multCuda.get();

    EXPECT_NEAR(C->lInfNorm(*ref), 0.0, 1e-5);
}
