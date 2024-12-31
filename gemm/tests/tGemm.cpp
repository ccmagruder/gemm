#include <cstddef>
#include <memory>
#include <tuple>

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

class tGemmCudaVsRef : public testing::TestWithParam<
                           std::tuple<int, int, int, std::tuple<int, int>>> {
   public:
    void SetUp() {
        const size_t M = std::get<0>(GetParam());
        const size_t K = std::get<1>(GetParam());
        const size_t N = std::get<2>(GetParam());

        this->A = Matrix::normalIID(M, K);
        this->B = Matrix::normalIID(K, N);

        GemmNaive multNaive(A->copy(), B->copy());
        multNaive.compute();
        this->Cref = multNaive.get();
    }

    static std::string nameGenerator(
        const testing::TestParamInfo<tGemmCudaVsRef::ParamType>& info) {
        return std::to_string(std::get<0>(info.param)) + "x" +
               std::to_string(std::get<1>(info.param)) + "x" +
               std::to_string(std::get<2>(info.param)) + "x" +
               std::to_string(std::get<0>(std::get<3>(info.param))) + "x" +
               std::to_string(std::get<1>(std::get<3>(info.param)));
    }

    std::unique_ptr<const Matrix> A;
    std::unique_ptr<const Matrix> B;
    std::shared_ptr<Matrix> Cref;
};

TEST_P(tGemmCudaVsRef, lInfError) {
    const std::tuple<int, int> blockDim = std::get<3>(GetParam());
    GemmCuda multCuda(this->A->copy(), this->B->copy());
    multCuda._setup();
    multCuda.__run(std::get<0>(blockDim), std::get<1>(blockDim));
    multCuda._teardown();
    std::shared_ptr<Matrix> C = multCuda.get();
    EXPECT_NEAR(C->lInfNorm(*this->Cref), 0.0, 1e-5);
}

INSTANTIATE_TEST_SUITE_P(
    OddsAndEvens,
    tGemmCudaVsRef,
    testing::Combine(testing::Values(1, 32, 33, 1025),            // M
                     testing::Values(1, 32, 33),                  // K
                     testing::Values(1, 2),                       // N
                     testing::Values(std::make_tuple(1, 1),       // blockDim
                                     std::make_tuple(1, 32),      // blockDim
                                     std::make_tuple(1, 1024),    // blockDim
                                     std::make_tuple(32, 1),      // blockDim
                                     std::make_tuple(32, 32),     // blockDim
                                     std::make_tuple(1024, 1))),  // blockDim
    tGemmCudaVsRef::nameGenerator);
