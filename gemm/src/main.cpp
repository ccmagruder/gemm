#include <chrono>
#include <iostream>

#include <gflags/gflags.h>

#include "Gemm.h"

DEFINE_int32(M, 4096, "Rows of A, Rows of C");
DEFINE_int32(K, 4096, "Rows of B, Cols of A");
DEFINE_int32(N, 4096, "Cols of B, Cols of C");
DEFINE_int32(blockDim_x, 32, "blockDim.x");
DEFINE_int32(blockDim_y, 32, "blockDim.y");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::unique_ptr<const Matrix> A = Matrix::normalIID(FLAGS_M, FLAGS_K);
    std::unique_ptr<const Matrix> B = Matrix::normalIID(FLAGS_K, FLAGS_N);

    GemmCuda mult(A->copy(), B->copy());
    mult._setup();

    auto start = std::chrono::high_resolution_clock::now();
    mult.__run(FLAGS_blockDim_x, FLAGS_blockDim_y);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    mult._teardown();
    std::shared_ptr<Matrix> Cref = mult.get();

    std::cout << "{\n"
              << "    \"M\": " << FLAGS_M << ",\n"
              << "    \"K\": " << FLAGS_K << ",\n"
              << "    \"N\": " << FLAGS_N << ",\n"
              << "    \"blockDim.x\": " << FLAGS_blockDim_x << ",\n"
              << "    \"blockDim.y\": " << FLAGS_blockDim_y << ",\n"
              << "    \"compute_time\": " << duration.count() << ",\n"
              << "    \"compute_time_units\": " << "\"ms\"" << ",\n"
              << "}\n";
    return 0;
}
