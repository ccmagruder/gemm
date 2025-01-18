#include <stdexcept>

void cudaCheck(cudaError_t code, const char* file, const int line) {
    if (code != cudaSuccess) {
        char msg[100];
        sprintf(msg, "GPU kernel assert: %s:%d \"%s\"\n", file, line,
                cudaGetErrorString(code));
        throw std::runtime_error(msg);
    }
}

void cudaCheck(const char* file, const int line) {
    cudaError_t code = cudaGetLastError();
    cudaCheck(code, file, line);
}
