#include "spgemm.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// 最普通的内积式矩阵乘：每个 C[row, col] 是 A 的一行和 B 的一列的内积。
// A 是 M x K，B 是 K x N，C 是 M x N，三个矩阵都按行优先存放。
__global__ void dense_matmul_kernel(const double *A, const double *B, double *C,
                                    int M, int K, int N) {
    size_t total = static_cast<size_t>(M) * N;
    int lane = threadIdx.x;

    // 一个完整 warp 共同计算一个输出元素，然后依次处理下一个元素。
    for (size_t index = 0; index < total; ++index) {
        int row = static_cast<int>(index / N);
        int col = static_cast<int>(index % N);

        double sum = 0.0f;
        for (int k = lane; k < K; k += 32) {
            sum += A[static_cast<size_t>(row) * K + k] *
                   B[static_cast<size_t>(k) * N + col];
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }

        if (lane == 0) {
            C[index] = sum;
        }
    }
}

static bool read_as_dense(const char *path, std::vector<double> &dense,
                          int &rows, int &cols) {
    void *buffer = nullptr;
    int *row_ptr = nullptr;
    int *col_idx = nullptr;
    double *val = nullptr;
    int nnz = 0;

    if (!read_matrix_market(path, &buffer, &row_ptr, &col_idx, &val,
                            &rows, &cols, &nnz)) {
        return false;
    }

    dense.assign(static_cast<size_t>(rows) * cols, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
            dense[static_cast<size_t>(row) * cols + col_idx[p]] = val[p];
        }
    }

    std::free(buffer);
    std::cout << "Read " << path << ": " << rows << " x " << cols
              << ", nnz = " << nnz << '\n';
    return true;
}

// 计算过程始终使用稠密 C；这里只在写文件前转回项目现有 writer 需要的 CSR。
static bool write_dense_result(const char *path, const std::vector<double> &dense,
                               int rows, int cols) {
    std::vector<int> row_ptr(rows + 1, 0);
    std::vector<int> col_idx;
    std::vector<double> val;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double x = dense[static_cast<size_t>(row) * cols + col];
            if (std::fabs(x) > 1e-12f) {
                if (col_idx.size() ==
                    static_cast<size_t>(std::numeric_limits<int>::max())) {
                    std::cerr << "Result has too many nonzero elements\n";
                    return false;
                }
                col_idx.push_back(col);
                val.push_back(x);
            }
        }
        row_ptr[row + 1] = static_cast<int>(col_idx.size());
    }

    return write_matrix_market(path, row_ptr.data(), col_idx.data(), val.data(),
                               rows, cols, static_cast<int>(val.size()));
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " A.mtx [output.mtx]\n"
                  << "  " << argv[0] << " A.mtx B.mtx output.mtx\n";
        return EXIT_FAILURE;
    }

    const char *A_path = argv[1];
    const char *B_path = (argc == 4) ? argv[2] : argv[1];
    const char *output_path = (argc == 4) ? argv[3]
                                           : (argc == 3 ? argv[2]
                                                        : "dense_result.mtx");

    std::vector<double> hA;
    std::vector<double> hB;
    int A_rows = 0, A_cols = 0;
    int B_rows = 0, B_cols = 0;

    if (!read_as_dense(A_path, hA, A_rows, A_cols)) {
        return EXIT_FAILURE;
    }

    if (argc == 4) {
        if (!read_as_dense(B_path, hB, B_rows, B_cols)) {
            return EXIT_FAILURE;
        }
    } else {
        hB = hA;
        B_rows = A_rows;
        B_cols = A_cols;
    }

    if (A_cols != B_rows) {
        std::cerr << "Dimension mismatch: A is " << A_rows << " x " << A_cols
                  << ", B is " << B_rows << " x " << B_cols << '\n';
        return EXIT_FAILURE;
    }

    size_t A_bytes = hA.size() * sizeof(double);
    size_t B_bytes = hB.size() * sizeof(double);
    size_t C_elements = static_cast<size_t>(A_rows) * B_cols;
    size_t C_bytes = C_elements * sizeof(double);
    std::vector<double> hC(C_elements);

    double *dA = nullptr;
    double *dB = nullptr;
    double *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, A_bytes));
    CHECK_CUDA(cudaMalloc(&dB, B_bytes));
    CHECK_CUDA(cudaMalloc(&dC, C_bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), B_bytes, cudaMemcpyHostToDevice));

    int block_size = 32;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    dense_matmul_kernel<<<1, block_size>>>(dA, dB, dC,
                                           A_rows, A_cols, B_cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, C_bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    if (!write_dense_result(output_path, hC, A_rows, B_cols)) {
        return EXIT_FAILURE;
    }

    double dense_mib = static_cast<double>(A_bytes + B_bytes + C_bytes) /
                       (1024.0 * 1024.0);
    std::cout << "C: " << A_rows << " x " << B_cols << '\n'
              << "Kernel time: " << milliseconds << " ms\n"
              << "Dense device memory: " << dense_mib << " MiB\n"
              << "Saved to " << output_path << '\n';
    return EXIT_SUCCESS;
}
