#include "spgemm.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <string>
#include <sys/stat.h>
#include <libgen.h>
#include <cuda_runtime.h>


static void ensure_directory(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        mkdir(path, 0755);
    }
}

static std::string get_basename(const char *path) {
    char *tmp = strdup(path);
    char *base = basename(tmp);
    std::string result(base);
    
    size_t pos = result.rfind(".mtx");
    if (pos != std::string::npos) {
        result = result.substr(0, pos);
    }
    
    free(tmp);
    return result;
}

int main(int argc, char **argv) {

    dbg("main entry, initializing CUDA context...\n");
    cudaFree(0);
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    dbg("CUDA context ready\n");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx>\n";
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];
    std::string basename = get_basename(input_path);
    
    ensure_directory("results");
    ensure_directory("results/matrices");          // ← 新增中间层
    std::string result_dir = "results/matrices/" + basename;
    ensure_directory(result_dir.c_str());

    
    std::string log_path = result_dir + "/performance.log";
    FILE *log_file = fopen(log_path.c_str(), "w");
    if (!log_file) {
        std::cerr << "Failed to create log file: " << log_path << "\n";
        return EXIT_FAILURE;
    }
    
    #define LOG_BOTH(fmt, ...) do { \
        printf(fmt, ##__VA_ARGS__); \
        fprintf(log_file, fmt, ##__VA_ARGS__); \
        fflush(log_file); \
    } while(0)

    // 读入矩阵 A（单块连续内存）
    void *A_buffer = nullptr;
    int *A_row_ptr = nullptr, *A_col_idx = nullptr;
    float *A_val = nullptr;
    int A_rows = 0, A_cols = 0, A_nnz = 0;

    LOG_BOTH("Reading matrix from %s...\n", input_path);
    dbg("start read_matrix_market: %s\n", input_path);
    if (!read_matrix_market(input_path, &A_buffer, &A_row_ptr, &A_col_idx, &A_val,
                            &A_rows, &A_cols, &A_nnz)) {
        LOG_BOTH("Failed to read matrix\n");
        fclose(log_file);
        return EXIT_FAILURE;
    }
    dbg("read done: %d x %d, nnz=%d\n", A_rows, A_cols, A_nnz);

    double sparsity = 100.0 * (1.0 - (double)A_nnz / ((double)A_rows * A_cols));
    LOG_BOTH("Input A: %d x %d, nnz = %d, sparsity = %.2f%%\n",
             A_rows, A_cols, A_nnz, sparsity);

    if(TEST_READ){
        return 0;//先测试读入函数是否正确
    }

    
    // 测试 1: C = A x A cuSparse
    {
        LOG_BOTH("\n=== Computing C = A x A (cuSPARSE) ===\n");

        if (A_rows != A_cols) {
            LOG_BOTH("Skip A x A: A is not square (%d x %d)\n", A_rows, A_cols);
        } else {
            void *C_buffer = nullptr;
            int C_rows = 0, C_cols = 0, C_nnz = 0;

            auto start = std::chrono::high_resolution_clock::now();
            dbg("T1 cuSPARSE self_product: start\n");
            spgemm_self_product(A_buffer, A_rows, A_cols, A_nnz,
                               &C_buffer, &C_rows, &C_cols, &C_nnz);
            dbg("T1 cuSPARSE self_product: done (C_nnz=%d)\n", C_nnz);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> elapsed = end - start;
            double C_sparsity = 100.0 * (1.0 - (double)C_nnz / ((double)C_rows * C_cols));
            LOG_BOTH("Result C: %d x %d, nnz = %d, sparsity = %.2f%%\n",
                     C_rows, C_cols, C_nnz, C_sparsity);
            LOG_BOTH("Time: %.3f ms\n", elapsed.count());

            char *C_base = (char*)C_buffer;
            size_t C_row_ptr_size = (C_rows + 1) * sizeof(int);
            size_t C_col_idx_size = C_nnz * sizeof(int);
            size_t C_row_ptr_size_aligned = (C_row_ptr_size + 3) & ~3;
            size_t C_col_idx_size_aligned = (C_col_idx_size + 3) & ~3;
            
            int *C_row_ptr = (int*)C_base;
            int *C_col_idx = (int*)(C_base + C_row_ptr_size_aligned);
            float *C_val = (float*)(C_base + C_row_ptr_size_aligned + C_col_idx_size_aligned);

            std::string output_path = result_dir + "/self_product.mtx";
            dbg("T1 writing %s (C_nnz=%d)...\n", output_path.c_str(), C_nnz);
            write_matrix_market(output_path.c_str(), C_row_ptr, C_col_idx,
                               C_val, C_rows, C_cols, C_nnz);
            LOG_BOTH("Saved to %s\n", output_path.c_str());
            dbg("T1 write done\n");

            cudaFreeHost(C_buffer);
        }
    }
/*
    // 测试 2: C = A x A^T (cuSPARSE)
    {
        LOG_BOTH("\n=== Computing C = A x A^T (cuSPARSE) ===\n");

        void *C_buffer = nullptr;
        int C_rows = 0, C_cols = 0, C_nnz = 0;

        auto start = std::chrono::high_resolution_clock::now();
        spgemm_transpose_product(A_buffer, A_rows, A_cols, A_nnz,
                                &C_buffer, &C_rows, &C_cols, &C_nnz);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        double C_sparsity = 100.0 * (1.0 - (double)C_nnz / ((double)C_rows * C_cols));
        LOG_BOTH("Result C: %d x %d, nnz = %d, sparsity = %.2f%%\n",
                 C_rows, C_cols, C_nnz, C_sparsity);
        LOG_BOTH("Time: %.3f ms\n", elapsed.count());

        char *C_base = (char*)C_buffer;
        size_t C_row_ptr_size = (C_rows + 1) * sizeof(int);
        size_t C_col_idx_size = C_nnz * sizeof(int);
        size_t C_row_ptr_size_aligned = (C_row_ptr_size + 3) & ~3;
        size_t C_col_idx_size_aligned = (C_col_idx_size + 3) & ~3;
        
        int *C_row_ptr = (int*)C_base;
        int *C_col_idx = (int*)(C_base + C_row_ptr_size_aligned);
        float *C_val = (float*)(C_base + C_row_ptr_size_aligned + C_col_idx_size_aligned);

        std::string output_path = result_dir + "/transpose_product_cusparse.mtx";
        write_matrix_market(output_path.c_str(), C_row_ptr, C_col_idx,
                           C_val, C_rows, C_cols, C_nnz);
        LOG_BOTH("Saved to %s\n", output_path.c_str());

        cudaFreeHost(C_buffer);
    }

    // 测试 3: C = A x A^T (手写对称优化)
    {
        LOG_BOTH("\n=== Computing C = A x A^T (Manual Symmetric) ===\n");

        void *C_buffer = nullptr;
        int C_rows = 0, C_cols = 0, C_nnz = 0;

        auto start = std::chrono::high_resolution_clock::now();
        spgemm_transpose_product_manual(A_buffer, A_rows, A_cols, A_nnz,
                                       &C_buffer, &C_rows, &C_cols, &C_nnz);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        double C_sparsity = 100.0 * (1.0 - (double)C_nnz / ((double)C_rows * C_cols));
        LOG_BOTH("Result C: %d x %d, nnz = %d, sparsity = %.2f%%\n",
                 C_rows, C_cols, C_nnz, C_sparsity);
        LOG_BOTH("Time: %.3f ms\n", elapsed.count());

        char *C_base = (char*)C_buffer;
        size_t C_row_ptr_size = (C_rows + 1) * sizeof(int);
        size_t C_col_idx_size = C_nnz * sizeof(int);
        size_t C_row_ptr_size_aligned = (C_row_ptr_size + 3) & ~3;
        size_t C_col_idx_size_aligned = (C_col_idx_size + 3) & ~3;
        
        int *C_row_ptr = (int*)C_base;
        int *C_col_idx = (int*)(C_base + C_row_ptr_size_aligned);
        float *C_val = (float*)(C_base + C_row_ptr_size_aligned + C_col_idx_size_aligned);

        std::string output_path = result_dir + "/transpose_product_manual.mtx";
        write_matrix_market(output_path.c_str(), C_row_ptr, C_col_idx,
                           C_val, C_rows, C_cols, C_nnz);
        LOG_BOTH("Saved to %s\n", output_path.c_str());

        cudaFreeHost(C_buffer);
    }
*/
    // 测试 4: C = A x A (手写实现)
    {
        LOG_BOTH("\n=== Computing C = A x A (Manual) ===\n");

        void *C_buffer = nullptr;
        int C_rows = 0, C_cols = 0, C_nnz = 0;

        auto start = std::chrono::high_resolution_clock::now();
        dbg("T4 manual self_product: start\n");
        spgemm_self_product_manual(A_buffer, A_rows, A_cols, A_nnz,
                                   &C_buffer, &C_rows, &C_cols, &C_nnz);
        dbg("T4 manual self_product: done (C_nnz=%d)\n", C_nnz);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        double C_sparsity = 100.0 * (1.0 - (double)C_nnz / ((double)C_rows * C_cols));
        LOG_BOTH("Result C: %d x %d, nnz = %d, sparsity = %.2f%%\n",
                 C_rows, C_cols, C_nnz, C_sparsity);
        LOG_BOTH("Time: %.3f ms\n", elapsed.count());

        char *C_base = (char*)C_buffer;
        size_t C_row_ptr_size = (C_rows + 1) * sizeof(int);
        size_t C_col_idx_size = C_nnz * sizeof(int);
        size_t C_row_ptr_size_aligned = (C_row_ptr_size + 3) & ~3;
        size_t C_col_idx_size_aligned = (C_col_idx_size + 3) & ~3;
        
        int *C_row_ptr = (int*)C_base;
        int *C_col_idx = (int*)(C_base + C_row_ptr_size_aligned);
        float *C_val = (float*)(C_base + C_row_ptr_size_aligned + C_col_idx_size_aligned);

        std::string output_path = result_dir + "/self_product_manual.mtx";
        dbg("T4 writing %s (C_nnz=%d)...\n", output_path.c_str(), C_nnz);
        write_matrix_market(output_path.c_str(), C_row_ptr, C_col_idx,
                           C_val, C_rows, C_cols, C_nnz);
        LOG_BOTH("Saved to %s\n", output_path.c_str());
        dbg("T4 write done\n");

        cudaFreeHost(C_buffer);
    }

    free(A_buffer);

    LOG_BOTH("\n=== All tests completed ===\n");

    //在LOG之后fclose,避免use-after-free
    fclose(log_file);
    return 0;
}