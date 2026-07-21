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

    {   // 默认由 spgemm.h 的 USE_MEMPOOL 宏决定;环境变量 USE_MEMPOOL=0/1 可覆盖(便于 A/B)
        const char* e = std::getenv("USE_MEMPOOL");
        g_use_mempool = e ? (std::atoi(e) > 0) : USE_MEMPOOL;
    }
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

    // 初始化 pinned host 内存池(USE_MEMPOOL=1 时生效;失败则退回原路径)
    if (!mempool_init()) {
        fprintf(stderr, "[main] mempool_init failed, falling back to legacy cudaMallocHost\n");
        g_use_mempool = false;
    }


    // ============ A·Aᵀ 上三角 profiling 模式 (argv[2]=="att") ============
    if (argc >= 3 && std::string(argv[2]) == "att") {
        printf("===============WARMING UP (att)===============");
        {
            void *wc = nullptr; int wr = 0, wcol = 0, wn = 0;
            spgemm_transpose_product(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn);
            if (wc) pinned_free(wc);
            wc = nullptr;
            spgemm_att_outer(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn);
            if (wc) pinned_free(wc);
            wc = nullptr;
            spgemm_att_gust(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn);
            if (wc) pinned_free(wc);
            wc = nullptr;
            spgemm_att_colw(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn);
            if (wc) pinned_free(wc);
            wc = nullptr;
            spgemm_att_inner(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn);
            if (wc) pinned_free(wc);
        }
        auto att_run = [&](const char *label, const char *kind, auto fn) {
            LOG_BOTH("\n=== Computing %s ===\n", label);
            void *C_buffer = nullptr; int Cr = 0, Cc = 0, Cn = 0;
            auto s = std::chrono::high_resolution_clock::now();
            fn(A_buffer, A_rows, A_cols, A_nnz, &C_buffer, &Cr, &Cc, &Cn);
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> el = e - s;
            LOG_BOTH("Result C (%s): %d x %d, nnz = %d\n", kind, Cr, Cc, Cn);
            LOG_BOTH("Time: %.3f ms\n", el.count());
            if (C_buffer) pinned_free(C_buffer);
        };
        att_run("C = A x A^T (cuSPARSE)", "A·Aᵀ 全量", spgemm_transpose_product);
        att_run("C = A x A^T upper (outer)", "A·Aᵀ 上三角", spgemm_att_outer);
        att_run("C = A x A^T upper (Gustavson)", "A·Aᵀ 上三角", spgemm_att_gust);
        att_run("C = A x A^T upper (colwise)", "A·Aᵀ 上三角", spgemm_att_colw);
        att_run("C = A x A^T upper (inner)", "A·Aᵀ 上三角", spgemm_att_inner);
        LOG_BOTH("\n=== All att tests completed ===\n");
        fclose(log_file);
        host_free(A_buffer);
        mempool_destroy();
        return 0;
    }


    // ---- 预热:正式计时前各方法空跑 3 轮,摊掉 thrust 工作区 / cuSPARSE handle /
    //   CUDA allocator 的一次性冷启动开销 + cache/TLB 预热,使 T1–T4 测的是稳态性能 ----
    printf("===============WARMING UP (3 rounds)===============");
    for (int warmup = 0; warmup < 3; warmup++) {
        void *wc = nullptr; int wr = 0, wcol = 0, wn = 0;
        if (should_run_method("cu"))      { spgemm_self_product(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn); if (wc) pinned_free(wc); wc = nullptr; }
        if (should_run_method("manual"))  { spgemm_self_product_manual(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn); if (wc) pinned_free(wc); wc = nullptr; }
        if (should_run_method("serial"))  { spgemm_self_product_merge(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn); if (wc) pinned_free(wc); wc = nullptr; }
        if (should_run_method("merge2"))  { spgemm_self_product_merge2(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn); if (wc) pinned_free(wc); wc = nullptr; }
        if (should_run_method("merge3"))  { spgemm_self_product_merge3(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn); if (wc) pinned_free(wc); wc = nullptr; }
        if (should_run_method("hash"))    { spgemm_self_product_hash(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn); if (wc) pinned_free(wc); wc = nullptr; }
        if (should_run_method("adaptive")){ spgemm_self_product_adaptive(A_buffer, A_rows, A_cols, A_nnz, &wc, &wr, &wcol, &wn); if (wc) pinned_free(wc); wc = nullptr; }
    }
    dbg("warmup done\n");

    // 测试 1: C = A x A cuSparse
    #if CU_REF
    if (should_run_method("cu")) {
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

            #if WRITE_MTX
                std::string output_path = result_dir + "/self_product.mtx";
                dbg("T1 writing %s (C_nnz=%d)...\n", output_path.c_str(), C_nnz);
                write_matrix_market(output_path.c_str(), C_row_ptr, C_col_idx,
                                     C_val, C_rows, C_cols, C_nnz);
                LOG_BOTH("Saved to  %s\n", output_path.c_str());
                dbg("T1 write done\n");
            #endif


            pinned_free(C_buffer);
        }
    }
    #endif

    // 测试 4: C = A x A (Gust)
    if (should_run_method("manual")) {
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

        #if WRITE_MTX
                    std::string output_path = result_dir + "/self_product_manual.mtx";
            dbg("T4 writing %s (C_nnz=%d)...\n", output_path.c_str(), C_nnz);
            write_matrix_market(output_path.c_str(), C_row_ptr, C_col_idx,
                                C_val, C_rows, C_cols, C_nnz);
            LOG_BOTH("Saved to %s\n", output_path.c_str());
            dbg("T4 write done\n");
        #endif


        pinned_free(C_buffer);
    }

    // 测试 4b: C = A x A (串行 k-way merge,与 ESC 对照)
    if (should_run_method("serial")) {
        LOG_BOTH("\n=== Computing C = A x A (Merge) ===\n");

        void *C_buffer = nullptr;
        int C_rows = 0, C_cols = 0, C_nnz = 0;

        auto start = std::chrono::high_resolution_clock::now();
        dbg("T4b merge self_product: start\n");
        spgemm_self_product_merge(A_buffer, A_rows, A_cols, A_nnz,
                                  &C_buffer, &C_rows, &C_cols, &C_nnz);
        dbg("T4b merge self_product: done (C_nnz=%d)\n", C_nnz);
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

        #if WRITE_MTX
            std::string output_path = result_dir + "/self_product_merge.mtx";
            dbg("T4b writing %s (C_nnz=%d)...\n", output_path.c_str(), C_nnz);
            write_matrix_market(output_path.c_str(), C_row_ptr, C_col_idx,
                                C_val, C_rows, C_cols, C_nnz);
            LOG_BOTH("Saved to %s\n", output_path.c_str());
            dbg("T4b write done\n");
        #endif

        pinned_free(C_buffer);
    }

    // 测试 4c: C = A x A (并行 k-way merge v2,与 ESC / serial merge 对照)
    if (should_run_method("merge2")) {
        LOG_BOTH("\n=== Computing C = A x A (Merge2) ===\n");

        void *C_buffer = nullptr;
        int C_rows = 0, C_cols = 0, C_nnz = 0;

        auto start = std::chrono::high_resolution_clock::now();
        dbg("T4c merge2 self_product: start\n");
        spgemm_self_product_merge2(A_buffer, A_rows, A_cols, A_nnz,
                                   &C_buffer, &C_rows, &C_cols, &C_nnz);
        dbg("T4c merge2 self_product: done (C_nnz=%d)\n", C_nnz);
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

        #if WRITE_MTX
            std::string output_path = result_dir + "/self_product_merge2.mtx";
            dbg("T4c writing %s (C_nnz=%d)...\n", output_path.c_str(), C_nnz);
            write_matrix_market(output_path.c_str(), C_row_ptr, C_col_idx,
                                C_val, C_rows, C_cols, C_nnz);
            LOG_BOTH("Saved to %s\n", output_path.c_str());
            dbg("T4c write done\n");
        #endif

        pinned_free(C_buffer);
    }

    // 测试 4d: C = A x A (分块 merge v3,与 merge2 对照)
    if (should_run_method("merge3")) {
        LOG_BOTH("\n=== Computing C = A x A (Merge3) ===\n");

        void *C_buffer = nullptr;
        int C_rows = 0, C_cols = 0, C_nnz = 0;

        auto start = std::chrono::high_resolution_clock::now();
        dbg("T4d merge3 self_product: start\n");
        spgemm_self_product_merge3(A_buffer, A_rows, A_cols, A_nnz,
                                   &C_buffer, &C_rows, &C_cols, &C_nnz);
        dbg("T4d merge3 self_product: done (C_nnz=%d)\n", C_nnz);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        double C_sparsity = 100.0 * (1.0 - (double)C_nnz / ((double)C_rows * C_cols));
        LOG_BOTH("Result C: %d x %d, nnz = %d, sparsity = %.2f%%\n",
                 C_rows, C_cols, C_nnz, C_sparsity);
        LOG_BOTH("Time: %.3f ms\n", elapsed.count());

        pinned_free(C_buffer);
    }

    // 测试 4e: C = A x A (hash SPA,大/稠密阵主场,与 merge 对照)
    if (should_run_method("hash")) {
        LOG_BOTH("\n=== Computing C = A x A (Hash SPA) ===\n");

        void *C_buffer = nullptr;
        int C_rows = 0, C_cols = 0, C_nnz = 0;

        auto start = std::chrono::high_resolution_clock::now();
        dbg("T4e hash self_product: start\n");
        spgemm_self_product_hash(A_buffer, A_rows, A_cols, A_nnz,
                                 &C_buffer, &C_rows, &C_cols, &C_nnz);
        dbg("T4e hash self_product: done (C_nnz=%d)\n", C_nnz);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        if (C_nnz < 0) {
            LOG_BOTH("Result C: OVERFLOW (某行 distinct>HASH_CAP),跳过\n");
        } else {
            double C_sparsity = 100.0 * (1.0 - (double)C_nnz / ((double)C_rows * C_cols));
            LOG_BOTH("Result C: %d x %d, nnz = %d, sparsity = %.2f%%\n",
                     C_rows, C_cols, C_nnz, C_sparsity);
            LOG_BOTH("Time: %.3f ms\n", elapsed.count());
        }

        if (C_buffer) pinned_free(C_buffer);
    }

    // 测试 4f: C = A x A (自适应 dispatcher:flop>thr→hash 否则 merge3,完整数据流)
    if (should_run_method("adaptive")) {
        LOG_BOTH("\n=== Computing C = A x A (Adaptive) ===\n");

        void *C_buffer = nullptr;
        int C_rows = 0, C_cols = 0, C_nnz = 0;

        auto start = std::chrono::high_resolution_clock::now();
        dbg("T4f adaptive self_product: start\n");
        spgemm_self_product_adaptive(A_buffer, A_rows, A_cols, A_nnz,
                                     &C_buffer, &C_rows, &C_cols, &C_nnz);
        dbg("T4f adaptive self_product: done (C_nnz=%d)\n", C_nnz);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        if (C_nnz < 0) {
            LOG_BOTH("Result C: OVERFLOW/错误\n");
        } else {
            double C_sparsity = 100.0 * (1.0 - (double)C_nnz / ((double)C_rows * C_cols));
            LOG_BOTH("Result C: %d x %d, nnz = %d, sparsity = %.2f%%\n",
                     C_rows, C_cols, C_nnz, C_sparsity);
            LOG_BOTH("Time: %.3f ms\n", elapsed.count());
        }

        if (C_buffer) pinned_free(C_buffer);
    }

    host_free(A_buffer);

    LOG_BOTH("\n=== All tests completed ===\n");

    //在LOG之后fclose,避免use-after-free
    fclose(log_file);
    mempool_destroy();
    return 0;
}