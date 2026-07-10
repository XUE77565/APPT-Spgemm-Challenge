#include "spgemm.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUSPARSE(call)                                                   \
    do {                                                                       \
        cusparseStatus_t st = (call);                                          \
        if (st != CUSPARSE_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuSPARSE error %s:%d: %d (%s)\n", __FILE__,       \
                    __LINE__, st, cusparseGetErrorString(st));                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static cusparseHandle_t g_handle = nullptr;

static void ensure_handle() {
    if (g_handle == nullptr) {
        CHECK_CUSPARSE(cusparseCreate(&g_handle));
    }
}

// 核心计算：输入已在 device 的单块内存，输出也是单块内存
static void spgemm_cusparse_device(
    void *dA_buffer, int A_rows, int A_cols, int A_nnz,
    void *dB_buffer, int B_rows, int B_cols, int B_nnz,
    void **dC_buffer_out, int *C_rows_out, int *C_cols_out, int *C_nnz_out) {
    
    if (A_cols != B_rows) {
        fprintf(stderr, "Dimension mismatch: A_cols(%d) != B_rows(%d)\n",
                A_cols, B_rows);
        exit(EXIT_FAILURE);
    }

    ensure_handle();

    // 从单块内存解析出三个指针
    char *dA_base = (char*)dA_buffer;
    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    float *dA_val = (float*)(dA_base + A_row_ptr_size + A_col_idx_size);

    char *dB_base = (char*)dB_buffer;
    size_t B_row_ptr_size = (B_rows + 1) * sizeof(int);
    size_t B_col_idx_size = B_nnz * sizeof(int);
    int *dB_row_ptr = (int*)dB_base;
    int *dB_col_idx = (int*)(dB_base + B_row_ptr_size);
    float *dB_val = (float*)(dB_base + B_row_ptr_size + B_col_idx_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaDataType computeType = CUDA_R_32F;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseSpMatDescr_t matA, matB, matC;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, A_rows, A_cols, A_nnz,
        dA_row_ptr, dA_col_idx, dA_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matB, B_rows, B_cols, B_nnz,
        dB_row_ptr, dB_col_idx, dB_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    int *dC_row_ptr;
    CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matC, A_rows, B_cols, 0, dC_row_ptr, nullptr, nullptr,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    void *dBuffer1 = nullptr, *dBuffer2 = nullptr;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    dbg("cusparse: workEstimation begin\n");
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        g_handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr));
    CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        g_handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[cu] workest (buf1=%zu B)\n", bufferSize1);

    dbg("cusparse: compute begin\n");
    CHECK_CUSPARSE(cusparseSpGEMM_compute(
        g_handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, nullptr));
    CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(
        g_handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[cu] compute (buf2=%zu B)\n", bufferSize2);

    int64_t C_rows64, C_cols64, C_nnz64;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_rows64, &C_cols64, &C_nnz64));
    int C_nnz = static_cast<int>(C_nnz64);
    dbg("cusparse: C size = %lld x %lld, nnz=%lld\n",
        (long long)C_rows64, (long long)C_cols64, (long long)C_nnz64);

    int *dC_col_idx;
    float *dC_val;
    CHECK_CUDA(cudaMalloc(&dC_col_idx, C_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dC_val, C_nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, dC_row_ptr, dC_col_idx, dC_val));

    dbg("cusparse: copy begin\n");
    CHECK_CUSPARSE(cusparseSpGEMM_copy(
        g_handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[cu] copy\n");

    // 分配对齐的 device 单块内存
    size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t C_col_idx_size = C_nnz * sizeof(int);
    size_t C_val_size = C_nnz * sizeof(float);
    
    // 对齐到 4 字节边界
    size_t C_row_ptr_size_aligned = (C_row_ptr_size + 3) & ~3;
    size_t C_col_idx_size_aligned = (C_col_idx_size + 3) & ~3;
    size_t C_total_size = C_row_ptr_size_aligned + C_col_idx_size_aligned + C_val_size;
    
    void *dC_buffer;
    CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));
    
    char *dC_base = (char*)dC_buffer;
    CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_size_aligned, dC_col_idx, C_col_idx_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_size_aligned + C_col_idx_size_aligned, dC_val, C_val_size, cudaMemcpyDeviceToDevice));
    dbg("[cu] pack\n");

    *dC_buffer_out = dC_buffer;
    *C_rows_out = A_rows;
    *C_cols_out = B_cols;
    *C_nnz_out = C_nnz;

    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    cudaFree(dC_row_ptr);
    cudaFree(dC_col_idx);
    cudaFree(dC_val);
}

void spgemm_self_product(void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    dbg("[cu] start\n");
    ensure_handle();

    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(float);
    size_t A_total_size = A_row_ptr_size + A_col_idx_size + A_val_size;

    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice));
    dbg("[cu] h2d (%zu B)\n", A_total_size);

    void *dC_buffer;
    int C_rows_tmp, C_cols_tmp, C_nnz_tmp;
    spgemm_cusparse_device(dA_buffer, A_rows, A_cols, A_nnz,
        dA_buffer, A_rows, A_cols, A_nnz,
        &dC_buffer, &C_rows_tmp, &C_cols_tmp, &C_nnz_tmp);

    // 改这里：用对齐后的大小
    size_t C_row_ptr_size = (C_rows_tmp + 1) * sizeof(int);
    size_t C_col_idx_size = C_nnz_tmp * sizeof(int);
    size_t C_val_size = C_nnz_tmp * sizeof(float);

    size_t C_row_ptr_size_aligned = (C_row_ptr_size + 3) & ~3;
    size_t C_col_idx_size_aligned = (C_col_idx_size + 3) & ~3;
    size_t C_total_size = C_row_ptr_size_aligned + C_col_idx_size_aligned + C_val_size;

    void *C_buffer;
    CHECK_CUDA(cudaMallocHost(&C_buffer, C_total_size));
    dbg("self_product: C D2H begin (%zu B)\n", C_total_size);
    CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, cudaMemcpyDeviceToHost));
    dbg("[cu] d2h\n");

    *C_buffer_out = C_buffer;
    *C_rows = C_rows_tmp;
    *C_cols = C_cols_tmp;
    *C_nnz = C_nnz_tmp;

    cudaFree(dA_buffer);
    cudaFree(dC_buffer);
}

void spgemm_transpose_product(void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    ensure_handle();

    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(float);
    size_t A_total_size = A_row_ptr_size + A_col_idx_size + A_val_size;

    // 单次 H2D
    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice));

    // 解析 A 的指针做转置
    char *dA_base = (char*)dA_buffer;
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    float *dA_val = (float*)(dA_base + A_row_ptr_size + A_col_idx_size);

    // 转置结果分别分配，保证对齐
    int *dAT_row_ptr, *dAT_col_idx;
    float *dAT_val;
    CHECK_CUDA(cudaMalloc(&dAT_row_ptr, (A_cols + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dAT_col_idx, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dAT_val, A_nnz * sizeof(float)));

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
    g_handle, A_rows, A_cols, A_nnz, dA_val, dA_row_ptr, dA_col_idx,
    dAT_val, dAT_row_ptr, dAT_col_idx, CUDA_R_32F,
    CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1, &bufferSize));

    void *dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    CHECK_CUSPARSE(cusparseCsr2cscEx2(
    g_handle, A_rows, A_cols, A_nnz, dA_val, dA_row_ptr, dA_col_idx,
    dAT_val, dAT_row_ptr, dAT_col_idx, CUDA_R_32F,
    CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
    CUSPARSE_CSR2CSC_ALG1, dBuffer));
    cudaFree(dBuffer);

    // 合并 A^T 到单块内存（保证对齐）
    size_t AT_row_ptr_size = (A_cols + 1) * sizeof(int);
    size_t AT_col_idx_size = A_nnz * sizeof(int);
    size_t AT_val_size = A_nnz * sizeof(float);

    // 对齐到 float（4 字节）边界
    size_t AT_row_ptr_size_aligned = (AT_row_ptr_size + 3) & ~3;
    size_t AT_col_idx_size_aligned = (AT_col_idx_size + 3) & ~3;
    size_t AT_total_size = AT_row_ptr_size_aligned + AT_col_idx_size_aligned + AT_val_size;

    void *dAT_buffer;
    CHECK_CUDA(cudaMalloc(&dAT_buffer, AT_total_size));

    char *dAT_base = (char*)dAT_buffer;
    CHECK_CUDA(cudaMemcpy(dAT_base, dAT_row_ptr, AT_row_ptr_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dAT_base + AT_row_ptr_size_aligned, dAT_col_idx, AT_col_idx_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dAT_base + AT_row_ptr_size_aligned + AT_col_idx_size_aligned, dAT_val, AT_val_size, cudaMemcpyDeviceToDevice));

    // 释放临时分配的三个数组
    cudaFree(dAT_row_ptr);
    cudaFree(dAT_col_idx);
    cudaFree(dAT_val);

    // 计算 A × A^T
    void *dC_buffer;
    int C_rows_tmp, C_cols_tmp, C_nnz_tmp;
    spgemm_cusparse_device(dA_buffer, A_rows, A_cols, A_nnz,
    dAT_buffer, A_cols, A_rows, A_nnz,
    &dC_buffer, &C_rows_tmp, &C_cols_tmp, &C_nnz_tmp);

    // 单次 D2H（C 内部已经对齐）
    size_t C_row_ptr_size = (C_rows_tmp + 1) * sizeof(int);
    size_t C_col_idx_size = C_nnz_tmp * sizeof(int);
    size_t C_val_size = C_nnz_tmp * sizeof(float);

    // 同样对齐
    size_t C_row_ptr_size_aligned = (C_row_ptr_size + 3) & ~3;
    size_t C_col_idx_size_aligned = (C_col_idx_size + 3) & ~3;
    size_t C_total_size = C_row_ptr_size_aligned + C_col_idx_size_aligned + C_val_size;

    void *C_buffer;
    CHECK_CUDA(cudaMallocHost(&C_buffer, C_total_size));
    CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, cudaMemcpyDeviceToHost));

    *C_buffer_out = C_buffer;
    *C_rows = C_rows_tmp;
    *C_cols = C_cols_tmp;
    *C_nnz = C_nnz_tmp;

    cudaFree(dA_buffer);
    cudaFree(dAT_buffer);
    cudaFree(dC_buffer);
}