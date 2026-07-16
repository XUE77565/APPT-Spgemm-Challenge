#ifndef SPGEMM_H
#define SPGEMM_H

#include <cstdio>
#include <chrono>
#include <cstdarg>
#include <cuda_runtime.h>
#include "mempool.h"

#define TEST_READ 0
#define DBG 1
#define WRITE_MTX 0
#define CU_REF 1
// pinned 内存池默认开关;运行时可用环境变量 USE_MEMPOOL=0/1 覆盖(便于 A/B)
#define USE_MEMPOOL 0

// 调试日志：带“程序启动以来毫秒数”时间戳，写 stderr（无缓冲，立刻可见，
// 即使被 timeout 杀掉也能看到最后一行）。每个翻译单元共享同一份 t0
// （inline 函数的 static 局部变量在 C++ 中跨 TU 唯一）。
// 由 DBG 宏控制：DBG=1 时打印；DBG=0 时 dbg(...) 展开为 ((void)0)，
// 连格式字符串都不编译进二进制。实现函数命名为 dbg_impl，再用宏 dbg(...) 转发，
// 避免宏与函数同名冲突。
#if DBG
inline void dbg(const char *fmt, ...) {
    static auto t0 = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - t0).count();
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dbg %9.3f ms] ", ms);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fflush(stderr);
}
#else
#define dbg(...) ((void)0)
#endif

// 读入 Matrix Market，返回单块连续 pinned memory
bool read_matrix_market(const char *filename, void **buffer_out,
                       int **row_ptr_out, int **col_idx_out, float **val_out,
                       int *rows, int *cols, int *nnz);

// 写出 Matrix Market
bool write_matrix_market(const char *filename, const int *row_ptr,
                        const int *col_idx, const float *val, int rows,
                        int cols, int nnz);

// C = A × A，输入输出都是单块连续内存
void spgemm_self_product(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                         void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A^T，输入输出都是单块连续内存
void spgemm_transpose_product(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                              void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A·Aᵀ 上三角(对称,只算 i≤j)。ESC 实现,返回上三角 CSR。
void spgemm_att_outer(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                      void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
void spgemm_att_gust(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
void spgemm_att_colw(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
void spgemm_att_inner(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                      void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

void spgemm_transpose_product_manual(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

void spgemm_self_product_manual(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A,Gustavson 行向,【串行 k-way merge】版:每行一个 block、thread 0 归并
// A[i,:] 各 k 贡献的有序列链,去重求和 → 替代 ESC 的 sort+reduce。与 manual(ESC)对照。
void spgemm_self_product_merge(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A,Gustavson 行向,【并行 k-way merge(v2)】版:warp-per-row 协作归并,
// 直接读 A(无 expand 阶段),warp-shuffle min/sum 归约。与 serial merge(merge)对照。
void spgemm_self_product_merge2(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// 三种公式对照(均为 ESC 合并;Gustavson=上面那个 manual)
// 外积(outer, 外层=k):读 A 的列k ⊗ 行k
void spgemm_self_product_outer(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                               void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
// 列向(column-wise, 外层=j):读 A 的若干列(作为"内积轴"对照)
void spgemm_self_product_colwise(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                                 void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
// 逐元素内积(inner product):C[i,j]=row_i·col_j,数值阶段逐元素归并点积(不走 ESC)
void spgemm_self_product_inner(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                               void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

#endif