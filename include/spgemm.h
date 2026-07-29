#ifndef SPGEMM_H
#define SPGEMM_H

#include <cstdio>
#include <chrono>
#include <cstdarg>
#include <cuda_runtime.h>
#include "mempool.h"

#define TEST_READ 0
#define DBG 1
#ifndef WRITE_MTX
#define WRITE_MTX 0
#endif

// 8 字节对齐:CSR buffer [row_ptr|col_idx|val] 的 val(double)偏移须 8 对齐。
#define ALIGN8(x) (((size_t)(x) + 7) & ~(size_t)7)
#define CU_REF 1
// pinned 内存池默认开关;运行时可用环境变量 USE_MEMPOOL=0/1 覆盖(便于 A/B)
#define USE_MEMPOOL 0

// 调试日志:带启动以来毫秒时间戳写 stderr(无缓冲)。DBG=1 打印;DBG=0 时 dbg(...) 展开为 ((void)0)。
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
                       int **row_ptr_out, int **col_idx_out, double **val_out,
                       int *rows, int *cols, int *nnz);

// 写出 Matrix Market
bool write_matrix_market(const char *filename, const int *row_ptr,
                        const int *col_idx, const double *val, int rows,
                        int cols, int nnz);

// C = A × A，输入输出都是单块连续内存
void spgemm_self_product(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                         void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A^T，输入输出都是单块连续内存
void spgemm_transpose_product(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                              void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A·Aᵀ 上三角(j≥i),hash SPA 版:复用 AA hash SPA 的 sizing/binning/compact_sort。
void spgemm_att_hash(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
// C = A·Aᵀ 上三角(j≥i),【merge3 列域分桶】版:AA merge3 的忠实拷贝(B=Aᵀ/CSC + j≥i)。
void spgemm_att_merge3(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                       void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
// C = A·Aᵀ 上三角(j≥i),自适应版:同 AA Auto 分流;hash 溢出回退 att_merge3。
void spgemm_att_adaptive(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                         void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

void spgemm_transpose_product_manual(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

void spgemm_self_product_manual(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A,Gustavson 行向,串行 k-way merge 版:每行一 block 归并有序列链去重求和。与 manual(ESC)对照。
void spgemm_self_product_merge(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A,并行 k-way merge(v2)版:warp-per-row 协作归并,warp-shuffle min/sum 归约。与 merge 对照。
void spgemm_self_product_merge2(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A,分块 merge(v3)版:每行列域分 K 桶,每桶一 block warp-merge 治 straggler。与 merge2 对照。
void spgemm_self_product_merge3(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A,hash SPA 版:每行 SMEM hash 累加器 dedup+sum,排序成 CSR;溢出时 C_nnz=-1 供 dispatcher 回退 merge。
void spgemm_self_product_hash(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// C = A × A,自适应 Auto 版:按规模 n / 重行不均分流 hash 或 merge3;hash 溢出回退 merge3。
void spgemm_self_product_adaptive(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

// METHOD 环境变量单方法门控:null/empty/"all" → 跑全部;否则只跑匹配的方法块
bool should_run_method(const char *key);

// GPU 端 CSR→CSC(formulations.cu;ATT 复用:A 的 CSC = Aᵀ 的 CSR)。调用方负责 free 三个输出。
void build_csc(const int *d_row_ptr, const int *d_col_idx, const double *d_val,
               int A_rows, int A_nnz,
               int **col_ptr_out, int **row_idx_out, double **val_out);

// 三种公式对照(均 ESC 合并)。外积(outer, 外层=k):读 A 的列k ⊗ 行k
void spgemm_self_product_outer(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                               void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
// 列向(column-wise, 外层=j):读 A 的若干列(作为"内积轴"对照)
void spgemm_self_product_colwise(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                                 void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);
// 逐元素内积(inner product):C[i,j]=row_i·col_j,数值阶段逐元素归并点积(不走 ESC)
void spgemm_self_product_inner(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                               void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

#endif