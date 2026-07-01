#ifndef SPGEMM_H
#define SPGEMM_H

#include <cstdio>
#include <cuda_runtime.h>

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

void spgemm_transpose_product_manual(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

void spgemm_self_product_manual(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                        void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz);

#endif