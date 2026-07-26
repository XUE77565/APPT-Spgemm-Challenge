#include "spgemm.h"
#include "hash_prof.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ double row_dot_product(
    const int *col_idx, const double *val,
    int start_i, int end_i,
    int start_j, int end_j)
{
    double dot = 0.0f;
    int pi = start_i, pj = start_j;

    while (pi < end_i && pj < end_j) {
        int ci = col_idx[pi];
        int cj = col_idx[pj];

        if (ci == cj) {
            dot += val[pi] * val[pj];
            pi++;
            pj++;
        } else if (ci < cj) {
            pi++;
        } else {
            pj++;
        }
    }

    return dot;
}

// ========== A x A^T ==========

__global__ void count_full_nnz_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_rows,
    int *row_nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;

    int A_i_start = A_row_ptr[i];
    int A_i_end = A_row_ptr[i + 1];

    int count = 0;
    for (int j = 0; j < A_rows; j++) {
        int A_j_start = A_row_ptr[j];
        int A_j_end = A_row_ptr[j + 1];

        double dot = row_dot_product(A_col_idx, A_val,
                                   A_i_start, A_i_end,
                                   A_j_start, A_j_end);

        if (fabsf(dot) > 1e-12f) {
            count++;
        }
    }

    row_nnz[i] = count;
}

__global__ void fill_full_result_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_rows,
    const int *C_row_ptr,
    int *C_col_idx, double *C_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;

    int A_i_start = A_row_ptr[i];
    int A_i_end = A_row_ptr[i + 1];

    int C_start = C_row_ptr[i];
    int write_pos = 0;

    for (int j = 0; j < A_rows; j++) {
        int A_j_start = A_row_ptr[j];
        int A_j_end = A_row_ptr[j + 1];

        double dot = row_dot_product(A_col_idx, A_val,
                                   A_i_start, A_i_end,
                                   A_j_start, A_j_end);

        if (fabsf(dot) > 1e-12f) {
            C_col_idx[C_start + write_pos] = j;
            C_val[C_start + write_pos] = dot;
            write_pos++;
        }
    }
}

// ========== A x A (self product) —— ESC: Expand–Sort–Compress ==========
// 设计(替掉旧的 shared-hash SPA):
//   每个中间乘积 a_ik * a_kj 看成三元组 (row=i, col=j, val=a_ik*a_kj)。
//   1) count_intermediates : 便宜的符号阶段,算每行"中间项数"= Σ_{k∈A[i,:]} nnz(A[k,:]),
//      用来分配展开阶段的写偏移(无需 hash,无需原子)。
//   2) expand_intermediates : 【唯一的重计算】一行一块,把所有中间项写进全局 COO,
//      key=(row<<32)|col 便于一次排序就按(行,列)有序;每行用 shared 计数器领号。
//   3) thrust::sort_by_key  : 按 key 排序。
//   4) thrust::reduce_by_key: 相邻同 key 求和去重 → 列有序、无重复的 CSR。
// 相比旧版:中间展开只做一遍(旧版 count+fill 各一遍);全局存储无 HASH_SIZE=4096 /
// temp[256] 上限,不再丢非零;排序/去重交给 thrust,正确性有保证。

// Stage 1: 每行中间乘积数(精确,非上界:每个 a_ik 与 a_kj 各产生一条)。
__global__ void count_intermediates_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int *ub)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    long long s = 0;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1]; //A的第i行有这么多非0的
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p]; //找这些非零的对应的k有没有非零的, 进而估算出nnz数量 
        s += (long long)(A_row_ptr[k + 1] - A_row_ptr[k]);   // nnz(A 的第 k 行)
    }
    ub[i] = (int)s;
}

// Stage 2: 唯一的重计算 —— 展开所有中间项到全局 COO。
__global__ void expand_intermediates_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_rows, const int *row_off,
    unsigned long long *key, double *val)
{
    int i = blockIdx.x;
    if (i >= A_rows) return;

    __shared__ int pos;
    if (threadIdx.x == 0) pos = row_off[i];   // 本行在 COO 里的起始偏移
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs + threadIdx.x; p < re; p += blockDim.x) {
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];// 找到第k行, 去乘
        for (int q = ks; q < ke; q++) {
            int slot = atomicAdd(&pos, 1);                 // 领一个写位置(本 block 私有)
            key[slot] = ((unsigned long long)i << 32) | (unsigned int)A_col_idx[q];
            val[slot] = a_ik * A_val[q];
        }
    }
}

// Stage 4: 把压缩后的 key 拆成 col,写 C_col_idx/C_val,并统计每行 nnz。
__global__ void finalize_csr_kernel(
    const unsigned long long *red_key, const double *red_val,
    int C_nnz, int A_rows,
    int *C_col_idx, double *C_val, int *C_row_nnz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= C_nnz) return;
    unsigned long long k = red_key[t];
    C_col_idx[t] = (int)(k & 0xffffffffu);                  // 列号 = key 低 32 位
    C_val[t]     = red_val[t];
    atomicAdd(&C_row_nnz[(int)(k >> 32)], 1);               // 行号 = key 高 32 位
}

// ========== A x A^T Host ==========

void spgemm_transpose_product_manual(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(double);
    size_t A_total_size = ALIGN8(A_row_ptr_size + A_col_idx_size) + A_val_size;

    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice));

    char *dA_base = (char*)dA_buffer;
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    double *dA_val = (double*)(dA_base + ALIGN8(A_row_ptr_size + A_col_idx_size));

    int *dC_row_nnz;
    CHECK_CUDA(cudaMalloc(&dC_row_nnz, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_nnz, 0, A_rows * sizeof(int)));

    int block_size = 256;
    int grid_size = (A_rows + block_size - 1) / block_size;

    count_full_nnz_kernel<<<grid_size, block_size>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows, dC_row_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());

    int *dC_row_ptr;
    CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));

    thrust::exclusive_scan(
        thrust::device_ptr<int>(dC_row_nnz),
        thrust::device_ptr<int>(dC_row_nnz + A_rows),
        thrust::device_ptr<int>(dC_row_ptr + 1));

    int C_nnz_result;
    CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_row_ptr + A_rows,
                         sizeof(int), cudaMemcpyDeviceToHost));

    int *dC_col_idx;
    double *dC_val;
    CHECK_CUDA(cudaMalloc(&dC_col_idx, C_nnz_result * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dC_val, C_nnz_result * sizeof(double)));

    fill_full_result_kernel<<<grid_size, block_size>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows,
        dC_row_ptr, dC_col_idx, dC_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 去掉逐行排序，改成批量排序
    // 注意：这里结果已经是按列号递增的（因为 j 递增遍历）
    // 所以不需要排序！

    size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t C_col_idx_size = C_nnz_result * sizeof(int);
    size_t C_val_size = C_nnz_result * sizeof(double);
    size_t C_row_ptr_aligned = ALIGN8(C_row_ptr_size);
    size_t C_col_idx_aligned = ALIGN8(C_col_idx_size);
    size_t C_total_size = C_row_ptr_aligned + C_col_idx_aligned + C_val_size;

    void *dC_buffer;
    CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));

    char *dC_base = (char*)dC_buffer;
    CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size,
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned, dC_col_idx,
                         C_col_idx_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned + C_col_idx_aligned,
                         dC_val, C_val_size, cudaMemcpyDeviceToDevice));

    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total_size));

    CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size,
                         cudaMemcpyDeviceToHost));

    *C_buffer_out = C_buffer;
    *C_rows = A_rows;
    *C_cols = A_rows;
    *C_nnz = C_nnz_result;

    cudaFree(dA_buffer);
    cudaFree(dC_row_nnz);
    cudaFree(dC_row_ptr);
    cudaFree(dC_col_idx);
    cudaFree(dC_val);
    cudaFree(dC_buffer);
}

// ========== A x A Host (ESC) ==========

void spgemm_self_product_manual(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    dbg("[gust] start\n");
    // ---- 0. A 上传到 GPU(与旧版相同)----
    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(double);
    size_t A_total_size = ALIGN8(A_row_ptr_size + A_col_idx_size) + A_val_size;

    HashProf prof("esc-prof");
    // 把整个A拷贝,gustavson
    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    prof("h2d", [&] {
        CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice));
        dbg("[gust] h2d\n");
    });

    char *dA_base = (char*)dA_buffer;
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    double *dA_val = (double*)(dA_base + ALIGN8(A_row_ptr_size + A_col_idx_size));

    const int block = 256;

    // ---- Stage 1: 每行中间乘积数(便宜符号阶段,无 hash/原子)----
    dbg("ESC: count intermediates begin\n");
    int *d_ub; //d_ub是每一行的upper_bound
    CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    prof("count", [&] {
        int grid = (A_rows + block - 1) / block;
        // 利用每一行的nnz来粗略计算C的nnz的上界
        count_intermediates_kernel<<<grid, block>>>(
            dA_row_ptr, dA_col_idx, A_rows, d_ub);
        CHECK_CUDA(cudaDeviceSynchronize());
        dbg("[gust] count\n");
    });

    // ---- Stage 1b: 前缀和得每行写偏移; off[A_rows] = 总中间项数 ----
    int *d_off;
    CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));                 // off[0] = 0
    int total_ub;
    prof("scan", [&] {
        // inclusive_scan 写到 off+1:得 off=[0, ub0, ub0+ub1, …, total],行偏移正确
        thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                               thrust::device_ptr<int>(d_ub + A_rows),
                               thrust::device_ptr<int>(d_off + 1));
        CHECK_CUDA(cudaMemcpy(&total_ub, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
        dbg("[gust] scan\n");
    });

    // ---- Stage 2: 展开(唯一的重计算)写全局 COO(key, val)----
    unsigned long long *d_key;
    double *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total_ub * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total_ub * sizeof(double)));
    prof("expand", [&] {
        dbg("ESC: expand begin (%d intermediates)\n", total_ub);
        expand_intermediates_kernel<<<A_rows, block>>>(
            dA_row_ptr, dA_col_idx, dA_val, A_rows, d_off, d_key, d_val);
        CHECK_CUDA(cudaDeviceSynchronize());
        dbg("[gust] expand\n");
    });

    // ---- Stage 3: 按 key=(row<<32|col) 全局排序 ----
    prof("sort", [&] {
        dbg("ESC: sort_by_key begin\n");
        thrust::sort_by_key(thrust::device_ptr<unsigned long long>(d_key),
                            thrust::device_ptr<unsigned long long>(d_key + total_ub),
                            thrust::device_ptr<double>(d_val));
        CHECK_CUDA(cudaDeviceSynchronize());
        dbg("[gust] sort\n");
    });

    // ---- Stage 3b: 相邻同 key 求和去重 → (red_key, red_val) ----
    unsigned long long *d_rk;
    double *d_rv;
    CHECK_CUDA(cudaMalloc(&d_rk, (size_t)total_ub * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_rv, (size_t)total_ub * sizeof(double)));
    int C_nnz_result;
    prof("reduce", [&] {
        dbg("ESC: reduce_by_key begin\n");
        thrust::pair<thrust::device_ptr<unsigned long long>,
                     thrust::device_ptr<double> > red_end =
            thrust::reduce_by_key(
                thrust::device_ptr<unsigned long long>(d_key),
                thrust::device_ptr<unsigned long long>(d_key + total_ub),
                thrust::device_ptr<double>(d_val),
                thrust::device_ptr<unsigned long long>(d_rk),
                thrust::device_ptr<double>(d_rv));
        C_nnz_result = (int)(red_end.first - thrust::device_ptr<unsigned long long>(d_rk));
        CHECK_CUDA(cudaDeviceSynchronize());
        dbg("[gust] reduce\n");
    });
    cudaFree(d_key);          // COO 不再需要
    cudaFree(d_val);

    // ---- Stage 4: 拆 key→col 写 C_col_idx/C_val,统计每行 nnz ----
    int *dC_col_idx;
    double *dC_val;
    int *dC_row_nnz;
    CHECK_CUDA(cudaMalloc(&dC_col_idx, (size_t)C_nnz_result * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dC_val, (size_t)C_nnz_result * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dC_row_nnz, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_nnz, 0, A_rows * sizeof(int)));
    // ---- Stage 4b: 每行 nnz → C_row_ptr ----
    int *dC_row_ptr;
    CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));
    prof("final", [&] {
        int grid = (C_nnz_result + block - 1) / block;
        finalize_csr_kernel<<<grid, block>>>(
            d_rk, d_rv, C_nnz_result, A_rows, dC_col_idx, dC_val, dC_row_nnz);
        CHECK_CUDA(cudaDeviceSynchronize());
        thrust::inclusive_scan(thrust::device_ptr<int>(dC_row_nnz),
                               thrust::device_ptr<int>(dC_row_nnz + A_rows),
                               thrust::device_ptr<int>(dC_row_ptr + 1));
        dbg("[gust] final\n");
    });
    cudaFree(d_rk);
    cudaFree(d_rv);

    // ---- 打包成单块 + D2H(与旧版相同)----
    size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t C_col_idx_size = (size_t)C_nnz_result * sizeof(int);
    size_t C_val_size = (size_t)C_nnz_result * sizeof(double);
    size_t C_row_ptr_aligned = ALIGN8(C_row_ptr_size);
    size_t C_col_idx_aligned = ALIGN8(C_col_idx_size);
    size_t C_total_size = C_row_ptr_aligned + C_col_idx_aligned + C_val_size;

    void *dC_buffer;
    CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));
    char *dC_base = (char*)dC_buffer;
    prof("pack", [&] {
        CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned, dC_col_idx,
                              C_col_idx_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned + C_col_idx_aligned,
                              dC_val, C_val_size, cudaMemcpyDeviceToDevice));
        dbg("[gust] pack\n");
    });

    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total_size));
    prof("d2h", [&] {
        CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, cudaMemcpyDeviceToHost));
        dbg("[gust] d2h\n");
    });

    *C_buffer_out = C_buffer;
    *C_rows = A_rows;
    *C_cols = A_cols;
    *C_nnz = C_nnz_result;

    cudaFree(dA_buffer);
    cudaFree(d_ub);
    cudaFree(d_off);
    cudaFree(dC_col_idx);
    cudaFree(dC_val);
    cudaFree(dC_row_nnz);
    cudaFree(dC_row_ptr);
    cudaFree(dC_buffer);
}
