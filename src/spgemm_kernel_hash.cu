#include "spgemm.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>

// ==========================================================================
//  hash SPA:C = A·A,每行一个 SMEM hash 累加器(atomicCAS 插列号 + atomicAdd 累值)
//  pipeline: count flop_ub → SMEM hash 累加+extract 去重 → compact → 全局 sort by (row,col) → CSR
//    · hash 负责 dedup+sum,sort 只排 C_nnz(远小于 flop_ub,~19× 省于 ESC 的排 flop_ub)
//    · 溢出检测:某行 distinct > HASH_CAP 时置 overflow_flag,host 返回 C_nnz=-1(dispatcher 回退)
// ==========================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

extern __global__ void count_intermediates_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int *ub);

#ifndef HASH_CAP
#define HASH_CAP 8192          // 每行 SMEM hash 表最大槽位(→ col+val = 64KB/块)
#endif
#define HASH_BLOCK 256

// 行内 SMEM hash 累加,extract 去重后的 (key=row<<32|col, val) 到 over-alloc tmp
__global__ void hash_spa_kernel(
    const int *A_row_ptr, const int *A_col_idx, const float *A_val,
    int A_rows, const int *row_off, const int *flop_ub,
    unsigned long long *tmp_key, float *tmp_val,
    int *row_nnz, int *overflow_flag)
{
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x;

    int fub = flop_ub[i];
    int ht = 64;
    while (ht < fub && ht < HASH_CAP) ht <<= 1;     // next pow2,封顶 HASH_CAP
    int mask = ht - 1;

    extern __shared__ int smem[];
    int   *sh_col = smem;                  // [ht]
    float *sh_val = (float*)(smem + ht);   // [ht]
    for (int s = tid; s < ht; s += HASH_BLOCK) { sh_col[s] = -1; sh_val[s] = 0.0f; }
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {                 // 整块顺序遍历 k
        int k = A_col_idx[p];
        float a_ik = A_val[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        for (int q = ks + tid; q < ke; q += HASH_BLOCK) {
            int j = A_col_idx[q];
            float v = a_ik * A_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;   // Knuth 乘法 hash
            int probes = 0;
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&sh_val[slot], v); break; }
                slot = (slot + 1) & mask;                            // 线性探测
                if (++probes >= ht) { atomicExch(overflow_flag, 1); break; }   // 溢出
            }
        }
    }
    __syncthreads();

    // extract 去重项 → tmp(行内无序,行间按 row_off 连续)
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    int base = row_off[i];
    for (int s = tid; s < ht; s += HASH_BLOCK) {
        int c = sh_col[s];
        if (c >= 0) {
            int pos = base + atomicAdd(&cnt, 1);
            tmp_key[pos] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[pos] = sh_val[s];
        }
    }
    __syncthreads();
    if (tid == 0) row_nnz[i] = cnt;
}

// compact:每行 row_nnz 项从 tmp(row_off 起)拷到连续 CSR(C_row_ptr 起)
__global__ void hash_compact_kernel(
    int A_rows, const int *row_off, const int *row_nnz, const int *C_row_ptr,
    const unsigned long long *tmp_key, const float *tmp_val,
    unsigned long long *out_key, float *out_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int src = row_off[i], dst = C_row_ptr[i], n = row_nnz[i];
    for (int t = 0; t < n; t++) {
        out_key[dst + t] = tmp_key[src + t];
        out_val[dst + t] = tmp_val[src + t];
    }
}

// 拆 key 低 32 位 → col
__global__ void split_key_kernel(const unsigned long long *key, int *col, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) col[t] = (int)(key[t] & 0xffffffffu);
}

// ========== Host ==========
void spgemm_self_product_hash(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    dbg("[hash] start (HASH_CAP=%d)\n", HASH_CAP);

    size_t A_rp_sz = (A_rows + 1) * sizeof(int);
    size_t A_ci_sz = (size_t)A_nnz * sizeof(int);
    size_t A_v_sz = (size_t)A_nnz * sizeof(float);
    size_t A_total = A_rp_sz + A_ci_sz + A_v_sz;
    void *dA; CHECK_CUDA(cudaMalloc(&dA, A_total));
    CHECK_CUDA(cudaMemcpy(dA, A_buffer, A_total, cudaMemcpyHostToDevice));
    char *b = (char*)dA;
    int   *dA_rp  = (int*)b;
    int   *dA_ci  = (int*)(b + A_rp_sz);
    float *dA_val = (float*)(b + A_rp_sz + A_ci_sz);
    dbg("[hash] h2d\n");

    // Stage 1: count flop_ub + scan → row_off + total_ub(输出上界)
    int *d_ub; CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    count_intermediates_kernel<<<(A_rows + 255) / 256, 256>>>(dA_rp, dA_ci, A_rows, d_ub);
    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total_ub; CHECK_CUDA(cudaMemcpy(&total_ub, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[hash] count+scan (total_ub=%d)\n", total_ub);

    // Stage 2: SMEM hash 累加 + extract
    int *d_row_nnz; CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
    int *d_overflow; CHECK_CUDA(cudaMalloc(&d_overflow, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
    unsigned long long *d_tmp_key; float *d_tmp_val;
    CHECK_CUDA(cudaMalloc(&d_tmp_key, (size_t)total_ub * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_tmp_val, (size_t)total_ub * sizeof(float)));
    size_t smem = (size_t)HASH_CAP * 2 * sizeof(int);          // col[] + val[]
    if (smem > 48 * 1024)
        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    hash_spa_kernel<<<A_rows, HASH_BLOCK, smem>>>(
        dA_rp, dA_ci, dA_val, A_rows, d_off, d_ub, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[hash] accumulate+extract\n");

    int overflow; CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (overflow) {
        fprintf(stderr, "[hash] OVERFLOW: 某行 distinct 列 > HASH_CAP=%d → 回退 merge(dispatcher 处理)\n", HASH_CAP);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        cudaFree(dA); cudaFree(d_ub); cudaFree(d_off); cudaFree(d_row_nnz);
        cudaFree(d_overflow); cudaFree(d_tmp_key); cudaFree(d_tmp_val);
        return;
    }

    // Stage 3: scan row_nnz → C_row_ptr + C_nnz(精确)
    int *dC_rp; CHECK_CUDA(cudaMalloc(&dC_rp, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_rp, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                           thrust::device_ptr<int>(d_row_nnz + A_rows),
                           thrust::device_ptr<int>(dC_rp + 1));
    int C_nnz_result; CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_rp + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[hash] scan (C_nnz=%d)\n", C_nnz_result);

    // Stage 4: compact → 连续 C_nnz 项(行内仍无序)
    unsigned long long *d_key; float *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)C_nnz_result * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)C_nnz_result * sizeof(float)));
    hash_compact_kernel<<<(A_rows + 255) / 256, 256>>>(
        A_rows, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Stage 5: 全局 sort by key(row<<32|col)→ 列有序 CSR
    thrust::sort_by_key(thrust::device_ptr<unsigned long long>(d_key),
                        thrust::device_ptr<unsigned long long>(d_key + C_nnz_result),
                        thrust::device_ptr<float>(d_val));
    dbg("[hash] sort\n");

    // Stage 6: 拆 key → C_col
    int *dC_ci; CHECK_CUDA(cudaMalloc(&dC_ci, (size_t)C_nnz_result * sizeof(int)));
    split_key_kernel<<<((size_t)C_nnz_result + 255) / 256, 256>>>(d_key, dC_ci, C_nnz_result);
    // d_val 已随 sort 排好,直接作 C_val

    // Stage 7: 打包成单块 + D2H
    size_t C_rp_sz = (A_rows + 1) * sizeof(int);
    size_t C_ci_sz = (size_t)C_nnz_result * sizeof(int);
    size_t C_v_sz = (size_t)C_nnz_result * sizeof(float);
    size_t C_rp_al = (C_rp_sz + 3) & ~3;
    size_t C_ci_al = (C_ci_sz + 3) & ~3;
    size_t C_total = C_rp_al + C_ci_al + C_v_sz;
    void *dC; CHECK_CUDA(cudaMalloc(&dC, C_total));
    char *cb = (char*)dC;
    CHECK_CUDA(cudaMemcpy(cb, dC_rp, C_rp_sz, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(cb + C_rp_al, dC_ci, C_ci_sz, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(cb + C_rp_al + C_ci_al, d_val, C_v_sz, cudaMemcpyDeviceToDevice));
    dbg("[hash] pack\n");

    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total));
    CHECK_CUDA(cudaMemcpy(C_buffer, dC, C_total, cudaMemcpyDeviceToHost));
    dbg("[hash] d2h\n");

    *C_buffer_out = C_buffer;
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = C_nnz_result;

    cudaFree(dA); cudaFree(d_ub); cudaFree(d_off); cudaFree(d_row_nnz);
    cudaFree(d_overflow); cudaFree(d_tmp_key); cudaFree(d_tmp_val);
    cudaFree(dC_rp); cudaFree(d_key); cudaFree(d_val); cudaFree(dC_ci); cudaFree(dC);
}
