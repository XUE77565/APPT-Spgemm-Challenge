#include "spgemm.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

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
#define HASH_CAP 16384         // 重行桶 hash 表最大槽位(→ col+val = 128KB/块)
#endif
#define HASH_BLOCK 256
#define N_BINS 11    // 10 hash ht-buckets(ht=32<<i, i=0..9) + 1 ultra(flop≤16)

// HLL(HyperLogLog)概率基数估计:替 flop_ub 定 tmp buffer 大小
#define HLL_P 10                      // precision bits → m=1024 寄存器,误差~3%
#define HLL_M (1 << HLL_P)
#define HLL_EXPAND 2.0                // expansion(覆盖 HLL 低估;×2 → 安全)

// 按【桶】跑:blockIdx.x = 桶内行索引,实际行号 = bucket_rows[idx];ht_size = 该桶 hash 表大小(2 的幂,全 launch 统一)。
// binning:轻行桶用小表 → 高 SMEM 占用率;重行桶用大表;distinct>ht_size → overflow_flag(上层回退 merge3)。
__global__ void hash_spa_kernel(
    const int *A_row_ptr, const int *A_col_idx, const float *A_val,
    const int *bucket_rows, int n_in_bucket, int ht_size,
    const int *row_off,
    unsigned long long *tmp_key, float *tmp_val,
    int *row_nnz, int *overflow_flag)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];                  // 实际行号
    int tid = threadIdx.x;
    int mask = ht_size - 1;

    extern __shared__ int smem[];
    int   *sh_col = smem;                      // [ht_size]
    float *sh_val = (float*)(smem + ht_size);  // [ht_size]
    for (int s = tid; s < ht_size; s += HASH_BLOCK) { sh_col[s] = -1; sh_val[s] = 0.0f; }
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {                     // 整块顺序遍历 k
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
                if (++probes >= ht_size) { atomicExch(overflow_flag, 1); break; }   // 溢出
            }
        }
    }
    __syncthreads();

    // extract 去重项 → tmp(行内无序,行间按 row_off 连续)
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    int base = row_off[i];
    for (int s = tid; s < ht_size; s += HASH_BLOCK) {
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

// ultrasparse:flop ≤ ULTRA_THR。不建 hash,每线程一行,寄存器小数组线性去重累加
//   (省 hash 建表/atomic/行内排序)。对应 Ocean 的 use_ultrasparse_workflow。
__global__ void hash_ultra_kernel(
    const int *A_row_ptr, const int *A_col_idx, const float *A_val,
    const int *ultra_rows, int n_ultra,
    const int *row_off,
    unsigned long long *tmp_key, float *tmp_val, int *row_nnz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ultra) return;
    int i = ultra_rows[idx];
    const int CAP = 16;                          // flop≤16 → distinct≤16
    int u_col[CAP]; float u_val[CAP]; int u_cnt = 0;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p]; float a_ik = A_val[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        for (int q = ks; q < ke; q++) {
            int j = A_col_idx[q]; float v = a_ik * A_val[q];
            int pos = -1;
            for (int t = 0; t < u_cnt; t++) if (u_col[t] == j) { pos = t; break; }
            if (pos >= 0) u_val[pos] += v;
            else { u_col[u_cnt] = j; u_val[u_cnt] = v; u_cnt++; }
        }
    }
    int base = row_off[i];
    for (int t = 0; t < u_cnt; t++) {
        tmp_key[base + t] = ((unsigned long long)i << 32) | (unsigned int)u_col[t];
        tmp_val[base + t] = u_val[t];
    }
    row_nnz[i] = u_cnt;
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

// ========== HLL 估计(Ocean 风格:替 flop_ub 收紧 tmp buffer)==========
// 每行一个 block,SMEM=m=1024 个 uint32 寄存器。
// 遍历行 i 的所有中间积(列 j)→ hash(j) → atomicMax 寄存器 → thread 0 算 HLL 公式。
__global__ void hll_estimate_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows,
    int *est_nnz)
{
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x;

    extern __shared__ unsigned int regs[];          // [HLL_M], init 0
    for (int j = tid; j < HLL_M; j += blockDim.x) regs[j] = 0;
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {                 // 顺序遍历 k(同 hash_spa_kernel)
        int k = A_col_idx[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        for (int q = ks + tid; q < ke; q += blockDim.x) {
            unsigned int h = (unsigned int)(A_col_idx[q] * 2654435761u);
            int idx = h >> (32 - HLL_P);
            unsigned int rest = h << HLL_P;
            int rho = (rest == 0) ? (33 - HLL_P) : (__clz(rest) + 1);
            atomicMax(&regs[idx], (unsigned int)rho);
        }
    }
    __syncthreads();

    if (tid == 0) {
        double sum = 0.0;
        int zeros = 0;
        for (int j = 0; j < HLL_M; j++) {
            sum += 1.0 / (double)(1ULL << regs[j]);
            if (regs[j] == 0) zeros++;
        }
        double alpha = 0.7213 / (1.0 + 1.079 / (double)HLL_M);
        double E = alpha * (double)HLL_M * (double)HLL_M / sum;
        if (E <= 2.5 * HLL_M && zeros > 0)          // 小范围修正(linear counting)
            E = (double)HLL_M * log((double)HLL_M / (double)zeros);
        int est = (int)(E * HLL_EXPAND);
        if (est < 1) est = 1;
        est_nnz[i] = est;
    }
}

// ========== GPU 端分桶(Ocean 风格:无 host 往返)==========
// 每行算 bucket_id:flop≤ULTRA_THR → ultra(bin 10);否则 ht=next_pow2(flop) → bin 0..9
__global__ void compute_bucket_kernel(
    const int *flop_ub, int A_rows, int ultra_thr, int *bucket_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int f = flop_ub[i];
    if (f <= ultra_thr) { bucket_id[i] = N_BINS - 1; return; }
    int bi = 0, ht = 32;
    while (ht < f && ht < HASH_CAP) { ht <<= 1; bi++; }
    bucket_id[i] = (bi < N_BINS - 1) ? bi : (N_BINS - 2);
}

// 直方图:每 bin 的行数(atomicAdd 到 N_BINS 个计数器)
__global__ void bucket_count_kernel(
    const int *bucket_id, int A_rows, int *counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    atomicAdd(&counts[bucket_id[i]], 1);
}

// scatter:把行号按 bucket 写到预分配大 buffer 的对应位置
__global__ void scatter_rows_kernel(
    const int *bucket_id, int A_rows,
    const int *offsets, int *pos, int *sorted_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int bid = bucket_id[i];
    int slot = offsets[bid] + atomicAdd(&pos[bid], 1);
    sorted_rows[slot] = i;
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
    dbg("[hash] count\n");
    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total_ub; CHECK_CUDA(cudaMemcpy(&total_ub, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[hash] scan\n");

    // HLL 估计 → 收紧 row_off(替 flop_ub):tmp buffer 从 total_ub(19× 过估)降到 ~2×C_nnz
    int *d_est; CHECK_CUDA(cudaMalloc(&d_est, A_rows * sizeof(int)));
    hll_estimate_kernel<<<A_rows, HASH_BLOCK, (size_t)HLL_M * sizeof(unsigned int)>>>(
        dA_rp, dA_ci, A_rows, d_est);
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_est),
                           thrust::device_ptr<int>(d_est + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total_est; CHECK_CUDA(cudaMemcpy(&total_est, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[hash] hll (est=%d vs ub=%d, %.1fx tighter)\n", total_est, total_ub,
        total_ub > 0 ? (double)total_ub / total_est : 0.0);

    // Stage 2: GPU 端分桶(Ocean 风格:全 device,无 host 往返)+ 预分配大 buffer(零 per-bucket malloc/free)
    int *d_row_nnz; CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
    int *d_overflow; CHECK_CUDA(cudaMalloc(&d_overflow, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
    unsigned long long *d_tmp_key; float *d_tmp_val;
    CHECK_CUDA(cudaMalloc(&d_tmp_key, (size_t)total_est * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_tmp_val, (size_t)total_est * sizeof(float)));
    // 预分配 compact 输出(与 tmp 同大小,免 runtime cudaMalloc)
    unsigned long long *d_key; float *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total_est * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total_est * sizeof(float)));

    // 2a: GPU 端分桶:bucket_id → count → exclusive scan → scatter(全 device)
    const int ULTRA_THR = 16;
    int *d_bkid; CHECK_CUDA(cudaMalloc(&d_bkid, A_rows * sizeof(int)));
    int *d_cnt;  CHECK_CUDA(cudaMalloc(&d_cnt,  N_BINS * sizeof(int)));
    int *d_offb; CHECK_CUDA(cudaMalloc(&d_offb, N_BINS * sizeof(int)));
    int *d_pos;  CHECK_CUDA(cudaMalloc(&d_pos,  N_BINS * sizeof(int)));
    int *d_sort; CHECK_CUDA(cudaMalloc(&d_sort, A_rows * sizeof(int)));   // 预分配:桶有序行号
    compute_bucket_kernel<<<(A_rows + 255) / 256, 256>>>(d_ub, A_rows, ULTRA_THR, d_bkid);
    CHECK_CUDA(cudaMemset(d_cnt, 0, N_BINS * sizeof(int)));
    bucket_count_kernel<<<(A_rows + 255) / 256, 256>>>(d_bkid, A_rows, d_cnt);
    thrust::exclusive_scan(thrust::device_ptr<int>(d_cnt),
                           thrust::device_ptr<int>(d_cnt + N_BINS),
                           thrust::device_ptr<int>(d_offb));
    CHECK_CUDA(cudaMemset(d_pos, 0, N_BINS * sizeof(int)));   // scatter 计数器从 0 起(非 offsets)
    scatter_rows_kernel<<<(A_rows + 255) / 256, 256>>>(d_bkid, A_rows, d_offb, d_pos, d_sort);

    // D2H counts + offsets(11 ints 各,trivial)
    int h_cnt[N_BINS], h_off[N_BINS];
    CHECK_CUDA(cudaMemcpy(h_cnt,  d_cnt,  N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_off,  d_offb, N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[hash] accumulate (GPU-binned)\n");

    // 2b: opt-in max SMEM
    {
        size_t maxsm = (size_t)HASH_CAP * 2 * sizeof(int);
        if (maxsm > 48 * 1024)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
    }
    // 2c: per-bin launch,预分配 d_sort + h_off[bi] 偏移(零 per-bucket malloc/free)
    for (int bi = 0; bi < N_BINS; bi++) {
        int n = h_cnt[bi];
        if (n == 0) continue;
        int *rows_ptr = d_sort + h_off[bi];
        if (bi == N_BINS - 1) {
            // ultra(flop≤16):线性,免 hash
            hash_ultra_kernel<<<(n + 255) / 256, 256>>>(
                dA_rp, dA_ci, dA_val, rows_ptr, n, d_off, d_tmp_key, d_tmp_val, d_row_nnz);
        } else {
            int ht = 32 << bi;
            size_t smem = (size_t)ht * 2 * sizeof(int);
            hash_spa_kernel<<<n, HASH_BLOCK, smem>>>(
                dA_rp, dA_ci, dA_val, rows_ptr, n, ht, d_off,
                d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    int overflow; CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (overflow) {
        fprintf(stderr, "[hash] OVERFLOW: 某行 distinct 列 > HASH_CAP=%d → 回退 merge(dispatcher 处理)\n", HASH_CAP);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        cudaFree(dA); cudaFree(d_ub); cudaFree(d_off); cudaFree(d_row_nnz);
        cudaFree(d_overflow); cudaFree(d_tmp_key); cudaFree(d_tmp_val);
        cudaFree(d_bkid); cudaFree(d_cnt); cudaFree(d_offb); cudaFree(d_pos); cudaFree(d_sort); cudaFree(d_est);
        cudaFree(d_key); cudaFree(d_val);
        return;
    }

    // Stage 3: scan row_nnz → C_row_ptr + C_nnz(精确)
    int *dC_rp; CHECK_CUDA(cudaMalloc(&dC_rp, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_rp, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                           thrust::device_ptr<int>(d_row_nnz + A_rows),
                           thrust::device_ptr<int>(dC_rp + 1));
    int C_nnz_result; CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_rp + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[hash] cnnz (C_nnz=%d)\n", C_nnz_result);

    // HLL underflow check: actual C_nnz > estimated total → 回退 merge3
    if (C_nnz_result > total_est) {
        fprintf(stderr, "[hash] HLL underflow: C_nnz=%d > total_est=%d → 回退 merge\n", C_nnz_result, total_est);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        cudaFree(dA); cudaFree(d_ub); cudaFree(d_off); cudaFree(d_row_nnz);
        cudaFree(d_overflow); cudaFree(d_tmp_key); cudaFree(d_tmp_val);
        cudaFree(d_bkid); cudaFree(d_cnt); cudaFree(d_offb); cudaFree(d_pos); cudaFree(d_sort); cudaFree(d_est);
        cudaFree(d_key); cudaFree(d_val); cudaFree(dC_rp);
        return;
    }

    // Stage 4: compact → d_key/d_val(预分配,免 runtime cudaMalloc)
    hash_compact_kernel<<<(A_rows + 255) / 256, 256>>>(
        A_rows, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[hash] compact\n");

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
