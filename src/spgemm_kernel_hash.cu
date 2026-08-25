#include "spgemm.h"
#include "hash_prof.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// hash SPA:C = A·A,SMEM hash 累加器(atomicCAS 插列+atomicAdd 累值);pipeline count→累加→compact→sort;overflow→回退

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

extern __global__ void count_intermediates_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int *ub);

// 并行 count flop_ub:warp-per-row(2026-08-25 重映射;旧版每行一 block,3.7M 行矩阵 launch 5.4ms → 此版 O(rows/8) blocks)
__global__ void count_intermediates_par_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int *ub)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int i = warp;
    if (i >= A_rows) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int s = 0;
    for (int p = rs + lane; p < re; p += 32) {
        int k = A_col_idx[p];
        s += A_row_ptr[k + 1] - A_row_ptr[k];
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
    if (lane == 0) ub[i] = s;
}

#ifndef HASH_CAP
#define HASH_CAP 16384         // 重行桶 hash 表最大槽位(→ col+val = 128KB/块)
#endif
#define HASH_BLOCK 256
#define N_BINS 12    // 10 hash ht-buckets(ht=32<<i, i=0..9) + ultra(10,flop≤16) + heavy(11,est>HASH_CAP 走全局表)
#define GLOBAL_HT_MAX_SLOTS (1 << 22)   // 单行全局表上限 4M 槽(50MB);est 超此 → 真溢出回退
#define AVG_FLOP_THR 64                  // avg_product ≤ 此值 → est=精确 flop 免 MinHash(Ocean Ana1 同款门)
#define CSORT_HT 1024   // ht≤此值的行(bin0-5)在 accumulate 内 count-sort 写有序;compact 阶段只 copy

// MinHash 概率基数估计:替 flop_ub 定 tmp buffer + hash 表大小
#define MH_M 128                      // partition 数(=128)
#define EST_EXPAND 1.15               // expansion(estimator 已无偏后收紧;原 1.5×2×snap×pow2≈4.2× 结构性膨胀)
#define EST_ULTRA_THR 16              // bin-snap:est≤此值 → ultrasparse(线性 kernel,CAP=32 留 2× 余量)
#define STREAMLINE_NNZ 100000         // 小阵 pipeline 精简:A_nnz<此值 → flop_ub(1 count kernel)替 MinHash 两阶段
#define WARP_SIZE 32

// MurmurHash3(full mix body+fmix32);MinHash 两阶段复用,SPA 累加暂仍用 Knuth 乘法
__device__ __forceinline__ unsigned int rotl32(unsigned int x, int r) { return (x << r) | (x >> (32 - r)); }
__device__ __forceinline__ unsigned int murmur_hash3(unsigned int key) {
    unsigned int h1 = 1234u;                       // MURMUR3_SEED
    unsigned int c1 = 0xcc9e2d51u, c2 = 0x1b873593u;
    unsigned int k1 = key;                         // index_t = uint32 → 单 block,len=4
    k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2;
    h1 ^= k1; h1 = rotl32(h1, 13); h1 = h1 * 5u + 0xe6546b64u;
    h1 ^= 4u;                                      // ^= len
    h1 ^= h1 >> 16; h1 *= 0x85ebca6bu;             // fmix32
    h1 ^= h1 >> 13; h1 *= 0xc2b2ae35u;
    h1 ^= h1 >> 16;
    return h1;
}

// MinHash 概率基数估计(partition bottom-k sketch)
#define MH_EMPTY 0xffffffffu

// Phase 1:对 B(=A)每行建 MinHash sketch(线性扫 CSR,O(nnz))。每元素 atomicMin keep 最小完整 hash(uint32)。
__global__ void mh_construct_kernel(
    const int *B_row_ptr, const int *B_col_ind,
    int B_rows, int B_nnz,
    unsigned int *global_mh,           // [B_rows * MH_M] uint32
    int rows_per_block)
{
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, B_rows);
    if (row_start >= B_rows) return;

    int elem_start = B_row_ptr[row_start];
    int elem_end = (row_end < B_rows) ? B_row_ptr[row_end] : B_nnz;
    int num_rows = row_end - row_start;

    extern __shared__ int smem_raw[];
    unsigned int *smem = (unsigned int*)smem_raw;          // [num_rows * MH_M] uint32 scratch(atomicMin 用)
    int total_items = num_rows * MH_M;
    for (int i = threadIdx.x; i < total_items; i += blockDim.x) smem[i] = MH_EMPTY;   // init 空(非 0)
    int *row_offsets = smem_raw + total_items;
    for (int i = threadIdx.x; i <= num_rows; i += blockDim.x)
        row_offsets[i] = (row_start + i < B_rows) ? B_row_ptr[row_start + i] : elem_end;
    __syncthreads();

    int current_row = 0;
    unsigned int *current_scratch = smem;
    for (int e = elem_start + threadIdx.x; e < elem_end; e += blockDim.x) {
        unsigned int col = (unsigned int)B_col_ind[e];
        while (e >= row_offsets[current_row + 1]) {
            current_row++;
            current_scratch = smem + current_row * MH_M;
        }
        unsigned int h = murmur_hash3(col);
        int idx = h & (MH_M - 1);                  // 低 log2(MH_M) 位 → partition
        atomicMin(&current_scratch[idx], h);       // MIN 完整 hash(keep 最小值)
    }
    __syncthreads();

    unsigned int *global_ptr = global_mh + row_start * MH_M;
    for (int i = threadIdx.x; i < num_rows * MH_M; i += blockDim.x) global_ptr[i] = smem[i];
}

// Phase 2:对 A 每行逐 partition min 合并(单 warp,vectorized uint4 merge)
__global__ void mh_merge_kernel(
    const int *A_row_ptr, const int *A_col_ind,
    int A_rows,
    const unsigned int *b_mh,           // [B_rows * MH_M] uint32 from Phase 1
    const int *row_flop,                // [A_rows] 每行精确乘积数(精确上界,封顶 MinHash 高估)
    int *est_nnz)                       // [A_rows] output
{
    int row = blockIdx.x;
    if (row >= A_rows) return;
    int tid = threadIdx.x;

    extern __shared__ int smem_merge_raw[];
    unsigned int *smem_merge = (unsigned int*)smem_merge_raw;   // [MH_M] uint32
    int base_i = tid * 4;                                       // 每线程 owning partitions [base_i..base_i+3]
    smem_merge[base_i + 0] = MH_EMPTY;
    smem_merge[base_i + 1] = MH_EMPTY;
    smem_merge[base_i + 2] = MH_EMPTY;
    smem_merge[base_i + 3] = MH_EMPTY;
    __syncthreads();

    int start_elem = A_row_ptr[row];
    int end_elem = A_row_ptr[row + 1];
    // 逐引用 B 行,vectorized uint4 min-merge(每线程固定 owning 4 个 partition → 无 race、无需内 sync)
    for (int e = start_elem; e < end_elem; e++) {
        int row_b = A_col_ind[e];
        const uint4 *bsk4 = (const uint4*)(b_mh + (size_t)row_b * MH_M);   // 行首 16B 对齐(512B/行)
        uint4 v = bsk4[tid];                                               // 16B coalesced
        unsigned int *sm = smem_merge + base_i;
        if (v.x < sm[0]) sm[0] = v.x;
        if (v.y < sm[1]) sm[1] = v.y;
        if (v.z < sm[2]) sm[2] = v.z;
        if (v.w < sm[3]) sm[3] = v.w;
    }
    __syncthreads();

    // 估计 reduce(单 warp):sum_min = Σ min_j(非空);V = 非空 partition 数
    double sum_min = 0.0;
    int V = 0;
    for (int k = 0; k < 4; k++) {
        unsigned int mv = smem_merge[base_i + k];
        if (mv != MH_EMPTY) { sum_min += (double)mv; V++; }
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_min += __shfl_down_sync(0xFFFFFFFF, sum_min, off);
        V += __shfl_down_sync(0xFFFFFFFF, V, off);
    }

    if (tid == 0) {
        double E;
        int v = V;
        if (v == 0) {
            E = 0.0;                                                      // 空行
        } else if (v < MH_M) {
            E = (double)MH_M * log((double)MH_M / (double)(MH_M - v));   // linear counting(小范围)
        } else {
            // min_j/2^32 ~ Exp(1/(D/m)) ⇒ E[min_j] = m·2^32/D ⇒ D ≈ m²·2^32/Σmin_j(算术均值,无偏)。
            // 旧式 2^32·Σ(1/min_j) 是调和均值,Jensen 高估 ~ln 倍(exdata_1 17.85×/TSOPF 20× 根因);
            // 首版修正漏了因子 m(低 EST 128 倍,exdata_1 est 254k vs C 11.3M)——都已修(2026-08-26)。
            E = (double)MH_M * MH_M * 0x100000000LL / sum_min;
        }
        int temp = (int)(E * EST_EXPAND);
        if (temp < 1) temp = 1;
        int est;
        if (temp <= EST_ULTRA_THR) {                                      // ultra:线性 kernel,est 保留紧 temp
            est = temp;
        } else {                                                          // snap 到 next_pow2 × 2 余量
            int ht = 32;                                                  // (2× 余量治小/中行低估表满;>HASH_CAP 交 heavy)
            while (ht < temp && ht < GLOBAL_HT_MAX_SLOTS) ht <<= 1;
            est = ht;
        }
        // flop = 每行乘积数 = distinct 的【精确上界】→ 封顶 MinHash 高估(TSOPF:Σest 从 147亿 → ≤Σflop,
        // 兼治 int 前缀和溢出)。min() 不引入新低估:flop ≥ distinct 恒真,低估风险仅剩原 2×snap 自身。
        int fl = row_flop[row];
        if (fl > 0 && est > fl) est = fl;
        est_nnz[row] = est;
    }
}

// ===================== 小行批量 kernel(2026-08-26,Phase 2 主杠杆) =====================
// est ≤ 64 的行(bin0-1):warp-per-row + SMEM warp 私有表(128 槽,2× 余量)+ 融合有序 extract。
// 治"1 CTA/行×256 线程伺候 ~6 个积"的每行固定开销(333SP 型 accumulate 55% 差距源);
// 有序直写 tmp → compact 阶段走 copy 免排序。设计:`inno/hash_batched_kernel_design.md`。
#define BATCH_HT 128        // per-warp 表槽(est≤64 → 2× 余量)
#define BATCH_WPB 8         // warp 数/CTA(256 线程)

__global__ void hash_spa_batched_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_in_bucket) return;
    int i = bucket_rows[warp];

    // 静态 SMEM:每 warp 段 = col[128] | val[128](8B 对齐,段首 512B✓) | pack_col[128] | pack_slot[128]
    __shared__ int bsmem[BATCH_WPB][4 * BATCH_HT];
    __shared__ int w_cnt[BATCH_WPB];
    int *t_col = bsmem[warp & (BATCH_WPB - 1)];
    double *t_val = (double*)(bsmem[warp & (BATCH_WPB - 1)] + BATCH_HT);
    int *pack_col = bsmem[warp & (BATCH_WPB - 1)] + 2 * BATCH_HT;
    int *pack_slot = bsmem[warp & (BATCH_WPB - 1)] + 3 * BATCH_HT;

    if (lane == 0) w_cnt[warp & (BATCH_WPB - 1)] = 0;
    for (int s = lane; s < BATCH_HT; s += 32) { t_col[s] = -1; t_val[s] = 0.0; }
    __syncwarp();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {                     // k 串行(小行 k 少)
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {      // j 32 路并行
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & (BATCH_HT - 1);
            int probes = 0;                             // 有界探测:必终止
            while (true) {
                int old = atomicCAS(&t_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&t_val[slot], v); break; }
                slot = (slot + 1) & (BATCH_HT - 1);
                if (++probes >= BATCH_HT) {
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
                    break;
                }
            }
        }
    }
    __syncwarp();

    // extract:pack 非空槽 → count-rank 排序 → 有序写 tmp(rank 是 0..n-1 置换)
    long long base = row_off[i];
    int cap = est[i];                                   // SAFETY:mh 低估时 n 可超 est → 守卫
    for (int s = lane; s < BATCH_HT; s += 32) {
        int c = t_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&w_cnt[warp & (BATCH_WPB - 1)], 1);
            if (pos < BATCH_HT) { pack_col[pos] = c; pack_slot[pos] = s; }
        }
    }
    __syncwarp();
    int n = w_cnt[warp & (BATCH_WPB - 1)];
    if (n > BATCH_HT) n = BATCH_HT;                     // pack 溢出截断(此时必已置 overflow)
    for (int idx = lane; idx < n; idx += 32) {
        int c = pack_col[idx];
        int rank = 0;
        for (int m = 0; m < n; m++) if (pack_col[m] < c) rank++;
        if (rank >= cap) { atomicExch(overflow_flag, 1); continue; }
        tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
        tmp_val[base + rank] = t_val[pack_slot[idx]];
    }
    __syncwarp();
    if (lane == 0) {
        if (n > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
        row_nnz[i] = (n <= cap) ? n : cap;
    }
}

// 按桶跑:ht_size = 该桶 hash 表大小;轻行小表、重行大表,distinct>ht_size → overflow_flag
__global__ void hash_spa_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,   // 内层 j 来源:AA=A,ATT=Aᵀ(CSC)
    int upper_tri,                                                      // 0=自乘;1=ATT 只算 j≥i
    const int *bucket_rows, int n_in_bucket, int ht_size,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];                  // 实际行号
    int tid = threadIdx.x;
    int mask = ht_size - 1;

    extern __shared__ __align__(8) int smem[];
    int   *sh_col = smem;                      // [ht_size]
    double *sh_val = (double*)(smem + ht_size);  // [ht_size]
    for (int s = tid; s < ht_size; s += HASH_BLOCK) { sh_col[s] = -1; sh_val[s] = 0.0f; }
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    // G 从 ht_size 推:重行→G 大,轻行→G 小(更多 k 并行);G ∈ {4,8,16,32}
    int G = (ht_size <= 64) ? 4 : (ht_size <= 256) ? 8 : (ht_size <= 1024) ? 16 : 32;
    int num_groups = HASH_BLOCK / G;
    int my_group = tid / G, my_id = tid % G;
    for (int p = rs + my_group; p < re; p += num_groups) {   // 并行 k(stride num_groups)
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];        // B 的行 k(AA=A;ATT=Aᵀ)
        for (int q = ks + my_id; q < ke; q += G) {           // 组内 G 线程并行 j
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;                // ATT 上三角过滤(j≥i)
            double v = a_ik * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;   // Knuth 乘法 hash
            int probes = 0;
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&sh_val[slot], v); break; }
                slot = (slot + 1) & mask;                            // 线性探测
                if (++probes >= ht_size) {                          // 溢出 → 记录行交重试
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
                    break;
                }
            }
        }
    }
    __syncthreads();

    // extract 去重项 → tmp:小行(ht≤CSORT_HT)SMEM 内 compact+count-sort 写有序,大行无序写(交 compact_sort)
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    if (ht_size <= CSORT_HT) {
        // compact 到前部:chunked 读进寄存器→sync→写前部(read-before-write 防 race)
        for (int it = 0; it < (ht_size + HASH_BLOCK - 1) / HASH_BLOCK; it++) {
            int s = it * HASH_BLOCK + tid;
            int c = -1; double v = 0.0f;
            if (s < ht_size) { c = sh_col[s]; v = sh_val[s]; }
            __syncthreads();
            if (c >= 0) { int pos = atomicAdd(&cnt, 1); sh_col[pos] = c; sh_val[pos] = v; }
            __syncthreads();
        }
        int count = cnt;
        // SAFETY:行槽位 cap = est(= row_off 差);欠估行(distinct>est)越界写会砸下一行 → 守卫+overflow
        long long cap = row_off[i + 1] - row_off[i];
        for (int k = tid; k < count; k += HASH_BLOCK) {      // count-sort:数 < 自己的 → rank → 落有序位
            int c = sh_col[k]; double v = sh_val[k]; int rank = 0;
            for (int j = 0; j < count; j++) if (sh_col[j] < c) rank++;
            if (rank >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + rank] = v;
        }
        __syncthreads();
        if (tid == 0) {
            if (count > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
            row_nnz[i] = (count <= cap) ? count : (int)cap;
        }
    } else {
        // 大行:无序 extract(交 compact_sort;SAFETY:cap=est 守卫同上)
        long long cap = row_off[i + 1] - row_off[i];
        for (int s = tid; s < ht_size; s += HASH_BLOCK) {
            int c = sh_col[s];
            if (c >= 0) {
                int pos = atomicAdd(&cnt, 1);
                if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
                tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
                tmp_val[base + pos] = sh_val[s];
            }
        }
        __syncthreads();
        if (tid == 0) {
            if (cnt > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
            row_nnz[i] = (cnt <= cap) ? cnt : (int)cap;
        }
    }
}

// hash_spa_priv_kernel:warp 私有 SPA(HASH_PRIV 门控):k 按 warp 分区→Phase A 无 atomicAdd 累加,Phase B 跨 warp merge,Phase C extract
__global__ void hash_spa_priv_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *bucket_rows, int n_in_bucket, int ht_size, int W,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int warp = tid >> 5, lane = tid & 31;                  // blockDim.x = W*32,warp ∈ [0,W)
    int mask = ht_size - 1;

    extern __shared__ __align__(8) int smem[];
    int   *sh_col = smem;                                  // [W*ht]
    double *sh_val = (double*)(smem + (size_t)W * ht_size);// [W*ht]
    int   *my_col = sh_col + (size_t)warp * ht_size;
    double *my_val = sh_val + (size_t)warp * ht_size;

    for (int s = lane; s < ht_size; s += 32) { my_col[s] = -1; my_val[s] = 0.0; }
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];

    // Phase A:warp 私有累加(k 按 warp 分区;plain +=,无 atomicAdd)
    for (int p = rs + warp; p < re; p += W) {              // 本 warp 的 k(stride W)
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {         // warp 内 32 lane 并行 j
            int j = A_col_idx[q];
            double v = a_ik * A_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&my_col[slot], -1, j);
                if (old == -1 || old == j) { my_val[slot] += v; break; }   // plain +=(同 warp 时间不并发)
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) { atomicExch(overflow_flag, 1); break; }
            }
        }
    }
    __syncthreads();

    // Phase B:merge warp 1..W-1 → warp 0 表(共享,atomicAdd,fan-in ≤ W)
    if (warp > 0) {
        for (int s = lane; s < ht_size; s += 32) {
            int c = my_col[s];
            if (c >= 0) {
                double v = my_val[s];
                unsigned slot = ((unsigned)(c * 2654435761u)) & mask;
                int probes = 0;
                while (true) {
                    int old = atomicCAS(&sh_col[slot], -1, c);
                    if (old == -1 || old == c) { atomicAdd(&sh_val[slot], v); break; }
                    slot = (slot + 1) & mask;
                    if (++probes >= ht_size) { atomicExch(overflow_flag, 1); break; }
                }
            }
        }
    }
    __syncthreads();

    // Phase C:extract warp 0 表 → tmp(同 flat)
    int *w0_col = sh_col; double *w0_val = sh_val;
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    if (ht_size <= CSORT_HT) {
        for (int it = 0; it < (ht_size + blockDim.x - 1) / blockDim.x; it++) {
            int s = it * blockDim.x + tid;
            int c = -1; double v = 0.0;
            if (s < ht_size) { c = w0_col[s]; v = w0_val[s]; }
            __syncthreads();
            if (c >= 0) { int pos = atomicAdd(&cnt, 1); sh_col[pos] = c; sh_val[pos] = v; }
            __syncthreads();
        }
        int count = cnt;
        for (int k = tid; k < count; k += blockDim.x) {    // count-sort:数 < 自己的 → rank → 落有序位
            int c = sh_col[k]; double v = sh_val[k]; int rank = 0;
            for (int j = 0; j < count; j++) if (sh_col[j] < c) rank++;
            tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + rank] = v;
        }
        __syncthreads();
        if (tid == 0) row_nnz[i] = count;
    } else {
        for (int s = tid; s < ht_size; s += blockDim.x) {
            int c = w0_col[s];
            if (c >= 0) {
                int pos = base + atomicAdd(&cnt, 1);
                tmp_key[pos] = ((unsigned long long)i << 32) | (unsigned int)c;
                tmp_val[pos] = w0_val[s];
            }
        }
        __syncthreads();
        if (tid == 0) row_nnz[i] = cnt;
    }
}

// ultrasparse:est≤EST_ULTRA_THR,不建 hash,每线程一行寄存器小数组线性去重累加;CAP=32 留 2× 余量,不够→overflow_flag
__global__ void hash_ultra_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *ultra_rows, int n_ultra,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ultra) return;
    int i = ultra_rows[idx];
    const int CAP = 32;                          // 2× 于 EST_ULTRA_THR,吸收估计低估
    int u_col[CAP]; double u_val[CAP]; int u_cnt = 0;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p]; double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];          // B 的行 k(AA=A;ATT=Aᵀ)
        for (int q = ks; q < ke; q++) {
            int j = B_col_idx[q]; double v = a_ik * B_val[q];
            if (upper_tri && j < i) continue;                   // ATT 上三角过滤
            int pos = -1;
            for (int t = 0; t < u_cnt; t++) if (u_col[t] == j) { pos = t; break; }
            if (pos >= 0) u_val[pos] += v;
            else {
                if (u_cnt >= CAP) {                                    // 超 CAP → 记录行交重试
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
                    return;
                }
                u_col[u_cnt] = j; u_val[u_cnt] = v; u_cnt++;
            }
        }
    }
    long long base = row_off[i];
    long long cap = row_off[i + 1] - row_off[i];   // SAFETY:est 槽位守卫(欠估行防越界砸下一行)
    int out = 0;
    for (int t = 0; t < u_cnt; t++) {
        if (out >= cap) {                                    // 欠估 → 记录行交重试
            atomicExch(overflow_flag, 1);
            if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
            break;
        }
        tmp_key[base + out] = ((unsigned long long)i << 32) | (unsigned int)u_col[t];
        tmp_val[base + out] = u_val[t];
        out++;
    }
    if (out < u_cnt) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
    row_nnz[i] = out;
}

// ===================== heavy 全局表路径(2026-08-25) =====================
// est > HASH_CAP 的重行不再回退 merge3(ocean337 上 67 阵 28.5× 灾难的主源),
// 改走 global-memory 开放寻址 hash 表(对标 Ocean numeric.overflow 的全局回退)。
// 每行一张 next_pow2(est) 表(≥2×HASH_CAP),表区 arena 按 scan 偏移切分;
// probe 有上界(必终止);extract 无序写 tmp → host 侧 cub 分段排序 → compact_copy。

__global__ void heavy_prep_kernel(
    const int *rows, int n, const int *est,
    long long *row_ht, int *overflow_flag)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int e = est[rows[t]];
    if (e > GLOBAL_HT_MAX_SLOTS) {            // 超过单表上限:真溢出,走原回退语义
        row_ht[t] = 0;
        atomicExch(overflow_flag, 1);
        if (t == 0 || (t & 255) == 0)        return;
    }
    long long h = 2 * HASH_CAP;               // 最小 32768
    while (h < 2LL * e) h <<= 1;              // 2× 余量:est 低估(MinHash 误差)时不炸
    row_ht[t] = h;
}

__global__ void hash_global_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_ht, const long long *tab_off,
    int *tab_col, double *tab_val,
    const long long *row_off, const int *est, long long tmp_total,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int ht_size = (int)row_ht[idx];
    int mask = ht_size - 1;
    int *sh_col = tab_col + tab_off[idx];      // 本行的全局表段
    double *sh_val = tab_val + tab_off[idx];
    for (int s = tid; s < ht_size; s += HASH_BLOCK) { sh_col[s] = -1; sh_val[s] = 0.0f; }
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int G = 32, num_groups = HASH_BLOCK / G, my_group = tid / G, my_id = tid % G;
    for (int p = rs + my_group; p < re; p += num_groups) {
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + my_id; q < ke; q += G) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a_ik * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;                    // 有界探测:必终止
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&sh_val[slot], v); break; }
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) {
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; }
                    break;
                }
            }
        }
    }
    __syncthreads();

    // 无序 extract → tmp(host 侧 cub 分段排序后交 compact_copy)
    // SAFETY:行槽位 = est[i](gapped buffer 按 est 分配);真实 distinct > est(MinHash 低估)时
    // 越界写会砸别的行/数组尾 → 守卫:超 est 只置 overflow 标志(整阵安全回退),绝不越界。
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    int cap = est[i];
    for (int s = tid; s < ht_size; s += HASH_BLOCK) {
        int c = sh_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&cnt, 1);
            if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + pos] = sh_val[s];
        }
    }
    __syncthreads();
    if (tid == 0) {
               row_nnz[i] = cnt > cap ? cap : cnt;
    }
}

// heavy 行分段排序的 begin/end 偏移(基于 gapped d_off + 真实 row_nnz)
__global__ void heavy_seg_kernel(
    const int *rows, int n, const long long *row_off, const int *row_nnz,
    int *seg_beg, int *seg_end)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int i = rows[t];
    seg_beg[t] = row_off[i];
    seg_end[t] = row_off[i] + row_nnz[i];
}

// ---- 行级重试 kernels(2026-08-26) ----
// prep:ht = pow2(2×flop)(精确上界的 2× 余量);flop 超全局表上限 → 不可重试(保持 overflow → 整阵回退)
__global__ void retry_prep_kernel(
    const int *rows, int n, const int *flop,
    long long *rht, int *row_ovf, int *overflow_flag)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int row = rows[t];
    long long f = flop[row];
    if (2 * f > (long long)GLOBAL_HT_MAX_SLOTS) return;   // 不可重试:overflow 已置
    long long h = 2 * HASH_CAP;
    while (h < 2 * f) h <<= 1;
    rht[t] = h;
    row_ovf[row] = 1;   // 已在 accumulate 记录时置位;此处幂等
}
// slot:重试区按行 id 的槽位数(=flop;非重试行为 0,先 memset)
__global__ void retry_slot_kernel(const int *rows, int n, const int *flop, int *rslot)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    rslot[rows[t]] = flop[rows[t]];
}

// compact:每行 row_nnz 项从 tmp(row_off 起)拷到连续 CSR(C_row_ptr 起)
__global__ void hash_compact_kernel(
    int A_rows, const long long *row_off, const int *row_nnz, const int *C_row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    unsigned long long *out_key, double *out_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    long long src = row_off[i]; int dst = C_row_ptr[i], n = row_nnz[i];
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

// per-row compact+sort:每 block 一行,BlockRadixSort 行内按 col 排→直接写 CSR;一次替掉 compact+全局 sort+split_key
template<int TPB, int IPT>
__global__ void hash_compact_sort_kernel(
    const int *rows, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip)
{
    using BlockRadixSort = cub::BlockRadixSort<unsigned int, TPB, IPT, double>;
    extern __shared__ char csort_smem[];   // 动态 shared(cap 大时 >48KB 需 opt-in,见 launch_csort)
    typename BlockRadixSort::TempStorage &temp_storage =
        *reinterpret_cast<typename BlockRadixSort::TempStorage *>(csort_smem);
    if (blockIdx.x >= n_rows) return;
    int row = rows[blockIdx.x];
    if (skip && skip[row]) return;               // 行级重试的行:数据在重试区,主区跳过
    long long start = row_off[row];
    int n = row_nnz[row];
    int out_start = row_ptr[row];

    unsigned int col[IPT];
    double val[IPT];
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;   // BLOCKED 排布(匹配 cub::BlockRadixSort 默认)
        if (idx < n) {
            col[i] = (unsigned int)(tmp_key[start + idx] & 0xffffffffu);   // key 低 32 = col
            val[i] = tmp_val[start + idx];
        } else {
            col[i] = 0xffffffffu;   // sentinel → 排到末尾(只存前 n 个真实项)
            val[i] = 0.0f;
        }
    }
    BlockRadixSort(temp_storage).Sort(col, val, 0, 32);   // 按 col 排(低 32 位)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;   // BLOCKED 排布(匹配 cub::BlockRadixSort 默认)
        if (idx < n) {
            out_col[out_start + idx] = (int)col[i];
            out_val[out_start + idx] = val[i];
        }
    }
}

// GPU 端分桶(无 host 往返):每行算 bucket_id 并直方图计数(融合省 1 launch)
__global__ void compute_bucket_kernel(
    const int *est, const int *A_row_ptr, int A_rows, int ultra_thr,
    int *bucket_id, int *counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int e = est[i];
    int bid;
    if (e <= ultra_thr) bid = N_BINS - 2;            // ultra:线性免 hash
    else if (e > HASH_CAP) bid = N_BINS - 1;         // heavy:全局表(2026-08-25,不再回退 merge)
    else {
        int bi = 0, ht = 32;
        // 小行表 2× 松弛(est=flop 紧界 → 满载原子争用,333SP accumulate +8ms 教训;槽位不变只放大表)
        int target = (e <= 4096) ? 2 * e : e;
        while (ht < target && ht < HASH_CAP) { ht <<= 1; bi++; }
        // 批量 kernel 门(2026-08-26):est≤64 且行长≤32(k 短,warp 串行 k 才划算;
        // 3Dspectralwave2 的 est 小但 k 数百的长链行回归 62→81ms 教训)→ bin0;
        // est≤64 但 k>32 → bin1 走原 per-row hash_spa(64 组 k 并行)
        if (e <= 64 && bi <= 1) {
            int rk = A_row_ptr[i + 1] - A_row_ptr[i];
            bid = (rk <= 32) ? 0 : 1;
        } else {
            bid = bi;
        }
    }
    bucket_id[i] = bid;
    atomicAdd(&counts[bid], 1);
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

// 单 block inclusive prefix-sum(Hillis-Steele);小阵专用,替 thrust::inclusive_scan
// T = int(cnnz)/ long long(est:ocean337 Σest 超 2^31)
template <typename T>
__global__ void scan_inclusive_kernel(const int *x, T *y, int n) {
    extern __shared__ unsigned char scan_smem_raw[];
    T *ss = (T*)scan_smem_raw;
    int tid = threadIdx.x;
    int N = blockDim.x;
    ss[tid] = (tid < n) ? (T)x[tid] : (T)0;
    __syncthreads();
    for (int off = 1; off < N; off <<= 1) {
        T v = (tid >= off) ? ss[tid - off] : (T)0;
        __syncthreads();
        ss[tid] += v;
        __syncthreads();
    }
    if (tid < n) y[tid] = ss[tid];
}

// 小行专用:accumulate 已在 SMEM 内 count-sort 写有序,这里只把 tmp(gapped)→ CSR(packed) 纯 copy。
__global__ void hash_compact_copy_kernel(
    const int *rows, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip)
{
    if (blockIdx.x >= n_rows) return;
    int row = rows[blockIdx.x];
    if (skip && skip[row]) return;               // 行级重试的行:数据在重试区,主区跳过
    long long src = row_off[row]; int n = row_nnz[row], dst = row_ptr[row];
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        out_col[dst + t] = (int)(tmp_key[src + t] & 0xffffffffu);
        out_val[dst + t] = tmp_val[src + t];
    }
}

// launch helper:按 config 查 BlockRadixSort TempStorage 大小,>48KB 自动 opt-in 动态 shared(H100 可 ~228KB)。
template<int TPB, int IPT>
static void launch_csort(int n, const int *rows_ptr, const long long *d_off, const int *d_row_nnz,
                         const int *dC_rp, const unsigned long long *d_tmp_key, const double *d_tmp_val,
                         int *dC_ci, double *d_val, const int *skip) {
    using BRS = cub::BlockRadixSort<unsigned int, TPB, IPT, double>;
    size_t smem = sizeof(typename BRS::TempStorage);
    if (smem > 48 * 1024)
        CHECK_CUDA(cudaFuncSetAttribute((const void*)hash_compact_sort_kernel<TPB, IPT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    hash_compact_sort_kernel<TPB, IPT><<<n, TPB, smem>>>(
        rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, skip);
}

// DBG 校验:输出 CSR 每行 col 严格升序(验证 compact_sort 排序正确)。
__global__ void hash_check_sorted_kernel(const int *row_ptr, const int *col, int A_rows, int *violations) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= A_rows) return;
    int s = row_ptr[r], e = row_ptr[r + 1];
    for (int i = s; i < e - 1; i++)
        if (col[i] >= col[i + 1]) atomicAdd(violations, 1);
}

// Host

static void hash_product(
    void *A_buffer, int A_rows, int A_cols, int A_nnz, bool att,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    const char *tag = att ? "atth" : "hash";
    dbg("[%s] start (HASH_CAP=%d)\n", tag, HASH_CAP);
    HashProf prof(att ? "atth-prof" : "hash-prof");
    dev_pool_reset();   // device arena:本调用所有 device buffer 复用(省 ~13 cudaMalloc/Free)

    size_t A_rp_sz = (A_rows + 1) * sizeof(int);
    size_t A_ci_sz = (size_t)A_nnz * sizeof(int);
    size_t A_v_sz = (size_t)A_nnz * sizeof(double);
    size_t A_total = ALIGN8(A_rp_sz + A_ci_sz) + A_v_sz;
    void *dA; dA = decltype(dA)(dev_alloc(A_total));
    prof("h2d", [&]{ CHECK_CUDA(cudaMemcpy(dA, A_buffer, A_total, cudaMemcpyHostToDevice)); });
    char *b = (char*)dA;
    int   *dA_rp  = (int*)b;
    int   *dA_ci  = (int*)(b + A_rp_sz);
    double *dA_val = (double*)(b + ALIGN8(A_rp_sz + A_ci_sz));
    dbg("[%s] h2d\n", tag);

    // 内层 j 的来源 B:AA → A 本身;ATT → Aᵀ(= A 的 CSC)。hash_spa/ultra/mh-construct 读 B 的行 k。
    int *dB_rp, *dB_ci; double *dB_val;
    int *d_csc_cp = nullptr, *d_csc_ri = nullptr; double *d_csc_val = nullptr;
    if (att) {
        build_csc(dA_rp, dA_ci, dA_val, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_val);
        dB_rp = d_csc_cp; dB_ci = d_csc_ri; dB_val = d_csc_val;
        dbg("[atth] csc\n");
    } else {
        dB_rp = dA_rp; dB_ci = dA_ci; dB_val = dA_val;
    }
    const int upper_tri = att ? 1 : 0;

    // Stage 1: row_off 由 est 的 scan 给出(64 位:ocean337 上 Σest 可超 2^31,TSOPF 教训)
    long long *d_off; d_off = decltype(d_off)(dev_alloc((A_rows + 1) * sizeof(long long)));

    // 估计每行 distinct(2026-08-25 重构,对标 Ocean Ana1):
    //   ① 全阵先算每行精确乘积数 flop(O(nnz) count,兼作 avg_product 门);
    //   ② avg_product ≤ AVG_FLOP_THR(64) → est = flop【精确上界,零低估】+ 免 MinHash
    //      (均度低的行 dup 因子≈1,上界即准界 —— 333SP 型省 mh 两遍,Ocean 同款免估计路径);
    //   ③ 否则 MinHash 两阶段(est = min(2×snap(E), flop):flop 封顶 MinHash 高估 —— TSOPF
    //      Σest 曾爆到 147 亿 = int 溢出 + 74× 过分配;min 不引入新低估:flop ≥ distinct 恒真)
    int *d_est; d_est = decltype(d_est)(dev_alloc(A_rows * sizeof(int)));
    int *d_flop; d_flop = decltype(d_flop)(dev_alloc(A_rows * sizeof(int)));
    long long total_flop = 0;
    prof("count_flop", [&]{
        count_intermediates_par_kernel<<<(A_rows * 32 + 255) / 256, 256>>>(dB_rp, dB_ci, A_rows, d_flop);
        CHECK_CUDA(cudaGetLastError());
        total_flop = thrust::reduce(thrust::device_ptr<int>(d_flop), thrust::device_ptr<int>(d_flop + A_rows), 0LL);
    });
    double avg_product = A_rows ? (double)total_flop / A_rows : 0.0;
    dbg("[%s] avg_product=%.1f (total_flop=%lld)\n", tag, avg_product, total_flop);
    if (avg_product <= AVG_FLOP_THR) {
        dbg("[%s] 低均度 → est=精确 flop(免 MinHash)\n", tag);
        CHECK_CUDA(cudaMemcpy(d_est, d_flop, (size_t)A_rows * sizeof(int), cudaMemcpyDeviceToDevice));
    } else {
        // MinHash 两阶段(partition bottom-k sketch;机制见 mh_construct/mh_merge)
        unsigned int *d_mh; d_mh = decltype(d_mh)(dev_alloc((size_t)A_rows * MH_M * sizeof(unsigned int)));
        int rows_per_block = 32;
        size_t smem_p1 = (size_t)rows_per_block * MH_M * sizeof(unsigned int) + (size_t)(rows_per_block + 1) * sizeof(int);
        if (smem_p1 > 48 * 1024)
            CHECK_CUDA(cudaFuncSetAttribute(mh_construct_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_p1));
        int grid_p1 = (A_rows + rows_per_block - 1) / rows_per_block;
        prof("mh_construct", [&]{
            mh_construct_kernel<<<grid_p1, HASH_BLOCK, smem_p1>>>(
                dB_rp, dB_ci, A_rows, A_nnz, d_mh, rows_per_block);
            CHECK_CUDA(cudaGetLastError());
        });
        int p2_block = MH_M / 4;                      // 单 warp(32):每线程 owning 4 partitions,vectorized uint4 merge
        int smem_p2 = MH_M * sizeof(unsigned int);   // [MH_M] uint32(smem_merge)
        prof("mh_merge", [&]{
            mh_merge_kernel<<<A_rows, p2_block, smem_p2>>>(
                dA_rp, dA_ci, A_rows, d_mh, d_flop, d_est);
            CHECK_CUDA(cudaGetLastError());
        });
        dev_free(d_mh);
    }
    long long total_est;
    prof("est_scan", [&]{
        CHECK_CUDA(cudaMemset(d_off, 0, sizeof(long long)));
        if (A_rows <= 1024) {
            int b = 1; while (b < A_rows) b <<= 1;                       // 小阵:单 block scan(1 launch)
            scan_inclusive_kernel<long long><<<1, b, b * sizeof(long long)>>>(d_est, d_off + 1, A_rows);
        } else {
            thrust::inclusive_scan(thrust::device_ptr<int>(d_est),
                                   thrust::device_ptr<int>(d_est + A_rows),
                                   thrust::device_ptr<long long>(d_off + 1));
        }
        CHECK_CUDA(cudaMemcpy(&total_est, d_off + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));   // D2H 纳入计时
    });
    dbg("[hash] total_est=%lld\n", total_est);

    // Stage 2: GPU 端分桶(全 device,无 host 往返)+ 预分配大 buffer(零 per-bucket malloc/free)
    int *d_row_nnz; d_row_nnz = decltype(d_row_nnz)(dev_alloc(A_rows * sizeof(int)));
    int *d_overflow; d_overflow = decltype(d_overflow)(dev_alloc(sizeof(int)));
    // 行级重试(2026-08-26,Ocean out_overflow_row_ids 同款):欠估行收集 → flop 定表重跑 → d_off[i] 改指重试区
    int *d_ovf_rows; d_ovf_rows = decltype(d_ovf_rows)(dev_alloc(A_rows * sizeof(int)));
    int *d_ovf_cnt;  d_ovf_cnt  = decltype(d_ovf_cnt)(dev_alloc(sizeof(int)));
    int *d_row_ovf;  d_row_ovf  = decltype(d_row_ovf)(dev_alloc(A_rows * sizeof(int)));   // 重试行标记(compact 路由)
    // 重试区(行级重试输出;compact 末段读)
    const int *d_retry_rows = nullptr; int d_retry_n = 0;
    const long long *d_retry_off = nullptr;
    const unsigned long long *d_retry_key = nullptr; const double *d_retry_val = nullptr;
    unsigned long long *d_tmp_key; double *d_tmp_val;
    d_tmp_key = decltype(d_tmp_key)(dev_alloc((size_t)total_est * sizeof(unsigned long long)));
    d_tmp_val = decltype(d_tmp_val)(dev_alloc((size_t)total_est * sizeof(double)));
    // d_val(C_val)现 alias 进连续 dC(compact+sort 直写,见 cnnz_scan 后),不再单独分配/释放。
    double *d_val = nullptr;

    // 2a: GPU 端分桶(MinHash est → bucket):bucket_id → count → exclusive scan → scatter(全 device)
    int *d_bkid; d_bkid = decltype(d_bkid)(dev_alloc(A_rows * sizeof(int)));
    int *d_cnt;  d_cnt = decltype(d_cnt)(dev_alloc( N_BINS * sizeof(int)));
    int *d_offb; d_offb = decltype(d_offb)(dev_alloc(N_BINS * sizeof(int)));
    int *d_pos;  d_pos = decltype(d_pos)(dev_alloc( N_BINS * sizeof(int)));
    int *d_sort; d_sort = decltype(d_sort)(dev_alloc(A_rows * sizeof(int)));   // 预分配:桶有序行号
    int h_cnt[N_BINS], h_off[N_BINS];

    // heavy 全局表路径 buffer(bin 11 = est > HASH_CAP,2026-08-25;不再回退 merge3)
    long long *d_heavy_ht = nullptr; long long *d_heavy_tab_off = nullptr;
    int *d_tab_col = nullptr; double *d_tab_val = nullptr;
    int *d_seg_beg = nullptr, *d_seg_end = nullptr;
    unsigned long long *d_scr_key = nullptr; double *d_scr_val = nullptr;
    void *d_cub_tmp = nullptr;
    prof("binning", [&]{
        CHECK_CUDA(cudaMemset(d_cnt, 0, N_BINS * sizeof(int)));   // 先清零(fused kernel 内 atomicAdd 累加)
        compute_bucket_kernel<<<(A_rows + 255) / 256, 256>>>(d_est, dA_rp, A_rows, EST_ULTRA_THR, d_bkid, d_cnt);
        thrust::exclusive_scan(thrust::device_ptr<int>(d_cnt),
                               thrust::device_ptr<int>(d_cnt + N_BINS),
                               thrust::device_ptr<int>(d_offb));
        CHECK_CUDA(cudaMemset(d_pos, 0, N_BINS * sizeof(int)));   // scatter 计数器从 0 起(非 offsets)
        scatter_rows_kernel<<<(A_rows + 255) / 256, 256>>>(d_bkid, A_rows, d_offb, d_pos, d_sort);
        // 2 个 D2H 合并:async + 单 sync(省 1 同步点;小阵 launch 开销友好)
        CHECK_CUDA(cudaMemcpyAsync(h_cnt, d_cnt,  N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpyAsync(h_off, d_offb, N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaStreamSynchronize(0));
    });

    // 2b: opt-in max SMEM
    {
        size_t maxsm = (size_t)HASH_CAP * (sizeof(int) + sizeof(double));   // sh_col[int]+sh_val[double]
        if (maxsm > 48 * 1024)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
    }
    // 2b-priv: warp 私有 SPA 门控(HASH_PRIV=1,HASH_PRIV_W 默认 8);仅 SMEM 可容纳的桶走 priv
    static int g_priv = -1, g_privW = 8;
    if (g_priv < 0) {
        const char *e = getenv("HASH_PRIV");
        g_priv = (e && *e && atoi(e) > 0) ? 1 : 0;
        const char *ew = getenv("HASH_PRIV_W");
        if (ew && *ew) { int w = atoi(ew); if (w >= 2 && w <= 8) g_privW = w; }
        if (g_priv)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_priv_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 196 * 1024));   // H100 ≤228KB;launch 时按 bin 再判
    }
    const int PRIV_W = g_privW;
    // 2c: per-bin launch(hash 插入 + extract;prof 内部已 sync);memset+overflow D2H 纳入计时
    int overflow;
    prof("accumulate", [&]{
        CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
        for (int bi = 0; bi < N_BINS; bi++) {
            int n = h_cnt[bi];
            if (n == 0) continue;
            int *rows_ptr = d_sort + h_off[bi];
            if (bi == 0) {
                // 小行批量(est≤64 且行长≤32):warp-per-row + 私有表 + 融合有序 extract(2026-08-26)
                hash_spa_batched_kernel<<<(n + BATCH_WPB - 1) / BATCH_WPB, BATCH_WPB * 32>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf);
            } else if (bi == N_BINS - 2) {
                // ultra(est≤EST_ULTRA_THR):线性,免 hash
                hash_ultra_kernel<<<(n + 255) / 256, 256>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf);
            } else if (bi == N_BINS - 1) {
                // heavy(est>HASH_CAP):全局内存 hash 表 + cub 分段排序(不回退 merge)
                d_heavy_ht     = decltype(d_heavy_ht)(dev_alloc(n * sizeof(long long)));   // ll:scan 同型(混型 scan 产出坏偏移的坑)
                d_heavy_tab_off= decltype(d_heavy_tab_off)(dev_alloc((n + 1) * sizeof(long long)));
                heavy_prep_kernel<<<(n + 255) / 256, 256>>>(rows_ptr, n, d_est, d_heavy_ht, d_overflow);
                // exclusive scan:[k] = Σ_{i<k} ht[i] —— 恰为行 k 的表偏移(inclusive 才需要 +1 偏移,别搞混)
                thrust::exclusive_scan(thrust::device_ptr<long long>(d_heavy_ht),
                                       thrust::device_ptr<long long>(d_heavy_ht + n),
                                       thrust::device_ptr<long long>(d_heavy_tab_off));
                long long total_slots, last_ht;
                CHECK_CUDA(cudaMemcpy(&total_slots, d_heavy_tab_off + (n - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(&last_ht, d_heavy_ht + (n - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                total_slots += last_ht;
                long long ght_cap = 24LL << 30;                       // SAFETY:表区 arena 上限(字节)
                if (const char *e = getenv("GHT_MAX_BYTES")) ght_cap = atoll(e);
                dbg("[hash] heavy rows=%d slots=%lld arena=%.2fGB\n", n, total_slots,
                    total_slots * 12.0 / (1 << 30));
                if (total_slots > 0 && total_slots * 12 > ght_cap) {
                    fprintf(stderr, "[hash] heavy arena %.2fGB 超 cap %.2fGB → 按 overflow 回退\n",
                            total_slots * 12.0 / (1 << 30), ght_cap / 1073741824.0);
                    CHECK_CUDA(cudaMemset(d_overflow, 1, sizeof(int)));
                } else if (total_slots > 0) {
                    d_tab_col = decltype(d_tab_col)(dev_alloc((size_t)total_slots * sizeof(int)));
                    d_tab_val = decltype(d_tab_val)(dev_alloc((size_t)total_slots * sizeof(double)));
                    d_seg_beg = decltype(d_seg_beg)(dev_alloc(n * sizeof(int)));
                    d_seg_end = decltype(d_seg_end)(dev_alloc(n * sizeof(int)));
                    d_scr_key = decltype(d_scr_key)(dev_alloc((size_t)total_est * sizeof(unsigned long long)));
                    d_scr_val = decltype(d_scr_val)(dev_alloc((size_t)total_est * sizeof(double)));
                    hash_global_kernel<<<n, HASH_BLOCK>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, d_heavy_ht, d_heavy_tab_off, d_tab_col, d_tab_val,
                        d_off, d_est, (long long)total_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                        d_ovf_rows, d_ovf_cnt, d_row_ovf);
                    heavy_seg_kernel<<<(n + 255) / 256, 256>>>(
                        rows_ptr, n, d_off, d_row_nnz, d_seg_beg, d_seg_end);
                    size_t cub_bytes = 0;
                    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                        nullptr, cub_bytes, d_tmp_key, d_scr_key, d_tmp_val, d_scr_val,
                        total_est, n, d_seg_beg, d_seg_end));
                    d_cub_tmp = dev_alloc(cub_bytes);
                    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                        d_cub_tmp, cub_bytes, d_tmp_key, d_scr_key, d_tmp_val, d_scr_val,
                        total_est, n, d_seg_beg, d_seg_end));
                }
            } else {
                int ht = 32 << bi;
                size_t smem_flat = (size_t)ht * (sizeof(int) + sizeof(double));
                size_t smem_priv = (size_t)PRIV_W * ht * (sizeof(int) + sizeof(double));
                if (!att && g_priv && smem_priv <= 196 * 1024) {
                    hash_spa_priv_kernel<<<n, PRIV_W * 32, smem_priv>>>(
                        dA_rp, dA_ci, dA_val, rows_ptr, n, ht, PRIV_W, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
                } else {
                    hash_spa_kernel<<<n, HASH_BLOCK, smem_flat>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, ht, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                }
            }
        }
        CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入
    });

    // ---- 行级重试(2026-08-26,Ocean out_overflow_row_ids 同款):欠估行(estimate<distinct)用
    //      flop(精确上界)定表重跑 hash_global,输出写独立重试区,compact 末段按 d_row_ovf 路由。
    //      全部重试成功 → 清 overflow 继续;存在不可重试行(flop 超表上限)→ 保持 overflow 整阵回退。----
    {
        int h_ovf = 0;
        CHECK_CUDA(cudaMemcpy(&h_ovf, d_ovf_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        dbg("[hash] dbg: overflow=%d ovf_cnt=%d\n", overflow, h_ovf);   // TEMP
        if (h_ovf > 0) {
            dbg("[hash] row-retry: %d 行 → flop 定表重跑\n", h_ovf);
            prof("retry", [&]{
                CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));   // 清 accumulate 的旧标志;不可重试行会重置
                long long *d_rht  = decltype(d_rht)(dev_alloc(h_ovf * sizeof(long long)));
                long long *d_rtab = decltype(d_rtab)(dev_alloc((h_ovf + 1) * sizeof(long long)));
                retry_prep_kernel<<<(h_ovf + 255) / 256, 256>>>(
                    d_ovf_rows, h_ovf, d_flop, d_rht, d_row_ovf, d_overflow);
                thrust::exclusive_scan(thrust::device_ptr<long long>(d_rht),
                                       thrust::device_ptr<long long>(d_rht + h_ovf),
                                       thrust::device_ptr<long long>(d_rtab));
                // 重试区槽位(按行 id;非重试行 0)
                int *d_rslot = decltype(d_rslot)(dev_alloc(A_rows * sizeof(int)));
                CHECK_CUDA(cudaMemset(d_rslot, 0, (size_t)A_rows * sizeof(int)));
                retry_slot_kernel<<<(h_ovf + 255) / 256, 256>>>(d_ovf_rows, h_ovf, d_flop, d_rslot);
                long long *d_roff = decltype(d_roff)(dev_alloc((A_rows + 1) * sizeof(long long)));
                CHECK_CUDA(cudaMemset(d_roff, 0, sizeof(long long)));
                thrust::inclusive_scan(thrust::device_ptr<int>(d_rslot),
                                       thrust::device_ptr<int>(d_rslot + A_rows),
                                       thrust::device_ptr<long long>(d_roff + 1));
                long long r_slots;
                CHECK_CUDA(cudaMemcpy(&r_slots, d_roff + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
                if (r_slots > 0) {
                    // 表 arena 总量 = rtab[h_ovf-1] + rht[h_ovf-1](exclusive scan 只写 [0..h_ovf-1])
                    long long last_h, rt_prev;
                    CHECK_CUDA(cudaMemcpy(&last_h, d_rht + (h_ovf - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(&rt_prev, d_rtab + (h_ovf - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                    long long rt_total = rt_prev + last_h;
                    int *rtc = decltype(rtc)(dev_alloc((size_t)rt_total * sizeof(int)));
                    double *rtv = decltype(rtv)(dev_alloc((size_t)rt_total * sizeof(double)));
                    unsigned long long *rk  = decltype(rk)(dev_alloc((size_t)r_slots * sizeof(unsigned long long)));
                    double *rv  = decltype(rv)(dev_alloc((size_t)r_slots * sizeof(double)));
                    unsigned long long *rk2 = decltype(rk2)(dev_alloc((size_t)r_slots * sizeof(unsigned long long)));
                    double *rv2  = decltype(rv2)(dev_alloc((size_t)r_slots * sizeof(double)));
                    // hash_global 逐行表偏移:exclusive_scan 已给 [0..h_ovf-1],[0]=0 ✓
                    hash_global_kernel<<<h_ovf, HASH_BLOCK>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        d_ovf_rows, h_ovf, d_rht, d_rtab, rtc, rtv,
                        d_roff, d_rslot, r_slots, rk, rv, d_row_nnz, d_overflow,
                        d_ovf_rows, d_ovf_cnt, d_row_ovf);
                    int *d_rsb = decltype(d_rsb)(dev_alloc(h_ovf * sizeof(int)));
                    int *d_rse = decltype(d_rse)(dev_alloc(h_ovf * sizeof(int)));
                    heavy_seg_kernel<<<(h_ovf + 255) / 256, 256>>>(
                        d_ovf_rows, h_ovf, d_roff, d_row_nnz, d_rsb, d_rse);
                    size_t rcb = 0;
                    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                        nullptr, rcb, rk, rk2, rv, rv2, r_slots, h_ovf, d_rsb, d_rse));
                    void *rct = dev_alloc(rcb);
                    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                        rct, rcb, rk, rk2, rv, rv2, r_slots, h_ovf, d_rsb, d_rse));
                    // retry_prep 遇不可重试行会重置 overflow → 仍有则整阵回退
                    d_retry_rows = d_ovf_rows; d_retry_n = h_ovf;
                    d_retry_off = d_roff; d_retry_key = rk2; d_retry_val = rv2;
                    dbg("[hash] row-retry: %d 行完成 → 继续(免整阵回退)\n", h_ovf);
                }
            });
        }
    }
    CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (overflow) {
        fprintf(stderr, "[hash] OVERFLOW: 某行 distinct 列 > HASH_CAP=%d → 回退 merge(dispatcher 处理)\n", HASH_CAP);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
        dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
        dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort); dev_free(d_est);
        dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val);
        dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
        dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
        dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
        return;
    }

    // Stage 3: scan row_nnz → C_row_ptr + C_nnz(精确)
    int *dC_rp; dC_rp = decltype(dC_rp)(dev_alloc((A_rows + 1) * sizeof(int)));
    int C_nnz_result;
    prof("cnnz_scan", [&]{
        CHECK_CUDA(cudaMemset(dC_rp, 0, sizeof(int)));
        if (A_rows <= 1024) {
            int b = 1; while (b < A_rows) b <<= 1;                       // 小阵:单 block scan(1 launch)
            scan_inclusive_kernel<int><<<1, b, b * sizeof(int)>>>(d_row_nnz, dC_rp + 1, A_rows);
        } else {
            thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                                   thrust::device_ptr<int>(d_row_nnz + A_rows),
                                   thrust::device_ptr<int>(dC_rp + 1));
        }
        CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_rp + A_rows, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入
    });
    dbg("[hash] C_nnz=%d (est=%d, %.2fx over-alloc)\n", C_nnz_result, total_est,
        total_est > 0 ? (double)total_est / C_nnz_result : 0.0);

    // est underflow check: actual C_nnz > estimated total → 回退 merge3
    if (C_nnz_result > total_est) {
        fprintf(stderr, "[hash] est underflow: C_nnz=%d > total_est=%d → 回退 merge\n", C_nnz_result, total_est);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
        dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
        dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort); dev_free(d_est);
        dev_free(dC_rp);
        dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val);
        dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
        dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
        dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
        return;
    }

    // Stage 4+5+6: per-row compact+sort(BlockRadixSort)替掉 compact+全局 sort+split_key;按 bin 选 config,直写连续 dC
    size_t C_rp_sz  = (A_rows + 1) * sizeof(int);
    size_t C_ci_sz  = (size_t)C_nnz_result * sizeof(int);
    size_t C_v_sz   = (size_t)C_nnz_result * sizeof(double);
    size_t C_rp_al  = ALIGN8(C_rp_sz);
    size_t C_ci_al  = ALIGN8(C_ci_sz);
    size_t C_total  = C_rp_al + C_ci_al + C_v_sz;
    void *dC; dC = decltype(dC)(dev_alloc(C_total));
    char *cb = (char*)dC;
    int   *dC_ci = (int*)(cb + C_rp_al);
    d_val        = (double*)(cb + C_rp_al + C_ci_al);
    prof("compact+sort", [&]{
        CHECK_CUDA(cudaMemcpy(cb, dC_rp, C_rp_sz, cudaMemcpyDeviceToDevice));   // row_ptr 落位(~A_rows ints,微秒级,纳入计时)
        for (int bi = 0; bi < N_BINS; bi++) {
            int n = h_cnt[bi];
            if (n == 0) continue;
            int *rows_ptr = d_sort + h_off[bi];
            if (bi <= 5) {
                // 小行(ht≤CSORT_HT):accumulate 已 count-sort,这里只 compact_copy(tmp→CSR)
                hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
            } else if (bi <= 7) {
                launch_csort<512, 8>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
            } else if (bi <= 9) {
                launch_csort<256, 64>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
            } else if (bi == N_BINS - 1) {
                // heavy:accumulate 内已分段排序(compact 读 scratch)
                hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_scr_key, d_scr_val, dC_ci, d_val, d_row_ovf);
            } else {
                // ultra(行≤32,无序):小 config sort
                launch_csort<64, 1>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
            }
        }
        if (d_retry_n > 0) {   // 行级重试的行:从重试区(已分段排序)拷到 CSR
            hash_compact_copy_kernel<<<d_retry_n, 256>>>(
                d_retry_rows, d_retry_n, d_retry_off, d_row_nnz, dC_rp,
                d_retry_key, d_retry_val, dC_ci, d_val, nullptr);
        }
        CHECK_CUDA(cudaGetLastError());
    });

#ifdef DBG
    {   // 校验:输出 CSR 每行 col 严格升序(验证 compact_sort 排序正确)
        int *d_viol; d_viol = decltype(d_viol)(dev_alloc(sizeof(int)));
        CHECK_CUDA(cudaMemset(d_viol, 0, sizeof(int)));
        hash_check_sorted_kernel<<<(A_rows + 255) / 256, 256>>>(dC_rp, dC_ci, A_rows, d_viol);
        int viol; CHECK_CUDA(cudaMemcpy(&viol, d_viol, sizeof(int), cudaMemcpyDeviceToHost));
        dbg("[hash] sorted check: %d 行内乱序违规\n", viol);
        dev_free(d_viol);
    }
#endif

    // Stage 7: 连续 dC 已就位(compact+sort 直写 col/val + row_ptr 已落位)→ 直接 D2H 前 C_total 字节 = packed [rp|ci|val]。免 pack。

    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total));
    prof("d2h", [&]{
        CHECK_CUDA(cudaMemcpy(C_buffer, dC, C_total, cudaMemcpyDeviceToHost));
    });

    *C_buffer_out = C_buffer;
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = C_nnz_result;

    dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
    dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
    dev_free(dC_rp); dev_free(dC);   // dC_ci/d_val 是 dC 的偏移别名,随 dC 一起释放
    dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val);   // ATT 的 Aᵀ(AA 时为 null,no-op)
    dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
    dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
    dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
}

void spgemm_self_product_hash(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                              void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    hash_product(A_buffer, A_rows, A_cols, A_nnz, /*att=*/false, C_buffer_out, C_rows, C_cols, C_nnz);
}

// C = A·Aᵀ 上三角(j≥i):AA hash SPA 的忠实拷贝,B=Aᵀ(CSC)+ j≥i 过滤。复用 MinHash sizing/binning/compact_sort。
void spgemm_att_hash(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    hash_product(A_buffer, A_rows, A_cols, A_nnz, /*att=*/true, C_buffer_out, C_rows, C_cols, C_nnz);
}
