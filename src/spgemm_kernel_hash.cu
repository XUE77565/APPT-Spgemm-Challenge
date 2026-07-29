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

// 并行 count flop_ub:每行一 block,Σ nnz(row k)+归约;小阵替 MinHash 两阶段,flop_ub 是确定性上界
__global__ void count_intermediates_par_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int *ub)
{
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int s = 0;
    for (int p = rs + threadIdx.x; p < re; p += blockDim.x) {
        int k = A_col_idx[p];
        s += A_row_ptr[k + 1] - A_row_ptr[k];
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);   // warp 归约
    __shared__ int warp_s[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_s[warp] = s;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    if (warp == 0) {
        s = (lane < nwarps) ? warp_s[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xFFFFFFFF, s, off);
        if (lane == 0) ub[i] = s;
    }
}

#ifndef HASH_CAP
#define HASH_CAP 16384         // 重行桶 hash 表最大槽位(→ col+val = 128KB/块)
#endif
#define HASH_BLOCK 256
#define N_BINS 11    // 10 hash ht-buckets(ht=32<<i, i=0..9) + 1 ultra(flop≤16)
#define CSORT_HT 1024   // ht≤此值的行(bin0-5)在 accumulate 内 count-sort 写有序;compact 阶段只 copy

// MinHash 概率基数估计:替 flop_ub 定 tmp buffer + hash 表大小
#define MH_M 128                      // partition 数(=128)
#define EST_EXPAND 1.5                // expansion(覆盖估计低估;溢出→回退 merge3 兜底)
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

    // 估计 reduce(单 warp):sum_inv = Σ 1/min_j(非空);V = 非空 partition 数
    double sum_inv = 0.0;
    int V = 0;
    for (int k = 0; k < 4; k++) {
        unsigned int mv = smem_merge[base_i + k];
        if (mv != MH_EMPTY) { sum_inv += 1.0 / (double)mv; V++; }
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_inv += __shfl_down_sync(0xFFFFFFFF, sum_inv, off);
        V += __shfl_down_sync(0xFFFFFFFF, V, off);
    }

    if (tid == 0) {
        double E;
        int v = V;
        if (v == 0) {
            E = 0.0;                                                      // 空行
        } else if (v < MH_M) {
            E = (double)MH_M * log((double)MH_M / (double)(MH_M - v));   // linear counting(小范围,偏保守上界)
        } else {
            E = (double)0x100000000LL * sum_inv - (double)MH_M;           // 2^32·Σ(1/min_j) − m
        }
        int temp = (int)(E * EST_EXPAND);
        if (temp < 1) temp = 1;
        int est;
        if (temp <= EST_ULTRA_THR) {                                      // ultra:线性 kernel,est 保留紧 temp
            est = temp;
        } else {                                                          // 否则 snap 到 next_pow2 ∈[32,HASH_CAP]
            int ht = 32;
            while (ht < temp && ht < HASH_CAP) ht <<= 1;
            est = ht;
        }
        est_nnz[row] = est;
    }
}

// 按桶跑:ht_size = 该桶 hash 表大小;轻行小表、重行大表,distinct>ht_size → overflow_flag
__global__ void hash_spa_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,   // 内层 j 来源:AA=A,ATT=Aᵀ(CSC)
    int upper_tri,                                                      // 0=自乘;1=ATT 只算 j≥i
    const int *bucket_rows, int n_in_bucket, int ht_size,
    const int *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag)
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
                if (++probes >= ht_size) { atomicExch(overflow_flag, 1); break; }   // 溢出
            }
        }
    }
    __syncthreads();

    // extract 去重项 → tmp:小行(ht≤CSORT_HT)SMEM 内 compact+count-sort 写有序,大行无序写(交 compact_sort)
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    int base = row_off[i];
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
        for (int k = tid; k < count; k += HASH_BLOCK) {      // count-sort:数 < 自己的 → rank → 落有序位
            int c = sh_col[k]; double v = sh_val[k]; int rank = 0;
            for (int j = 0; j < count; j++) if (sh_col[j] < c) rank++;
            tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + rank] = v;
        }
        __syncthreads();
        if (tid == 0) row_nnz[i] = count;
    } else {
        // 大行:无序 extract(交 compact_sort)
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
}

// hash_spa_priv_kernel:warp 私有 SPA(HASH_PRIV 门控):k 按 warp 分区→Phase A 无 atomicAdd 累加,Phase B 跨 warp merge,Phase C extract
__global__ void hash_spa_priv_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *bucket_rows, int n_in_bucket, int ht_size, int W,
    const int *row_off,
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
    int base = row_off[i];
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
    const int *row_off,
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz, int *overflow_flag)
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
                if (u_cnt >= CAP) { atomicExch(overflow_flag, 1); return; }   // 超 CAP → 回退 merge3
                u_col[u_cnt] = j; u_val[u_cnt] = v; u_cnt++;
            }
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
    const unsigned long long *tmp_key, const double *tmp_val,
    unsigned long long *out_key, double *out_val)
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

// per-row compact+sort:每 block 一行,BlockRadixSort 行内按 col 排→直接写 CSR;一次替掉 compact+全局 sort+split_key
template<int TPB, int IPT>
__global__ void hash_compact_sort_kernel(
    const int *rows, int n_rows,
    const int *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val)
{
    using BlockRadixSort = cub::BlockRadixSort<unsigned int, TPB, IPT, double>;
    extern __shared__ char csort_smem[];   // 动态 shared(cap 大时 >48KB 需 opt-in,见 launch_csort)
    typename BlockRadixSort::TempStorage &temp_storage =
        *reinterpret_cast<typename BlockRadixSort::TempStorage *>(csort_smem);
    if (blockIdx.x >= n_rows) return;
    int row = rows[blockIdx.x];
    int start = row_off[row];
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
    const int *est, int A_rows, int ultra_thr, int *bucket_id, int *counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int e = est[i];
    int bid;
    if (e <= ultra_thr) bid = N_BINS - 1;
    else {
        int bi = 0, ht = 32;
        while (ht < e && ht < HASH_CAP) { ht <<= 1; bi++; }
        bid = (bi < N_BINS - 1) ? bi : (N_BINS - 2);
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
__global__ void scan_inclusive_kernel(const int *x, int *y, int n) {
    extern __shared__ int ss[];
    int tid = threadIdx.x;
    int N = blockDim.x;
    ss[tid] = (tid < n) ? x[tid] : 0;
    __syncthreads();
    for (int off = 1; off < N; off <<= 1) {
        int v = (tid >= off) ? ss[tid - off] : 0;
        __syncthreads();
        ss[tid] += v;
        __syncthreads();
    }
    if (tid < n) y[tid] = ss[tid];
}

// 小行专用:accumulate 已在 SMEM 内 count-sort 写有序,这里只把 tmp(gapped)→ CSR(packed) 纯 copy。
__global__ void hash_compact_copy_kernel(
    const int *rows, int n_rows,
    const int *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val)
{
    if (blockIdx.x >= n_rows) return;
    int row = rows[blockIdx.x];
    int src = row_off[row], n = row_nnz[row], dst = row_ptr[row];
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        out_col[dst + t] = (int)(tmp_key[src + t] & 0xffffffffu);
        out_val[dst + t] = tmp_val[src + t];
    }
}

// launch helper:按 config 查 BlockRadixSort TempStorage 大小,>48KB 自动 opt-in 动态 shared(H100 可 ~228KB)。
template<int TPB, int IPT>
static void launch_csort(int n, const int *rows_ptr, const int *d_off, const int *d_row_nnz,
                         const int *dC_rp, const unsigned long long *d_tmp_key, const double *d_tmp_val,
                         int *dC_ci, double *d_val) {
    using BRS = cub::BlockRadixSort<unsigned int, TPB, IPT, double>;
    size_t smem = sizeof(typename BRS::TempStorage);
    if (smem > 48 * 1024)
        CHECK_CUDA(cudaFuncSetAttribute((const void*)hash_compact_sort_kernel<TPB, IPT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    hash_compact_sort_kernel<TPB, IPT><<<n, TPB, smem>>>(
        rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
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

    // Stage 1: row_off 由 est 的 scan 给出(小阵 streamline 用 flop_ub count)
    int *d_off; d_off = decltype(d_off)(dev_alloc((A_rows + 1) * sizeof(int)));

    // 估计每行 distinct:小阵(A_nnz<STREAMLINE_NNZ)→ flop_ub(1 并行 count kernel);大阵 → MinHash 两阶段
    int *d_est; d_est = decltype(d_est)(dev_alloc(A_rows * sizeof(int)));
    if (A_nnz < STREAMLINE_NNZ) {
        prof("count_flop", [&]{
            count_intermediates_par_kernel<<<A_rows, 256>>>(dB_rp, dB_ci, A_rows, d_est);
            CHECK_CUDA(cudaGetLastError());
        });
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
                dA_rp, dA_ci, A_rows, d_mh, d_est);
            CHECK_CUDA(cudaGetLastError());
        });
        dev_free(d_mh);
    }
    int total_est;
    prof("est_scan", [&]{
        CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
        if (A_rows <= 1024) {
            int b = 1; while (b < A_rows) b <<= 1;                       // 小阵:单 block scan(1 launch)
            scan_inclusive_kernel<<<1, b, b * sizeof(int)>>>(d_est, d_off + 1, A_rows);
        } else {
            thrust::inclusive_scan(thrust::device_ptr<int>(d_est),
                                   thrust::device_ptr<int>(d_est + A_rows),
                                   thrust::device_ptr<int>(d_off + 1));
        }
        CHECK_CUDA(cudaMemcpy(&total_est, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入计时
    });
    dbg("[hash] total_est=%d\n", total_est);

    // Stage 2: GPU 端分桶(全 device,无 host 往返)+ 预分配大 buffer(零 per-bucket malloc/free)
    int *d_row_nnz; d_row_nnz = decltype(d_row_nnz)(dev_alloc(A_rows * sizeof(int)));
    int *d_overflow; d_overflow = decltype(d_overflow)(dev_alloc(sizeof(int)));
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
    prof("binning", [&]{
        CHECK_CUDA(cudaMemset(d_cnt, 0, N_BINS * sizeof(int)));   // 先清零(fused kernel 内 atomicAdd 累加)
        compute_bucket_kernel<<<(A_rows + 255) / 256, 256>>>(d_est, A_rows, EST_ULTRA_THR, d_bkid, d_cnt);
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
        for (int bi = 0; bi < N_BINS; bi++) {
            int n = h_cnt[bi];
            if (n == 0) continue;
            int *rows_ptr = d_sort + h_off[bi];
            if (bi == N_BINS - 1) {
                // ultra(est≤EST_ULTRA_THR):线性,免 hash
                hash_ultra_kernel<<<(n + 255) / 256, 256>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
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
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
                }
            }
        }
        CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入
    });
    if (overflow) {
        fprintf(stderr, "[hash] OVERFLOW: 某行 distinct 列 > HASH_CAP=%d → 回退 merge(dispatcher 处理)\n", HASH_CAP);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
        dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
        dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort); dev_free(d_est);
        dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val);
        return;
    }

    // Stage 3: scan row_nnz → C_row_ptr + C_nnz(精确)
    int *dC_rp; dC_rp = decltype(dC_rp)(dev_alloc((A_rows + 1) * sizeof(int)));
    int C_nnz_result;
    prof("cnnz_scan", [&]{
        CHECK_CUDA(cudaMemset(dC_rp, 0, sizeof(int)));
        if (A_rows <= 1024) {
            int b = 1; while (b < A_rows) b <<= 1;                       // 小阵:单 block scan(1 launch)
            scan_inclusive_kernel<<<1, b, b * sizeof(int)>>>(d_row_nnz, dC_rp + 1, A_rows);
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
                hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
            } else if (bi <= 7) {
                launch_csort<512, 8>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
            } else if (bi <= 9) {
                launch_csort<256, 64>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
            } else {
                // ultra(行≤32,无序):小 config sort
                launch_csort<64, 1>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val);
            }
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
