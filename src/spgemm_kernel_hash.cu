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

// 二分:返回 [lo,hi) 内第一个 arr[idx] >= val 的下标(与 spgemm_merge.cu 同款,跨 TU 内联用本地拷贝)
__device__ __forceinline__ int dev_lower_bound(const int *arr, int lo, int hi, int val) {
    while (lo < hi) { int mid = (lo + hi) >> 1; if (arr[mid] < val) lo = mid + 1; else hi = mid; }
    return lo;
}

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

// 融合版(docs/27 §4.6):warp-per-row 一趟出 flop + span_lo/len + max_b_len(省 1 launch + 一趟 A×B 遍历)
// ⚠ 仅 AA 路径(count 的外层 = dB_rp 与 row_span 的 dA_rp 在 att 下语义不同,att 保留双核)
__global__ void count_flop_span_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows,
    const int *B_row_ptr, const int *B_col_idx,
    int *ub, int *span_lo, int *span_len, int *max_b_len)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int i = warp;
    if (i >= A_rows) return;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int s = 0, lo = INT_MAX, hi = -1, mb = 0;
    for (int p = rs + lane; p < re; p += 32) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        s += b1 - b0;
        if (b0 < b1) {
            int c0 = B_col_idx[b0], c1 = B_col_idx[b1 - 1];
            lo = min(lo, c0); hi = max(hi, c1); mb = max(mb, b1 - b0);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xFFFFFFFF, s, off);
        lo = min(lo, __shfl_down_sync(0xFFFFFFFF, lo, off));
        hi = max(hi, __shfl_down_sync(0xFFFFFFFF, hi, off));
        mb = max(mb, __shfl_down_sync(0xFFFFFFFF, mb, off));
    }
    if (lane == 0) {
        ub[i] = s;
        span_lo[i] = (lo == INT_MAX) ? 0 : lo;
        span_len[i] = (hi < lo) ? 0 : (hi - lo + 1);
        if (max_b_len) max_b_len[i] = mb;
    }
}

#ifndef HASH_CAP
#define HASH_CAP 16384         // 重行桶 hash 表最大槽位(→ col+val = 128KB/块)
#endif
#define HASH_BLOCK 256
#define N_BINS 18    // 0..10:hash 梯 + ultra(11) + heavy(12) + dense-iter(13) + 小跨度 dense 4 子桶(14-17,docs/37)
#define BIN_SSPAN0 14   // span≤256(64线程/3.3KB,32+行/SM —— Ocean dense bin0 同位)
#define BIN_ULTRA 11
#define BIN_HEAVY 12
#define BIN_DITER 13
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
    double expand,                      // EST_EXPAND(运行时:小阵 1.4 免重试 / 大阵 1.15 省内存)
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
        int temp = (int)(E * expand);
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

// localLoadBalance 移植(docs/27 §4.2,Ocean AccumulatorCommon.cuh:67 同款语义):
// 按行 (a_len, flop, max_b_len) 动态选 2^log_nthr 线程/k —— 起步均值【排除最长 B 行】(它是
// straggler,由后续双向夹逼吸收),再按 max_sub_iter vs num_iters 的 2× 失衡双向调 G。
// 我们无 warp 内在依赖,G 上限 = HASH_BLOCK(整 block 伺候一个 k)。返回 log2(G)。
__device__ __forceinline__ int local_load_balance(int n_element_a, int num_products, int max_elements_b,
                                                  int lbound, int ubound)
{
    if (num_products <= 0) return lbound;
    int avg_ops = (num_products - max_elements_b) / (n_element_a > 1 ? n_element_a - 1 : 1);
    if (avg_ops < 1) avg_ops = 1;
    int log_nthr = 31 - __clz((unsigned)avg_ops);
    if (log_nthr < 1) log_nthr = 1;
    if ((1 << log_nthr) * 3 < avg_ops * 2) log_nthr += 1;   // 取最近 2^k(非线性截断)
    if (log_nthr > ubound) log_nthr = ubound;               // 预钳位:防 1<<(ubound-log_nthr) 负移位 UB(Ocean 原版隐患)
    int a_elem_per_iter = 1 << (ubound - log_nthr);
    int num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    int max_sub_iter = (max_elements_b + (1 << log_nthr) - 1) >> log_nthr;
    while (max_sub_iter > num_iters * 2) {
        if (log_nthr >= ubound) break;
        log_nthr++;
        max_sub_iter = (max_elements_b + (1 << log_nthr) - 1) >> log_nthr;
        a_elem_per_iter = 1 << (ubound - log_nthr);
        num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    }
    while (num_iters > max_sub_iter * 2) {
        if (log_nthr <= lbound) break;
        log_nthr--;
        max_sub_iter = (max_elements_b + (1 << log_nthr) - 1) >> log_nthr;
        a_elem_per_iter = 1 << (ubound - log_nthr);
        num_iters = (n_element_a + a_elem_per_iter - 1) / a_elem_per_iter;
    }
    while (a_elem_per_iter > n_element_a && (1 << log_nthr) < max_elements_b) {
        log_nthr++;
        a_elem_per_iter = 1 << (ubound - log_nthr);
    }
    if (log_nthr < lbound) log_nthr = lbound;
    if (log_nthr > ubound) log_nthr = ubound;
    return log_nthr;
}

// ===================== dense 累积器路径(2026-08-26,超越点) =====================
// 触发:n ≤ DENSE_MAX_N 且 输出足够稠密(avg_est/n ≥ DENSE_MIN_FRAC)—— exdata_1 类
// (6001²,est 1878/行 = 31% 稠密,76× dup)。SMEM 直接寻址表 vals[n]+flags[n]:
// 零探测、零 CAS(槽位即列号),纯 atomicAdd —— 高 dup 下原子链是 hash 与 Ocean 共有的
// 结构瓶颈,稠密阵直接绕开。对标 Ocean AccumulatorDense(他们的 FAT 工作流)。
// 有序性免费:j 升序扫 flags → 输出行有序,compact 全走 copy。
#define DENSE_MAX_N 14980            // SMEM 预算:8n + n + 4n ≈ 190KB
#define DENSE_MIN_FRAC 0.15          // avg_est/n ≥ 此值才走 dense

__global__ void hash_dense_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri, int A_rows, int n,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x;
    extern __shared__ __align__(8) unsigned char dsmem[];
    double *dval = (double*)dsmem;                    // [n]
    unsigned char *dflag = dsmem + (size_t)n * sizeof(double);   // [n]
    int *dpref = (int*)(dsmem + (((size_t)n * 9) + 3) / 4 * 4);  // [n](9n 向上 4 对齐;exdata n=6001 曾错位崩)
    for (int j = tid; j < n; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
    __syncthreads();

    // 累加:warp 取 k(stride nwarps),lane 并行 j;k 内 B 行列唯一 → 同 warp 内无同 j;
    // 跨 warp 同 j → atomicAdd(链长 = dup 因子,已是下界)。
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int p = rs + warp; p < re; p += nw) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            atomicAdd(&dval[j], a * B_val[q]);
            dflag[j] = 1;                              // 幂等字节写,无竞争语义
        }
    }
    __syncthreads();

    // 有序 extract:warp0 做 flags 的全局 exclusive 前缀(分段串行 + warp 段基址扫),全块写
    if (warp == 0) {
        int seglen = (n + 31) / 32;
        int lo = lane * seglen, hi = min(lo + seglen, n);
        int run = 0;
        for (int j = lo; j < hi; j++) { dpref[j] = run; run += dflag[j]; }
        int excl = run;
        for (int off = 1; off < 32; off <<= 1) {
            int v = __shfl_up_sync(0xffffffff, excl, off);
            if (lane >= off) excl += v;
        }
        // excl(lane) = lanes < lane 的段和总和 → 写最终 dpref
        for (int j = lo; j < hi; j++) dpref[j] += excl - run;   // excl - run = 前面所有段之和
    }
    __syncthreads();
    int total_nz = dpref[n - 1] + dflag[n - 1];
    long long base2 = row_off[i];
    int cap = est[i];
    for (int j = tid; j < n; j += blockDim.x) {
        if (dflag[j]) {
            int rank = dpref[j];
            if (rank >= cap) {                          // SAFETY:欠估守卫(同族)
                if (!atomicExch(&row_ovf[i], 1)) { int q = atomicAdd(ovf_cnt, 1); ovf_rows[q] = i; }
                atomicExch(overflow_flag, 1);
                continue;
            }
            tmp_key[base2 + rank] = ((unsigned long long)i << 32) | (unsigned int)j;
            tmp_val[base2 + rank] = dval[j];
        }
    }
    if (tid == 0) row_nnz[i] = (total_nz <= cap) ? total_nz : cap;
}

// ===================== 小跨度 dense 4 子桶(docs/37 = 35 §3 设计 + 36 号 spECK/nsparse 弹药)=====================
// span∈{256,512,1024,2048} × 块{64,128,256,256}:1 行/块但小块+小 SMEM → 8-32 行/SM(Ocean dense 梯同位);
// 直接寻址免 CAS 免探测;warp0 前缀有序 extract → compact 走 copy。est≥span/2 由路由门保证利用率。
template<int SPAN, int TPB>
__global__ void hash_sspan2_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri, int A_rows,
    const int *bucket_rows,
    const int *span_lo, const int *span_len,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int i = bucket_rows[blockIdx.x];
    if (i >= A_rows) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = TPB >> 5;
    extern __shared__ __align__(8) unsigned char ssm2[];
    double *dval = (double*)ssm2;                                    // [SPAN]
    unsigned char *dflag = ssm2 + (size_t)SPAN * sizeof(double);     // [SPAN]
    int *dpref = (int*)(ssm2 + (((size_t)SPAN * 9) + 3) / 4 * 4);    // [SPAN]
    int lo = span_lo[i];
    int w_ = span_len[i];
    if (w_ <= 0) w_ = 1;
    if (w_ > SPAN) w_ = SPAN;   // SAFETY:路由门已限
    for (int j = tid; j < w_; j += TPB) { dval[j] = 0.0; dflag[j] = 0; }
    __syncthreads();
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs + warp; p < re; p += nw) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int j = B_col_idx[q] - lo;
            if (j < 0 || j >= w_) continue;
            int col = lo + j;
            if (upper_tri && col < i) continue;
            atomicAdd(&dval[j], a * B_val[q]);
            dflag[j] = 1;
        }
    }
    __syncthreads();
    if (warp == 0) {
        int seglen = (w_ + 31) / 32;
        int slo = lane * seglen, shi = min(slo + seglen, w_);
        int run = 0;
        for (int j = slo; j < shi; j++) { dpref[j] = run; run += dflag[j]; }
        int excl = run;
        for (int off = 1; off < 32; off <<= 1) {
            int v = __shfl_up_sync(0xffffffff, excl, off);
            if (lane >= off) excl += v;
        }
        for (int j = slo; j < shi; j++) dpref[j] += excl - run;
    }
    __syncthreads();
    long long base = row_off[i];
    long long cap = row_off[i + 1] - row_off[i];
    int count = dpref[w_ - 1] + dflag[w_ - 1];
    for (int j = tid; j < w_; j += TPB) {
        if (dflag[j]) {
            int rank = dpref[j];
            if (rank >= cap) {
                atomicExch(overflow_flag, 1);
                if (!atomicExch(&row_ovf[i], 1)) { int q2 = atomicAdd(ovf_cnt, 1); ovf_rows[q2] = i; }
                break;
            }
            tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)(lo + j);
            tmp_val[base + rank] = dval[j];
        }
    }
    if (tid == 0) row_nnz[i] = (count <= cap) ? count : (int)cap;
}

// ===================== dense 窗口版(2026-08-26 Step3,治 TSOPF 类) =====================
// n > SMEM 预算但输出稠密的矩阵:每行一 CTA,列域切成 ≤ DENSE_MAX_N 的窗口,
// 逐窗口 SMEM 稠密累加(每 k 用 lower_bound 切出窗口内切片,免重复处理),
// 窗口升序 → 提取拼接天然全局有序,零跨 CTA 同步。对标 Ocean denseNumericIterKernel。
__global__ void hash_dense_window_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri, int A_rows, int n,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    const int *bucket_rows /*nullable:bin 行列表;null=矩阵级模式,blockIdx 即行号*/,
    const int *span_lo /*nullable:每行列跨度起点;null=0(矩阵级模式扫全 [0,n))*/,
    const int *row_flop = nullptr, const int *row_maxbl = nullptr /*Phase B v2-lite 动态组*/)
{
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x;
    int lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    extern __shared__ __align__(8) unsigned char wsmem[];
    double *dval = (double*)wsmem;                                       // [W]
    unsigned char *dflag = wsmem + (size_t)DENSE_MAX_N * sizeof(double);
    int *dpref = (int*)(wsmem + (((size_t)DENSE_MAX_N * 9) + 3) / 4 * 4); // [W]

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    long long base = row_off[i];
    int cap = est[i];
    int out = 0;
    bool ovf = false;
    // 跨度窗口 v2(docs/24):窗口对准本行【真实列跨度】[span_lo, n)(由 span_kernel 预计算),
    // 不再从 0 扫到 n —— F2 类带状 FEM(span~3k vs n=71k)从 5 个全窗(99% 空扫)变 1 个紧窗。
    // 矩阵级模式(span_lo=null)保持扫全 [0,n)。extract 仍窗口升序 → 全局有序契约不变。
    int lo0 = span_lo ? span_lo[i] : 0;
    int nw_win = span_lo ? (n - lo0 + DENSE_MAX_N - 1) / DENSE_MAX_N
                         : (n + DENSE_MAX_N - 1) / DENSE_MAX_N;
    for (int wi = 0; wi < nw_win; wi++) {
        int c0 = lo0 + wi * DENSE_MAX_N, c1 = min(c0 + DENSE_MAX_N, n);
        int w = c1 - c0;
        for (int j = tid; j < w; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
        __syncthreads();
        for (int p = rs + warp; p < re; p += nw) {
            int k = A_col_idx[p];
            double a = A_val[p];
            int ks = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], c0);
            int ke = dev_lower_bound(B_col_idx, ks, B_row_ptr[k + 1], c1);
            for (int q = ks + lane; q < ke; q += 32) {
                int j = B_col_idx[q] - c0;
                int col = c0 + j;
                if (upper_tri && col < i) continue;
                atomicAdd(&dval[j], a * B_val[q]);
                dflag[j] = 1;
            }
        }
        __syncthreads();
        // 窗口内有序前缀(warp0 分段 + 段基址扫,同 dense v1)
        if (warp == 0) {
            int seglen = (w + 31) / 32;
            int lo = lane * seglen, hi = min(lo + seglen, w);
            int run = 0;
            for (int j = lo; j < hi; j++) { dpref[j] = run; run += dflag[j]; }
            int excl = run;
            for (int off = 1; off < 32; off <<= 1) {
                int v = __shfl_up_sync(0xffffffff, excl, off);
                if (lane >= off) excl += v;
            }
            for (int j = lo; j < hi; j++) dpref[j] += excl - run;
        }
        __syncthreads();
        for (int j = tid; j < w; j += blockDim.x) {
            if (dflag[j]) {
                int col = c0 + j;
                if (upper_tri && col < i) continue;
                int rank = out + dpref[j];
                if (rank >= cap) { ovf = true; continue; }
                tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)col;
                tmp_val[base + rank] = dval[j];
            }
        }
        out += dpref[w - 1] + dflag[w - 1];   // 窗口计数(w>0 因 n≥1 时首窗非空)
        __syncthreads();
    }
    if (ovf) {
        atomicExch(overflow_flag, 1);
        if (!atomicExch(&row_ovf[i], 1)) { int q = atomicAdd(ovf_cnt, 1); ovf_rows[q] = i; }
    }
    if (tid == 0) row_nnz[i] = (out <= cap) ? out : cap;
}

// ===================== 方案5:dense 行精确计数 + 直写终态(docs/27 §4.1,2026-08-27)=====================
// Ocean type1 工作流的核心经济性:两遍直写 vs 单遍 gapped 的交通量临界 = dup≈6~8(v4 门 dup<8 对齐)。
//   count pass:只标记 flag(1B/列,免 dval 8B+atomicAdd)→ row_nnz = 精确 distinct(无 cap,永不 ovf);
//   numeric pass:窗口升序直写 dC 精确偏移 —— dense 行的 gapped 写 + compact copy 整体消失。
// hash 行不动(结构已同 Ocean type2 est 工作流)。DIRECT5=1 开启,默认关。
#define DENSE_CNT_W 65536   // 计数窗口(SMEM 1B/列;数值 pass 窗口 DENSE_MAX_N 不同不影响总数)

// count pass:窗口扫描标 flag + 块归约;窗口划分与数值 pass 不同也无妨( disjoint cover 计数同)。
__global__ void hash_dense_count_kernel(
    const int *A_row_ptr, const int *A_col_idx,
    const int *B_row_ptr, const int *B_col_idx,
    int upper_tri, int A_rows, int n,
    int *row_nnz,
    const int *bucket_rows /*nullable,同 window kernel*/,
    const int *span_lo /*nullable*/,
    const int *row_flop = nullptr, const int *row_maxbl = nullptr /*Phase B v2-lite*/)
{
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    extern __shared__ __align__(8) unsigned char csmem[];   // [DENSE_CNT_W] flags
    __shared__ int wcnt[16];                                 // blockDim 512 = 16 warp
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int total = 0;
    int lo0 = span_lo ? span_lo[i] : 0;
    int nw_win = (n - lo0 + DENSE_CNT_W - 1) / DENSE_CNT_W;
    for (int wi = 0; wi < nw_win; wi++) {
        int c0 = lo0 + wi * DENSE_CNT_W, c1 = min(c0 + DENSE_CNT_W, n);
        int w = c1 - c0;
        for (int j = tid; j < w; j += blockDim.x) csmem[j] = 0;
        __syncthreads();
        for (int p = rs + warp; p < re; p += nw) {
            int k = A_col_idx[p];
            int ks = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], c0);
            int ke = dev_lower_bound(B_col_idx, ks, B_row_ptr[k + 1], c1);
            for (int q = ks + lane; q < ke; q += 32) {
                int j = B_col_idx[q] - c0;
                int col = c0 + j;
                if (upper_tri && col < i) continue;   // 与数值 pass 同过滤位置(标记前)
                csmem[j] = 1;
            }
        }
        __syncthreads();
        int cnt = 0;
        for (int j = tid; j < w; j += blockDim.x) cnt += csmem[j];
        for (int off = 16; off; off >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, off);
        if (lane == 0) wcnt[warp] = cnt;
        __syncthreads();
        if (tid == 0) { int s = 0; for (int x = 0; x < nw; x++) s += wcnt[x]; total += s; }
        __syncthreads();   // 读完 wcnt 才可进入下一窗口清 flags
    }
    if (tid == 0) row_nnz[i] = total;
}

#ifndef PB2_STAGE
#define PB2_STAGE 4
#endif
// dense 行 a_len 采集/散射(全局 cursor 区偏移构造)
__global__ void alen_gather_kernel(const int *rows, int nr, const int *A_rp, int *alen) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nr) { int r = rows ? rows[j] : j; alen[j] = A_rp[r + 1] - A_rp[r]; }
}
__global__ void scatter_alen_kernel(const int *rows, int nr, const int *alen, int *off_by_row /*[A_rows],已清零*/) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < nr) off_by_row[rows ? rows[j] : j] = alen[j];
}

// numeric pass:直写终态 + Ocean iter 内核机制(docs/30 Phase B v2):start_map 游标 + 动态组
// + 数据驱动窗口。三件套消除旧窗口内核的两个结构税:①每窗每 k 的 lower_bound 重复搜索
// (G>32 时冗余 = G×2log(avgB),c-58/case39/gupta1/TSOPF 回归主因);②固定 warp-per-k 的
// 线程利用率 32×a_len/512(mult_dcop 31% 占用 = 6.9× 差距主因)。游标:每 k 一个 SMEM 槽,
// 首窗一次 lower_bound 定位到 span_lo,越窗列回写 atomicMin(= O(1) 推进,免重复二分);
// 下一窗起点 = 全体越窗列的最小列(数据驱动,跳空窗)。窗口宽 PB2_W(SMEM 17B/列:
// val8+flag1+pref4+cursor4 → 12900 列 ≈ 219KB)。
#define PB2_W 12900   // 游标版窗口宽(SMEM 预算 13B×W + 双游标区 ≤ ~227KB)
#define SMAP_SMEM_MAX 7300   // a_len ≤ 此值的行游标进 SMEM(docs/35 §3:TSOPF 修法;预算 2×4B×7300)
__global__ void hash_dense_direct_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri, int A_rows, int n,
    const int *exact_rp, int *dC_ci, double *dC_val,
    int *row_nnz,
    const int *bucket_rows /*nullable*/,
    const int *span_lo /*nullable*/,
    const int *row_flop = nullptr, const int *row_maxbl = nullptr,
    int *smap_base = nullptr, const int *smap_off = nullptr, int smir_bias = 0,
    int use_cursor = 1)
{
    int i = bucket_rows ? bucket_rows[blockIdx.x] : blockIdx.x;
    if (i >= A_rows) return;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    extern __shared__ __align__(8) unsigned char wsmem[];
    double *dval = (double*)wsmem;
    unsigned char *dflag = wsmem + (size_t)PB2_W * sizeof(double);
    int *dpref = (int*)(wsmem + (((size_t)PB2_W * 9) + 3) / 4 * 4);
    int *smap_g = smap_base + smap_off[i];                  // 全局游标(大 a_len 回退)
    int *smir_g = smap_base + smir_bias + smap_off[i];
    int *smap_s = dpref + PB2_W;                            // SMEM 游标区(与 dpref 分立)
    int *smir_s = smap_s + SMAP_SMEM_MAX;
    __shared__ int s_next_lo;

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int a_len = re - rs;
    int *smap = (a_len <= SMAP_SMEM_MAX) ? smap_s : smap_g; // docs/35 §3:SMEM 消每窗 O(a_len) 全局往返
    int *smir = (smap == smap_s) ? smir_s : smir_g;
    int base = exact_rp[i];
    int cap = row_nnz[i];
    int out = 0;
    int lo0 = span_lo ? span_lo[i] : 0;

    int logG = (row_flop && row_maxbl) ? local_load_balance(a_len, row_flop[i], row_maxbl[i], 5, 9)
                                       : 5;
    int G = 1 << logG, num_groups = blockDim.x >> logG;
    int my_group = tid >> logG, my_id = tid & (G - 1);

    for (int p = rs + tid; p < re; p += blockDim.x) {
        int k = A_col_idx[p];
        smap[p - rs] = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], lo0);
    }
    if (tid == 0) s_next_lo = n;
    __syncthreads();
    if (PB2_STAGE < 1) return;

    int avgB = (row_flop && a_len > 0) ? row_flop[i] / a_len : 0;
    if (avgB < 64 || use_cursor != 1) {   // docs/39 路由 v3 判据收回 avgB(host 级 dup 门见下)
        // 混合路由(docs/30):avgB(=flop/a_len,平均 B 行长)< 64 的行走旧固定窗搜索路径 ——
        // 游标镜像/复位是每窗 O(a_len) 全局往返,开销/工作量 ∝ a_len×窗数/flop = 1/avgB;
        // TSOPF(avgB≈16)游标版 +22% 实锤,mult_dcop(avgB≈4000)游标 -42%。旧路径 = 52b89da 行为。
        int nw_win = span_lo ? (n - lo0 + PB2_W - 1) / PB2_W
                             : (n + PB2_W - 1) / PB2_W;
        for (int wi = 0; wi < nw_win; wi++) {
            int c0 = lo0 + wi * PB2_W, c1 = min(c0 + PB2_W, n);
            int w = min(PB2_W, n - c0);
            for (int j = tid; j < w; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
            __syncthreads();
            for (int p = rs + warp; p < re; p += nw) {
                int k = A_col_idx[p];
                double a = A_val[p];
                int ks = dev_lower_bound(B_col_idx, B_row_ptr[k], B_row_ptr[k + 1], c0);
                int ke = dev_lower_bound(B_col_idx, ks, B_row_ptr[k + 1], c1);
                for (int q = ks + lane; q < ke; q += 32) {
                    int j = B_col_idx[q] - c0;
                    if (upper_tri && (c0 + j) < i) continue;
                    atomicAdd(&dval[j], a * B_val[q]);
                    dflag[j] = 1;
                }
            }
            __syncthreads();
            if (warp == 0) {
                int seglen = (w + 31) / 32;
                int lo = lane * seglen, hi = min(lo + seglen, w);
                int run = 0;
                for (int j = lo; j < hi; j++) { dpref[j] = run; run += dflag[j]; }
                int excl = run;
                for (int off = 1; off < 32; off <<= 1) {
                    int v = __shfl_up_sync(0xffffffff, excl, off);
                    if (lane >= off) excl += v;
                }
                for (int j = lo; j < hi; j++) dpref[j] += excl - run;
            }
            __syncthreads();
            for (int j = tid; j < w; j += blockDim.x) {
                if (dflag[j]) {
                    int rank = out + dpref[j];
                    if (rank >= cap) continue;
                    dC_ci[base + rank] = c0 + j;
                    dC_val[base + rank] = dval[j];
                }
            }
            out += dpref[w - 1] + dflag[w - 1];
            __syncthreads();
        }
        if (tid == 0 && out != cap) row_nnz[i] = out;
        return;
    }
    int win_lo = lo0;
    while (win_lo < n) {
        int w = min(PB2_W, n - win_lo);
        for (int j = tid; j < w; j += blockDim.x) { dval[j] = 0.0; dflag[j] = 0; }
        __syncthreads();
        if (tid == 0) s_next_lo = n;
        __syncthreads();
        // 每窗读游标进镜像 + 原槽复位 INT_MAX(Ocean 原版 reset-to-∞ 语义):否则旧游标恒小于
        // 新 break 值 → atomicMin 永不更新 → 游标不前进 → 下窗重读旧位置 → col<win_lo → j<0
        // → SMEM 下越界 → illegal instruction(今晚实锤的坑,docs/30)
        for (int p = rs + tid; p < re; p += blockDim.x) {
            int v = smap[p - rs];
            smir[p - rs] = v;
            smap[p - rs] = 0x7fffffff;
        }
        __syncthreads();
        int my_next = n;
        if (PB2_STAGE >= 2) {
            for (int p = rs + my_group; p < re; p += num_groups) {
                int st = smir[p - rs];
                if (st == 0x7fffffff) continue;   // 哨兵:该 k 已耗尽(INT_MAX+my_id 会溢出为负 → 全局越界)
                int k = A_col_idx[p];
                double a = A_val[p];
                int b_end = B_row_ptr[k + 1];
                for (int q = st + my_id; q < b_end; q += G) {
                    int j = B_col_idx[q] - win_lo;
                    if (j >= w) {
                        atomicMin(&smap[p - rs], q);
                        if (B_col_idx[q] < my_next) my_next = B_col_idx[q];
                        break;
                    }
                    int col = win_lo + j;
                    if (upper_tri && col < i) continue;
                    atomicAdd(&dval[j], a * B_val[q]);
                    dflag[j] = 1;
                }
            }
            if (my_next < n) atomicMin(&s_next_lo, my_next);
        }
        __syncthreads();
        if (PB2_STAGE >= 3) {
            if (warp == 0) {
                int seglen = (w + 31) / 32;
                int lo = lane * seglen, hi = min(lo + seglen, w);
                int run = 0;
                for (int j = lo; j < hi; j++) { dpref[j] = run; run += dflag[j]; }
                int excl = run;
                for (int off = 1; off < 32; off <<= 1) {
                    int v = __shfl_up_sync(0xffffffff, excl, off);
                    if (lane >= off) excl += v;
                }
                for (int j = lo; j < hi; j++) dpref[j] += excl - run;
            }
        }
        __syncthreads();
        if (PB2_STAGE >= 4) {
            for (int j = tid; j < w; j += blockDim.x) {
                if (dflag[j]) {
                    int rank = out + dpref[j];
                    if (rank >= cap) continue;
                    dC_ci[base + rank] = win_lo + j;
                    dC_val[base + rank] = dval[j];
                }
            }
            out += dpref[w - 1] + dflag[w - 1];
        }
        int next = s_next_lo;
        __syncthreads();
        win_lo = next;
    }
    if (tid == 0 && out != cap)
        row_nnz[i] = out;
}

// dense 行集合的 Σest / Σrow_nnz / Σflop(方案5:tmp 缩容 + est 下溢检查改 hash 侧口径 + dup 门)
// sum 用 unsigned ll 原子(sm_90 无 signed ll atomicAdd 重载;值为非负,位型等同)
__global__ void dense_sum_kernel(const int *rows, int nr, const int *est, const int *row_nnz,
                                 const int *flop /*nullable:方案5 dup 门 pre-pass*/,
                                 unsigned long long *sum_est, unsigned long long *sum_nnz) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nr) return;
    if (est) atomicAdd(sum_est, (unsigned long long)est[rows ? rows[r] : r]);
    if (row_nnz) atomicAdd(sum_nnz, (unsigned long long)row_nnz[rows ? rows[r] : r]);
    if (flop) atomicAdd(sum_nnz, (unsigned long long)flop[rows ? rows[r] : r]);   // 门 pre-pass 复用 sum_nnz 槽
}

// dense 行 est 置 0(方案5:d_off 重扫前坍缩 gapped 空间;binning 已完成,est 无后续消费者)
__global__ void zero_est_kernel(int *est, const int *rows, int nr) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < nr) est[rows[r]] = 0;
}

// ===================== sub-warp 批量 kernel(2026-08-26 方向B,巨型图第二程) =====================
// est ≤ 8 的极稀疏行(germany_osm/road 类 avg_product≈5):每行只派 8 线程(4 行/warp),
// 去重累加在**寄存器**里(≤8 个 col/val 对,零 SMEM 零原子),warp shfl 去重+求和。
// 对标 Ocean hashNumericSubWarpKernel<4>(它 4 线程/行,hash 表在 SMEM;我们寄存器更轻)。
#define SUBW_HT 8    // 寄存器表容量(每行 ≤ 8 distinct)

// 干净版:8 线程/行,SMEM 小表(32 slot),warp 内 4 行不共享表
__global__ void hash_subwarp8_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf)
{
    int lane = threadIdx.x & 31;
    int row_slot = lane >> 3;                    // 0-3
    int sub_lane = lane & 7;                     // 0-7
    int rows_per_warp = 32 >> 3;                 // 4
    int row_idx = ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * rows_per_warp + row_slot;
    if (row_idx >= n_in_bucket) return;
    int i = bucket_rows[row_idx];

    // SMEM 表:每 warp 4 行 × 32 slot × 12B = 1.5KB/warp,256 线程 8 warp = 12KB
    __shared__ int s_col[8][4][32];              // [warp][row_slot][slot]
    __shared__ double s_val[8][4][32];
    int warp = threadIdx.x >> 5;
    int *my_col = s_col[warp][row_slot];
    double *my_val = s_val[warp][row_slot];
    for (int s = sub_lane; s < 32; s += 8) { my_col[s] = -1; my_val[s] = 0.0; }
    __syncwarp(0xFFu << (row_slot * 8));         // 同行 8 线程同步(warp 内子组)

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + sub_lane; q < ke; q += 8) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & 31;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&my_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&my_val[slot], v); break; }
                slot = (slot + 1) & 31;
                if (++probes >= 32) {
                    atomicExch(overflow_flag, 1);
                    if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
                    break;
                }
            }
        }
    }
    __syncwarp(0xFFu << (row_slot * 8));

    // 提取 + 插入排序(单线程,8 线程中 sub_lane==0 做)
    long long base = row_off[i];
    long long cap = row_off[i + 1] - row_off[i];
    if (sub_lane == 0) {
        int cols[32]; double vals[32]; int cnt = 0;
        for (int s = 0; s < 32; s++) {
            if (my_col[s] >= 0) { cols[cnt] = my_col[s]; vals[cnt] = my_val[s]; cnt++; }
        }
        // 插入排序
        for (int t = 1; t < cnt; t++) {
            int c = cols[t]; double v = vals[t];
            int s2 = t - 1;
            while (s2 >= 0 && cols[s2] > c) { cols[s2+1] = cols[s2]; vals[s2+1] = vals[s2]; s2--; }
            cols[s2+1] = c; vals[s2+1] = v;
        }
        if (cnt > cap) {
            atomicExch(overflow_flag, 1);
            if (!atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
            cnt = (int)cap;
        }
        for (int t = 0; t < cnt; t++) {
            tmp_key[base + t] = ((unsigned long long)i << 32) | (unsigned int)cols[t];
            tmp_val[base + t] = vals[t];
        }
        row_nnz[i] = cnt;
    }
}


#define BATCH_HT 128        // per-warp 表槽(est≤64 → 2× 余量)
#define BATCH_WPB 8         // warp 数/CTA(256 线程)
// ===================== 方案1: 预排序二分寻址 (HSMU HPCA'25 思路) =====================
// 两阶段: symbolic 收集列集合并排序 -> numeric 二分定位 + 直接 atomicAdd
// 零 CAS(位置由二分确定), 零冲突(列集合唯一), 输出天然有序

__global__ void bsearch_symbolic_kernel(
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
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            unsigned slot = ((unsigned)(j * 2654435761u)) & (BATCH_HT - 1);
            int probes = 0;
            while (true) {
                int old = atomicCAS(&t_col[slot], -1, j);
                if (old == -1 || old == j) break;
                slot = (slot + 1) & (BATCH_HT - 1);
                if (++probes >= BATCH_HT) { atomicExch(overflow_flag, 1); break; }
            }
        }
    }
    __syncwarp();

    __shared__ int w_cnt2[BATCH_WPB];
    if (lane == 0) w_cnt2[warp & (BATCH_WPB - 1)] = 0;
    __syncwarp();
    long long base = row_off[i];
    int cap = est[i];
    for (int s = lane; s < BATCH_HT; s += 32) {
        int c = t_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&w_cnt2[warp & (BATCH_WPB - 1)], 1);
            if (pos < BATCH_HT) { pack_col[pos] = c; pack_slot[pos] = s; }
        }
    }
    __syncwarp();
    int n = w_cnt2[warp & (BATCH_WPB - 1)];
    if (n > BATCH_HT) n = BATCH_HT;
    for (int idx = lane; idx < n; idx += 32) {
        int c = pack_col[idx];
        int rank = 0;
        for (int m = 0; m < n; m++) if (pack_col[m] < c) rank++;
        if (rank >= cap) { atomicExch(overflow_flag, 1); continue; }
        tmp_key[base + rank] = ((unsigned long long)i << 32) | (unsigned int)c;
        tmp_val[base + rank] = 0.0;
    }
    __syncwarp();
    if (lane == 0) {
        if (n > cap && !atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
        row_nnz[i] = (n <= cap) ? n : cap;
    }
}

__global__ void bsearch_numeric_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *row_nnz,
    unsigned long long *tmp_key, double *tmp_val)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_in_bucket) return;
    int i = bucket_rows[warp];

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    long long base = row_off[i];
    int nnz = row_nnz[i];

    __shared__ int scol[BATCH_WPB][BATCH_HT];
    for (int s = lane; s < nnz; s += 32) {
        scol[warp & (BATCH_WPB - 1)][s] = (int)(tmp_key[base + s] & 0xffffffffu);
    }
    __syncwarp();
    int *cols = scol[warp & (BATCH_WPB - 1)];

    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + lane; q < ke; q += 32) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            int lo = 0, hi = nnz - 1, pos = -1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (cols[mid] == j) { pos = mid; break; }
                else if (cols[mid] < j) lo = mid + 1;
                else hi = mid - 1;
            }
            if (pos >= 0) atomicAdd(&tmp_val[base + pos], v);
        }
    }
}

// ===================== 小行批量 kernel(2026-08-26,Phase 2 主杠杆) =====================
// est ≤ 64 的行(bin0-1):warp-per-row + SMEM warp 私有表(128 槽,2× 余量)+ 融合有序 extract。
// 治"1 CTA/行×256 线程伺候 ~6 个积"的每行固定开销(333SP 型 accumulate 55% 差距源);
// 有序直写 tmp → compact 阶段走 copy 免排序。设计:`inno/hash_batched_kernel_design.md`。

__global__ void hash_spa_batched_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_off, const int *est,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    double *global_val = nullptr)    // HYBRID=1: 全局 value 池(L2 原子,吞吐 > SMEM)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_in_bucket) return;
    int i = bucket_rows[warp];

    // 静态 SMEM:每 warp 段 = col[128] | val[128](8B 对齐,段首 512B✓) | pack_col[128] | pack_slot[128]
    __shared__ int bsmem[BATCH_WPB][4 * BATCH_HT];
    __shared__ int w_cnt[BATCH_WPB];
    int *t_col = bsmem[warp & (BATCH_WPB - 1)];
    // 方案4修正: HYBRID 模式下 value 在全局(L2 原子吞吐高,CAS 仍在 SMEM)
    double *t_val;
    if (global_val) {
        t_val = global_val + (size_t)(warp) * BATCH_HT;  // 全局池,每 warp 一段
    } else {
        t_val = (double*)(bsmem[warp & (BATCH_WPB - 1)] + BATCH_HT);
    }
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
// MODE:0=legacy(gapped tmp + packed key,compact 阶段善后)1=count-only(只插 key 出精确 row_nnz,
// 方案5 扩展到 hash 行,docs/31)2=direct(exact_rp 精确偏移直写 dC_ci/d_val,行内无序,原地 csort 善后)
template<int MODE>
__global__ void hash_spa_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,   // 内层 j 来源:AA=A,ATT=Aᵀ(CSC)
    int upper_tri,                                                      // 0=自乘;1=ATT 只算 j≥i
    const int *bucket_rows, int n_in_bucket, int ht_size,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    const int *row_flop = nullptr,      // localLoadBalance 输入(行 flop);null=旧静态 G
    const int *row_maxbl = nullptr,     // 行 k 集内 B 行最大长;null=旧静态 G
    int llb = 0,                        // LLB=1:按行动态 G(docs/27 §4.2)
    const int *exact_rp = nullptr, int *dC_ci = nullptr, double *dC_val = nullptr)   // MODE=2
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];                  // 实际行号
    if (MODE == 2 && row_ovf[i]) return;       // count pass 溢出行已由 retry 处理(compact 末段拷)
    int tid = threadIdx.x;
    int mask = ht_size - 1;

    extern __shared__ __align__(8) int smem[];
    int   *sh_col = smem;                      // [ht_size]
    double *sh_val = (double*)(smem + ht_size);  // [ht_size]
    for (int s = tid; s < ht_size; s += HASH_BLOCK) { sh_col[s] = -1; sh_val[s] = 0.0f; }
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    // G 选择:默认按 ht_size 静态梯(重行→G 大);LLB=1 改按行 (a_len, flop, max_b_len) 动态 2^k
    // (Ocean localLoadBalance 同款语义,治 Ge99 类 4.4× 的负载不均;G 可到 HASH_BLOCK=整 block/k)
    int logG;
    if (llb && row_flop && row_maxbl)
        logG = local_load_balance(re - rs, row_flop[i], row_maxbl[i], 2, 31 - __clz((unsigned)HASH_BLOCK));
    else
        logG = (ht_size <= 64) ? 2 : (ht_size <= 256) ? 3 : (ht_size <= 1024) ? 4 : 5;
    int G = 1 << logG;
    int num_groups = HASH_BLOCK / G;
    int my_group = tid / G, my_id = tid % G;
    for (int p = rs + my_group; p < re; p += num_groups) {   // 并行 k(stride num_groups)
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];        // B 的行 k(AA=A;ATT=Aᵀ)
        for (int q = ks + my_id; q < ke; q += G) {           // 组内 G 线程并行 j
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;                // ATT 上三角过滤(j≥i)
            double v = (MODE == 1) ? 0.0 : a_ik * B_val[q];  // count-only 免乘免值写(docs/31)
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;   // Knuth 乘法 hash
            int probes = 0;
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { if (MODE != 1) atomicAdd(&sh_val[slot], v); break; }
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
    if (MODE == 1) {   // count-only:occupancy 即精确 distinct;溢出行已在上面标记(row_nnz 留给 retry)
        int cnt = 0;
        for (int s = tid; s < ht_size; s += HASH_BLOCK) cnt += (sh_col[s] != -1);
        for (int off = 16; off; off >>= 1) cnt += __shfl_down_sync(0xffffffff, cnt, off);
        __shared__ int ccnt[HASH_BLOCK / 32 > 16 ? HASH_BLOCK / 32 : 16];
        if ((tid & 31) == 0) ccnt[tid >> 5] = cnt;
        __syncthreads();
        if (tid == 0) { int t = 0; for (int x = 0; x < (HASH_BLOCK >> 5); x++) t += ccnt[x]; row_nnz[i] = t; }
        return;
    }

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
        if (MODE == 2) {   // 精确偏移直写(有序,免 compact;count pass 已保证无溢出)
            int base2 = exact_rp[i];
            for (int k = tid; k < count; k += HASH_BLOCK) {
                int c = sh_col[k]; double v = sh_val[k]; int rank = 0;
                for (int j = 0; j < count; j++) if (sh_col[j] < c) rank++;
                dC_ci[base2 + rank] = c;
                dC_val[base2 + rank] = v;
            }
            __syncthreads();
            return;   // row_nnz 已由 count pass 写过(精确)
        }
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
    } else if (MODE == 2) {
        // 大行 MODE=2:无序直写 dC(精确偏移;行内乱序 → 原地 csort 善后)
        int base2 = exact_rp[i];
        for (int s = tid; s < ht_size; s += HASH_BLOCK) {
            int c = sh_col[s];
            if (c >= 0) {
                int pos = atomicAdd(&cnt, 1);
                dC_ci[base2 + pos] = c;
                dC_val[base2 + pos] = sh_val[s];
            }
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

// hash_spa_hkv_kernel:Hybrid Value 全 bin 版(docs/24 §4.1,治大表行低占用):
// keys 留 SMEM(4B/槽,occupancy ×3)+ values 走全局 L2 atomicAdd(Ocean HYBRID_HASHMAP 同款)。
// 仅用于 ht ≥ 4096 的 bin(ht > CSORT_HT 恒真)→ 只保留无序 extract 路径(compact 走 csort)。
__global__ void hash_spa_hkv_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket, int ht_size,
    const long long *row_off,
    unsigned long long *tmp_key, double *tmp_val,
    int *row_nnz, int *overflow_flag,
    int *ovf_rows, int *ovf_cnt, int *row_ovf,
    double *gval, const long long *gval_off)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int mask = ht_size - 1;

    extern __shared__ __align__(8) int smem[];
    int *sh_col = smem;                        // [ht_size] 仅 keys(12B→4B/槽)
    for (int s = tid; s < ht_size; s += HASH_BLOCK) sh_col[s] = -1;
    __syncthreads();

    long long gb = gval_off[i];                // 本行 value 池基址(host 已 memset 0)
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int G = (ht_size <= 64) ? 4 : (ht_size <= 256) ? 8 : (ht_size <= 1024) ? 16 : 32;
    int num_groups = HASH_BLOCK / G;
    int my_group = tid / G, my_id = tid % G;
    for (int p = rs + my_group; p < re; p += num_groups) {
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + my_id; q < ke; q += G) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a_ik * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&gval[gb + slot], v); break; }
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

    // 无序 extract(ht ≥ 4096 > CSORT_HT;compact 侧 csort 排序)
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    long long cap = row_off[i + 1] - row_off[i];
    for (int s = tid; s < ht_size; s += HASH_BLOCK) {
        int c = sh_col[s];
        if (c >= 0) {
            int pos = atomicAdd(&cnt, 1);
            if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + pos] = gval[gb + s];
        }
    }
    __syncthreads();
    if (tid == 0) {
        if (cnt > cap) { if (!atomicExch(&row_ovf[i], 1)) { int p = atomicAdd(ovf_cnt, 1); ovf_rows[p] = i; } }
        row_nnz[i] = (cnt <= cap) ? cnt : (int)cap;
    }
}

// compact 原地化守卫:margin[j] = E_off[j] − C_rp[j](≥0 全体成立 → 前缀紧缩安全,docs/21 Fix2)
__global__ void margin_kernel(const long long *eoff, const int *crp, int n, long long *marg)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    marg[j] = eoff[j] - (long long)crp[j];
}

// 每行 HT 梯表尺寸(与 compute_bucket 同梯;hybrid value 池的 scan 输入)
__global__ void ladder_ht_kernel(const int *est, int A_rows, int *lht)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int e = est[i];
    int target = (e <= 4096) ? 2 * e : e;
    int ht = 32;
    while (ht < target && ht < HASH_CAP) ht <<= 1;
    lht[i] = ht;
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
    // 插入排序(u_cnt ≤ CAP=32,单线程 ~512 次比较):输出有序 → compact 走纯 copy,
    // 免去每行一个 BlockRadixSort CTA(germany_osm 11.5M ultra 行 × csort = 62ms 教训)
    for (int t = 1; t < u_cnt; t++) {
        int c = u_col[t]; double v = u_val[t];
        int s = t - 1;
        while (s >= 0 && u_col[s] > c) { u_col[s + 1] = u_col[s]; u_val[s + 1] = u_val[s]; s--; }
        u_col[s + 1] = c; u_val[s + 1] = v;
    }
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

// heavy 混合版(2026-08-26,Step2):keys 进 SMEM(ht ≤ HYBRID_HT_MAX,128KB)、values 留全局
// arena —— SMEM atomicCAS 比全局快 ~4-8×,治 TSOPF 类 heavy 全局表 3.86× 的主源。
// 对标 Ocean HYBRID_HASHMAP(keys SMEM + values 全局池)。smem 每 launch 均一 →
// host 按 ht ≤ HYBRID_HT_MAX 分两批launch。
#define HYBRID_HT_MAX 32768

__global__ void hash_global_hybrid_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    const int *bucket_rows, int n_in_bucket,
    const long long *row_ht, const long long *tab_off,
    int *tab_col_global, double *tab_val,     // tab_val 全局;tab_col_global 仅大行用(此 kernel 不写)
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
    double *sh_val = tab_val + tab_off[idx];               // 值:全局(与纯全局版共用 arena)
    extern __shared__ __align__(8) int hkey[];             // 键:SMEM [ht_size]
    for (int s = tid; s < ht_size; s += blockDim.x) hkey[s] = -1;
    __syncthreads();

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int G = 32, num_groups = blockDim.x / G, my_group = tid / G, my_id = tid % G;
    for (int p = rs + my_group; p < re; p += num_groups) {
        int k = A_col_idx[p];
        double a = A_val[p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        for (int q = ks + my_id; q < ke; q += G) {
            int j = B_col_idx[q];
            if (upper_tri && j < i) continue;
            double v = a * B_val[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;
            int probes = 0;
            while (true) {
                int old = atomicCAS(&hkey[slot], -1, j);                   // SMEM CAS
                if (old == -1 || old == j) { atomicAdd(&sh_val[slot], v); break; }  // 全局 val
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) {
                    atomicExch(overflow_flag, 1);
                    int p2 = atomicAdd(ovf_cnt, 1); if (!atomicExch(&row_ovf[i], 1)) ovf_rows[p2] = i;
                    break;
                }
            }
        }
    }
    __syncthreads();

    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    long long base = row_off[i];
    int cap = est[i];
    for (int s = tid; s < ht_size; s += blockDim.x) {
        int c = hkey[s];
        if (c >= 0) {
            int pos = atomicAdd(&cnt, 1);
            if (pos >= cap) { atomicExch(overflow_flag, 1); continue; }
            tmp_key[base + pos] = ((unsigned long long)i << 32) | (unsigned int)c;
            tmp_val[base + pos] = sh_val[s];
        }
    }
    __syncthreads();
    if (tid == 0) {
        if (cnt > cap && !atomicExch(&row_ovf[i], 1)) { int p2 = atomicAdd(ovf_cnt, 1); ovf_rows[p2] = i; }
        row_nnz[i] = cnt > cap ? cap : cnt;
    }
}

// heavy 拆分辅助:从全表 gather (ht, tab_off) 到子表(按子表行号)
__global__ void heavy_gather_kernel(
    int *sub_rows, int n_sub, const int *full_rows,
    const long long *full_ht, const long long *full_off,
    long long *sub_ht, long long *sub_off)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_sub) return;
    int idx = sub_rows[t];          // heavy 表内位置
    sub_ht[t] = full_ht[idx];
    sub_off[t] = full_off[idx];
    sub_rows[t] = full_rows[idx];   // 原地换成真实行号(kernel 按此读 A/B)
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
// 每行真实列跨度(docs/24 跨度窗口):B(CSR 有序)每行首/末元素即该行最小/最大列 →
// 行 i 的乘积列集合 ⊆ [min_k ci[rp[k]], max_k ci[rp[k+1]-1]]。O(nnz_row) 两次寻址,免费。
__global__ void row_span_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows,
    const int *B_row_ptr, const int *B_col_idx,
    int *span_lo, int *span_len,
    int *max_b_len /*nullable:本行 k 集内 B 行最大长(localLoadBalance 输入)*/)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int lo = INT_MAX, hi = -1, mb = 0;
    for (int p = A_row_ptr[i]; p < A_row_ptr[i + 1]; p++) {
        int k = A_col_idx[p];
        int b0 = B_row_ptr[k], b1 = B_row_ptr[k + 1];
        if (b0 < b1) {
            int c0 = B_col_idx[b0], c1 = B_col_idx[b1 - 1];
            if (c0 < lo) lo = c0;
            if (c1 > hi) hi = c1;
            if (b1 - b0 > mb) mb = b1 - b0;
        }
    }
    span_lo[i]  = (lo == INT_MAX) ? 0 : lo;
    span_len[i] = (hi < lo) ? 0 : (hi - lo + 1);
    if (max_b_len) max_b_len[i] = mb;
}


struct ToLL {   // thrust scan 输入升 64 位(thrust 按输入 value_type 累加,int 输入 Σ>2^31 回绕)
    __host__ __device__ long long operator()(int x) const { return (long long)x; }
};

// prep:ht = pow2(2×min(flop, ncols))(每行 distinct ≤ min(flop,n),高 dup 行 flop 是 distinct 的
// 几千倍,裸 flop 定表 = 内存爆炸根源之一,docs/21);超全局表上限 → 不可重试(保持 overflow → 整阵回退)
__global__ void retry_prep_kernel(
    const int *rows, int n, const int *flop,
    long long *rht, int *row_ovf, int *overflow_flag, int ncols)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int row = rows[t];
    long long f = flop[row];
    if (f > ncols) f = ncols;                              // distinct ≤ ncols(数学封顶)
    if (2 * f > (long long)GLOBAL_HT_MAX_SLOTS) return;   // 不可重试:overflow 已置
    long long h = 2 * HASH_CAP;
    while (h < 2 * f) h <<= 1;
    rht[t] = h;
    row_ovf[row] = 1;   // 已在 accumulate 记录时置位;此处幂等
}
// slot:重试区按行 id 的槽位数(=min(flop, ncols);非重试行为 0,先 memset)
__global__ void retry_slot_kernel(const int *rows, int n, const int *flop, int *rslot, int ncols)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int f = flop[rows[t]];
    if (f > ncols) f = ncols;
    rslot[rows[t]] = f;
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
    const int *flop, int diter_thr,
    const int *span_len,   // 每行真实列跨度(docs/24:替代 n 近似)
    int *bucket_id, int *counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int e = est[i];
    int bid;
    int sl0 = span_len[i];
    // 门 v2(333SP +160%/F2 +13% 教训:窄跨度但 est 小的行在 batched/hash 本来就快,勿偷):
    // 极窄 span≤512:数组比任何 hash 表都小,只要 e>64 就值得;中段 512<span≤2048:须 est≥2048
    // (= v4 门的"hash 伺候不了的大表行")。两档都要求 est≥span/2(数组利用率)。
    if (e > 64 && sl0 > 0 && sl0 <= 256 && 2LL * e >= (long long)sl0) {
        // 小跨度 dense(docs/37;门 v3=收缩到 Ocean dense0 精确槽位:span≤256):pwtk +145%/F2 +14%
        // 教训 = 高 dup 行 dense 付逐乘积原子、hash 只付逐 distinct,中窄跨度(257-2048)反复被偷;
        // 256 内数组比一切 hash 表小且密集,是唯一稳定赢的槽位(nemeth18 全族实证)。
        bid = 14;
    }
    else if (e <= ultra_thr) bid = BIN_ULTRA;             // ultra:线性免 hash
    // dense-iter(docs/22 Phase A + 24 裁决):重行 + 低 dup(<8×)+ **est 大(≥2048)**。
    // v3 门控演进:flop/span 密度作判据被实测否决(输家 pkustk 0.61 vs 赢家 c-58 0.16 完全重叠);
    // 真判据 = 我方 hash 对该行的速度 ∝ 表大小:F2 的行 est~1-2k → hash 9.7 G/s 本来就快,
    // 送窗口反而亏;c-58 的重行 est 3-10k → 大表低占用 hash 0.96 G/s,窗口大胜。est≥2048
    // = "hash 伺候不了的大表行"(docs/24 §门控三版)
    else if (diter_thr > 0 && flop[i] >= diter_thr && e >= 2048 && 16LL * flop[i] >= (long long)span_len[i] &&
             (long long)flop[i] < 8LL * e) bid = BIN_DITER;
    else if (e > HASH_CAP) bid = BIN_HEAVY;          // heavy:全局表(2026-08-25,不再回退 merge)
    else {
        int bi = 0, ht = 32;
        // 小行表 2× 松弛(est=flop 紧界 → 满载原子争用,333SP accumulate +8ms 教训;槽位不变只放大表)
        int target = (e <= 4096) ? 2 * e : e;
        while (ht < target && ht < HASH_CAP) { ht <<= 1; bi++; }
        // 批量 kernel 门(2026-08-26):est≤64 且行长≤32(k 短,warp 串行 k 才划算;
        // 3Dspectralwave2 的 est 小但 k 数百的长链行回归教训)→ bin0(BATCH_HT=128 覆盖 est≤64);
        // 否则走原 per-row 梯。333SP 教训:门须独立于 2× 表目标的梯子(否则 est~36 被推到 bi2)
        if (false) {  // subwarp8 否决(germany_osm 49→55ms,warp idle 不是瓶颈;kernel 保留待查;dispatch 分支已删,勿再加在 BIN_HEAVY 前——曾挡死 heavy 导致 DNF)
            bid = BIN_DITER;
        } else if (e <= 64) {
            int rk = A_row_ptr[i + 1] - A_row_ptr[i];
            bid = (rk <= 32) ? 0 : bi;
        } else {
            bid = bi;
        }
    }
    bucket_id[i] = bid;
    // warp 聚合:同 warp 内同 bin 的行由 leader 一次 atomicAdd(3.7M 行×12 计数器争用 → 333SP 2.9ms 教训)
    unsigned mask = __activemask();
    unsigned peers = __match_any_sync(mask, (unsigned)bid);
    int leader = __ffs(peers) - 1;
    if ((int)(threadIdx.x & 31) == leader) atomicAdd(&counts[bid], __popc(peers));
}

// scatter:把行号按 bucket 写到预分配大 buffer 的对应位置
__global__ void scatter_rows_kernel(
    const int *bucket_id, int A_rows,
    const int *offsets, int *pos, int *sorted_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int bid = bucket_id[i];
    // warp 聚合取槽:leader 一次 atomicAdd(popc),成员按 warp 内 rank 分配(0.65→聚合后原子减 ~32×)
    unsigned mask = __activemask();
    unsigned peers = __match_any_sync(mask, (unsigned)bid);
    int leader = __ffs(peers) - 1;
    int lane = threadIdx.x & 31;
    int base = 0;
    if (lane == leader) base = atomicAdd(&pos[bid], __popc(peers));
    base = __shfl_sync(mask, base, leader);
    int rank = __popc(peers & ((1u << lane) - 1));    // 同 bid 组内低于自己的 lane 数
    sorted_rows[offsets[bid] + base + rank] = i;
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

// warp-per-row 批量 copy(小行,2026-08-26):bin0 的 ~3M 行×19 项,1CTA/行 launch-bound → 8 行/CTA
__global__ void hash_compact_copy_warp_kernel(
    const int *rows, int n_rows,
    const long long *row_off, const int *row_nnz, const int *row_ptr,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *out_col, double *out_val, const int *skip)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= n_rows) return;
    int row = rows[warp];
    if (skip && skip[row]) return;
    long long src = row_off[row]; int n = row_nnz[row], dst = row_ptr[row];
    for (int t = lane; t < n; t += 32) {
        out_col[dst + t] = (int)(tmp_key[src + t] & 0xffffffffu);
        out_val[dst + t] = tmp_val[src + t];
    }
}

// D5H 原地行排序(MODE=2 大行:无序直写 dC 后,行内 BlockRadixSort col/val in==out)
template<int TPB, int IPT>
__global__ void csort_inplace_kernel(const int *rows, const int *rp, int *col, double *val) {
    int r = rows[blockIdx.x];
    int s = rp[r], n = rp[r + 1] - rp[r];
    if (n <= 1) return;
    using BRS = cub::BlockRadixSort<unsigned, TPB, IPT, double>;
    extern __shared__ char cip_smem[];
    unsigned kc[IPT]; double kv[IPT];
    using BTmp = typename BRS::TempStorage;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        kc[i] = (idx < n) ? (unsigned)col[s + idx] : 0xffffffffu;
        kv[i] = (idx < n) ? val[s + idx] : 0.0;
    }
    __syncthreads();
    BRS(reinterpret_cast<BTmp&>(*cip_smem)).Sort(kc, kv);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n) { col[s + idx] = (int)kc[i]; val[s + idx] = kv[i]; }
    }
}
template<int TPB, int IPT>
static void launch_csort_ip(int n, const int *rows_ptr, const int *d_rp, int *d_col, double *d_val) {
    using BRS = cub::BlockRadixSort<unsigned, TPB, IPT, double>;
    size_t smem = sizeof(typename BRS::TempStorage);
    if (smem > 48 * 1024)
        CHECK_CUDA(cudaFuncSetAttribute((const void*)csort_inplace_kernel<TPB, IPT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    csort_inplace_kernel<TPB, IPT><<<n, TPB, smem>>>(rows_ptr, d_rp, d_col, d_val);
    CHECK_CUDA(cudaGetLastError());
}

// 融合键排序(docs/38,Ocean sortOutputFused):键 = (src_pos << valid_bits) | col,radix 只扫
// col 位;值不进排序流 —— 先灌 SMEM,排序后按键内 src_pos gather。省一半排序交通。
// 约束:n < 2^valid_bits 且 CAP=TPB×IPT ≤ 2^(32−valid_bits)(大 n 回退旧双数组内核)。
template<int TPB, int IPT>
__global__ void csort_fused_kernel(
    const int *rows, const long long *d_off, const int *d_row_nnz, const int *dC_rp,
    const unsigned long long *tmp_key, const double *tmp_val,
    int *dC_ci, double *dC_val, int valid_bits)
{
    constexpr int CAP = TPB * IPT;
    int r = rows[blockIdx.x];
    long long s = d_off[r];
    int n_el = d_row_nnz[r];
    int out = dC_rp[r];
    if (n_el == 0) return;
    using BRS = cub::BlockRadixSort<unsigned, TPB, IPT>;
    extern __shared__ __align__(16) char cfs[];
    union CFU { typename BRS::TempStorage sort; double vals[CAP]; };
    CFU &u = reinterpret_cast<CFU&>(*cfs);
    unsigned k[IPT];
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n_el) {
            unsigned col = (unsigned)(tmp_key[s + idx] & 0xffffffffu);
            k[i] = ((unsigned)idx << valid_bits) | col;  // src_pos 高位免费搭车
        } else {
            k[i] = 0xffffffffu;
        }
    }
    __syncthreads();
    BRS(u.sort).Sort(k, /*begin_bit=*/0, /*end_bit=*/valid_bits);   // 只扫 col 位
    __syncthreads();
    // 值在排序【后】装 SMEM(union 区排序时被 TempStorage 占用,Ocean 同款时序)
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n_el) u.vals[idx] = tmp_val[s + idx];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = threadIdx.x * IPT + i;
        if (idx < n_el) {
            unsigned key = k[i];
            unsigned col = key & ((1u << valid_bits) - 1u);
            int src = (int)(key >> valid_bits);
            dC_ci[out + idx] = (int)col;
            dC_val[out + idx] = u.vals[src];           // 值按源位 gather
        }
    }
}
template<int TPB, int IPT>
static void launch_csort_fused(int n, const int *rows_ptr, const long long *d_off, const int *d_row_nnz,
                               const int *dC_rp, const unsigned long long *k, const double *v,
                               int *dC_ci, double *d_val, int valid_bits) {
    using BRS = cub::BlockRadixSort<unsigned, TPB, IPT>;
    constexpr int CAP = TPB * IPT;
    size_t smem = std::max(sizeof(typename BRS::TempStorage), (size_t)CAP * sizeof(double));
    if (smem > 48 * 1024)
        CHECK_CUDA(cudaFuncSetAttribute((const void*)csort_fused_kernel<TPB, IPT>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    csort_fused_kernel<TPB, IPT><<<n, TPB, smem>>>(rows_ptr, d_off, d_row_nnz, dC_rp, k, v, dC_ci, d_val, valid_bits);
    CHECK_CUDA(cudaGetLastError());
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
    if (getenv("CSORT_DBG")) fprintf(stderr, "[csort-dbg] TPB=%d IPT=%d n=%d smem=%zu\n", TPB, IPT, n, smem);
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
    int *d_span_lo; d_span_lo = decltype(d_span_lo)(dev_alloc(A_rows * sizeof(int)));    // docs/24 跨度窗口
    int *d_span_len; d_span_len = decltype(d_span_len)(dev_alloc(A_rows * sizeof(int)));
    int *d_maxbl;   d_maxbl   = decltype(d_maxbl)(dev_alloc(A_rows * sizeof(int)));   // 行 k 集内 B 行最大长(LLB 输入)
    long long total_flop = 0;
    prof("count_flop", [&]{
        if (!att)
            count_flop_span_kernel<<<(A_rows * 32 + 255) / 256, 256>>>(
                dA_rp, dA_ci, A_rows, dB_rp, dB_ci, d_flop, d_span_lo, d_span_len, d_maxbl);
        else {   // att:count 外层=dB_rp(CSC)与 row_span 外层=dA_rp 语义不同,保留双核
            count_intermediates_par_kernel<<<(A_rows * 32 + 255) / 256, 256>>>(dB_rp, dB_ci, A_rows, d_flop);
            row_span_kernel<<<(A_rows + 255) / 256, 256>>>(dA_rp, dA_ci, A_rows, dB_rp, dB_ci, d_span_lo, d_span_len, d_maxbl);
        }
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
        // 自适应 EXPAND(2026-08-26):小计算量矩阵是延迟域 —— retry 的 ~0.7ms 固定延迟占比大,
        // 保留 1.4 松弛让欠估尾部消失;大矩阵是内存域 —— 1.15 紧 est,retry 摊薄可忽略。
        // (bcsstk30 教训:1.15 下 143 行欠估 → retry 0.74ms = 总预算 30%,1.76× 落后主因)
        double est_expand = 1.15;   // EXPAND 扫描裁决(1.15/1.20/1.25/1.30/1.40 → 9.03/9.46/9.50/10.44/9.64ms):紧 est 完胜,retry-vs-compact 零和
        if (const char *e = getenv("HASH_EXPAND")) est_expand = atof(e);
        dbg("[%s] est_expand=%.2f(total_flop=%lld)\n", tag, est_expand, total_flop);
        prof("mh_merge", [&]{
            mh_merge_kernel<<<A_rows, p2_block, smem_p2>>>(
                dA_rp, dA_ci, A_rows, d_mh, d_flop, est_expand, d_est);
            CHECK_CUDA(cudaGetLastError());
        });
        dev_free(d_mh);
    }
    long long total_est = 0;
    prof("est_scan", [&]{
        CHECK_CUDA(cudaMemset(d_off, 0, sizeof(long long)));
        if (A_rows <= 1024) {
            int b = 1; while (b < A_rows) b <<= 1;                       // 小阵:单 block scan(1 launch)
            scan_inclusive_kernel<long long><<<1, b, b * sizeof(long long)>>>(d_est, d_off + 1, A_rows);
        } else {
            // ⚠ 累加必须以 ll 为输入类型:thrust 的 scan 按【输入 value_type】累加,输出 ll + plus<ll>
            // 都不够(实测 c-73 Σest=2.63G 仍 32 位回绕变负)→ transform_iterator 先升 ll 再 scan
            thrust::inclusive_scan(
                thrust::make_transform_iterator(thrust::device_ptr<int>(d_est), ToLL()),
                thrust::make_transform_iterator(thrust::device_ptr<int>(d_est + A_rows), ToLL()),
                thrust::device_ptr<long long>(d_off + 1));
        }
        // total_est 的 D2H 延后并入 binning 的 sync(小阵省一次同步往返,docs/34)
    });
    // (TEMP est 诊断块已删:total_est D2H 延后到 binning sync,此处未读)

    // Stage 2: GPU 端分桶(全 device,无 host 往返)+ 预分配大 buffer(零 per-bucket malloc/free)
    int *d_row_nnz; d_row_nnz = decltype(d_row_nnz)(dev_alloc(A_rows * sizeof(int)));
    int *d_overflow; d_overflow = decltype(d_overflow)(dev_alloc(sizeof(int)));
    // 行级重试(2026-08-26,Ocean out_overflow_row_ids 同款):欠估行收集 → flop 定表重跑 → d_off[i] 改指重试区
    int *d_ovf_rows; d_ovf_rows = decltype(d_ovf_rows)(dev_alloc(A_rows * sizeof(int)));
    int *d_ovf_cnt;  d_ovf_cnt  = decltype(d_ovf_cnt)(dev_alloc(sizeof(int)));
    int *d_row_ovf;  d_row_ovf  = decltype(d_row_ovf)(dev_alloc(A_rows * sizeof(int)));   // 重试行标记(compact 路由)
    // 重试区(行级重试输出;compact 末段读)+ 重试中间缓冲(docs/21 Fix0:此前从不释放,每调用泄漏
    // r_slots×32B + rt_total×12B,warmup 多轮累加 → rajat 类 OOM 崩溃的帮凶)
    const int *d_retry_rows = nullptr; int d_retry_n = 0;
    const long long *d_retry_off = nullptr;
    const unsigned long long *d_retry_key = nullptr; const double *d_retry_val = nullptr;
    long long *d_rht = nullptr, *d_rtab = nullptr, *d_roff = nullptr;
    int *d_rslot = nullptr, *d_rsb = nullptr, *d_rse = nullptr;
    int *rtc = nullptr; double *rtv = nullptr;
    unsigned long long *rk = nullptr, *rk2 = nullptr;
    double *rv = nullptr, *rv2 = nullptr;
    void *rct = nullptr;
    unsigned long long *d_tmp_key; double *d_tmp_val;
    // (tmp 分配挪到 binning/方案5 之后:dense 行直写不占 gapped 槽 → 缩容 Σest_dense,见 tmp_slots)
    // d_val(C_val)现 alias 进连续 dC(compact+sort 直写,见 cnnz_scan 后),不再单独分配/释放。
    double *d_val = nullptr;
    double *d_hybrid_val = nullptr;   // Fix#6(审查):bin0 hybrid 池句柄(尾部释放)

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
        static int g_diter = -1;   // dense-iter 重行阈值(DITER_MIN_FLOP,默认 4096;0=关)
        if (g_diter < 0) { const char *e = getenv("DITER_MIN_FLOP"); g_diter = (e && *e) ? atoi(e) : 4096; }
        compute_bucket_kernel<<<(A_rows + 255) / 256, 256>>>(d_est, dA_rp, A_rows, EST_ULTRA_THR, d_flop, g_diter, d_span_len, d_bkid, d_cnt);
        thrust::exclusive_scan(thrust::device_ptr<int>(d_cnt),
                               thrust::device_ptr<int>(d_cnt + N_BINS),
                               thrust::device_ptr<int>(d_offb));
        CHECK_CUDA(cudaMemset(d_pos, 0, N_BINS * sizeof(int)));   // scatter 计数器从 0 起(非 offsets)
        scatter_rows_kernel<<<(A_rows + 255) / 256, 256>>>(d_bkid, A_rows, d_offb, d_pos, d_sort);
        // 3 个 D2H 合并:async + 单 sync(省 2 同步点;total_est 从 est_scan 延后到此)
        CHECK_CUDA(cudaMemcpyAsync(h_cnt, d_cnt,  N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpyAsync(h_off, d_offb, N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpyAsync(&total_est, d_off + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaStreamSynchronize(0));
        dbg("[%s] diter bin=%d 行 / %d\n", tag, h_cnt[BIN_DITER], A_rows);   // TEMP:docs/24 路由观测
    });

    // ---- 全 bin Hybrid Value(docs/24 §4.1):ht≥4096 的大表 bin 的 value 池(keys 留 SMEM,occupancy ×3)。
    //      池 = 每行梯表尺寸 × 8B 的 scan;>8GB(鲸鱼阵)或 HYB_TIER=0 跳过。 ----
    double *d_gv = nullptr; long long *d_gv_off = nullptr;
    {
        static int g_hyb_tier = -1;
        // 默认 0(2026-08-27 A/B 否决:Ge99 +16%/Ga3As3H12 2.9× 恶化 —— 每乘积一趟全局 L2 原子的
        // 往返成本 > SMEM 争用成本;与 docs/23 bin0-only 收益微弱一致。保留代码供后续组合实验)
        if (g_hyb_tier < 0) { const char *e = getenv("HYB_TIER"); g_hyb_tier = (e && *e) ? atoi(e) : 0; }
        if (g_hyb_tier && !att) {
            int *d_lht = decltype(d_lht)(dev_alloc(A_rows * sizeof(int)));
            d_gv_off = decltype(d_gv_off)(dev_alloc((A_rows + 1) * sizeof(long long)));
            ladder_ht_kernel<<<(A_rows + 255) / 256, 256>>>(d_est, A_rows, d_lht);
            CHECK_CUDA(cudaMemset(d_gv_off, 0, sizeof(long long)));
            thrust::inclusive_scan(thrust::make_transform_iterator(thrust::device_ptr<int>(d_lht), ToLL()),
                                   thrust::make_transform_iterator(thrust::device_ptr<int>(d_lht + A_rows), ToLL()),
                                   thrust::device_ptr<long long>(d_gv_off + 1));
            long long gv_slots;
            CHECK_CUDA(cudaMemcpy(&gv_slots, d_gv_off + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
            dev_free(d_lht);
            long long gv_cap = 8LL << 30;                       // 池上限 8GB(鲸鱼阵 est 巨大,跳过)
            if (gv_slots > 0 && gv_slots * (long long)sizeof(double) <= gv_cap) {
                d_gv = decltype(d_gv)(dev_alloc((size_t)gv_slots * sizeof(double)));
                CHECK_CUDA(cudaMemset(d_gv, 0, (size_t)gv_slots * sizeof(double)));
                dbg("[%s] hybrid value pool: %.2f GB\n", tag, gv_slots * 8.0 / (1 << 30));
            } else {
                dev_free(d_gv_off); d_gv_off = nullptr;
                dbg("[%s] hybrid value pool 跳过(%.2f GB 超 cap)\n", tag, gv_slots * 8.0 / (1 << 30));
            }
        }
    }

    // 2b: opt-in max SMEM
    {
        size_t maxsm = (size_t)HASH_CAP * (sizeof(int) + sizeof(double));   // sh_col[int]+sh_val[double]
        if (maxsm > 48 * 1024)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
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
    // dense 累积器门(2026-08-26,exdata_1 类):小 n + 输出 ≥15% 稠密 → 直接寻址免探测免 CAS
    #define DENSE_WIN_MAX_N 200000
    const bool dense_mode = (A_cols <= DENSE_MAX_N) && A_rows > 0 &&
        ((double)total_est / ((double)A_rows * A_cols) >= DENSE_MIN_FRAC);
    const bool dense_win_mode = !dense_mode && (A_cols <= DENSE_WIN_MAX_N) && A_rows > 0 &&
        ((double)total_est / ((double)A_rows * A_cols) >= DENSE_MIN_FRAC);
    // ---- 方案5(docs/27 §4.1):dense 行精确计数 + 直写终态。DIRECT5=1 开,默认关。----
    // dense 行集合 = 矩阵级 dense/dense_win(全体行)或 Phase A bin(v4 门含 dup<8 = 两遍直写
    // 交通量赢面);hash 行不动(结构同 Ocean type2 est 工作流)。count pass 在 accumulate 前
    // 出精确 row_nnz → cnnz_scan 后数值 pass 直写 dC;dense 行不再占 gapped tmp 槽(缩容 Σest_dense)
    // 且永不 ovf(计数精确,免 retry)。
    static int g_direct5 = -1;
    if (g_direct5 < 0) { const char *e = getenv("DIRECT5"); g_direct5 = (e && *e) ? atoi(e) : 1; }   // refresh9 全量验证 -0.99% → 默认开;DIRECT5=0 关
    static int g_d5h = -1;
    if (g_d5h < 0) { const char *e = getenv("D5H"); g_d5h = (e && *e && atoi(e) > 0) ? 1 : 0; }
    int dense_nr = 0; const int *dense_rows = nullptr;   // dense 行集合(null = 全体行)
    long long dense_est_sum = 0, dense_nnz_sum = 0, tmp_slots_dev = 0;
    if (g_direct5) {
        // 矩阵级只取 dense_win:n≤DENSE_MAX_N 的 dense_mode 用按 n 尺寸的 v1 内核(occupancy 高),
        // direct 是固定 14980 窗口(195KB,1CTA/SM)——exdata_1 实测 13.5→33.4ms 占用减半;且
        // dense_mode 的 compact 仅 ~1.4% 无税可省。dense_win 两路同为固定窗口 → 纯赢(A/B 实证)。
        if (dense_win_mode) dense_nr = A_rows;
        else if (!dense_mode && h_cnt[BIN_DITER] > 0) { dense_nr = h_cnt[BIN_DITER]; dense_rows = d_sort + h_off[BIN_DITER]; }
        if (dense_nr > 0) {
            // 矩阵级路径不再走下方 accumulate 分支(那三处 memset 挪到这里,计时内);mixed 路径 per-bin 头会再清,无害
            prof("dense_count", [&]{
                CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
                CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
                CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
                // 规模门 pre-pass(refresh9 实证:dup 门不成立 —— 赢家 dup 1.54/2.11/6.14 与输家
                // <1.5/1.74/4.64 完全重叠,docs/29 §8 开放问题;机械可解释的只有微型 dense 集:
                // cnr-2000(73 行/0.3% est)/ohne2(69 行/0.13%)纯固定开销亏损)。
                // 门 = dense 行数 ≥ 1000 且 Σest_dense ≥ 5% total_est,否则方案5 整体跳过回 legacy。
                {
                    unsigned long long *d_s2 = decltype(d_s2)(dev_alloc(2 * sizeof(long long)));
                    CHECK_CUDA(cudaMemset(d_s2, 0, 2 * sizeof(long long)));
                    dense_sum_kernel<<<(dense_nr + 255) / 256, 256>>>(
                        dense_rows, dense_nr, d_est, nullptr, d_flop, d_s2, d_s2 + 1);
                    unsigned long long h2[2];
                    CHECK_CUDA(cudaMemcpy(h2, d_s2, 2 * sizeof(long long), cudaMemcpyDeviceToHost));
                    dev_free(d_s2);
                    long long est_d = (long long)h2[0], flop_d = (long long)h2[1];
                    if (dense_nr < 1000 || est_d * 20 < total_est) {
                        dbg("[%s] 方案5 规模门:跳过(dense_nr=%d Σest=%lld/%lld, flop=%lld)\n",
                            tag, dense_nr, est_d, total_est, flop_d);
                        dense_nr = 0;
                        return;
                    }
                }
                size_t csm = (size_t)DENSE_CNT_W;
                if (csm > 48 * 1024)
                    CHECK_CUDA(cudaFuncSetAttribute(hash_dense_count_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)csm));
                hash_dense_count_kernel<<<dense_nr, 512, csm>>>(
                    dA_rp, dA_ci, dB_rp, dB_ci, upper_tri, A_rows, A_cols,
                    d_row_nnz, dense_rows, dense_rows ? d_span_lo : nullptr, d_flop, d_maxbl);
                CHECK_CUDA(cudaGetLastError());
                unsigned long long *d_sums = decltype(d_sums)(dev_alloc(2 * sizeof(long long)));
                CHECK_CUDA(cudaMemset(d_sums, 0, 2 * sizeof(long long)));
                dense_sum_kernel<<<(dense_nr + 255) / 256, 256>>>(
                    dense_rows, dense_nr, d_est, d_row_nnz, nullptr, d_sums, d_sums + 1);
                CHECK_CUDA(cudaMemcpy(&dense_est_sum, d_sums, sizeof(long long), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(&dense_nnz_sum, d_sums + 1, sizeof(long long), cudaMemcpyDeviceToHost));
                dev_free(d_sums);
                // dense 行 est 置 0 → d_off 重扫:gapped 偏移坍缩进 hash 侧空间(tmp 才能按缩容量分配;
                // d_off 是全行 est 前缀,不重扫则 heavy/hash 行偏移仍指旧空间 → 越界)。
                // 置零安全:binning 已完成,后续无消费者按 dense 行的 est 寻址(它们不走 hash/compact)。
                // ⚠ D5H(方案5 扩展到 hash 行,docs/31):bins 1-10 先 MODE=1 count-only 出精确 row_nnz,
                //    数值 pass 改 MODE=2 直写 dC(小行有序免排/大行原地 csort),compact 对这些行消失。
                //    门 = dup = total_flop/total_est < 8(Ocean type1 的 compaction 经济学同款)。
                if (dense_rows)
                    zero_est_kernel<<<(dense_nr + 255) / 256, 256>>>(d_est, dense_rows, dense_nr);
                else
                    CHECK_CUDA(cudaMemset(d_est, 0, (size_t)A_rows * sizeof(int)));
                CHECK_CUDA(cudaMemset(d_off, 0, sizeof(long long)));
                if (A_rows <= 1024) {
                    int b = 1; while (b < A_rows) b <<= 1;
                    scan_inclusive_kernel<long long><<<1, b, b * sizeof(long long)>>>(d_est, d_off + 1, A_rows);
                } else {
                    thrust::inclusive_scan(
                        thrust::make_transform_iterator(thrust::device_ptr<int>(d_est), ToLL()),
                        thrust::make_transform_iterator(thrust::device_ptr<int>(d_est + A_rows), ToLL()),
                        thrust::device_ptr<long long>(d_off + 1));
                }
                CHECK_CUDA(cudaMemcpy(&tmp_slots_dev, d_off + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
            });
            dbg("[%s] 方案5:count pass %d dense 行(Σest=%lld Σnnz=%lld)→ cnnz_scan 后直写终态\n",
                tag, dense_nr, dense_est_sum, dense_nnz_sum);
        }
    }
    // ---- D5H:方案5 扩展到 hash 行(docs/31;D5H=1 开,默认关)----
    // bins 1-10 行 MODE=1 count-only → row_nnz 精确 → MODE=2 数值 pass 直写 dC。
    // 小行(ht≤CSORT_HT)count-sort 有序直写免善后;大行无序直写 + 原地 csort。
    // 计入 tmp 缩容(与 dense 行共用置零+重扫)。
    int d5h_rows = 0; long long d5h_nnz_sum = 0;
    const bool g_d5h_on = g_d5h && !att && !(dense_nr > 0 && (dense_mode || dense_win_mode))
        && (total_est > 0 ? (double)total_flop / (double)total_est < 8.0 : false);   // dup<8(Ocean type1 经济学)
    if (g_d5h_on) {
        prof("hash_count", [&]{
            for (int bi = 1; bi <= 10; bi++) {
                int n = h_cnt[bi];
                if (n == 0) continue;
                int *rows_ptr = d_sort + h_off[bi];
                int ht = 32 << bi;
                size_t smem_flat = (size_t)ht * (sizeof(int) + sizeof(double));
                {   // 一次性无条件 opt-in(49152 恰在 48KB 边界也会失败;legacy 对 <0> 即此做法)
                    static bool done1 = false;
                    if (!done1) {
                        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, HASH_CAP * (int)(sizeof(int) + sizeof(double))));
                        done1 = true;
                    }
                }
                hash_spa_kernel<1><<<n, HASH_BLOCK, smem_flat>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, ht, d_off,
                    d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                    d_flop, d_maxbl);
                d5h_rows += n;
                CHECK_CUDA(cudaGetLastError());
            }
            if (false) {   // 见上:dense_sum launch 确定性 invalid-configuration,已绕过
                // Σnnz(count 后)供下溢检查改口径
                unsigned long long *d_s3 = decltype(d_s3)(dev_alloc(sizeof(long long)));
                CHECK_CUDA(cudaMemset(d_s3, 0, sizeof(long long)));
                for (int bi = 1; bi <= 10; bi++) {
                    if (h_cnt[bi] == 0) continue;   // Fix#8(审查):空 bin 的 <<<0,256>>> = invalid-config
                                                      // = docs/31/35 "invalid argument" 之谜根因(grid=0)
                    dense_sum_kernel<<<(h_cnt[bi] + 255) / 256, 256>>>(
                        d_sort + h_off[bi], h_cnt[bi], nullptr, d_row_nnz, nullptr, nullptr, d_s3);
                }
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaMemcpy(&d5h_nnz_sum, d_s3, sizeof(long long), cudaMemcpyDeviceToHost));
                dev_free(d_s3);
            }
            if (d5h_rows > 0) {   // 置零 bins 1-10 行的 est + d_off 重扫(与 dense 共用机制)
                for (int bi = 1; bi <= 10; bi++) {
                    int n = h_cnt[bi];
                    if (n == 0) continue;
                    zero_est_kernel<<<(n + 255) / 256, 256>>>(d_est, d_sort + h_off[bi], n);
                }
                CHECK_CUDA(cudaMemset(d_off, 0, sizeof(long long)));
                if (A_rows <= 1024) {
                    int b = 1; while (b < A_rows) b <<= 1;
                    scan_inclusive_kernel<long long><<<1, b, b * sizeof(long long)>>>(d_est, d_off + 1, A_rows);
                } else {
                    thrust::inclusive_scan(
                        thrust::make_transform_iterator(thrust::device_ptr<int>(d_est), ToLL()),
                        thrust::make_transform_iterator(thrust::device_ptr<int>(d_est + A_rows), ToLL()),
                        thrust::device_ptr<long long>(d_off + 1));
                }
                CHECK_CUDA(cudaMemcpy(&tmp_slots_dev, d_off + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
            }
        });
        dbg("[%s] D5H:bins1-10 count-only %d 行 → MODE=2 直写\n", tag, d5h_rows);
    }
    // tmp 缩容:dense 行直写不占 gapped 槽(heavy scratch 同口径;鲸鱼阵 Phase A 行恰是 est 大头)
    const long long tmp_slots = (dense_nr > 0 || d5h_rows > 0) ? tmp_slots_dev : total_est;
    d_tmp_key = decltype(d_tmp_key)(dev_alloc((size_t)tmp_slots * sizeof(unsigned long long)));
    d_tmp_val = decltype(d_tmp_val)(dev_alloc((size_t)tmp_slots * sizeof(double)));
    if (dense_nr > 0 && (dense_mode || dense_win_mode)) {
        // 方案5 矩阵级:count 已出精确 row_nnz,数值 pass 在 cnnz_scan 后直写(此分支无 accumulate)
    } else
    if (dense_mode) {
        dbg("[%s] dense 累积器路径(n=%d, est 占比 %.1f%%)\n", tag, A_cols,
            100.0 * total_est / ((double)A_rows * A_cols));
        prof("accumulate", [&]{
            CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
            size_t dsm = ((size_t)A_cols * 9 + 3) / 4 * 4 + (size_t)A_cols * sizeof(int);
            if (dsm > 48 * 1024)
                CHECK_CUDA(cudaFuncSetAttribute(hash_dense_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsm));
            hash_dense_kernel<<<A_rows, 512, dsm>>>(
                dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, A_cols,
                d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                d_ovf_rows, d_ovf_cnt, d_row_ovf);
            CHECK_CUDA(cudaGetLastError());
        });
    } else if (dense_win_mode) {
        dbg("[%s] dense 窗口路径(n=%d, est 占比 %.1f%%)\n", tag, A_cols,
            100.0 * total_est / ((double)A_rows * A_cols));
        prof("accumulate", [&]{
            CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
            CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
            size_t wsm = ((size_t)DENSE_MAX_N * 9 + 3) / 4 * 4 + (size_t)DENSE_MAX_N * sizeof(int);
            CHECK_CUDA(cudaFuncSetAttribute(hash_dense_window_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsm));
            hash_dense_window_kernel<<<A_rows, 512, wsm>>>(
                dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, A_cols,
                d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                d_ovf_rows, d_ovf_cnt, d_row_ovf, nullptr, nullptr, d_flop, d_maxbl);
            CHECK_CUDA(cudaGetLastError());
        });
    } else
    prof("accumulate", [&]{
        CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_ovf_cnt, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_row_ovf, 0, (size_t)A_rows * sizeof(int)));
        static int g_bsearch_s = -1;
    if (g_bsearch_s < 0) { const char *e = getenv("BSEARCH"); g_bsearch_s = (e && *e && atoi(e) > 0) ? 1 : 0; }
    // LLB=1:hash_spa_kernel 按行动态 G(localLoadBalance 移植,docs/27 §4.2;默认关待 A/B)
    static int g_llb = -1;
    if (g_llb < 0) { const char *e = getenv("LLB"); g_llb = (e && *e && atoi(e) > 0) ? 1 : 0; }
    const bool g_bsearch = g_bsearch_s;
    // 方案4修正: HYBRID=1 时 value 走全局 L2 原子(学 Ocean HYBRID_HASHMAP)
    static int g_hybrid_env = -2;
    if (g_hybrid_env == -2) { const char *e = getenv("HYBRID"); g_hybrid_env = (e && *e) ? atoi(e) : -1; }
    // 方案4: dup 因子 > 5 时自动开 hybrid(L2 原子吞吐 > SMEM,高 dup 阵受益 5-29%)
    // 小阵不开(全局池分配 + L2 延退 > 收益);HYBRID=0/1 强制覆盖
    const bool g_hybrid = (g_hybrid_env >= 0) ? g_hybrid_env
        : (total_est > 0 && (double)total_flop / total_est > 5.0 && A_rows > 50000);
    // 多 stream per-bin(HASH_NSTREAMS env 调优,0=单流对照;对标 Ocean 20-stream)
        static int g_nstream = -2;
        if (g_nstream == -2) {
            const char *e = getenv("HASH_NSTREAMS");
            g_nstream = (e && *e) ? atoi(e) : 4;
            if (g_nstream < 0 || g_nstream > 16) g_nstream = 8;
        }
        const bool use_ms = (g_nstream > 0);
        static cudaStream_t bin_s[16];
        static bool bin_s_init = false;
        if (use_ms && !bin_s_init) {
            for (int i = 0; i < g_nstream; i++) CHECK_CUDA(cudaStreamCreate(&bin_s[i]));
            bin_s_init = true;
        }
        for (int bi = 0; bi < N_BINS; bi++) {
            cudaStream_t cur_s = use_ms ? bin_s[bi % g_nstream] : (cudaStream_t)0;
            int n = h_cnt[bi];
            if (n == 0) continue;
            if (g_d5h_on && bi >= 1 && bi <= 10) continue;   // D5H:bins1-10 延后 MODE=2(scan 后直写)
            int *rows_ptr = d_sort + h_off[bi];
            // ⚠ ca7bb79(subwarp8 实验)曾把 subwarp8 分支插在 heavy 前面挡死了 BIN_HEAVY 的 dispatch
            //   (est>16384 的重行全走 32 槽小表 → 必然溢出 → 全体进 flop 定表 retry = DNF 内存爆炸推手)。
            //   subwarp8 已否决(ca7bb79 自述"路由回退"但 dispatch 漏改),此分支删除,bin 布局见 N_BINS 注释。
            if (bi == 0 && g_bsearch) {
                // 方案1: 预排序二分寻址(HSMU 思路): symbolic 排序列 + numeric 二分定位
                bsearch_symbolic_kernel<<<(n + BATCH_WPB - 1) / BATCH_WPB, BATCH_WPB * 32, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf);
                bsearch_numeric_kernel<<<(n + BATCH_WPB - 1) / BATCH_WPB, BATCH_WPB * 32, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_row_nnz, d_tmp_key, d_tmp_val);
            } else if (bi == 0) {
                // 小行批量: HYBRID=1 时 value 走全局 L2(Ocean 杀手锏;池句柄 d_hybrid_val 尾部统一释放,Fix#6)
                if (g_hybrid && !d_hybrid_val)
                    d_hybrid_val = decltype(d_hybrid_val)(dev_alloc((size_t)n * BATCH_HT * sizeof(double)));
                hash_spa_batched_kernel<<<(n + BATCH_WPB - 1) / BATCH_WPB, BATCH_WPB * 32, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf, d_hybrid_val);
            } else if (bi == BIN_ULTRA) {
                // ultra(est≤EST_ULTRA_THR):线性,免 hash
                hash_ultra_kernel<<<(n + 255) / 256, 256, 0, cur_s>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, d_off, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                    d_ovf_rows, d_ovf_cnt, d_row_ovf);
            } else if (bi == BIN_DITER) {
                // dense-iter(docs/22 Phase A):重行列窗口累积,直接寻址免探测;
                // 窗口升序 → 行内天然有序 → compact 走 copy。输出仍写 tmp(est-gapped)+ ovf 行照旧 retry。
                // 方案5(DIRECT5):count pass 已出精确 row_nnz,数值 pass 在 cnnz_scan 后直写 → 此 bin 跳过
                if (dense_nr == 0) {
                    size_t wsm = ((size_t)DENSE_MAX_N * 9 + 3) / 4 * 4 + (size_t)DENSE_MAX_N * sizeof(int);
                    CHECK_CUDA(cudaFuncSetAttribute(hash_dense_window_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsm));
                    hash_dense_window_kernel<<<n, 512, wsm, cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, A_cols,
                        d_off, d_est, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow,
                        d_ovf_rows, d_ovf_cnt, d_row_ovf, rows_ptr, d_span_lo, d_flop, d_maxbl);
                }
            }
            else if (bi >= BIN_SSPAN0 && bi <= BIN_SSPAN0 + 3) {
                // 小跨度 dense 4 子桶(docs/37):span 定 SMEM/块型,小块高行密度
                int sb = bi - BIN_SSPAN0;
                if (sb == 0)
                    hash_sspan2_kernel<256, 64><<<n, 64, (((size_t)256 * 9) + 3) / 4 * 4 + (size_t)256 * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        rows_ptr, d_span_lo, d_span_len, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                else if (sb == 1)
                    hash_sspan2_kernel<512, 128><<<n, 128, (((size_t)512 * 9) + 3) / 4 * 4 + (size_t)512 * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        rows_ptr, d_span_lo, d_span_len, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                else if (sb == 2)
                    hash_sspan2_kernel<1024, 256><<<n, 256, (((size_t)1024 * 9) + 3) / 4 * 4 + (size_t)1024 * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        rows_ptr, d_span_lo, d_span_len, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                else
                    hash_sspan2_kernel<2048, 256><<<n, 256, (((size_t)2048 * 9) + 3) / 4 * 4 + (size_t)2048 * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows,
                        rows_ptr, d_span_lo, d_span_len, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());   // 即时检查(invalid-argument 之谜防复发,docs/35 §4)
            }
            else if (bi == BIN_HEAVY) {
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
                    d_scr_key = decltype(d_scr_key)(dev_alloc((size_t)tmp_slots * sizeof(unsigned long long)));
                    d_scr_val = decltype(d_scr_val)(dev_alloc((size_t)tmp_slots * sizeof(double)));
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
                if (d_gv && ht >= 4096) {
                    // Hybrid Value 大表 bin:keys SMEM(4B/槽)+ values 全局 L2 原子(docs/24)
                    hash_spa_hkv_kernel<<<n, HASH_BLOCK, (size_t)ht * sizeof(int), cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, ht, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                        d_gv, d_gv_off);
                } else if (!att && g_priv && smem_priv <= 196 * 1024) {
                    hash_spa_priv_kernel<<<n, PRIV_W * 32, smem_priv>>>(
                        dA_rp, dA_ci, dA_val, rows_ptr, n, ht, PRIV_W, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
                } else {
                    hash_spa_kernel<0><<<n, HASH_BLOCK, smem_flat, cur_s>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        rows_ptr, n, ht, d_off,
                        d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                        d_flop, d_maxbl, g_llb);
                }
            }
        }
        if (use_ms)
            for (int i = 0; i < g_nstream; i++) CHECK_CUDA(cudaStreamSynchronize(bin_s[i]));
        CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入
    });

    // ---- 行级重试(2026-08-26,Ocean out_overflow_row_ids 同款):欠估行(estimate<distinct)用
    //      flop(精确上界)定表重跑 hash_global,输出写独立重试区,compact 末段按 d_row_ovf 路由。
    //      全部重试成功 → 清 overflow 继续;存在不可重试行(flop 超表上限)→ 保持 overflow 整阵回退。----
    auto run_retry = [&] {
        int h_ovf = 0;
        CHECK_CUDA(cudaMemcpy(&h_ovf, d_ovf_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        dbg("[hash] dbg: overflow=%d ovf_cnt=%d\n", overflow, h_ovf);   // TEMP
        if (h_ovf > 0) {
            dbg("[hash] row-retry: %d 行 → flop 定表重跑\n", h_ovf);
            prof("retry", [&]{
                CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));   // 清 accumulate 的旧标志;不可重试行会重置
                d_rht  = decltype(d_rht)(dev_alloc(h_ovf * sizeof(long long)));
                d_rtab = decltype(d_rtab)(dev_alloc((h_ovf + 1) * sizeof(long long)));
                retry_prep_kernel<<<(h_ovf + 255) / 256, 256>>>(
                    d_ovf_rows, h_ovf, d_flop, d_rht, d_row_ovf, d_overflow, A_cols);
                thrust::exclusive_scan(thrust::device_ptr<long long>(d_rht),
                                       thrust::device_ptr<long long>(d_rht + h_ovf),
                                       thrust::device_ptr<long long>(d_rtab));
                // 重试区槽位(按行 id;非重试行 0)
                d_rslot = decltype(d_rslot)(dev_alloc(A_rows * sizeof(int)));
                CHECK_CUDA(cudaMemset(d_rslot, 0, (size_t)A_rows * sizeof(int)));
                retry_slot_kernel<<<(h_ovf + 255) / 256, 256>>>(d_ovf_rows, h_ovf, d_flop, d_rslot, A_cols);
                d_roff = decltype(d_roff)(dev_alloc((A_rows + 1) * sizeof(long long)));
                CHECK_CUDA(cudaMemset(d_roff, 0, sizeof(long long)));
                thrust::inclusive_scan(thrust::make_transform_iterator(thrust::device_ptr<int>(d_rslot), ToLL()),
                                       thrust::make_transform_iterator(thrust::device_ptr<int>(d_rslot + A_rows), ToLL()),
                                       thrust::device_ptr<long long>(d_roff + 1));   // Fix#3(审查):int 累加 2^31 回绕,同 est_scan 坑
                long long r_slots;
                CHECK_CUDA(cudaMemcpy(&r_slots, d_roff + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
                if (r_slots > 0) {
                    // 表 arena 总量 = rtab[h_ovf-1] + rht[h_ovf-1](exclusive scan 只写 [0..h_ovf-1])
                    long long last_h, rt_prev;
                    CHECK_CUDA(cudaMemcpy(&last_h, d_rht + (h_ovf - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                    CHECK_CUDA(cudaMemcpy(&rt_prev, d_rtab + (h_ovf - 1), sizeof(long long), cudaMemcpyDeviceToHost));
                    long long rt_total = rt_prev + last_h;
                    rtc = decltype(rtc)(dev_alloc((size_t)rt_total * sizeof(int)));
                    rtv = decltype(rtv)(dev_alloc((size_t)rt_total * sizeof(double)));
                    rk  = decltype(rk)(dev_alloc((size_t)r_slots * sizeof(unsigned long long)));
                    rv  = decltype(rv)(dev_alloc((size_t)r_slots * sizeof(double)));
                    rk2 = decltype(rk2)(dev_alloc((size_t)r_slots * sizeof(unsigned long long)));
                    rv2  = decltype(rv2)(dev_alloc((size_t)r_slots * sizeof(double)));
                    // hash_global 逐行表偏移:exclusive_scan 已给 [0..h_ovf-1],[0]=0 ✓
                    hash_global_kernel<<<h_ovf, HASH_BLOCK>>>(
                        dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                        d_ovf_rows, h_ovf, d_rht, d_rtab, rtc, rtv,
                        d_roff, d_rslot, r_slots, rk, rv, d_row_nnz, d_overflow,
                        d_ovf_rows, d_ovf_cnt, d_row_ovf);
                    d_rsb = decltype(d_rsb)(dev_alloc(h_ovf * sizeof(int)));
                    d_rse = decltype(d_rse)(dev_alloc(h_ovf * sizeof(int)));
                    heavy_seg_kernel<<<(h_ovf + 255) / 256, 256>>>(
                        d_ovf_rows, h_ovf, d_roff, d_row_nnz, d_rsb, d_rse);
                    size_t rcb = 0;
                    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                        nullptr, rcb, rk, rk2, rv, rv2, r_slots, h_ovf, d_rsb, d_rse));
                    rct = dev_alloc(rcb);
                    CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
                        rct, rcb, rk, rk2, rv, rv2, r_slots, h_ovf, d_rsb, d_rse));
                    // retry_prep 遇不可重试行会重置 overflow → 仍有则整阵回退
                    d_retry_rows = d_ovf_rows; d_retry_n = h_ovf;
                    d_retry_off = d_roff; d_retry_key = rk2; d_retry_val = rv2;
                    dbg("[hash] row-retry: %d 行完成(r_slots=%lld) → 继续(免整阵回退)\n", h_ovf, r_slots);
                }
            });
        }
    };  // D5H:count/legacy 的 ovf 都在 scan 前覆盖(retry 产精确 row_nnz;MODE=2 跳过 ovf 行)
    run_retry();

    CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (!g_d5h_on && overflow) {
        fprintf(stderr, "[hash] OVERFLOW: 某行 distinct 列 > HASH_CAP=%d → 回退 merge(dispatcher 处理)\n", HASH_CAP);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        dev_free(rtc); dev_free(rtv); dev_free(rk); dev_free(rv); dev_free(rk2); dev_free(rv2); dev_free(rct);   // docs/21 Fix0
        dev_free(d_rht); dev_free(d_rtab); dev_free(d_rslot); dev_free(d_roff); dev_free(d_rsb); dev_free(d_rse);
        dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
        dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
        dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort); dev_free(d_est); dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl); dev_free(d_gv); dev_free(d_gv_off);
        dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val);
        dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
        dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
        dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
        dev_free(d_flop);
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
    dbg("[hash] C_nnz=%d (est=%lld, %.2fx over-alloc)\n", C_nnz_result, total_est,
        total_est > 0 ? (double)total_est / C_nnz_result : 0.0);

    // est underflow check: actual C_nnz > estimated total → 回退 merge3
    // 方案5:dense 行计数精确(可 > 其 est,合法),检查改 hash 侧 = C_nnz − Σnnz_dense ≤ tmp_slots
    if ((dense_nr > 0 || d5h_rows > 0)
            ? ((long long)C_nnz_result - dense_nnz_sum - d5h_nnz_sum > tmp_slots)
            : ((long long)C_nnz_result > total_est)) {
        fprintf(stderr, "[hash] est underflow: C_nnz=%d > total_est=%d → 回退 merge\n", C_nnz_result, total_est);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        dev_free(rtc); dev_free(rtv); dev_free(rk); dev_free(rv); dev_free(rk2); dev_free(rv2); dev_free(rct);   // docs/21 Fix0
        dev_free(d_rht); dev_free(d_rtab); dev_free(d_rslot); dev_free(d_roff); dev_free(d_rsb); dev_free(d_rse);
        dev_free(dA); dev_free(d_off); dev_free(d_row_nnz);
        dev_free(d_overflow); dev_free(d_tmp_key); dev_free(d_tmp_val);
        dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort); dev_free(d_est); dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl); dev_free(d_gv); dev_free(d_gv_off);
        dev_free(dC_rp);
        dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val);
        dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
        dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
        dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
        dev_free(d_flop);
        return;
    }

    // Stage 4+5+6: per-row compact+sort(BlockRadixSort)替掉 compact+全局 sort+split_key;按 bin 选 config,直写连续 dC
    size_t C_rp_sz  = (A_rows + 1) * sizeof(int);
    size_t C_ci_sz  = (size_t)C_nnz_result * sizeof(int);
    size_t C_v_sz   = (size_t)C_nnz_result * sizeof(double);
    size_t C_rp_al  = ALIGN8(C_rp_sz);
    size_t C_ci_al  = ALIGN8(C_ci_sz);
    size_t C_total  = C_rp_al + C_ci_al + C_v_sz;
    // ---- compact 原地化(docs/21 Fix2):dC(12B×C)不再新分配,col/val 直接前缀紧缩进 tmp 自己。
    //      安全条件 = ∀j: 前缀 A(j)=Σactual ≤ E(j)=Σest(d_est 原始值,retry 行的欠估已计入)——
    //      此时行 i 的写区间 [4A(i), 4A(i+1)) 恒低于任何 j>i 行的读区间 [8E(j),…) ✓(含 retry 行
    //      从 rk2 的写)。违例(retry 欠估累计 > 正常行富余)→ 回退旧路径。whale 阵省 C×12GB 级峰值。
    bool in_place = false;
    {
        long long *d_eoff = decltype(d_eoff)(dev_alloc((A_rows + 1) * sizeof(long long)));
        CHECK_CUDA(cudaMemset(d_eoff, 0, sizeof(long long)));
        thrust::inclusive_scan(thrust::make_transform_iterator(thrust::device_ptr<int>(d_est), ToLL()),
                               thrust::make_transform_iterator(thrust::device_ptr<int>(d_est + A_rows), ToLL()),
                               thrust::device_ptr<long long>(d_eoff + 1));
        // margin = min_j (E(j) − C_rp[j]);dC_rp 是 int(实际行和 ≤ 2^31 保证:C_nnz 是 int)
        long long *d_marg = decltype(d_marg)(dev_alloc((A_rows + 1) * sizeof(long long)));
        margin_kernel<<<(A_rows + 255) / 256, 256>>>(d_eoff, dC_rp, A_rows + 1, d_marg);
        long long min_marg;
        thrust::device_ptr<long long> mmin = thrust::min_element(thrust::device_ptr<long long>(d_marg), thrust::device_ptr<long long>(d_marg) + A_rows + 1);
        CHECK_CUDA(cudaMemcpy(&min_marg, mmin.get(), sizeof(long long), cudaMemcpyDeviceToHost));
        // ⚠ 2026-08-27 rajat16 实测 294411 行内乱序(原 0)= 某紧凑路径交叉写读(疑 csort 多趟/
        //   gapped 区间重读)。默认关闭,守卫+代码保留,FIX2=1 可开(普通阵 margin 普遍 ≥0)
        static int g_fix2 = -1;
        if (g_fix2 < 0) { const char *e = getenv("FIX2"); g_fix2 = (e && *e) ? atoi(e) : 0; }
        in_place = (min_marg >= 0) && g_fix2 && ((long long)C_nnz_result > 50LL * 1000 * 1000)
                   && (dense_nr == 0) && !g_d5h_on;   // 方案5/D5H 互斥:direct 写终态与 tmp 前缀紧缩重叠,且 est 已置 0 致 margin 失义
        dev_free(d_eoff); dev_free(d_marg);
        dbg("[%s] compact in_place=%d (min_margin=%lld)\n", tag, (int)in_place, min_marg);
    }
    void *dC = nullptr;
    int   *dC_ci; double *d_val_ip;
    if (in_place) {
        dC_ci = (int*)d_tmp_key;          // col 前缀紧缩进 key 区(4B 写 ≤ 8B 源,同一守卫)
        d_val_ip = d_tmp_val;             // val 原地(等宽,守卫直接适用)
    } else {
        dC = decltype(dC)(dev_alloc(C_total));
        char *cb = (char*)dC;
        dC_ci = (int*)(cb + C_rp_al);
        d_val_ip = (double*)(cb + C_rp_al + C_ci_al);
    }
    d_val = d_val_ip;
    // 方案5 数值 pass(Phase B v2 游标版):dense 行数据驱动窗口直写终态
    int *d_smap = nullptr, *d_smap_off = nullptr;
    if (dense_nr > 0) {
        prof("dense_direct", [&]{
            // 全局 cursor 区:每 dense 行 a_len 个 int,偏移 = alen 按行散射后的 inclusive scan(Σ ≤ nnz)
            int *d_alen = decltype(d_alen)(dev_alloc((size_t)dense_nr * sizeof(int)));
            d_smap_off = decltype(d_smap_off)(dev_alloc((A_rows + 1) * sizeof(int)));
            alen_gather_kernel<<<(dense_nr + 255) / 256, 256>>>(dense_rows, dense_nr, dA_rp, d_alen);
            CHECK_CUDA(cudaMemset(d_smap_off, 0, (size_t)(A_rows + 1) * sizeof(int)));
            scatter_alen_kernel<<<(dense_nr + 255) / 256, 256>>>(dense_rows, dense_nr, d_alen, d_smap_off);
            // ⚠ exclusive(偏移语义)+ 无槽位移位:off[i] = Σ_{r<i} alen[r];off[A_rows] = 总量。
            //   (首版 inclusive+1 槽双错:行 chunk 右移互相覆盖 → 游标读到别行 B 位置 → j<0 →
            //    SMEM 下越界 → "illegal instruction";docs/21 同款陷阱再现)
            thrust::exclusive_scan(thrust::device_ptr<int>(d_smap_off),
                                   thrust::device_ptr<int>(d_smap_off + A_rows + 1),
                                   thrust::device_ptr<int>(d_smap_off));
            int sm_tot;
            CHECK_CUDA(cudaMemcpy(&sm_tot, d_smap_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
#ifdef DBG
            {   // TEMP:offset 验证(首 dense 行 off 应=0;Σ 应=sm_tot)
                int h0 = -1, h1 = -1, h2 = -1, r0 = -1, r1 = -1;
                CHECK_CUDA(cudaMemcpy(&h0, d_smap_off, sizeof(int), cudaMemcpyDeviceToHost));
                if (dense_rows) { CHECK_CUDA(cudaMemcpy(&r0, dense_rows, sizeof(int), cudaMemcpyDeviceToHost));
                                 CHECK_CUDA(cudaMemcpy(&h1, d_smap_off + r0, sizeof(int), cudaMemcpyDeviceToHost)); }
                CHECK_CUDA(cudaMemcpy(&h2, d_smap_off + 1, sizeof(int), cudaMemcpyDeviceToHost));
                fprintf(stderr, "[pb2dbg] sm_tot=%d off[0]=%d first_dense_row=%d off[row0]=%d off[1]=%d\n",
                        sm_tot, h0, r0, h1, h2);
            }
#endif
            d_smap = decltype(d_smap)(dev_alloc((size_t)(sm_tot > 0 ? 2 * sm_tot : 1) * sizeof(int)));   // 2×:活动游标 + 本窗镜像
            dev_free(d_alen);
            size_t wsm = ((size_t)PB2_W * 9 + 3) / 4 * 4 + (size_t)PB2_W * sizeof(int)
                       + 2 * (size_t)SMAP_SMEM_MAX * sizeof(int);
            CHECK_CUDA(cudaFuncSetAttribute(hash_dense_direct_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)wsm));
            static int g_pb2cur = -1;
            if (g_pb2cur < 0) { const char *e = getenv("PB2_CURSOR"); g_pb2cur = (e && *e && atoi(e) == 0) ? 0 : 1; }
            // docs/39 路由 v3:矩阵级 dense_win ∧ 高 dup(≥4)→ search(TSOPF_FS 族实测 -26/-13%;
            // 低 dup 的 bloweya/mult_dcop/vsp/brainpc2 游标大胜,勿动);bin 行恒 cursor。
            int uc = g_pb2cur;
            if (uc == 1 && !dense_rows && total_est > 0 && (double)total_flop / (double)total_est >= 4.0)
                uc = 0;
            hash_dense_direct_kernel<<<dense_nr, 512, wsm>>>(
                dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri, A_rows, A_cols,
                dC_rp, dC_ci, d_val, d_row_nnz, dense_rows, dense_rows ? d_span_lo : nullptr,
                d_flop, d_maxbl, d_smap, d_smap_off, sm_tot, uc);
            CHECK_CUDA(cudaGetLastError());
        });
    }
    // D5H 数值 pass:bins 1-10 MODE=2 直写 dC(小行 count-sort 有序;大行无序 → 原地 csort)
    if (g_d5h_on) {
        prof("hash_direct", [&]{
            for (int bi = 1; bi <= 10; bi++) {
                int n = h_cnt[bi];
                if (n == 0) continue;
                int *rows_ptr = d_sort + h_off[bi];
                int ht = 32 << bi;
                size_t smem_flat = (size_t)ht * (sizeof(int) + sizeof(double));
                {
                    static bool done2 = false;
                    if (!done2) {
                        CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, HASH_CAP * (int)(sizeof(int) + sizeof(double))));
                        done2 = true;
                    }
                }
                hash_spa_kernel<2><<<n, HASH_BLOCK, smem_flat>>>(
                    dA_rp, dA_ci, dA_val, dB_rp, dB_ci, dB_val, upper_tri,
                    rows_ptr, n, ht, d_off,
                    d_tmp_key, d_tmp_val, d_row_nnz, d_overflow, d_ovf_rows, d_ovf_cnt, d_row_ovf,
                    d_flop, d_maxbl, 0, dC_rp, dC_ci, d_val);
                CHECK_CUDA(cudaGetLastError());
            }
        });
        CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));   // MODE=2 理论不再溢出(count 同表已插过);防御复查
        if (overflow) {   // 不可重试行(flop 超表上限)→ 整阵回退(与 legacy 口径一致)
            fprintf(stderr, "[hash] OVERFLOW(D5H): 某 hash 行 distinct > HASH_CAP=%d → 回退 merge\n", HASH_CAP);
            *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
            dev_free(dA); dev_free(d_off); dev_free(d_row_nnz); dev_free(d_overflow);
            dev_free(d_tmp_key); dev_free(d_tmp_val); dev_free(dC_rp); dev_free(dC);
            dev_free(d_bkid); dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos); dev_free(d_sort);
            dev_free(d_est); dev_free(d_flop); dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl);
            dev_free(d_gv); dev_free(d_gv_off); dev_free(d_ovf_rows); dev_free(d_ovf_cnt); dev_free(d_row_ovf);
            dev_free(d_heavy_ht); dev_free(d_heavy_tab_off); dev_free(d_tab_col); dev_free(d_tab_val);
            dev_free(d_seg_beg); dev_free(d_seg_end); dev_free(d_scr_key); dev_free(d_scr_val); dev_free(d_cub_tmp);
            dev_free(d_smap); dev_free(d_smap_off); dev_free(d_csc_cp); dev_free(d_csc_ri); dev_free(d_csc_val);
            return;
        }
    }
    prof("compact+sort", [&]{
        if (!in_place)
            CHECK_CUDA(cudaMemcpy(dC, dC_rp, C_rp_sz, cudaMemcpyDeviceToDevice));   // row_ptr 落位(~A_rows ints,微秒级,纳入计时)
        for (int bi = 0; bi < ((dense_mode || dense_win_mode) ? 1 : N_BINS); bi++) {
            int n = (dense_mode || dense_win_mode) ? A_rows : h_cnt[bi];
            int *rows_ptr = (dense_mode || dense_win_mode) ? nullptr : (d_sort + h_off[bi]);
            if (dense_mode || dense_win_mode) {
                // 方案5:矩阵级 dense 行已在 dense_direct 直写终态 → copy 整段跳过
                if (dense_nr > 0) break;
                // dense 输出全局有序:identity 行表 + 一趟 warp-batched copy
                int *d_all = decltype(d_all)(dev_alloc((size_t)A_rows * sizeof(int)));
                thrust::sequence(thrust::device_ptr<int>(d_all), thrust::device_ptr<int>(d_all + A_rows));
                hash_compact_copy_warp_kernel<<<(A_rows + 7) / 8, 256>>>(
                    d_all, A_rows, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                break;
            }
            if (n == 0) continue;               // 空 bin 免 <<<0,TPB>>> 非法 launch(dense 重构曾吃掉此守卫)
            if (g_d5h_on && bi >= 1 && bi <= 10) {   // D5H:小行已有序直写免善后;大行原地 csort
                if (bi <= 5) continue;
                if (bi <= 7) launch_csort_ip<512, 8>(n, rows_ptr, dC_rp, dC_ci, d_val);
                else         launch_csort_ip<256, 64>(n, rows_ptr, dC_rp, dC_ci, d_val);
                CHECK_CUDA(cudaGetLastError());
                continue;
            }
            if (bi == 0) {
                // 批量行(est≤64 且 k≤32):warp-per-row copy
                hash_compact_copy_warp_kernel<<<(n + 7) / 8, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());   // TEMP:定位
            } else if (bi <= 5) {
                // 小行(ht≤CSORT_HT):accumulate 已 count-sort,这里只 compact_copy(tmp→CSR)
                // Fix#11(审查):warp-per-row(8 行/CTA)替 1CTA/行 —— 万行级 bin 的 CTA 数砍 8×
                hash_compact_copy_warp_kernel<<<(n + 7) / 8, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
            } else if (bi <= 7) {
                if (A_cols < (1 << 20) && !getenv("NOCSF"))   // 融合键:pos 12 位(512×8=4096)→ n<2^20(docs/38)
                    launch_csort_fused<512, 8>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, 31 - __builtin_clz(A_cols) + 1);
                else
                    launch_csort<512, 8>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else if (bi <= 9) {
                if (A_cols < (1 << 18) && !getenv("NOCSF"))   // pos 14 位(256×64=16384)→ n<2^18
                    launch_csort_fused<256, 64>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, 31 - __builtin_clz(A_cols) + 1);
                else
                    launch_csort<256, 64>(n, rows_ptr, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else if (bi == BIN_HEAVY) {
                // heavy:accumulate 内已分段排序(compact 读 scratch)
                hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_scr_key, d_scr_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());   // TEMP:定位
            } else if (bi >= BIN_SSPAN0 && bi <= BIN_SSPAN0 + 3) {
                hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else if (bi == BIN_DITER) {
                // dense-iter:窗口升序 = 行内天然有序,只 copy。方案5:direct 已直写 → 跳过
                if (dense_nr == 0)
                    hash_compact_copy_kernel<<<n, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            } else {
                // ultra(行≤32)和 subwarp8(bin12):kernel 内已插入排序 → warp copy
                hash_compact_copy_warp_kernel<<<(n + 7) / 8, 256>>>(rows_ptr, n, d_off, d_row_nnz, dC_rp, d_tmp_key, d_tmp_val, dC_ci, d_val, d_row_ovf);
                CHECK_CUDA(cudaGetLastError());
            }
        }
        if (d_retry_n > 0) {   // 行级重试的行:从重试区(已分段排序)拷到 CSR
            if (d_retry_n > 0 && d_retry_n <= A_rows)   // TEMP:守卫+定位
                hash_compact_copy_kernel<<<d_retry_n, 256>>>(
                    d_retry_rows, d_retry_n, d_retry_off, d_row_nnz, dC_rp,
                    d_retry_key, d_retry_val, dC_ci, d_val, nullptr);
        }
        CHECK_CUDA(cudaGetLastError());
    });
    // docs/21 Fix0:重试中间缓冲(compact 消费完)立即释放 —— 此前从不释放,高 flop 阵每调用
    // 泄漏 r_slots×32B + rt_total×12B,warmup 多轮累加直接 OOM(rajat 类 DNF 帮凶)
    dev_free(rtc); dev_free(rtv); dev_free(rk); dev_free(rv); dev_free(rk2); dev_free(rv2); dev_free(rct);
    dev_free(d_rht); dev_free(d_rtab); dev_free(d_rslot); dev_free(d_roff); dev_free(d_rsb); dev_free(d_rse);

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
    //          in_place 模式:col/val 在 tmp 里 → 3 段拷进同一 pinned 布局(rq: rp 段从 dC_rp)。

    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total));
    prof("d2h", [&]{
        if (in_place) {
            char *hb = (char*)C_buffer;
            CHECK_CUDA(cudaMemcpyAsync(hb, dC_rp, C_rp_sz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpyAsync(hb + C_rp_al, d_tmp_key, C_ci_sz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpyAsync(hb + C_rp_al + C_ci_al, d_tmp_val, C_v_sz, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaStreamSynchronize(0));
        } else {
            CHECK_CUDA(cudaMemcpy(C_buffer, dC, C_total, cudaMemcpyDeviceToHost));
        }
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
    // 成功路径逐行数组泄漏修复(2026-08-27):d_bkid/d_sort/d_est/d_flop/d_span_*/d_maxbl 此前只在
    // 早退路径释放,成功路径漏 ~28B×A_rows/调用(USE_DEV_POOL 默认关 = cudaMalloc 模式;whale 阵
    // warmup+bench 多轮累加 GB 级 —— c-73 峰值 ~62GB 的隐性推手之一)。尾部释放与 d2h 后既有
    // 12 buffer 同区域,不进 hash-prof 计时相位。
    dev_free(d_hybrid_val);   // Fix#6(审查):bin0 hybrid 值池从不释放(n×1KB/调用)
    dev_free(d_bkid); dev_free(d_sort); dev_free(d_est); dev_free(d_flop);
    dev_free(d_span_lo); dev_free(d_span_len); dev_free(d_maxbl); dev_free(d_gv); dev_free(d_gv_off);
    dev_free(d_cnt); dev_free(d_offb); dev_free(d_pos);
    dev_free(d_smap); dev_free(d_smap_off);   // Phase B v2 全局 cursor 区
    dev_free(d_hybrid_val);
    { cudaError_t pe = cudaGetLastError(); if (pe != cudaSuccess) fprintf(stderr, "[probe] frees 后: %s\n", cudaGetErrorString(pe)); }                    // Fix#6(审查)
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
