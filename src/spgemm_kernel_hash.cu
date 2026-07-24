#include "spgemm.h"
#include "hash_prof.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
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

// 并行 count flop_ub:每行一个 block,线程并行 Σ nnz(row k) + block 归约(无 straggler,O(nnz_A))。
//   小阵 streamline 用它替 HLL 两阶段(1 kernel vs 2),flop_ub 是确定性上界(≥distinct,无 underflow)。
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

// HLL(HyperLogLog)概率基数估计:替 flop_ub 定 tmp buffer + hash 表大小(统一 sizing)
// Ocean 两阶段架构:Phase1 对 B(=A 自乘)每行建 sketch(O(nnz));Phase2 对 A 每行读 B 的 sketch 做 packed merge(O(nnz))
// 原理详见 worklog/hll_estimation_explained.md
#define HLL_P 7                       // precision bits → m=128 寄存器(对齐 Ocean HLL_CONSTANT 表上限),误差~9.2%
#define HLL_M (1 << HLL_P)
#define HLL_EXPAND 2.0                // expansion(覆盖 HLL 低估;×2 → 安全;溢出 → 回退 merge3 兜底)
#define HLL_ULTRA_THR 16              // bin-snap:est≤此值 → ultrasparse(线性 kernel,CAP=32 留 2× 余量)
#define STREAMLINE_NNZ 100000         // 小阵 pipeline 精简:A_nnz<此值 → flop_ub(1 count kernel)替 HLL 两阶段
#define WARP_SIZE 32
// HLL 偏差校正 α(标准公式 0.7213/(1+1.079/m) 预算;对齐 Ocean HLL_CONSTANT[Common.h])
constexpr double HLL_CONSTANT[8] = {0.0, 0.0, 0.0, 0.0, 0.673, 0.697, 0.709, 0.715};

// MurmurHash3(Ocean MurmurHash.cuh 同款:full mix = body(c1/c2 rotl) + len + fmix32,seed=1234)
// 替掉旧版只有 fmix32 finalizer 的弱 hash。HLL 两阶段复用此函数(SPA 累加暂仍用 Knuth 乘法,隔离测量)
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

// ============ Phase 1: 对 B(=A)的每行建 HLL sketch(O(nnz),线性扫描) ============
// 每个 block 处理 rows_per_block 行,线性遍历 B 的 CSR → hash(col) → atomicMax SMEM 寄存器 → 写 global
__global__ void hll_construct_kernel(
    const int *B_row_ptr, const int *B_col_ind,
    int B_rows, int B_nnz,
    unsigned char *global_hll,           // [B_rows * HLL_M] uint8 寄存器
    int rows_per_block)
{
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, B_rows);
    if (row_start >= B_rows) return;

    int elem_start = B_row_ptr[row_start];
    int elem_end = (row_end < B_rows) ? B_row_ptr[row_end] : B_nnz;
    int num_rows = row_end - row_start;

    extern __shared__ int smem_raw[];    // 共享 extern(全文件统一 int 基类型)
    unsigned int *smem = (unsigned int*)smem_raw;    // [num_rows * HLL_M] scratch(uint32,atomicMax 用)
    int total_items = num_rows * HLL_M;
    for (int i = threadIdx.x; i < total_items; i += blockDim.x) smem[i] = 0;

    // 行偏移表(用于判断当前元素属于哪行)
    int *row_offsets = smem_raw + total_items;
    for (int i = threadIdx.x; i <= num_rows; i += blockDim.x)
        row_offsets[i] = (row_start + i < B_rows) ? B_row_ptr[row_start + i] : elem_end;
    __syncthreads();

    // 线性遍历 B 的元素(coalesced!)
    int current_row = 0;
    unsigned int *current_scratch = smem;
    for (int e = elem_start + threadIdx.x; e < elem_end; e += blockDim.x) {
        unsigned int col = (unsigned int)B_col_ind[e];
        // 找到当前元素属于哪行
        while (e >= row_offsets[current_row + 1]) {
            current_row++;
            current_scratch = smem + current_row * HLL_M;
        }
        unsigned int h = murmur_hash3(col);
        int idx = h & (HLL_M - 1);                  // 低 P 位 → 寄存器索引
        unsigned int rho = __clz(h) + 1;            // 全 32 位前导零+1(高位决定;对齐 Ocean hllAdd)
        const unsigned int rho_max = (32 - HLL_P) + 1;
        if (rho > rho_max) rho = rho_max;           // 高 (32-P) 位全 0 时封顶(等价旧 rest==0 分支)
        atomicMax(&current_scratch[idx], rho);
    }
    __syncthreads();

    // 收缩 uint32 → uint8 写回 global
    unsigned char *global_ptr = global_hll + row_start * HLL_M;
    for (int i = threadIdx.x; i < num_rows * HLL_M; i += blockDim.x)
        global_ptr[i] = (unsigned char)min(smem[i], 255u);
}

// ============ Phase 2: 对 A 的每行 merge B 的 HLL sketch(O(nnz_A),packed __vmaxu4) ============
// 每个 block 处理 A 的一行。读 A 引用的 B 行的 HLL sketch(uint8),packed 4-byte max → SMEM reduce → 估计公式
__global__ void hll_merge_kernel(
    const int *A_row_ptr, const int *A_col_ind,
    int A_rows,
    const unsigned char *b_hll,           // [B_rows * HLL_M] from Phase 1
    int *est_nnz)                         // [A_rows] output
{
    int row = blockIdx.x;
    if (row >= A_rows) return;
    int tid = threadIdx.x;

    // packed merge:每个线程维护一个 uint32 packed max(4 个 uint8 寄存器)
    // 每次迭代读 4 个 B 行的 4 字节,做 __vmaxu4
    constexpr int bytes_per_thread = 4;   // 每线程每次读 4 字节
    int elements_per_iter = blockDim.x * bytes_per_thread;
    int b_rows_per_iter = elements_per_iter / HLL_M;
    if (b_rows_per_iter == 0) b_rows_per_iter = 1;

    extern __shared__ int smem_merge_raw[];
    unsigned char *smem_merge = (unsigned char*)smem_merge_raw;
    for (int i = tid; i < HLL_M; i += blockDim.x) smem_merge[i] = 0;
    __syncthreads();

    int start_elem = A_row_ptr[row];
    int end_elem = A_row_ptr[row + 1];

    // packed merge:每个线程跨 B 行步进,读 4 字节做 __vmaxu4
    int my_row_offset = tid / (blockDim.x / b_rows_per_iter);
    int my_col_offset = bytes_per_thread * (tid % (blockDim.x / b_rows_per_iter));

    unsigned int packed_max = 0;
    for (int e = start_elem + my_row_offset; e < end_elem; e += b_rows_per_iter) {
        int row_b = A_col_ind[e];
        int byte_idx = row_b * HLL_M + my_col_offset;
        // 读 4 字节(packed),__vmaxu4 4 路并行 max
        if (byte_idx + 4 <= (row_b + 1) * HLL_M) {
            unsigned int buf = *(const unsigned int*)(b_hll + byte_idx);
            packed_max = __vmaxu4(packed_max, buf);
        }
    }

    // 展开到 SMEM(uint8)
    #pragma unroll
    for (int i = 0; i < bytes_per_thread; i++) {
        smem_merge[tid * bytes_per_thread + i] = (unsigned char)(packed_max & 0xFF);
        packed_max >>= 8;
    }
    __syncthreads();

    // 并行估计 reduce(对齐 Ocean hllMerge:for 累加 → warp shuffle → atomicAdd 跨 warp)
    constexpr int items_padded = HLL_M < 32 ? 32 : HLL_M;   // P=7 → 128
    constexpr double a = HLL_CONSTANT[HLL_P];               // α 校正(查表,等价 0.7213/(1+1.079/m))
    double z = 0.0;
    int empty_reg = 0;
    for (int i = tid; i < items_padded; i += blockDim.x) {
        unsigned char val = 0;
        if (i < HLL_M) {
            for (int j = 0; j < b_rows_per_iter; j++)        // 跨 batch 取 max(b_rows_per_iter 路写)
                if (val < smem_merge[j * HLL_M + i]) val = smem_merge[j * HLL_M + i];
        }
        z += 1.0 / (double)(1ULL << val);
        if (val == 0) empty_reg++;
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {       // warp 内归约 z / empty_reg
        z += __shfl_down_sync(0xFFFFFFFF, z, off);
        empty_reg += __shfl_down_sync(0xFFFFFFFF, empty_reg, off);
    }
    __shared__ double total_z;
    __shared__ int total_empty_reg;
    if (tid == 0) { total_z = 0.0; total_empty_reg = 0; }
    __syncthreads();
    if (tid % WARP_SIZE == 0) { atomicAdd(&total_z, z); atomicAdd(&total_empty_reg, empty_reg); }
    __syncthreads();

    // HLL 估计公式(thread 0):调和平均 + 小范围 linear counting
    if (tid == 0) {
        z = total_z; empty_reg = total_empty_reg;
        double E = a * (double)HLL_M * (double)HLL_M / z;
        if (empty_reg != 0 && E <= 2.5 * HLL_M)
            E = (double)HLL_M * log((double)HLL_M / (double)empty_reg);
        int temp = (int)(E * HLL_EXPAND);
        if (temp < 1) temp = 1;
        // bin-snap(对齐 Ocean hllMerge:估计 → hash 表大小)。统一 sizing:此 est 同时定
        //   ① tmp buffer 每行槽位(row_off = est 的 prefix sum)② hash 表大小(compute_bucket 直接读)。
        //   ultra(temp≤HLL_ULTRA_THR)→ 线性 kernel,est 保留紧 temp;否则 snap 到 next_pow2 ∈[32,HASH_CAP]。
        int est;
        if (temp <= HLL_ULTRA_THR) {
            est = temp;
        } else {
            int ht = 32;
            while (ht < temp && ht < HASH_CAP) ht <<= 1;
            est = ht;
        }
        est_nnz[row] = est;
    }
}

// 按【桶】跑:blockIdx.x = 桶内行索引,实际行号 = bucket_rows[idx];ht_size = 该桶 hash 表大小(2 的幂,全 launch 统一)。
// binning:轻行桶用小表 → 高 SMEM 占用率;重行桶用大表;distinct>ht_size → overflow_flag(上层回退 merge3)。
__global__ void hash_spa_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
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
    // G 从 ht_size 推(零 scan):ht_size 大 = 重行(更多 distinct → 更长 B-row → 更多 j) → G 大;
    //   ht_size 小 = 轻行(短 B-row) → G 小(更多 k 并行)。G ∈ {4,8,16,32}。
    int G = (ht_size <= 64) ? 4 : (ht_size <= 256) ? 8 : (ht_size <= 1024) ? 16 : 32;
    int num_groups = HASH_BLOCK / G;
    int my_group = tid / G, my_id = tid % G;
    for (int p = rs + my_group; p < re; p += num_groups) {   // 并行 k(stride num_groups)
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        for (int q = ks + my_id; q < ke; q += G) {           // 组内 G 线程并行 j
            int j = A_col_idx[q];
            double v = a_ik * A_val[q];
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

    // extract 去重项 → tmp。小行(ht≤CSORT_HT)在 SMEM 内 compact+count-sort 写有序(Ocean compactAndSort 式);
    //   大行无序写(交 compact_sort 的 BlockRadixSort)。
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

// ultrasparse:est ≤ HLL_ULTRA_THR(HLL 估计 ≤ 16)。不建 hash,每线程一行,寄存器小数组线性去重累加
//   (省 hash 建表/atomic/行内排序)。对应 Ocean 的 use_ultrasparse_workflow。
//   CAP=32 给 est≤16(真 distinct ~≤16)留 2× 余量;若仍不够(HLL 低估)→ overflow_flag → 上层回退 merge3。
__global__ void hash_ultra_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *ultra_rows, int n_ultra,
    const int *row_off,
    unsigned long long *tmp_key, double *tmp_val, int *row_nnz, int *overflow_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_ultra) return;
    int i = ultra_rows[idx];
    const int CAP = 32;                          // 2× 于 HLL_ULTRA_THR,吸收 HLL 低估
    int u_col[CAP]; double u_val[CAP]; int u_cnt = 0;
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p]; double a_ik = A_val[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        for (int q = ks; q < ke; q++) {
            int j = A_col_idx[q]; double v = a_ik * A_val[q];
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

// per-row compact+sort(Ocean sortOutputDyn 式):每 block 一行。
//   从 tmp(gapped,row_off)读 (col,val) → cub::BlockRadixSort 行内按 col 排 → 直接写 CSR(packed,row_ptr)。
//   一次替掉 hash_compact_kernel + 全局 thrust::sort + split_key_kernel。
//   要求 TPB*IPT ≥ ht(该行 hash 表大小);因非溢出行 ht ≥ row_nnz,故装得下。col=key 低 32 位,排序位 [0,32)。
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
// 每行算 bucket_id:est≤HLL_ULTRA_THR → ultra(bin N_BINS-1);否则 est(已是 next_pow2)→ bin 0..9
//   统一 sizing:bucket 直接由 HLL est 决定(不再依赖 flop_ub)
__global__ void compute_bucket_kernel(
    const int *est, int A_rows, int ultra_thr, int *bucket_id)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int e = est[i];
    if (e <= ultra_thr) { bucket_id[i] = N_BINS - 1; return; }
    int bi = 0, ht = 32;
    while (ht < e && ht < HASH_CAP) { ht <<= 1; bi++; }
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

// ========== Host ==========

void spgemm_self_product_hash(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    dbg("[hash] start (HASH_CAP=%d, HLL_P=%d)\n", HASH_CAP, HLL_P);
    HashProf prof("hash-prof");

    size_t A_rp_sz = (A_rows + 1) * sizeof(int);
    size_t A_ci_sz = (size_t)A_nnz * sizeof(int);
    size_t A_v_sz = (size_t)A_nnz * sizeof(double);
    size_t A_total = ALIGN8(A_rp_sz + A_ci_sz) + A_v_sz;
    void *dA; CHECK_CUDA(cudaMalloc(&dA, A_total));
    prof("h2d", [&]{ CHECK_CUDA(cudaMemcpy(dA, A_buffer, A_total, cudaMemcpyHostToDevice)); });
    char *b = (char*)dA;
    int   *dA_rp  = (int*)b;
    int   *dA_ci  = (int*)(b + A_rp_sz);
    double *dA_val = (double*)(b + ALIGN8(A_rp_sz + A_ci_sz));
    dbg("[hash] h2d\n");

    // Stage 1: row_off 由 HLL est 的 scan 给出。
    //   (streamline:小阵用 flop_ub count 替 HLL 两阶段。bp_0 有效(0.36→0.32);bin8-9 sort
    //    landmine 已修(<256,64>);count 实测 0.011ms 快(旧 0.85 是 clock 抖动)。现已重开。)
    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));

    // 估计每行 distinct:小阵(A_nnz<STREAMLINE_NNZ)→ flop_ub(1 并行 count kernel);大阵 → HLL 两阶段
    int *d_est; CHECK_CUDA(cudaMalloc(&d_est, A_rows * sizeof(int)));
    if (A_nnz < STREAMLINE_NNZ) {
        prof("count_flop", [&]{
            count_intermediates_par_kernel<<<A_rows, 256>>>(dA_rp, dA_ci, A_rows, d_est);
            CHECK_CUDA(cudaGetLastError());
        });
    } else {
        // Phase 1: 对 A(=B 自乘)每行建 HLL sketch。线性扫 CSR,O(nnz) 非 O(flop)。
        unsigned char *d_hll; CHECK_CUDA(cudaMalloc(&d_hll, (size_t)A_rows * HLL_M));
        int rows_per_block = 32;  // 大 rows_per_block 摊薄 SMEM init 开销(32×1024×4=128KB → opt-in)
        size_t smem_p1 = (size_t)rows_per_block * HLL_M * sizeof(unsigned int) + (size_t)(rows_per_block + 1) * sizeof(int);
        if (smem_p1 > 48 * 1024)
            CHECK_CUDA(cudaFuncSetAttribute(hll_construct_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_p1));
        int grid_p1 = (A_rows + rows_per_block - 1) / rows_per_block;
        prof("hll_construct", [&]{
            hll_construct_kernel<<<grid_p1, HASH_BLOCK, smem_p1>>>(
                dA_rp, dA_ci, A_rows, A_nnz, d_hll, rows_per_block);
            CHECK_CUDA(cudaGetLastError());
        });
        // Phase 2: 对 A 每行,读 B 的 HLL sketch 做 packed __vmaxu4 merge。O(nnz_A)。
        int p2_block = HLL_M / 2;   // blockDim×4 = 2×HLL_M → b_rows_per_iter=2(Ocean 同款,2× 吞吐)
        int smem_p2 = HLL_M * 2;   // 2 批 × HLL_M(smem_merge 用)
        prof("hll_merge", [&]{
            hll_merge_kernel<<<A_rows, p2_block, smem_p2>>>(
                dA_rp, dA_ci, A_rows, d_hll, d_est);
            CHECK_CUDA(cudaGetLastError());
        });
        cudaFree(d_hll);
    }
    int total_est;
    prof("est_scan", [&]{
        CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
        thrust::inclusive_scan(thrust::device_ptr<int>(d_est),
                               thrust::device_ptr<int>(d_est + A_rows),
                               thrust::device_ptr<int>(d_off + 1));
        CHECK_CUDA(cudaMemcpy(&total_est, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入计时(对齐 Ocean)
    });
    dbg("[hash] total_est=%d\n", total_est);

    // Stage 2: GPU 端分桶(Ocean 风格:全 device,无 host 往返)+ 预分配大 buffer(零 per-bucket malloc/free)
    int *d_row_nnz; CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
    int *d_overflow; CHECK_CUDA(cudaMalloc(&d_overflow, sizeof(int)));
    unsigned long long *d_tmp_key; double *d_tmp_val;
    CHECK_CUDA(cudaMalloc(&d_tmp_key, (size_t)total_est * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_tmp_val, (size_t)total_est * sizeof(double)));
    // d_val(C_val)现 alias 进连续 dC(compact+sort 直写,见 cnnz_scan 后),不再单独分配/释放。
    double *d_val = nullptr;

    // 2a: GPU 端分桶(HLL est → bucket):bucket_id → count → exclusive scan → scatter(全 device)
    int *d_bkid; CHECK_CUDA(cudaMalloc(&d_bkid, A_rows * sizeof(int)));
    int *d_cnt;  CHECK_CUDA(cudaMalloc(&d_cnt,  N_BINS * sizeof(int)));
    int *d_offb; CHECK_CUDA(cudaMalloc(&d_offb, N_BINS * sizeof(int)));
    int *d_pos;  CHECK_CUDA(cudaMalloc(&d_pos,  N_BINS * sizeof(int)));
    int *d_sort; CHECK_CUDA(cudaMalloc(&d_sort, A_rows * sizeof(int)));   // 预分配:桶有序行号
    int h_cnt[N_BINS], h_off[N_BINS];
    prof("binning", [&]{
        compute_bucket_kernel<<<(A_rows + 255) / 256, 256>>>(d_est, A_rows, HLL_ULTRA_THR, d_bkid);
        CHECK_CUDA(cudaMemset(d_cnt, 0, N_BINS * sizeof(int)));
        bucket_count_kernel<<<(A_rows + 255) / 256, 256>>>(d_bkid, A_rows, d_cnt);
        thrust::exclusive_scan(thrust::device_ptr<int>(d_cnt),
                               thrust::device_ptr<int>(d_cnt + N_BINS),
                               thrust::device_ptr<int>(d_offb));
        CHECK_CUDA(cudaMemset(d_pos, 0, N_BINS * sizeof(int)));   // scatter 计数器从 0 起(非 offsets)
        scatter_rows_kernel<<<(A_rows + 255) / 256, 256>>>(d_bkid, A_rows, d_offb, d_pos, d_sort);
        CHECK_CUDA(cudaMemcpy(h_cnt,  d_cnt,  N_BINS * sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入(对齐 Ocean)
        CHECK_CUDA(cudaMemcpy(h_off,  d_offb, N_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    });

    // 2b: opt-in max SMEM
    {
        size_t maxsm = (size_t)HASH_CAP * (sizeof(int) + sizeof(double));   // sh_col[int]+sh_val[double]
        if (maxsm > 48 * 1024)
            CHECK_CUDA(cudaFuncSetAttribute(hash_spa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)maxsm));
    }
    // 2c: per-bin launch(hash 插入 + extract;prof 内部已 sync)。memset+overflow D2H 纳入计时(对齐 Ocean)
    int overflow;
    prof("accumulate", [&]{
        CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
        for (int bi = 0; bi < N_BINS; bi++) {
            int n = h_cnt[bi];
            if (n == 0) continue;
            int *rows_ptr = d_sort + h_off[bi];
            if (bi == N_BINS - 1) {
                // ultra(est≤HLL_ULTRA_THR):线性,免 hash
                hash_ultra_kernel<<<(n + 255) / 256, 256>>>(
                    dA_rp, dA_ci, dA_val, rows_ptr, n, d_off, d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
            } else {
                int ht = 32 << bi;
                size_t smem = (size_t)ht * (sizeof(int) + sizeof(double));   // sh_col[int]+sh_val[double]
                hash_spa_kernel<<<n, HASH_BLOCK, smem>>>(
                    dA_rp, dA_ci, dA_val, rows_ptr, n, ht, d_off,
                    d_tmp_key, d_tmp_val, d_row_nnz, d_overflow);
            }
        }
        CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入
    });
    if (overflow) {
        fprintf(stderr, "[hash] OVERFLOW: 某行 distinct 列 > HASH_CAP=%d → 回退 merge(dispatcher 处理)\n", HASH_CAP);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        cudaFree(dA); cudaFree(d_off); cudaFree(d_row_nnz);
        cudaFree(d_overflow); cudaFree(d_tmp_key); cudaFree(d_tmp_val);
        cudaFree(d_bkid); cudaFree(d_cnt); cudaFree(d_offb); cudaFree(d_pos); cudaFree(d_sort); cudaFree(d_est);
        return;
    }

    // Stage 3: scan row_nnz → C_row_ptr + C_nnz(精确)
    int *dC_rp; CHECK_CUDA(cudaMalloc(&dC_rp, (A_rows + 1) * sizeof(int)));
    int C_nnz_result;
    prof("cnnz_scan", [&]{
        CHECK_CUDA(cudaMemset(dC_rp, 0, sizeof(int)));
        thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                               thrust::device_ptr<int>(d_row_nnz + A_rows),
                               thrust::device_ptr<int>(dC_rp + 1));
        CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_rp + A_rows, sizeof(int), cudaMemcpyDeviceToHost));   // D2H 纳入
    });
    dbg("[hash] C_nnz=%d (est=%d, %.2fx over-alloc)\n", C_nnz_result, total_est,
        total_est > 0 ? (double)total_est / C_nnz_result : 0.0);

    // HLL underflow check: actual C_nnz > estimated total → 回退 merge3
    if (C_nnz_result > total_est) {
        fprintf(stderr, "[hash] HLL underflow: C_nnz=%d > total_est=%d → 回退 merge\n", C_nnz_result, total_est);
        *C_buffer_out = nullptr; *C_rows = A_rows; *C_cols = A_cols; *C_nnz = -1;
        cudaFree(dA); cudaFree(d_off); cudaFree(d_row_nnz);
        cudaFree(d_overflow); cudaFree(d_tmp_key); cudaFree(d_tmp_val);
        cudaFree(d_bkid); cudaFree(d_cnt); cudaFree(d_offb); cudaFree(d_pos); cudaFree(d_sort); cudaFree(d_est);
        cudaFree(dC_rp);
        return;
    }

    // Stage 4+5+6: per-row compact+sort(Ocean sortOutputDyn 式 BlockRadixSort)一次替掉 compact + 全局 sort + split_key。
    //   每 block 一行:从 tmp(gapped,row_off)读 → BlockRadixSort 行内按 col 排 → 写 CSR(packed,row_ptr)。
    //   复用 est-分桶,按 bin 选 config(cap ≥ ht ≥ row_nnz):bi0-1/ultra→64×1, bi2-3→128×2, bi4-5→256×4, bi6-7→512×8, bi8-9→1024×16。
    //   输出连续 dC=[row_ptr|col|val](精确大小):dC_ci/d_val 是 dC 偏移别名 → compact+sort 直写连续布局,
    //   免去旧 pack 的 col/val 两个大 D2D gather(对齐 Ocean 连续 C)。dC_rp(scan 产物)落位 dC 头部用 1 个小 D2D。
    size_t C_rp_sz  = (A_rows + 1) * sizeof(int);
    size_t C_ci_sz  = (size_t)C_nnz_result * sizeof(int);
    size_t C_v_sz   = (size_t)C_nnz_result * sizeof(double);
    size_t C_rp_al  = ALIGN8(C_rp_sz);
    size_t C_ci_al  = ALIGN8(C_ci_sz);
    size_t C_total  = C_rp_al + C_ci_al + C_v_sz;
    void *dC; CHECK_CUDA(cudaMalloc(&dC, C_total));
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
        int *d_viol; CHECK_CUDA(cudaMalloc(&d_viol, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_viol, 0, sizeof(int)));
        hash_check_sorted_kernel<<<(A_rows + 255) / 256, 256>>>(dC_rp, dC_ci, A_rows, d_viol);
        int viol; CHECK_CUDA(cudaMemcpy(&viol, d_viol, sizeof(int), cudaMemcpyDeviceToHost));
        dbg("[hash] sorted check: %d 行内乱序违规\n", viol);
        cudaFree(d_viol);
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

    cudaFree(dA); cudaFree(d_off); cudaFree(d_row_nnz);
    cudaFree(d_overflow); cudaFree(d_tmp_key); cudaFree(d_tmp_val);
    cudaFree(dC_rp); cudaFree(dC);   // dC_ci/d_val 是 dC 的偏移别名,随 dC 一起释放
}
