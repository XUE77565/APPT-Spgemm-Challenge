#include "spgemm.h"
#include "hash_prof.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ==========================================================================
//  Serial k-way Merge 版 Gustavson 自乘 C = A·A
// --------------------------------------------------------------------------
//  与 spgemm_kernel_manual.cu 的 ESC(expand→sort→reduce→finalize)对照。
//  关键观察:A 是 CSR → 每行的列有序;行 i 的中间项列 = 若干【有序列链】
//  {A[k,:] : k∈A[i,:]} 的并集。对这些有序链做 k-way merge,即可直接得到
//  行内列有序,merge 时同列自然求和去重 → sort + reduce 一起省掉。
//
//  本文件是【最简串行版】:每行一个 block,只 thread 0 做 expand / merge。
//  目的:验证 merge 路线可行性、找出与 CUB sort 的性能交叉点,后续再提并行度。
//  count / scan / pack / D2H 复用与 manual 相同的结构;ESC baseline 不动。
// ==========================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 复用 manual.cu 的 count kernel(只数每行中间项数,无 hash/原子)
extern __global__ void count_intermediates_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int *ub);

// ---- Stage 2(串行):把所有中间项写进全局 COO,key=(row<<32|col),val=a_ik*a_kj。
// 与 manual 的 expand 区别:这里【串行写】→ 每个 k 的贡献是一段【连续、col 有序】的段,
// 这样 merge_serial 才能把它当成一条有序列链来归并。
__global__ void expand_serial_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_rows, const int *row_off,
    unsigned long long *key, double *val)
{
    int i = blockIdx.x;            // 每行一个 block
    if (i >= A_rows) return;       // blockDim = 1,只有 thread 0
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int slot = row_off[i];
    for (int p = rs; p < re; p++) {
        int k = A_col_idx[p];
        double a_ik = A_val[p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];   // A 的第 k 行(列有序)
        for (int q = ks; q < ke; q++) {
            key[slot] = ((unsigned long long)i << 32) | (unsigned int)A_col_idx[q];
            val[slot] = a_ik * A_val[q];
            slot++;
        }
    }
}

// ---- Stage 3(串行 k-way merge + dedup + sum):thread 0 对本行的 num_k 条有序
// 列链做归并,输出每行去重+有序的 (col, val),并记 per-row nnz。
// shared mem = seg_cur[num_k] 紧接 seg_end[num_k]。
__global__ void merge_serial_kernel(
    const int *A_row_ptr, const int *A_col_idx,
    int A_rows, const int *row_off,
    const unsigned long long *key, const double *val,
    int *out_col, double *out_val, int *row_nnz)
{
    int i = blockIdx.x;            // 每行一个 block
    if (i >= A_rows) return;       // blockDim = 1,thread 0
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;           // 要归并的列链数
    int base = row_off[i];

    extern __shared__ __align__(8) int smem[];
    int *seg_cur = smem;           // [num_k] 每条链当前读位置
    int *seg_end = smem + num_k;   // [num_k] 每条链结束位置

    // 初始化 num_k 条链的边界:第 p 段 = A[k_p,:],k_p=A_col_idx[rs+p],长=nnz(A[k_p,:])
    int cum = base;
    for (int p = 0; p < num_k; p++) {
        int k = A_col_idx[rs + p];
        seg_cur[p] = cum;
        cum += A_row_ptr[k + 1] - A_row_ptr[k];
        seg_end[p] = cum;
    }
    // 此处 cum == row_off[i] + d_ub[i] == row_off[i+1]

    int out_idx = 0;
    while (true) {
        // 1) 扫所有链头部取最小 col
        int min_col = 0x7fffffff;
        for (int p = 0; p < num_k; p++) {
            int pos = seg_cur[p];
            if (pos < seg_end[p]) {
                int col = (int)(key[pos] & 0xffffffffu);
                if (col < min_col) min_col = col;
            }
        }
        if (min_col == 0x7fffffff) break;          // 所有链耗尽

        // 2) 求和所有头部 == min_col 的链,并前进
        double sum = 0.0f;
        for (int p = 0; p < num_k; p++) {
            int pos = seg_cur[p];
            if (pos < seg_end[p] && (int)(key[pos] & 0xffffffffu) == min_col) {
                sum += val[pos];
                seg_cur[p] = pos + 1;
            }
        }

        // 3) 输出(写到本行的 upper-bound slot,行内有间隙,之后 compact 压紧)
        out_col[base + out_idx] = min_col;
        out_val[base + out_idx] = sum;
        out_idx++;
    }
    row_nnz[i] = out_idx;
}

// ---- Stage 3b(compact):merge 的输出写在 row_off[i] 起的 upper-bound slot(行间有
// 间隙),按 C_row_ptr 压成连续 CSR。每行一块,块内线程跨步拷贝。
__global__ void compact_kernel(
    const int *row_off, const int *row_nnz, const int *C_row_ptr,
    const int *in_col, const double *in_val,
    int *out_col, double *out_val, int A_rows)
{
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int src = row_off[i];
    int dst = C_row_ptr[i];
    int n = row_nnz[i];
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        out_col[dst + t] = in_col[src + t];
        out_val[dst + t] = in_val[src + t];
    }
}

// ========== Host ==========

void spgemm_self_product_merge(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    dbg("[merge] start\n");

    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(double);
    size_t A_total_size = ALIGN8(A_row_ptr_size + A_col_idx_size) + A_val_size;

    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice));
    dbg("[merge] h2d\n");

    char *dA_base = (char*)dA_buffer;
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    double *dA_val = (double*)(dA_base + ALIGN8(A_row_ptr_size + A_col_idx_size));

    const int block = 256;

    // host 扫 pinned A_row_ptr 得 max_row_nnz → merge kernel 的 shared mem(seg_cur+seg_end)
    const int *h_row_ptr = (const int*)A_buffer;
    int max_row_nnz = 0;
    for (int i = 0; i < A_rows; i++) {
        int nn = h_row_ptr[i + 1] - h_row_ptr[i];
        if (nn > max_row_nnz) max_row_nnz = nn;
    }
    size_t merge_smem = (size_t)max_row_nnz * 2 * sizeof(int);
    // 动态 shared > 48KB 时需显式 opt-in(H100 上限 ~228KB);本数据集 max_row_nnz 很小,留防御
    if (merge_smem > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(merge_serial_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)merge_smem));
    }

    // ---- Stage 1: 每行中间项数(复用 count kernel)----
    dbg("merge: count begin\n");
    int *d_ub;
    CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    {
        int grid = (A_rows + block - 1) / block;
        count_intermediates_kernel<<<grid, block>>>(
            dA_row_ptr, dA_col_idx, A_rows, d_ub);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[merge] count\n");

    // ---- Stage 1b: 前缀和得每行写偏移; off[A_rows] = 总中间项数 ----
    int *d_off;
    CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total_ub;
    CHECK_CUDA(cudaMemcpy(&total_ub, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[merge] scan\n");

    // ---- Stage 2: 串行展开(每 k 连续有序段)写全局 COO(key, val)----
    unsigned long long *d_key;
    double *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total_ub * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total_ub * sizeof(double)));
    dbg("merge: expand_serial begin (%d intermediates)\n", total_ub);
    expand_serial_kernel<<<A_rows, 1>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows, d_off, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[merge] expand\n");

    // ---- Stage 3: 串行 k-way merge + dedup + sum → (out_col, out_val), row_nnz ----
    int *d_out_col;
    double *d_out_val;
    int *d_row_nnz;
    CHECK_CUDA(cudaMalloc(&d_out_col, (size_t)total_ub * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out_val, (size_t)total_ub * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
    dbg("merge: merge_serial begin\n");
    merge_serial_kernel<<<A_rows, 1, merge_smem>>>(
        dA_row_ptr, dA_col_idx, A_rows, d_off, d_key, d_val,
        d_out_col, d_out_val, d_row_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[merge] merge\n");
    cudaFree(d_key);
    cudaFree(d_val);

    // ---- Stage 3b: scan row_nnz → C_row_ptr; 读 C_nnz ----
    int *dC_row_ptr;
    CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                           thrust::device_ptr<int>(d_row_nnz + A_rows),
                           thrust::device_ptr<int>(dC_row_ptr + 1));
    int C_nnz_result;
    CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_row_ptr + A_rows,
                         sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[merge] final\n");

    // ---- Stage 3c: compact 到连续 CSR ----
    int *dC_col_idx;
    double *dC_val;
    CHECK_CUDA(cudaMalloc(&dC_col_idx, (size_t)C_nnz_result * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dC_val, (size_t)C_nnz_result * sizeof(double)));
    compact_kernel<<<A_rows, block>>>(
        d_off, d_row_nnz, dC_row_ptr, d_out_col, d_out_val, dC_col_idx, dC_val, A_rows);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_out_col);
    cudaFree(d_out_val);
    cudaFree(d_row_nnz);
    dbg("[merge] compact\n");

    // ---- 打包成单块 + D2H(与 manual 相同)----
    size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t C_col_idx_size = (size_t)C_nnz_result * sizeof(int);
    size_t C_val_size = (size_t)C_nnz_result * sizeof(double);
    size_t C_row_ptr_aligned = ALIGN8(C_row_ptr_size);
    size_t C_col_idx_aligned = ALIGN8(C_col_idx_size);
    size_t C_total_size = C_row_ptr_aligned + C_col_idx_aligned + C_val_size;

    void *dC_buffer;
    CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));
    char *dC_base = (char*)dC_buffer;
    CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned, dC_col_idx,
                          C_col_idx_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned + C_col_idx_aligned,
                          dC_val, C_val_size, cudaMemcpyDeviceToDevice));
    dbg("[merge] pack\n");

    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total_size));
    CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, cudaMemcpyDeviceToHost));
    dbg("[merge] d2h\n");

    *C_buffer_out = C_buffer;
    *C_rows = A_rows;
    *C_cols = A_cols;
    *C_nnz = C_nnz_result;

    cudaFree(dA_buffer);
    cudaFree(d_ub);
    cudaFree(d_off);
    cudaFree(dC_row_ptr);
    cudaFree(dC_col_idx);
    cudaFree(dC_val);
    cudaFree(dC_buffer);
}

// ==========================================================================
//  v2:并行 k-way merge(warp-per-row 协作,【直接读 A,无 expand 阶段】)
// --------------------------------------------------------------------------
//  与 serial v1 的差异:
//   ① 无 expand —— merge 直接读 A 的 CSR 行(段 p = A[k_p,:]),
//      砍掉「写 COO 一遍 + 读回一遍」的来回 traffic(Tier 0 融合)。
//   ② 行内并行 —— 每行一个 block(1 warp=32 线程),每步 32 路并发取各链头、
//      warp-shuffle min/sum 归约 → 隐藏散列全局读延迟,治重行 straggler(Tier 2)。
//  正确性:与 serial 展开的是同一批中间项(行 i 的 k=A_col_idx[p]、weight=A_val[p]、
//      列=A[k_p:]、贡献=weight*A_val[q]);逐次输出全局最小列并合并同列 → 等价 sort+reduce。
// ==========================================================================
__global__ void merge_warp_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_rows, const int *row_off,
    int *out_col, double *out_val, int *row_nnz)
{
    int i = blockIdx.x;                 // 每行一个 block
    if (i >= A_rows) return;
    int lane = threadIdx.x;             // blockDim = 32(1 warp)

    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;                // 要归并的列链数
    int base = row_off[i];

    extern __shared__ __align__(8) int smem[];
    int   *seg_ptr = smem;                  // [num_k] 每条链当前读位置(指向 A)
    int   *seg_end = smem + num_k;          // [num_k] 每条链结束位置
    double *weight  = (double*)(smem + 2*num_k); // [num_k] a_{i,k_p}

    // (装载)每个 lane 认领 ⌈num_k/32⌉ 条链的描述符:k_p=A_col_idx[rs+p]
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        seg_ptr[p] = A_row_ptr[k];
        seg_end[p] = A_row_ptr[k + 1];
        weight[p]  = A_val[rs + p];
    }
    __syncwarp();

    int out_idx = 0;
    while (true) {
        // (1) 每个 lane 在自己持有的链里找局部最小头列(直接读 A,32 路并发隐藏延迟)
        int mymin = 0x7fffffff;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p]) {
                int col = A_col_idx[pos];
                if (col < mymin) mymin = col;
            }
        }
        // (2) warp shuffle min-归约 → wmin(全 warp 一致)
        int wmin = mymin;
        for (int off = 16; off > 0; off >>= 1) {
            int v = __shfl_xor_sync(0xffffffff, wmin, off);
            if (v < wmin) wmin = v;
        }
        if (wmin == 0x7fffffff) break;       // 所有链耗尽

        // (3) 每个 lane:头部==wmin 的链,累加 weight*A_val 并前进
        double mysum = 0.0f;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p] && A_col_idx[pos] == wmin) {
                mysum += weight[p] * A_val[pos];
                seg_ptr[p] = pos + 1;
            }
        }
        // (4) warp shuffle sum-归约 → 全 warp 都拿到总和
        for (int off = 16; off > 0; off >>= 1)
            mysum += __shfl_xor_sync(0xffffffff, mysum, off);

        // (5) lane0 输出(out_idx 全 warp 同步递增)
        if (lane == 0) {
            out_col[base + out_idx] = wmin;
            out_val[base + out_idx] = mysum;
        }
        out_idx++;
    }
    if (lane == 0) row_nnz[i] = out_idx;
}

// ========== v2 Host ==========

void spgemm_self_product_merge2(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    dbg("[mrg2] start\n");

    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(double);
    size_t A_total_size = ALIGN8(A_row_ptr_size + A_col_idx_size) + A_val_size;

    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice));
    dbg("[mrg2] h2d\n");

    char *dA_base = (char*)dA_buffer;
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    double *dA_val = (double*)(dA_base + ALIGN8(A_row_ptr_size + A_col_idx_size));

    const int block = 256;

    // host 扫 pinned A_row_ptr 得 max_row_nnz → merge_warp shared(seg_ptr+seg_end+weight)
    const int *h_row_ptr = (const int*)A_buffer;
    int max_row_nnz = 0;
    for (int i = 0; i < A_rows; i++) {
        int nn = h_row_ptr[i + 1] - h_row_ptr[i];
        if (nn > max_row_nnz) max_row_nnz = nn;
    }
    size_t merge_smem = (size_t)max_row_nnz * 3 * sizeof(int);
    if (merge_smem > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(merge_warp_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)merge_smem));
    }

    // ---- Stage 1: 每行中间项数(复用 count kernel)----
    dbg("merge2: count begin\n");
    int *d_ub;
    CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    {
        int grid = (A_rows + block - 1) / block;
        count_intermediates_kernel<<<grid, block>>>(
            dA_row_ptr, dA_col_idx, A_rows, d_ub);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[mrg2] count\n");

    // ---- Stage 1b: 前缀和得每行写偏移; off[A_rows] = 总中间项数(=输出上界)----
    int *d_off;
    CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total_ub;
    CHECK_CUDA(cudaMemcpy(&total_ub, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[mrg2] scan\n");

    // ---- Stage 2: 并行 k-way merge(直接读 A)→ out_col/out_val + row_nnz ----
    int *d_out_col;
    double *d_out_val;
    int *d_row_nnz;
    CHECK_CUDA(cudaMalloc(&d_out_col, (size_t)total_ub * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out_val, (size_t)total_ub * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
    dbg("merge2: merge_warp begin (%d intermediates)\n", total_ub);
    merge_warp_kernel<<<A_rows, 32, merge_smem>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows, d_off,
        d_out_col, d_out_val, d_row_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[mrg2] merge\n");

    // ---- Stage 2b: scan row_nnz → C_row_ptr; 读 C_nnz ----
    int *dC_row_ptr;
    CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                           thrust::device_ptr<int>(d_row_nnz + A_rows),
                           thrust::device_ptr<int>(dC_row_ptr + 1));
    int C_nnz_result;
    CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_row_ptr + A_rows,
                         sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[mrg2] final\n");

    // ---- Stage 2c: compact 到连续 CSR(复用 compact_kernel)----
    int *dC_col_idx;
    double *dC_val;
    CHECK_CUDA(cudaMalloc(&dC_col_idx, (size_t)C_nnz_result * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dC_val, (size_t)C_nnz_result * sizeof(double)));
    compact_kernel<<<A_rows, block>>>(
        d_off, d_row_nnz, dC_row_ptr, d_out_col, d_out_val, dC_col_idx, dC_val, A_rows);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_out_col);
    cudaFree(d_out_val);
    cudaFree(d_row_nnz);
    dbg("[mrg2] compact\n");

    // ---- 打包成单块 + D2H ----
    size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t C_col_idx_size = (size_t)C_nnz_result * sizeof(int);
    size_t C_val_size = (size_t)C_nnz_result * sizeof(double);
    size_t C_row_ptr_aligned = ALIGN8(C_row_ptr_size);
    size_t C_col_idx_aligned = ALIGN8(C_col_idx_size);
    size_t C_total_size = C_row_ptr_aligned + C_col_idx_aligned + C_val_size;

    void *dC_buffer;
    CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));
    char *dC_base = (char*)dC_buffer;
    CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned, dC_col_idx,
                          C_col_idx_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned + C_col_idx_aligned,
                          dC_val, C_val_size, cudaMemcpyDeviceToDevice));
    dbg("[mrg2] pack\n");

    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total_size));
    CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, cudaMemcpyDeviceToHost));
    dbg("[mrg2] d2h\n");

    *C_buffer_out = C_buffer;
    *C_rows = A_rows;
    *C_cols = A_cols;
    *C_nnz = C_nnz_result;

    cudaFree(dA_buffer);
    cudaFree(d_ub);
    cudaFree(d_off);
    cudaFree(dC_row_ptr);
    cudaFree(dC_col_idx);
    cudaFree(dC_val);
    cudaFree(dC_buffer);
}

// ==========================================================================
//  merge3:分块 merge(列域分桶 col-bucketing)
// --------------------------------------------------------------------------
//  每行把列域 [0, A_cols) 分成 K 个桶,每个 (row, bucket) 一个 block(1 warp),
//  warp-merge 该桶内各链的子区间(用 lower_bound 二分定位 [blo,bhi))。
//  · 桶内天然列有序 → 各桶按序拼接 = 行内列有序 CSR。
//  · 先 count 每桶 distinct 列 → scan 得桶偏移 + C_nnz(精确,无需 compact)。
//  · 目的:把单行(尤其重行)的串行链切成 K 段并行 → 治 straggler。
//    全行均匀分 K 桶(轻行会有 K 块开销,留待自适应优化)。
// ==========================================================================

// 二分:返回 [lo,hi) 内第一个 arr[idx] >= val 的下标(都 < val 则返回 hi)
__device__ __forceinline__ int dev_lower_bound(const int *arr, int lo, int hi, int val) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// (row,bucket) 一块:count 该桶内 distinct 列数(merge-count,不写值)
__global__ void bucket_count_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int A_cols,
    int K, int *bucket_nnz)   // [A_rows * K],行主序 [i*K + b]
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    long long n = A_cols;
    int blo = (int)(b * n / K);
    int bhi = (int)((b + 1) * n / K);
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;
    extern __shared__ __align__(8) int smem[];
    int *seg_ptr = smem;            // [num_k]
    int *seg_end = smem + num_k;    // [num_k]
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        seg_ptr[p] = dev_lower_bound(A_col_idx, ks, ke, blo);
        seg_end[p] = dev_lower_bound(A_col_idx, ks, ke, bhi);
    }
    __syncwarp();
    int cnt = 0;
    while (true) {
        int mymin = 0x7fffffff;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p]) { int col = A_col_idx[pos]; if (col < mymin) mymin = col; }
        }
        int wmin = mymin;
        for (int off = 16; off > 0; off >>= 1) { int v = __shfl_xor_sync(0xffffffff, wmin, off); if (v < wmin) wmin = v; }
        if (wmin == 0x7fffffff) break;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p] && A_col_idx[pos] == wmin) seg_ptr[p] = pos + 1;
        }
        cnt++;
    }
    if (lane == 0) bucket_nnz[i * K + b] = cnt;
}

// 行内桶偏移:bucket_off[i*K+b] = 行内 exclusive scan;row_nnz[i] = 行总和
__global__ void bucket_scan_kernel(int A_rows, int K, const int *bucket_nnz,
                                   int *bucket_off, int *row_nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    int acc = 0;
    for (int b = 0; b < K; b++) {
        bucket_off[i * K + b] = acc;
        acc += bucket_nnz[i * K + b];
    }
    row_nnz[i] = acc;
}

// (row,bucket) 一块:warp-merge 该桶子区间,写 (col, val) 到全局偏移
__global__ void bucket_merge_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_rows, int A_cols, int K, const int *C_row_ptr, const int *bucket_off,
    int *out_col, double *out_val)
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    long long n = A_cols;
    int blo = (int)(b * n / K);
    int bhi = (int)((b + 1) * n / K);
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;
    extern __shared__ __align__(8) int smem[];
    int   *seg_ptr = smem;
    int   *seg_end = smem + num_k;
    double *weight  = (double*)(smem + 2 * num_k);
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        seg_ptr[p] = dev_lower_bound(A_col_idx, ks, ke, blo);
        seg_end[p] = dev_lower_bound(A_col_idx, ks, ke, bhi);
        weight[p]  = A_val[rs + p];
    }
    __syncwarp();
    int base = C_row_ptr[i] + bucket_off[i * K + b];
    int out_idx = 0;
    while (true) {
        int mymin = 0x7fffffff;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p]) { int col = A_col_idx[pos]; if (col < mymin) mymin = col; }
        }
        int wmin = mymin;
        for (int off = 16; off > 0; off >>= 1) { int v = __shfl_xor_sync(0xffffffff, wmin, off); if (v < wmin) wmin = v; }
        if (wmin == 0x7fffffff) break;
        double mysum = 0.0f;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p] && A_col_idx[pos] == wmin) {
                mysum += weight[p] * A_val[pos];
                seg_ptr[p] = pos + 1;
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            mysum += __shfl_xor_sync(0xffffffff, mysum, off);
        if (lane == 0) { out_col[base + out_idx] = wmin; out_val[base + out_idx] = mysum; }
        out_idx++;
    }
}

// ==========================================================================
//  merge3 + flop_ub sizing(设计点② 收尾 / C):
//  原版 bucket_count 为拿【精确】per-bucket distinct,把整遍 warp-merge 跑了一次(只数不写)
//  → 占 merge3 ~37%(bcsstk30 count 6.2ms vs merge 10.4ms)。此处改用【flop_ub 上界】定桶区,
//  省 count pass 的 merge 迭代(仅 lower_bound 定位 + 求和);merge 写到 gapped flop 区 + 记真实数;
//  再 scan 真实数 + compact 压紧。代价:gapped buffer(=Σflop,大阵 ~2GB,H100 可容)+ 一个 compact pass。
//  host 端 getenv("MRG3_FLOP_UB") 门控,默认关(走精确 count 原 path,便于 A/B)。
// ==========================================================================

// (C-1) 每 (row,bucket) 算 flop_ub = Σ_k (seg_end-seg_ptr)(lower_bound 定 [blo,bhi),无 merge 迭代)。
//   这是该桶 distinct 的确定性上界(≥ distinct)。比 bucket_count 的 merge-iter count 快 ~30×。
__global__ void bucket_flop_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int A_cols,
    int K, int *bucket_flop)   // [A_rows * K]
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    long long n = A_cols;
    int blo = (int)(b * n / K);
    int bhi = (int)((b + 1) * n / K);
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;
    extern __shared__ __align__(8) int smem[];
    int *seg_ptr = smem;            // [num_k]
    int *seg_end = smem + num_k;    // [num_k]
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        seg_ptr[p] = dev_lower_bound(A_col_idx, ks, ke, blo);
        seg_end[p] = dev_lower_bound(A_col_idx, ks, ke, bhi);
    }
    __syncwarp();
    int s = 0;
    for (int p = lane; p < num_k; p += 32) s += seg_end[p] - seg_ptr[p];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if (lane == 0) bucket_flop[i * K + b] = s;
}

// (C-2) 每 (row,bucket) warp-merge 写 (col,val) 到【gapped flop 区】(base = row_off + bucket_flop_off),
//   并记真实 distinct 数 bucket_real_nnz(= 该桶 merge 产出项数)。逻辑同 bucket_merge_kernel,
//   仅 base 改 flop-off、末尾多写一个 real 计数。
__global__ void bucket_merge_flop_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_rows, int A_cols, int K, const int *row_off, const int *bucket_flop_off,
    int *out_col, double *out_val, int *bucket_real_nnz)   // [A_rows*K]
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    long long n = A_cols;
    int blo = (int)(b * n / K);
    int bhi = (int)((b + 1) * n / K);
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;
    extern __shared__ __align__(8) int smem[];
    int   *seg_ptr = smem;
    int   *seg_end = smem + num_k;
    double *weight  = (double*)(smem + 2 * num_k);
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int ks = A_row_ptr[k], ke = A_row_ptr[k + 1];
        seg_ptr[p] = dev_lower_bound(A_col_idx, ks, ke, blo);
        seg_end[p] = dev_lower_bound(A_col_idx, ks, ke, bhi);
        weight[p]  = A_val[rs + p];
    }
    __syncwarp();
    int base = row_off[i] + bucket_flop_off[i * K + b];
    int out_idx = 0;
    while (true) {
        int mymin = 0x7fffffff;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p]) { int col = A_col_idx[pos]; if (col < mymin) mymin = col; }
        }
        int wmin = mymin;
        for (int off = 16; off > 0; off >>= 1) { int v = __shfl_xor_sync(0xffffffff, wmin, off); if (v < wmin) wmin = v; }
        if (wmin == 0x7fffffff) break;
        double mysum = 0.0f;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p] && A_col_idx[pos] == wmin) {
                mysum += weight[p] * A_val[pos];
                seg_ptr[p] = pos + 1;
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            mysum += __shfl_xor_sync(0xffffffff, mysum, off);
        if (lane == 0) { out_col[base + out_idx] = wmin; out_val[base + out_idx] = mysum; }
        out_idx++;
    }
    if (lane == 0) bucket_real_nnz[i * K + b] = out_idx;
}

// (C-3) compact:把每 (row,bucket) 的真实项从 gapped flop 区拷到精确 CSR(base = C_row_ptr + bucket_off_exact)。
__global__ void bucket_compact_kernel(
    int A_rows, int K, const int *row_off, const int *bucket_flop_off,
    const int *bucket_real_nnz, const int *C_row_ptr, const int *bucket_off_exact,
    const int *in_col, const double *in_val, int *out_col, double *out_val)
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    int src = row_off[i] + bucket_flop_off[i * K + b];
    int dst = C_row_ptr[i] + bucket_off_exact[i * K + b];
    int n = bucket_real_nnz[i * K + b];
    for (int t = lane; t < n; t += 32) {
        out_col[dst + t] = in_col[src + t];
        out_val[dst + t] = in_val[src + t];
    }
}

// ========== merge3 Host ==========
void spgemm_self_product_merge3(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    const int K = 5;   // 每行列域分桶数(可调;大→并行度高、开销大)
    dbg("[mrg3] start (K=%d)\n", K);
    HashProf prof("mrg3-prof");

    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(double);
    size_t A_total_size = ALIGN8(A_row_ptr_size + A_col_idx_size) + A_val_size;
    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    prof("h2d", [&]{ CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice)); });
    char *dA_base = (char*)dA_buffer;
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    double *dA_val = (double*)(dA_base + ALIGN8(A_row_ptr_size + A_col_idx_size));

    const int block = 256;
    const int *h_row_ptr = (const int*)A_buffer;
    int max_row_nnz = 0;
    for (int i = 0; i < A_rows; i++) {
        int nn = h_row_ptr[i + 1] - h_row_ptr[i];
        if (nn > max_row_nnz) max_row_nnz = nn;
    }
    //根据最长的可能子链来分配空间
    size_t smem_count = (size_t)max_row_nnz * 2 * sizeof(int);
    size_t smem_merge = (size_t)max_row_nnz * (2 * sizeof(int) + sizeof(double));   // seg_ptr+seg_end[int]+weight[double]
    if (smem_merge > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(bucket_merge_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_merge));
        CHECK_CUDA(cudaFuncSetAttribute(bucket_count_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_count));
    }

    dim3 grid(A_rows, K), block32(32);

    // 门控:flop_ub sizing(省 count pass ~37%)vs 精确 count(原版)。默认 flop_ub(净赢 24-32%、无回归);MRG3_FLOP_UB=0 关回精确 count。
    static int g_flop = -1;
    if (g_flop < 0) { const char *e = getenv("MRG3_FLOP_UB"); g_flop = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }
    if (g_flop && smem_merge > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(bucket_flop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_count));
        CHECK_CUDA(cudaFuncSetAttribute(bucket_merge_flop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_merge));
    }

    int *dC_row_ptr; CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    int C_nnz_result;
    void *dC_buffer = nullptr;
    size_t C_total_size = 0;

    if (g_flop) {
        // ============ flop_ub path:省 count 的 merge 迭代,代价 = gapped buffer + compact ============
        int *d_bflop, *d_bflop_off, *d_row_flop, *d_row_off;
        CHECK_CUDA(cudaMalloc(&d_bflop, (size_t)A_rows * K * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_bflop_off, (size_t)A_rows * K * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_row_flop, A_rows * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_row_off, (A_rows + 1) * sizeof(int)));
        int total_flop;
        prof("flop", [&]{   // (C-1) 每 (row,bucket) flop_ub(lower_bound+sum,无 merge 迭代)
            bucket_flop_kernel<<<grid, block32, smem_count>>>(dA_row_ptr, dA_col_idx, A_rows, A_cols, K, d_bflop);
            CHECK_CUDA(cudaGetLastError());
        });
        prof("fscan", [&]{  // 行内 scan(flop)→ bucket_flop_off + row_flop;全局 scan → row_off(gapped) + total_flop
            bucket_scan_kernel<<<(A_rows + block - 1) / block, block>>>(A_rows, K, d_bflop, d_bflop_off, d_row_flop);
            CHECK_CUDA(cudaMemset(d_row_off, 0, sizeof(int)));
            thrust::inclusive_scan(thrust::device_ptr<int>(d_row_flop), thrust::device_ptr<int>(d_row_flop + A_rows),
                                   thrust::device_ptr<int>(d_row_off + 1));
            CHECK_CUDA(cudaMemcpy(&total_flop, d_row_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
        });
        // gapped out buffer [total_flop](=Σflop,大阵 ~2GB,H100 可容)
        int *d_gcol; double *d_gval; int *d_breal;
        CHECK_CUDA(cudaMalloc(&d_gcol, (size_t)total_flop * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_gval, (size_t)total_flop * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_breal, (size_t)A_rows * K * sizeof(int)));
        prof("merge", [&]{   // (C-2) warp-merge 写 gapped 区 + 记真实数 bucket_real_nnz
            bucket_merge_flop_kernel<<<grid, block32, smem_merge>>>(
                dA_row_ptr, dA_col_idx, dA_val, A_rows, A_cols, K, d_row_off, d_bflop_off,
                d_gcol, d_gval, d_breal);
            CHECK_CUDA(cudaGetLastError());
        });
        int *d_boff_ex, *d_row_nnz;
        CHECK_CUDA(cudaMalloc(&d_boff_ex, (size_t)A_rows * K * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
        prof("rscan", [&]{  // scan(真实数)→ exact bucket_off + row_nnz → C_row_ptr + C_nnz
            bucket_scan_kernel<<<(A_rows + block - 1) / block, block>>>(A_rows, K, d_breal, d_boff_ex, d_row_nnz);
            CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));
            thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz), thrust::device_ptr<int>(d_row_nnz + A_rows),
                                   thrust::device_ptr<int>(dC_row_ptr + 1));
            CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_row_ptr + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
        });
        // alloc 精确 dC=[row_ptr|col|val],compact gapped → 精确
        size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
        size_t C_rp_al = ALIGN8(C_row_ptr_size);
        size_t C_ci_al = ALIGN8((size_t)C_nnz_result * sizeof(int));
        C_total_size = C_rp_al + C_ci_al + (size_t)C_nnz_result * sizeof(double);
        CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));
        char *cb = (char*)dC_buffer;
        int   *dC_ci = (int*)(cb + C_rp_al);
        double *dC_val = (double*)(cb + C_rp_al + C_ci_al);
        prof("compact", [&]{
            CHECK_CUDA(cudaMemcpy(cb, dC_row_ptr, C_row_ptr_size, cudaMemcpyDeviceToDevice));
            bucket_compact_kernel<<<grid, block32>>>(A_rows, K, d_row_off, d_bflop_off, d_breal, dC_row_ptr, d_boff_ex,
                                                      d_gcol, d_gval, dC_ci, dC_val);
            CHECK_CUDA(cudaGetLastError());
        });
        cudaFree(d_bflop); cudaFree(d_bflop_off); cudaFree(d_row_flop); cudaFree(d_row_off);
        cudaFree(d_gcol); cudaFree(d_gval); cudaFree(d_breal); cudaFree(d_boff_ex); cudaFree(d_row_nnz);
    } else {
        // ============ exact-count path(原版)============
        // ---- Stage 1: 每桶 count distinct 列 ----
        int *d_bucket_nnz;
        CHECK_CUDA(cudaMalloc(&d_bucket_nnz, (size_t)A_rows * K * sizeof(int)));
        prof("count", [&]{
            bucket_count_kernel<<<grid, block32, smem_count>>>(
                dA_row_ptr, dA_col_idx, A_rows, A_cols, K, d_bucket_nnz);
            CHECK_CUDA(cudaGetLastError());
        });
        // ---- Stage 2: 行内桶偏移 + row_nnz + C_row_ptr(D2H 纳入 scan 块对齐 Ocean)----
        int *d_bucket_off, *d_row_nnz;
        CHECK_CUDA(cudaMalloc(&d_bucket_off, (size_t)A_rows * K * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
        prof("scan", [&]{
            bucket_scan_kernel<<<(A_rows + block - 1) / block, block>>>(
                A_rows, K, d_bucket_nnz, d_bucket_off, d_row_nnz);
            CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));
            thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                                   thrust::device_ptr<int>(d_row_nnz + A_rows),
                                   thrust::device_ptr<int>(dC_row_ptr + 1));
            CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_row_ptr + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
        });
        // ---- Stage 3: 每桶 merge 写值 + 连续输出 dC_buffer=[row_ptr|col|val] ----
        size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
        size_t C_col_idx_aligned = ALIGN8((size_t)C_nnz_result * sizeof(int));
        size_t C_rp_al = ALIGN8(C_row_ptr_size);
        C_total_size = C_rp_al + C_col_idx_aligned + (size_t)C_nnz_result * sizeof(double);
        CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));
        char *dC_base = (char*)dC_buffer;
        int   *dC_col_idx = (int*)(dC_base + C_rp_al);
        double *dC_val     = (double*)(dC_base + C_rp_al + C_col_idx_aligned);
        prof("merge", [&]{
            CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size, cudaMemcpyDeviceToDevice));
            bucket_merge_kernel<<<grid, block32, smem_merge>>>(
                dA_row_ptr, dA_col_idx, dA_val, A_rows, A_cols, K, dC_row_ptr, d_bucket_off,
                dC_col_idx, dC_val);
            CHECK_CUDA(cudaGetLastError());
        });
        cudaFree(d_bucket_nnz); cudaFree(d_bucket_off); cudaFree(d_row_nnz);
    }


    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total_size));
    prof("d2h", [&]{ CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, cudaMemcpyDeviceToHost)); });

    *C_buffer_out = C_buffer;
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = C_nnz_result;

    cudaFree(dA_buffer); cudaFree(dC_row_ptr); cudaFree(dC_buffer);   // 各 path 的临时数组已在分支内 free;dC_ci/dC_val 是 dC_buffer 别名
}
