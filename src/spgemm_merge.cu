#include "spgemm.h"
#include "hash_prof.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Serial k-way Merge 版自乘 C=A·A:对行内有序列链做 merge,省掉 sort+reduce。

// DEFENSE(2026-08-25,GPU wedge 三连教训):warp-merge 主循环硬上界。
// 每次迭代至少消费 1 个中间积 ⇒ 合法迭代数 ≤ 行 flop,远低于此 cap;
// 不变量破坏时输出错误而非挂死 —— 保证 kernel 必然终止(同 att_tiered 的 ATT_LOOP_CAP 手法)。
#define MRG3_LOOP_CAP (1 << 28)

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

// Stage 2(串行):中间项写全局 COO,key=(row<<32|col);每个 k 是一段连续 col-有序段供 merge 归并。
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

// Stage 3(串行 k-way merge + dedup + sum):thread 0 归并 num_k 条有序链 → 去重有序 (col,val) + per-row nnz。
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

// Stage 3b(compact):merge 输出在 upper-bound slot(行间有间隙),按 C_row_ptr 压成连续 CSR。
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

// Host

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

    // Stage 1: 每行中间项数(复用 count kernel)
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

    // Stage 1b: 前缀和得每行写偏移;off[A_rows]=总中间项数
    int *d_off;
    CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total_ub;
    CHECK_CUDA(cudaMemcpy(&total_ub, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[merge] scan\n");

    // Stage 2: 串行展开(每 k 连续有序段)写全局 COO(key,val)
    unsigned long long *d_key;
    double *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total_ub * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total_ub * sizeof(double)));
    dbg("merge: expand_serial begin (%d intermediates)\n", total_ub);
    expand_serial_kernel<<<A_rows, 1>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows, d_off, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[merge] expand\n");

    // Stage 3: 串行 k-way merge + dedup + sum → (out_col,out_val),row_nnz
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

    // Stage 3b: scan row_nnz → C_row_ptr;读 C_nnz
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

    // Stage 3c: compact 到连续 CSR
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

    // 打包成单块 + D2H(与 manual 相同)
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

// v2:并行 k-way merge(warp-per-row 协作,直接读 A 无 expand;warp-shuffle min/sum 归约治重行 straggler)。
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

// v2 Host

void spgemm_self_product_merge2(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    dbg("[mrg2] start\n");
    HashProf prof("mrg2-prof");

    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(double);
    size_t A_total_size = ALIGN8(A_row_ptr_size + A_col_idx_size) + A_val_size;

    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    prof("h2d", [&]{ CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice)); });
    dbg("[mrg2] h2d\n");

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
    size_t merge_smem = (size_t)max_row_nnz * (2 * sizeof(int) + sizeof(double));   // seg_ptr+seg_end[int]+weight[double]
    if (merge_smem > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(merge_warp_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)merge_smem));
    }

    // Stage 1: 每行中间项数(复用 count kernel)
    int *d_ub;
    CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    prof("count", [&]{
        int grid = (A_rows + block - 1) / block;
        count_intermediates_kernel<<<grid, block>>>(
            dA_row_ptr, dA_col_idx, A_rows, d_ub);
    });
    dbg("[mrg2] count\n");

    // Stage 1b: 前缀和得每行写偏移;off[A_rows]=总中间项数(=输出上界)
    int *d_off;
    CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    int total_ub;
    prof("scan", [&]{
        thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                               thrust::device_ptr<int>(d_ub + A_rows),
                               thrust::device_ptr<int>(d_off + 1));
        CHECK_CUDA(cudaMemcpy(&total_ub, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    });
    dbg("[mrg2] scan\n");

    // Stage 2: 并行 k-way merge(1 warp/row)→ out_col/out_val + row_nnz
    int *d_out_col;
    double *d_out_val;
    int *d_row_nnz;
    CHECK_CUDA(cudaMalloc(&d_out_col, (size_t)total_ub * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_out_val, (size_t)total_ub * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
    prof("merge", [&]{
        merge_warp_kernel<<<A_rows, 32, merge_smem>>>(
            dA_row_ptr, dA_col_idx, dA_val, A_rows, d_off,
            d_out_col, d_out_val, d_row_nnz);
    });
    dbg("[mrg2] merge\n");

    // Stage 2b: scan row_nnz → C_row_ptr;读 C_nnz
    int *dC_row_ptr;
    CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));
    int C_nnz_result;
    prof("final", [&]{
        thrust::inclusive_scan(thrust::device_ptr<int>(d_row_nnz),
                               thrust::device_ptr<int>(d_row_nnz + A_rows),
                               thrust::device_ptr<int>(dC_row_ptr + 1));
        CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_row_ptr + A_rows,
                             sizeof(int), cudaMemcpyDeviceToHost));
    });
    dbg("[mrg2] final\n");

    // Stage 2c: compact 到连续 CSR(复用 compact_kernel)
    int *dC_col_idx;
    double *dC_val;
    CHECK_CUDA(cudaMalloc(&dC_col_idx, (size_t)C_nnz_result * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dC_val, (size_t)C_nnz_result * sizeof(double)));
    prof("compact", [&]{
        compact_kernel<<<A_rows, block>>>(
            d_off, d_row_nnz, dC_row_ptr, d_out_col, d_out_val, dC_col_idx, dC_val, A_rows);
    });
    cudaFree(d_out_col);
    cudaFree(d_out_val);
    cudaFree(d_row_nnz);
    dbg("[mrg2] compact\n");

    // 打包成单块 + D2H(pack D2D 归入 d2h,不计 compute)
    size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t C_col_idx_size = (size_t)C_nnz_result * sizeof(int);
    size_t C_val_size = (size_t)C_nnz_result * sizeof(double);
    size_t C_row_ptr_aligned = ALIGN8(C_row_ptr_size);
    size_t C_col_idx_aligned = ALIGN8(C_col_idx_size);
    size_t C_total_size = C_row_ptr_aligned + C_col_idx_aligned + C_val_size;

    void *dC_buffer;
    CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));
    char *dC_base = (char*)dC_buffer;
    void *C_buffer = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total_size));
    prof("d2h", [&]{
        CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned, dC_col_idx,
                              C_col_idx_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned + C_col_idx_aligned,
                              dC_val, C_val_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, cudaMemcpyDeviceToHost));
    });
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

// merge3:列域分 K 桶 × (row,bucket) 一块 warp-merge,把重行串行链切 K 段并行治 straggler。

// 二分:返回 [lo,hi) 内第一个 arr[idx] >= val 的下标(都 < val 则返回 hi)
__device__ __forceinline__ int dev_lower_bound(const int *arr, int lo, int hi, int val) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// 动态负载均衡 v2(docs/66):每行把 K 个桶均分在【乘积列实际跨度】[min,max] 上,非 [0,n)。
// min/max = 各输入 k 的 B 行首/末列(行有序)→ O(num_k) 顺序负载、零二分。宽带阵等宽桶宽 ≫
// bw 时全部工作落一桶(32000/5=6400 ≫ 2048),K 路切分失效 —— 本 kernel 治此病。
// v1(值域二分求精确等 flop 切点)实测 +374% 否决:rank 评估 O(num_k·log(avgB)) × log(n) 轮,
// 成本 ≈ merge 本身。v2 近似(跨度内均匀假设)对带状/聚集行已是 5× 均衡,开销 <1%。
// 轻行(total < dyn_min)回退等宽 [0,n) = 旧行为逐位一致。
__global__ void bucket_bnd_kernel(
    const int *A_row_ptr, const int *A_col_idx,
    const int *B_row_ptr, const int *B_col_idx,
    int A_rows, int A_cols, int K, long long dyn_min,
    int *bnd,                      // [A_rows * (K+1)]
    int *d_max_span = nullptr)     // docs/66 §7:host D2H 一个 int → dense 桶宽上限精确已知
{
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    int *rb = bnd + (size_t)i * (K + 1);
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;
    if (num_k <= 0) {   // 空行:任意合法边界(全空桶)
        if (lane == 0) { rb[0] = 0; rb[K] = A_cols; for (int j = 1; j < K; j++) rb[j] = (int)((long long)j * A_cols / K); }
        return;
    }
    int mn = 0x7fffffff, mx = -1;
    long long total = 0;
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];
        total += ke - ks;
        if (ks < ke) {
            int c0 = B_col_idx[ks], c1 = B_col_idx[ke - 1];
            if (c0 < mn) mn = c0;
            if (c1 > mx) mx = c1;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        int om = __shfl_xor_sync(0xffffffff, mn, off); if (om < mn) mn = om;
        int ox = __shfl_xor_sync(0xffffffff, mx, off); if (ox > mx) mx = ox;
    }
    for (int off = 16; off > 0; off >>= 1) total += __shfl_xor_sync(0xffffffff, total, off);
    if (total < dyn_min || mx < mn) {   // 轻行/无乘积:等宽回退(= 旧 blo=b*n/K)
        if (lane == 0) { rb[0] = 0; rb[K] = A_cols; for (int j = 1; j < K; j++) rb[j] = (int)((long long)j * A_cols / K); }
        return;
    }
    if (lane == 0) {
        long long span = (long long)mx - mn + 1;
        rb[0] = mn;        // 桶 0 左端收紧到 min(乘积不可能 < mn)
        rb[K] = mx + 1;    // 末桶右端 = max+1(覆盖 max;乘积不可能 > max)
        for (int j = 1; j < K; j++) rb[j] = mn + (int)(j * span / K);
        if (d_max_span) atomicMax(d_max_span, (int)span);
    }
}

// (row,bucket) 一块:count 该桶内 distinct 列数。小跨度桶走 flags 直计(docs/66 §7:
// O(flop) 幂等置位 + O(span) popcount,替代 O(distinct×num_k) 的 k 段扫描 merge 迭代),
// 跨度放不下回退 merge-count。smem_bytes = 实际 launch 的动态 SMEM(运行时定 flags 上限)。
__global__ void bucket_count_kernel(
    const int *A_row_ptr, const int *A_col_idx, int A_rows, int A_cols,
    int K, const int *bnd, int smem_bytes, int *bucket_nnz)   // [A_rows * K],行主序 [i*K + b]
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    long long n = A_cols;
    int blo, bhi;   // bnd 非空 = 动态工作量边界(docs/66);空 = 等宽(旧行为)
    if (bnd) { const int *rb = bnd + (size_t)i * (K + 1); blo = rb[b]; bhi = rb[b + 1]; }
    else     { blo = (int)(b * n / K); bhi = (int)((b + 1) * n / K); }
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
    {   // flags 直计分支:seg 区后剩余 SMEM 全给 flags(1B/列)
        unsigned char *flags = (unsigned char*)(smem + 2 * num_k);
        long usable = (long)smem_bytes - 8L * num_k;
        if (usable > 0 && bhi - blo <= usable) {
            for (int c = lane; c < bhi - blo; c += 32) flags[c] = 0;
            __syncwarp();
            for (int p = lane; p < num_k; p += 32)
                for (int pos = seg_ptr[p]; pos < seg_end[p]; pos++)
                    flags[A_col_idx[pos] - blo] = 1;
            __syncwarp();
            int cnt = 0;
            for (int c = lane; c < bhi - blo; c += 32) cnt += (flags[c] != 0);
            for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(0xffffffff, cnt, off);
            if (lane == 0) bucket_nnz[i * K + b] = cnt;
            return;
        }
    }
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
// T = int(精确 nnz 路)/ long long(flop 路:Σflop 可超 int,ocean337 的 Ga/band 族)
template <typename T>
__global__ void bucket_scan_kernel(int A_rows, int K, const int *bucket_nnz,
                                   int *bucket_off, T *row_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    T acc = 0;
    for (int b = 0; b < K; b++) {
        bucket_off[i * K + b] = (int)acc;
        acc += (T)bucket_nnz[i * K + b];
    }
    row_sum[i] = acc;
}

// (row,bucket) 一块:warp-merge 该桶子区间,写 (col, val) 到全局偏移
__global__ void bucket_merge_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    int A_rows, int A_cols, int K, const int *bnd, const int *C_row_ptr, const int *bucket_off,
    int *out_col, double *out_val,
    int dmode = 0,        // 0=merge 模式(处理 L<128 或 L>dcap 的桶);1=dense 模式(128≤L≤dcap)
    int dcap = 0)         // host 由 bnd 的 max_span 定的桶宽上限(docs/66 §7)
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    long long n = A_cols;
    int blo, bhi;   // bnd 非空 = 动态工作量边界(docs/66);空 = 等宽(旧行为)
    if (bnd) { const int *rb = bnd + (size_t)i * (K + 1); blo = rb[b]; bhi = rb[b + 1]; }
    else     { blo = (int)(b * n / K); bhi = (int)((b + 1) * n / K); }
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
    {   // docs/66 §7 dense 直写(精确 CSR 版):小跨度桶免 k 段扫描,32 lane 连续段顺序发射
        int L = bhi - blo;
        bool eligible = (L >= 128 && L <= dcap);
        if (eligible && dmode == 0) return;   // 归 dense launch(双 launch 各司其职,防双跑)
        if (eligible) {
            unsigned char *dfl = (unsigned char*)(smem + 2 * num_k);
            double *dval = (double*)(((uintptr_t)(smem + 2 * num_k) + L + 7) & ~(uintptr_t)7);
            for (int c = lane; c < L; c += 32) { dfl[c] = 0; dval[c] = 0.0; }
            __syncwarp();
            for (int p = lane; p < num_k; p += 32) {
                double w = weight[p];
                for (int pos = seg_ptr[p]; pos < seg_end[p]; pos++) {
                    int c = A_col_idx[pos] - blo;
                    atomicAdd(&dval[c], w * A_val[pos]);
                    dfl[c] = 1;
                }
            }
            __syncwarp();
            int lo_c = (int)((long long)lane * L / 32), hi_c = (int)((long long)(lane + 1) * L / 32);
            int c0 = 0;
            for (int c = lo_c; c < hi_c; c++) c0 += (dfl[c] != 0);
            int inc = c0;
            for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffff, inc, d); if (lane >= d) inc += v; }
            int t = inc - c0;
            for (int c = lo_c; c < hi_c; c++)
                if (dfl[c]) { out_col[base + t] = blo + c; out_val[base + t] = dval[c]; t++; }
            return;
        }
        if (dmode == 1) return;   // 非合格桶(L<128 争用地板 / L>dcap 放不下)归 merge 模式 launch
    }
    int out_idx = 0;
    for (int __guard = 0; __guard < MRG3_LOOP_CAP; ++__guard) {   // 硬上界:必终止
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

// merge3 + flop_ub sizing:flop 上界定桶区省 count 的 merge 迭代,代价 = gapped buffer + compact(MRG3_FLOP_UB 门控)。

// (C-1) 每 (row,bucket) flop_ub = Σ_k(seg_end-seg_ptr)(lower_bound 定 [blo,bhi),无 merge 迭代),distinct 的确定性上界。
__global__ void bucket_flop_kernel(
    const int *A_row_ptr, const int *A_col_idx,
    const int *B_row_ptr, const int *B_col_idx,
    int A_rows, int A_cols,
    int K, const int *bnd, int *bucket_flop)   // [A_rows * K]
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    long long n = A_cols;
    int blo, bhi;   // bnd 非空 = 动态工作量边界(docs/66);空 = 等宽(旧行为)
    if (bnd) { const int *rb = bnd + (size_t)i * (K + 1); blo = rb[b]; bhi = rb[b + 1]; }
    else     { blo = (int)(b * n / K); bhi = (int)((b + 1) * n / K); }
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;
    extern __shared__ __align__(8) int smem[];
    int *seg_ptr = smem;            // [num_k]
    int *seg_end = smem + num_k;    // [num_k]
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];            // B 的行 k
        seg_ptr[p] = dev_lower_bound(B_col_idx, ks, ke, blo);
        seg_end[p] = dev_lower_bound(B_col_idx, ks, ke, bhi);
    }
    __syncwarp();
    int s = 0;
    for (int p = lane; p < num_k; p += 32) s += seg_end[p] - seg_ptr[p];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if (lane == 0) bucket_flop[i * K + b] = s;
}

// (C-2) warp-merge 写 (col,val) 到 gapped flop 区 + 记真实 distinct(bucket_real_nnz);upper_tri 跳过 wmin<i。
__global__ void bucket_merge_flop_kernel(
    const int *A_row_ptr, const int *A_col_idx, const double *A_val,
    const int *B_row_ptr, const int *B_col_idx, const double *B_val,
    int upper_tri,
    int A_rows, int A_cols, int K, const int *bnd, const long long *row_off, const int *bucket_flop_off,
    int *out_col, double *out_val, int *bucket_real_nnz,   // [A_rows*K]
    int dmode = 0,                     // 0=merge 模式(处理 L<128 或 L>dcap 的桶);1=dense 模式(128≤L≤dcap)
    int dcap = 0,                      // host 由 bnd 的 max_span 定的桶宽上限(docs/66 §7)
    unsigned long long *rowtime = nullptr)   // MRG3_ROWTIME:每 (row,bucket) 执行周期(docs/66)
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    unsigned long long rt0 = rowtime ? clock64() : 0;   // 周期数对 DVFS 免疫(工作量的稳定度量)
    long long n = A_cols;
    int blo, bhi;   // bnd 非空 = 动态工作量边界(docs/66);空 = 等宽(旧行为)
    if (bnd) { const int *rb = bnd + (size_t)i * (K + 1); blo = rb[b]; bhi = rb[b + 1]; }
    else     { blo = (int)(b * n / K); bhi = (int)((b + 1) * n / K); }
    int rs = A_row_ptr[i], re = A_row_ptr[i + 1];
    int num_k = re - rs;
    extern __shared__ __align__(8) int smem[];
    int   *seg_ptr = smem;
    int   *seg_end = smem + num_k;
    double *weight  = (double*)(smem + 2 * num_k);
    for (int p = lane; p < num_k; p += 32) {
        int k = A_col_idx[rs + p];
        int ks = B_row_ptr[k], ke = B_row_ptr[k + 1];            // B 的行 k
        seg_ptr[p] = dev_lower_bound(B_col_idx, ks, ke, blo);
        seg_end[p] = dev_lower_bound(B_col_idx, ks, ke, bhi);
        weight[p]  = A_val[rs + p];                              // a_ik(外层 A)
    }
    __syncwarp();
    long long base = (long long)row_off[i] + bucket_flop_off[i * K + b];   // 64位:Σflop 可超 int
    {   // docs/66 §7 dense 直写分支:小跨度桶免 k 段扫描 —— SMEM flags+val 累加 O(flop),
        // 32 lane 连续列区间顺序发射(桶内天然有序,compact 机器零改动),real_nnz 副产物。
        // 布局:[seg|seg_end|weight] 后 flags[L] pad8 [dval[L]]。dmode=1 只处理 128≤L≤cap 的桶
        // (L≥128 = 原子地址地板,band128 L=51 争用集中实测 2× 劣);dmode=0 处理其余(merge)。
        int L = bhi - blo;
        bool eligible = (L >= 128 && L <= dcap);
        if (eligible && dmode == 0) return;   // 归 dense launch(双 launch 各司其职,防双跑)
        if (eligible) {
            unsigned char *dfl = (unsigned char*)(smem + 2 * num_k);
            double *dval = (double*)(((uintptr_t)(smem + 2 * num_k) + L + 7) & ~(uintptr_t)7);
            for (int c = lane; c < L; c += 32) { dfl[c] = 0; dval[c] = 0.0; }
            __syncwarp();
            for (int p = lane; p < num_k; p += 32) {
                double w = weight[p];
                for (int pos = seg_ptr[p]; pos < seg_end[p]; pos++) {
                    int c = B_col_idx[pos] - blo;
                    atomicAdd(&dval[c], w * B_val[pos]);
                    dfl[c] = 1;
                }
            }
            __syncwarp();
            // 32 lane 各认连续列段:先数(含 upper_tri 过滤)→ warp 前缀 → 段内顺序发射
            int lo_c = (int)((long long)lane * L / 32), hi_c = (int)((long long)(lane + 1) * L / 32);
            int c0 = 0;
            for (int c = lo_c; c < hi_c; c++)
                c0 += (dfl[c] && !(upper_tri && (blo + c) < i));
            int inc = c0;
            for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffff, inc, d); if (lane >= d) inc += v; }
            int ex = inc - c0;                       // 段起点的桶内偏移
            int t = ex;
            for (int c = lo_c; c < hi_c; c++)
                if (dfl[c] && !(upper_tri && (blo + c) < i)) {
                    out_col[base + t] = blo + c;
                    out_val[base + t] = dval[c];
                    t++;
                }
            int tot = __shfl_sync(0xffffffff, inc, 31);
            if (lane == 0) {
                bucket_real_nnz[i * K + b] = tot;
                if (rowtime) rowtime[i * K + b] = clock64() - rt0;
            }
            return;
        }
        if (dmode == 1) return;   // 非合格桶(L<128 争用地板 / L>dcap 放不下)归 merge 模式 launch
    }
    int out_idx = 0;
    for (int __guard = 0; __guard < MRG3_LOOP_CAP; ++__guard) {   // 硬上界:必终止
        int mymin = 0x7fffffff;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p]) { int col = B_col_idx[pos]; if (col < mymin) mymin = col; }
        }
        int wmin = mymin;
        for (int off = 16; off > 0; off >>= 1) { int v = __shfl_xor_sync(0xffffffff, wmin, off); if (v < wmin) wmin = v; }
        if (wmin == 0x7fffffff) break;
        double mysum = 0.0f;
        for (int p = lane; p < num_k; p += 32) {
            int pos = seg_ptr[p];
            if (pos < seg_end[p] && B_col_idx[pos] == wmin) {
                mysum += weight[p] * B_val[pos];
                seg_ptr[p] = pos + 1;
            }
        }
        for (int off = 16; off > 0; off >>= 1)
            mysum += __shfl_xor_sync(0xffffffff, mysum, off);
        if (lane == 0) {
            if (!(upper_tri && wmin < i)) {                       // ATT 上三角:跳过 j<i(仍消费)
                out_col[base + out_idx] = wmin;
                out_val[base + out_idx] = mysum;
                out_idx++;
            }
        }
    }
    if (lane == 0) {
        bucket_real_nnz[i * K + b] = out_idx;
        if (rowtime) rowtime[i * K + b] = clock64() - rt0;
    }
}

// (C-3) compact:把每 (row,bucket) 的真实项从 gapped flop 区拷到精确 CSR(base = C_row_ptr + bucket_off_exact)。
__global__ void bucket_compact_kernel(
    int A_rows, int K, const long long *row_off, const int *bucket_flop_off,
    const int *bucket_real_nnz, const int *C_row_ptr, const int *bucket_off_exact,
    const int *in_col, const double *in_val, int *out_col, double *out_val)
{
    int i = blockIdx.x, b = blockIdx.y;
    if (i >= A_rows) return;
    int lane = threadIdx.x;
    long long src = (long long)row_off[i] + bucket_flop_off[i * K + b];
    int dst = C_row_ptr[i] + bucket_off_exact[i * K + b];
    int n = bucket_real_nnz[i * K + b];
    for (int t = lane; t < n; t += 32) {
        out_col[dst + t] = in_col[src + t];
        out_val[dst + t] = in_val[src + t];
    }
}

// merge3 Host
static void merge3_product(
    void *A_buffer, int A_rows, int A_cols, int A_nnz, bool att,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    const int K = 5;   // 每行列域分桶数(可调;大→并行度高、开销大)
    const char *tag = att ? "attm" : "mrg3";
    dbg("[%s] start (K=%d)\n", tag, K);
    HashProf prof(att ? "attm-prof" : "mrg3-prof");

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
    size_t smem_count = (size_t)max_row_nnz * 2 * sizeof(int);                     // 基础:count/flop kernel
    size_t smem_merge = (size_t)max_row_nnz * (2 * sizeof(int) + sizeof(double));  // 基础:numeric merge 模式(seg+weight)
    // docs/66 §7 dense 分支:预算【不盲定】—— bnd kernel 顺带 atomicMax 行 span → D2H 一个 int
    // → dcap(桶宽上界)精确已知。dense launch 只在 dcap≥128(有合格桶)且 SMEM 装得下(≤48KB,
    // 免 SetAttribute)时发射;merge 模式恒保基础 SMEM(band128 教训:盲目扩容 → occupancy
    // 14→3 blocks/SM,flop 相位 0.85→2.0ms;空 dense launch 也有 ~5ms 块调度税)。
    // L≥128 = 原子地址地板(band128 L=51 争用集中实测 2× 劣)。MRG3_DENSE_SPAN=0 全关。
    static int g_dspan = -1;
    if (g_dspan < 0) { const char *e = getenv("MRG3_DENSE_SPAN"); g_dspan = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }
    int dense_dcap = 0;            // ≥128 才有 dense launch;0 = 全 merge(旧行为)
    size_t smem_count_d = 0, smem_merge_d = 0;
    if (smem_merge > 48 * 1024)
        CHECK_CUDA(cudaFuncSetAttribute(bucket_merge_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_merge));
    if (smem_count > 48 * 1024)
        CHECK_CUDA(cudaFuncSetAttribute(bucket_count_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_count));

    dim3 grid(A_rows, K), block32(32);

    // 内层 j 来源 B:AA=A;ATT=Aᵀ(A 的 CSC)。ATT 强制走 flop path(count path 未泛化为 B)。
    int *dB_row_ptr, *dB_col_idx; double *dB_val;
    int *d_csc_cp = nullptr, *d_csc_ri = nullptr; double *d_csc_val = nullptr;
    if (att) {
        build_csc(dA_row_ptr, dA_col_idx, dA_val, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_val);
        dB_row_ptr = d_csc_cp; dB_col_idx = d_csc_ri; dB_val = d_csc_val;
        dbg("[attm] csc\n");
    } else {
        dB_row_ptr = dA_row_ptr; dB_col_idx = dA_col_idx; dB_val = dA_val;
    }
    const int upper_tri = att ? 1 : 0;

    // 动态负载均衡(docs/66):值域二分算每行 K+1 边界(轻行等宽回退)。MRG3_DYN_BND=0 关回旧行为。
    // B 口径:!att 时 dB==dA(count 路径与 flop 路径同源);att 时只有 flop 路径跑,dB=CSC 正确。
    static int g_dyn = -1;
    if (g_dyn < 0) { const char *e = getenv("MRG3_DYN_BND"); g_dyn = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }
    long long dyn_min = 8192;
    if (const char *e = getenv("MRG3_DYN_MIN")) dyn_min = atoll(e);
    int *d_bnd = nullptr;
    int *d_max_span = nullptr;   // docs/66 §7:bnd 顺带 atomicMax 行 span → host 精确 dense 桶宽
    if (g_dyn && g_dspan) {
        CHECK_CUDA(cudaMalloc(&d_max_span, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_max_span, 0, sizeof(int)));
    }
    if (g_dyn) {
        CHECK_CUDA(cudaMalloc(&d_bnd, (size_t)A_rows * (K + 1) * sizeof(int)));
        prof("bnd", [&]{
            bucket_bnd_kernel<<<A_rows, block32>>>(
                dA_row_ptr, dA_col_idx, dB_row_ptr, dB_col_idx, A_rows, A_cols, K, dyn_min, d_bnd, d_max_span);
            CHECK_CUDA(cudaGetLastError());
        });
        // docs/66 §7:D2H 一个 int(~10μs)→ dcap 精确;只在可能 dense 时付这笔同步
        if (d_max_span) {
            int hmaxs = 0;
            CHECK_CUDA(cudaMemcpy(&hmaxs, d_max_span, sizeof(int), cudaMemcpyDeviceToHost));
            long long L1 = ((long long)hmaxs + K - 1) / K;            // 动态行桶宽上界
            long long L2 = ((long long)A_cols + K - 1) / K;            // 等宽回退行桶宽
            long long L = L1 > L2 ? L1 : L2;
            if (L >= 128 && smem_merge + 9 * L + 16 <= 48 * 1024) {
                dense_dcap = (int)L;
                smem_merge_d = smem_merge + 9 * L + 16;                // numeric dense launch
                smem_count_d = smem_count + L;                         // count flags(1B/col)
                if (smem_count_d > 48 * 1024) smem_count_d = 48 * 1024;
                dbg("[mrg3] dense dcap=%d(Σspan_max=%d)merge_d=%zuKB\n", dense_dcap, hmaxs, smem_merge_d / 1024);
            }
            cudaFree(d_max_span);
        }
    }

    // 门控:flop_ub sizing(省 count pass ~37%)vs 精确 count(原版)。默认 flop_ub(净赢 24-32%、无回归);MRG3_FLOP_UB=0 关回精确 count。
    static int g_flop = -1;
    if (g_flop < 0) { const char *e = getenv("MRG3_FLOP_UB"); g_flop = (e && *e) ? (atoi(e) > 0 ? 1 : 0) : 1; }
    if (att) g_flop = 1;   // ATT 只走 flop path(bucket_count/merge 未泛化为 B)
    if (g_flop && smem_merge > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(bucket_flop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_count));
        CHECK_CUDA(cudaFuncSetAttribute(bucket_merge_flop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_merge));
    }

    int *dC_row_ptr; CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    int C_nnz_result;
    void *dC_buffer = nullptr;
    size_t C_total_size = 0;

    if (g_flop) {
        // flop_ub path:省 count 的 merge 迭代,代价 = gapped buffer + compact
        // MRG3_ROWTIME=path:逐 (row,bucket) 执行周期落盘(docs/66 per-row 采数;周期对 DVFS 免疫)
        unsigned long long *d_rtime = nullptr; FILE *rtf = nullptr;
        const char *rt_path = getenv("MRG3_ROWTIME");
        if (rt_path && *rt_path) {
            rtf = fopen(rt_path, "w");
            if (rtf) CHECK_CUDA(cudaMalloc(&d_rtime, (size_t)A_rows * K * sizeof(unsigned long long)));
        }
        int *d_bflop, *d_bflop_off;
        long long *d_row_flop, *d_row_off;   // 64 位:ocean337 上 Σflop > 2^31(Ga/band 族),int 会变负 → 灾难分配
        CHECK_CUDA(cudaMalloc(&d_bflop, (size_t)A_rows * K * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_bflop_off, (size_t)A_rows * K * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_row_flop, A_rows * sizeof(long long)));
        CHECK_CUDA(cudaMalloc(&d_row_off, (A_rows + 1) * sizeof(long long)));
        long long total_flop;
        prof("flop", [&]{   // (C-1) 每 (row,bucket) flop_ub(lower_bound+sum,无 merge 迭代)
            bucket_flop_kernel<<<grid, block32, smem_count>>>(dA_row_ptr, dA_col_idx, dB_row_ptr, dB_col_idx, A_rows, A_cols, K, d_bnd, d_bflop);
            CHECK_CUDA(cudaGetLastError());
        });
        prof("fscan", [&]{  // 行内 scan(flop)→ bucket_flop_off + row_flop;全局 scan → row_off(gapped) + total_flop
            bucket_scan_kernel<long long><<<(A_rows + block - 1) / block, block>>>(A_rows, K, d_bflop, d_bflop_off, d_row_flop);
            CHECK_CUDA(cudaMemset(d_row_off, 0, sizeof(long long)));
            thrust::inclusive_scan(thrust::device_ptr<long long>(d_row_flop), thrust::device_ptr<long long>(d_row_flop + A_rows),
                                   thrust::device_ptr<long long>(d_row_off + 1));
            CHECK_CUDA(cudaMemcpy(&total_flop, d_row_off + A_rows, sizeof(long long), cudaMemcpyDeviceToHost));
        });
        // SAFETY:分配前 sanity —— 非法/超界 total_flop 干净报错退出,绝不把天文数字送进 cudaMalloc
        // (2026-08-25 GPU1 wedge 的根因之一:int 溢出 → (size_t)负数 → 灾难分配)。
        long long mrg3_max = 3000000000LL;   // 3e9 项 ≈ 36GB gapped,80GB 卡的上限
        if (const char *e = getenv("MRG3_MAX_ENTRIES")) mrg3_max = atoll(e);
        if (total_flop <= 0 || total_flop > mrg3_max) {
            fprintf(stderr, "[mrg3] SAFETY: total_flop=%lld 非法/超界(cap=%lld,MRG3_MAX_ENTRIES 可调)"
                            " → 干净退出,不进分配\n", total_flop, mrg3_max);
            exit(EXIT_FAILURE);
        }
        // gapped out buffer [total_flop](=Σflop,64位计;大阵数 GB,H100 可容)
        int *d_gcol; double *d_gval; int *d_breal;
        CHECK_CUDA(cudaMalloc(&d_gcol, (size_t)total_flop * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_gval, (size_t)total_flop * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_breal, (size_t)A_rows * K * sizeof(int)));
        prof("merge", [&]{   // (C-2) warp-merge 写 gapped 区 + 记真实数 bucket_real_nnz
            // numeric:双 launch(count 路径同构;dmode/dcap 详见 kernel 注释)
            bucket_merge_flop_kernel<<<grid, block32, smem_merge>>>(
                dA_row_ptr, dA_col_idx, dA_val, dB_row_ptr, dB_col_idx, dB_val, upper_tri,
                A_rows, A_cols, K, d_bnd, d_row_off, d_bflop_off, d_gcol, d_gval, d_breal, 0, dense_dcap, d_rtime);
            CHECK_CUDA(cudaGetLastError());
            if (dense_dcap) {
                bucket_merge_flop_kernel<<<grid, block32, smem_merge_d>>>(
                    dA_row_ptr, dA_col_idx, dA_val, dB_row_ptr, dB_col_idx, dB_val, upper_tri,
                    A_rows, A_cols, K, d_bnd, d_row_off, d_bflop_off, d_gcol, d_gval, d_breal, 1, dense_dcap, d_rtime);
                CHECK_CUDA(cudaGetLastError());
            }
            CHECK_CUDA(cudaGetLastError());
        });
        if (d_rtime) {   // 逐桶周期 → 文本 "row bucket cycles";python 侧 join 特征做判据
            std::vector<unsigned long long> h_rt((size_t)A_rows * K);
            CHECK_CUDA(cudaMemcpy(h_rt.data(), d_rtime, h_rt.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            for (int i = 0; i < A_rows; i++)
                for (int b = 0; b < K; b++)
                    fprintf(rtf, "%d %d %llu\n", i, b, h_rt[(size_t)i * K + b]);
            fclose(rtf);
            dbg("[mrg3] rowtime → %s (%d×%d)\n", rt_path, A_rows, K);
            cudaFree(d_rtime);
        }
        int *d_boff_ex, *d_row_nnz;
        CHECK_CUDA(cudaMalloc(&d_boff_ex, (size_t)A_rows * K * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_row_nnz, A_rows * sizeof(int)));
        prof("rscan", [&]{  // scan(真实数)→ exact bucket_off + row_nnz → C_row_ptr + C_nnz
            bucket_scan_kernel<int><<<(A_rows + block - 1) / block, block>>>(A_rows, K, d_breal, d_boff_ex, d_row_nnz);
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
        // exact-count path(原版)
        // Stage 1: 每桶 count distinct 列
        int *d_bucket_nnz;
        CHECK_CUDA(cudaMalloc(&d_bucket_nnz, (size_t)A_rows * K * sizeof(int)));
        prof("count", [&]{
            {   // count:单 launch,SMEM 带 flags 余量(纯置位 dense,无原子争用地板问题)
                size_t scnt = dense_dcap ? smem_count_d : smem_count;
                bucket_count_kernel<<<grid, block32, scnt>>>(
                    dA_row_ptr, dA_col_idx, A_rows, A_cols, K, d_bnd, dense_dcap ? (int)scnt : 0, d_bucket_nnz);
                CHECK_CUDA(cudaGetLastError());
            }
            CHECK_CUDA(cudaGetLastError());
        });
        // Stage 2: 行内桶偏移 + row_nnz + C_row_ptr(D2H 纳入 scan 块)
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
        // Stage 3: 每桶 merge 写值 + 连续输出 dC_buffer=[row_ptr|col|val]
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
            // numeric:双 launch —— merge 模式(基础 SMEM)处理 L<128/L>dcap 的桶;
            // dense 模式(扩容 SMEM)处理 128≤L≤dcap 的桶。各桶恰被一个 launch 处理。
            bucket_merge_kernel<<<grid, block32, smem_merge>>>(
                dA_row_ptr, dA_col_idx, dA_val, A_rows, A_cols, K, d_bnd, dC_row_ptr, d_bucket_off,
                dC_col_idx, dC_val, 0, dense_dcap);
            CHECK_CUDA(cudaGetLastError());
            if (dense_dcap) {
                bucket_merge_kernel<<<grid, block32, smem_merge_d>>>(
                    dA_row_ptr, dA_col_idx, dA_val, A_rows, A_cols, K, d_bnd, dC_row_ptr, d_bucket_off,
                    dC_col_idx, dC_val, 1, dense_dcap);
                CHECK_CUDA(cudaGetLastError());
            }
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
    cudaFree(d_bnd);                                                  // 动态边界(docs/66;g_dyn=0 时为 null,no-op)
    cudaFree(d_csc_cp); cudaFree(d_csc_ri); cudaFree(d_csc_val);      // ATT 的 Aᵀ(AA 时 null,no-op)
}

void spgemm_self_product_merge3(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                                void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    merge3_product(A_buffer, A_rows, A_cols, A_nnz, /*att=*/false, C_buffer_out, C_rows, C_cols, C_nnz);
}

// C = A·Aᵀ 上三角(j≥i):AA merge3(列域分桶)的忠实拷贝,B=Aᵀ(CSC)+ j≥i 过滤。返回上三角 CSR。
void spgemm_att_merge3(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                       void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    merge3_product(A_buffer, A_rows, A_cols, A_nnz, /*att=*/true, C_buffer_out, C_rows, C_cols, C_nnz);
}
