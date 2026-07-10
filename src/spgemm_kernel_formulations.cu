#include "spgemm.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>

// 三种 SpGEMM 公式(自乘 C=A·A)的 ESC 对照实现:
//   - Gustavson(行向, 外层=i): 见 spgemm_kernel_manual.cu, 保留不动
//   - 外积  (outer, 外层=k): 本文件 spgemm_self_product_outer
//   - 列向  (colwise, 外层=j):本文件 spgemm_self_product_colwise
// 三者展开的是同一批中间项 (i,j,A[i,k]*A[k,j]),区别只在【外层并行轴 / 访存模式】;
// 合并阶段(排序+去重求和)完全相同 → 共用 esc_merge()。
// 外积与列向都要读 A 的"列",所以先把 CSR 转成 CSC。

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ===================== CSR -> CSC(自包含,不依赖 cusparse 版本)=====================

__global__ void csc_count_kernel(const int *col_idx, int nnz, int *col_count)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nnz) return;
    atomicAdd(&col_count[col_idx[t]], 1);
}

// 逐行把 (i, col_idx[p], val[p]) 散到 CSC:用 tmp_off[col] 作每列写指针
__global__ void csc_fill_kernel(
    const int *row_ptr, const int *col_idx, const float *val,
    int A_rows, int *tmp_off,
    int *csc_row_idx, float *csc_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; p++) {
        int c = col_idx[p];
        int pos = atomicAdd(&tmp_off[c], 1);
        csc_row_idx[pos] = i;
        csc_val[pos] = val[p];
    }
}

static void build_csc(const int *d_row_ptr, const int *d_col_idx, const float *d_val,
                      int A_rows, int A_nnz,
                      int **col_ptr_out, int **row_idx_out, float **val_out)
{
    int *d_col_count;
    CHECK_CUDA(cudaMalloc(&d_col_count, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_col_count, 0, A_rows * sizeof(int)));
    {
        int grid = (A_nnz + 255) / 256;
        csc_count_kernel<<<grid, 256>>>(d_col_idx, A_nnz, d_col_count);
    }
    int *d_col_ptr;
    CHECK_CUDA(cudaMalloc(&d_col_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_col_ptr, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_col_count),
                           thrust::device_ptr<int>(d_col_count + A_rows),
                           thrust::device_ptr<int>(d_col_ptr + 1));

    // tmp_off = 复制一份 col_ptr(去掉最后一个元素语义),作为每列写指针
    int *d_tmp;
    CHECK_CUDA(cudaMalloc(&d_tmp, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_tmp, d_col_ptr, A_rows * sizeof(int), cudaMemcpyDeviceToDevice));

    int *d_csc_row; float *d_csc_val;
    CHECK_CUDA(cudaMalloc(&d_csc_row, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csc_val, A_nnz * sizeof(float)));
    {
        int grid = (A_rows + 255) / 256;
        csc_fill_kernel<<<grid, 256>>>(d_row_ptr, d_col_idx, d_val, A_rows,
                                       d_tmp, d_csc_row, d_csc_val);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_col_count);
    cudaFree(d_tmp);
    *col_ptr_out = d_col_ptr;
    *row_idx_out = d_csc_row;
    *val_out = d_csc_val;
}

// ===================== 共用:ESC 合并(排序 + 去重求和 → CSR)=====================

__global__ void esc_finalize_kernel(
    const unsigned long long *red_key, const float *red_val,
    int C_nnz, int *C_col_idx, float *C_val, int *C_row_nnz)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= C_nnz) return;
    unsigned long long k = red_key[t];
    C_col_idx[t] = (int)(k & 0xffffffffu);
    C_val[t] = red_val[t];
    atomicAdd(&C_row_nnz[(int)(k >> 32)], 1);
}

// 输入展开好的 d_key[total]=(row<<32|col), d_val[total];
// 输出(设备)*dC_col_idx[Cnnz], *dC_val[Cnnz], *dC_row_ptr[A_rows+1]; 返回 Cnnz。
static int esc_merge(unsigned long long *d_key, float *d_val, int total, int A_rows,
                     int **dC_col_idx, float **dC_val, int **dC_row_ptr)
{
    thrust::sort_by_key(thrust::device_ptr<unsigned long long>(d_key),
                        thrust::device_ptr<unsigned long long>(d_key + total),
                        thrust::device_ptr<float>(d_val));
    unsigned long long *d_rk; float *d_rv;
    CHECK_CUDA(cudaMalloc(&d_rk, (size_t)total * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_rv, (size_t)total * sizeof(float)));
    thrust::pair<thrust::device_ptr<unsigned long long>,
                 thrust::device_ptr<float> > e =
        thrust::reduce_by_key(
            thrust::device_ptr<unsigned long long>(d_key),
            thrust::device_ptr<unsigned long long>(d_key + total),
            thrust::device_ptr<float>(d_val),
            thrust::device_ptr<unsigned long long>(d_rk),
            thrust::device_ptr<float>(d_rv));
    int Cnnz = (int)(e.first - thrust::device_ptr<unsigned long long>(d_rk));
    CHECK_CUDA(cudaDeviceSynchronize());

    int *c_col; float *c_val; int *c_row_nnz;
    CHECK_CUDA(cudaMalloc(&c_col, (size_t)Cnnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&c_val, (size_t)Cnnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&c_row_nnz, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(c_row_nnz, 0, A_rows * sizeof(int)));
    {
        esc_finalize_kernel<<<(Cnnz + 255) / 256, 256>>>(
            d_rk, d_rv, Cnnz, c_col, c_val, c_row_nnz);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_rk); cudaFree(d_rv);

    int *c_row_ptr;
    CHECK_CUDA(cudaMalloc(&c_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(c_row_ptr, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(c_row_nnz),
                           thrust::device_ptr<int>(c_row_nnz + A_rows),
                           thrust::device_ptr<int>(c_row_ptr + 1));
    cudaFree(c_row_nnz);
    *dC_col_idx = c_col; *dC_val = c_val; *dC_row_ptr = c_row_ptr;
    return Cnnz;
}

// 把 (row_ptr|col_idx|val) 三段打包成单块并 D2H(与 Gustavson 版一致)
static void *pack_and_download(int *d_row_ptr, int *d_col_idx, float *d_val,
                               int A_rows, int Cnnz)
{
    size_t rp = (A_rows + 1) * sizeof(int);
    size_t ci = (size_t)Cnnz * sizeof(int);
    size_t vv = (size_t)Cnnz * sizeof(float);
    size_t rp_a = (rp + 3) & ~3;
    size_t ci_a = (ci + 3) & ~3;
    size_t total = rp_a + ci_a + vv;
    void *db;
    CHECK_CUDA(cudaMalloc(&db, total));
    char *base = (char*)db;
    CHECK_CUDA(cudaMemcpy(base, d_row_ptr, rp, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(base + rp_a, d_col_idx, ci, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(base + rp_a + ci_a, d_val, vv, cudaMemcpyDeviceToDevice));
    void *hb = nullptr;
    CHECK_CUDA(cudaMallocHost(&hb, total));
    CHECK_CUDA(cudaMemcpy(hb, db, total, cudaMemcpyDeviceToHost));
    cudaFree(db);
    return hb;
}

// ===================== 外积 (outer product, 外层 = k) =====================

// 每个收缩维 k 的中间项数 = nnz(列k) * nnz(行k)
__global__ void count_outer_kernel(
    const int *csr_row_ptr, const int *csc_col_ptr, int A_rows, int *ub)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= A_rows) return;
    long long r = csr_row_ptr[k + 1] - csr_row_ptr[k];
    long long c = csc_col_ptr[k + 1] - csc_col_ptr[k];
    ub[k] = (int)(r * c);
}

// 每个 k 一块:把 (列k 的每个 i) × (行k 的每个 j) 散出去
__global__ void expand_outer_kernel(
    const int *csr_row_ptr, const int *csr_col_idx, const float *csr_val,
    const int *csc_col_ptr, const int *csc_row_idx, const float *csc_val,
    int A_rows, const int *off,
    unsigned long long *key, float *val)
{
    int k = blockIdx.x;
    if (k >= A_rows) return;
    __shared__ int pos;
    if (threadIdx.x == 0) pos = off[k];
    __syncthreads();

    int cs = csc_col_ptr[k], ce = csc_col_ptr[k + 1];   // 列 k 的 i 们
    int rs = csr_row_ptr[k], re = csr_row_ptr[k + 1];   // 行 k 的 j 们
    for (int pi = cs + threadIdx.x; pi < ce; pi += blockDim.x) {
        int i = csc_row_idx[pi];
        float a_ik = csc_val[pi];
        for (int q = rs; q < re; q++) {
            int slot = atomicAdd(&pos, 1);
            key[slot] = ((unsigned long long)i << 32) | (unsigned int)csr_col_idx[q];
            val[slot] = a_ik * csr_val[q];
        }
    }
}

void spgemm_self_product_outer(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    size_t rp = (A_rows + 1) * sizeof(int);
    size_t ci = A_nnz * sizeof(int);
    size_t vv = A_nnz * sizeof(float);
    size_t totalA = rp + ci + vv;
    void *dA; CHECK_CUDA(cudaMalloc(&dA, totalA));
    CHECK_CUDA(cudaMemcpy(dA, A_buffer, totalA, cudaMemcpyHostToDevice));
    char *b = (char*)dA;
    int *d_rp = (int*)b; int *d_ci = (int*)(b + rp); float *d_v = (float*)(b + rp + ci);

    int *d_csc_cp, *d_csc_ri; float *d_csc_v;
    build_csc(d_rp, d_ci, d_v, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_v);

    dbg("outer: count begin\n");
    int *d_ub; CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    count_outer_kernel<<<(A_rows + 255) / 256, 256>>>(d_rp, d_csc_cp, A_rows, d_ub);
    CHECK_CUDA(cudaDeviceSynchronize());

    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total; CHECK_CUDA(cudaMemcpy(&total, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("outer: total intermediates = %d\n", total);

    unsigned long long *d_key; float *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total * sizeof(float)));
    dbg("outer: expand begin\n");
    expand_outer_kernel<<<A_rows, 256>>>(d_rp, d_ci, d_v, d_csc_cp, d_csc_ri, d_csc_v,
                                         A_rows, d_off, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("outer: expand done\n");

    int *c_col; float *c_val; int *c_rp;
    dbg("outer: merge begin\n");
    int Cnnz = esc_merge(d_key, d_val, total, A_rows, &c_col, &c_val, &c_rp);
    dbg("outer: merge done, C_nnz=%d\n", Cnnz);

    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;

    cudaFree(dA); cudaFree(d_csc_cp); cudaFree(d_csc_ri); cudaFree(d_csc_v);
    cudaFree(d_ub); cudaFree(d_off); cudaFree(d_key); cudaFree(d_val);
    cudaFree(c_col); cudaFree(c_val); cudaFree(c_rp);
}

// ===================== 列向 (column-wise, 外层 = j) =====================
// 注:经典"内积"(逐元素点积)与外积展开同一批中间项;这里以【输出列 j】为外层
// 并行作为第三种收缩轴,访存全走 CSC。

// 每个输出列 j 的中间项数 = Σ_{k ∈ 列j} nnz(列k)
__global__ void count_colwise_kernel(
    const int *csc_col_ptr, const int *csc_row_idx, int A_rows, int *ub)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= A_rows) return;
    long long s = 0;
    for (int p = csc_col_ptr[j]; p < csc_col_ptr[j + 1]; p++) {
        int k = csc_row_idx[p];
        s += (long long)(csc_col_ptr[k + 1] - csc_col_ptr[k]);
    }
    ub[j] = (int)s;
}

// 每个 j 一块:对 列j 的每个 k,再把 列k 的每个 i 散到 (i,j)
__global__ void expand_colwise_kernel(
    const int *csc_col_ptr, const int *csc_row_idx, const float *csc_val,
    int A_rows, const int *off,
    unsigned long long *key, float *val)
{
    int j = blockIdx.x;
    if (j >= A_rows) return;
    __shared__ int pos;
    if (threadIdx.x == 0) pos = off[j];
    __syncthreads();

    int js = csc_col_ptr[j], je = csc_col_ptr[j + 1];   // 列 j 的 k 们
    for (int p = js + threadIdx.x; p < je; p += blockDim.x) {
        int k = csc_row_idx[p];
        float a_kj = csc_val[p];
        int ks = csc_col_ptr[k], ke = csc_col_ptr[k + 1];   // 列 k 的 i 们
        for (int q = ks; q < ke; q++) {
            int slot = atomicAdd(&pos, 1);
            key[slot] = ((unsigned long long)csc_row_idx[q] << 32) | (unsigned int)j;
            val[slot] = csc_val[q] * a_kj;
        }
    }
}

void spgemm_self_product_colwise(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    size_t rp = (A_rows + 1) * sizeof(int);
    size_t ci = A_nnz * sizeof(int);
    size_t vv = A_nnz * sizeof(float);
    size_t totalA = rp + ci + vv;
    void *dA; CHECK_CUDA(cudaMalloc(&dA, totalA));
    CHECK_CUDA(cudaMemcpy(dA, A_buffer, totalA, cudaMemcpyHostToDevice));
    char *b = (char*)dA;
    int *d_rp = (int*)b; int *d_ci = (int*)(b + rp); float *d_v = (float*)(b + rp + ci);

    int *d_csc_cp, *d_csc_ri; float *d_csc_v;
    build_csc(d_rp, d_ci, d_v, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_v);

    dbg("colwise: count begin\n");
    int *d_ub; CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    count_colwise_kernel<<<(A_rows + 255) / 256, 256>>>(d_csc_cp, d_csc_ri, A_rows, d_ub);
    CHECK_CUDA(cudaDeviceSynchronize());

    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total; CHECK_CUDA(cudaMemcpy(&total, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("colwise: total intermediates = %d\n", total);

    unsigned long long *d_key; float *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total * sizeof(float)));
    dbg("colwise: expand begin\n");
    expand_colwise_kernel<<<A_rows, 256>>>(d_csc_cp, d_csc_ri, d_csc_v, A_rows, d_off, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("colwise: expand done\n");

    int *c_col; float *c_val; int *c_rp;
    dbg("colwise: merge begin\n");
    int Cnnz = esc_merge(d_key, d_val, total, A_rows, &c_col, &c_val, &c_rp);
    dbg("colwise: merge done, C_nnz=%d\n", Cnnz);

    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;

    cudaFree(dA); cudaFree(d_csc_cp); cudaFree(d_csc_ri); cudaFree(d_csc_v);
    cudaFree(d_ub); cudaFree(d_off); cudaFree(d_key); cudaFree(d_val);
    cudaFree(c_col); cudaFree(c_val); cudaFree(c_rp);
}

// ===================== 逐元素内积 (inner product) =====================
// C[i,j] = row_i(A) · col_j(A),数值阶段对【每个输出元素】独立归并 row_i 与 col_j。
// 关键:内积要"逐元素",必须先知道有哪些 (i,j) ——而"找结构"本身就是 SpGEMM,
// 没法用廉价的全局 mark 并行去重(并发行会互相覆盖)。所以这里:
//   符号阶段:借用 ESC(count+expand+sort+reduce)拿到【结构】(哪些 (i,j) 存在);
//   数值阶段:才是真正的"逐元素内积"——每个 C[i,j] 用归并点积重算,
//            row_i 会被该行每个输出列重复读取(这正是内积低效的根源)。
// 因此 inner 比 ESC 多一遍数值归并,用它和前三种对照能看出"逐元素点积"的代价。

// 复用 manual.cu 里 Gustavson 的符号展开(只取结构,值会被数值阶段覆盖)
extern __global__ void count_intermediates_kernel(const int *A_row_ptr,
                                                  const int *A_col_idx, int A_rows, int *ub);
extern __global__ void expand_intermediates_kernel(const int *A_row_ptr, const int *A_col_idx,
                                                   const float *A_val, int A_rows,
                                                   const int *row_off,
                                                   unsigned long long *key, float *val);

// 归并点积:row_i(csr, 按 col 有序) ∩ col_j(csc, 按 row 有序)
__device__ __forceinline__ float merge_dot(
    const int *csr_ci, const float *csr_v, int rs, int re,
    const int *csc_ri, const float *csc_v, int cs, int ce)
{
    float dot = 0.0f;
    int p = rs, q = cs;
    while (p < re && q < ce) {
        int kp = csr_ci[p];
        int kq = csc_ri[q];
        if (kp == kq) { dot += csr_v[p] * csc_v[q]; p++; q++; }
        else if (kp < kq) p++;
        else q++;
    }
    return dot;
}

// 数值阶段(内积本体):每个输出元素独立归并 row_i 与 col_j
__global__ void inner_numeric_kernel(
    const int *csr_rp, const int *csr_ci, const float *csr_v,
    const int *csc_cp, const int *csc_ri, const float *csc_v,
    int A_rows, const int *C_rp, const int *C_ci, float *C_val)
{
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int start = C_rp[i], end = C_rp[i + 1];
    int rs = csr_rp[i], re = csr_rp[i + 1];
    for (int t = start + threadIdx.x; t < end; t += blockDim.x) {
        int j = C_ci[t];
        C_val[t] = merge_dot(csr_ci, csr_v, rs, re,
                             csc_ri, csc_v, csc_cp[j], csc_cp[j + 1]);
    }
}

void spgemm_self_product_inner(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    size_t rp = (A_rows + 1) * sizeof(int);
    size_t ci = A_nnz * sizeof(int);
    size_t vv = A_nnz * sizeof(float);
    size_t totalA = rp + ci + vv;
    void *dA; CHECK_CUDA(cudaMalloc(&dA, totalA));
    CHECK_CUDA(cudaMemcpy(dA, A_buffer, totalA, cudaMemcpyHostToDevice));
    char *b = (char*)dA;
    int *d_rp = (int*)b; int *d_ci = (int*)(b + rp); float *d_v = (float*)(b + rp + ci);

    int *d_csc_cp, *d_csc_ri; float *d_csc_v;
    build_csc(d_rp, d_ci, d_v, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_v);

    // ---- 符号阶段:ESC 展开+排序+去重,只取结构 (c_col 已按 (行,列) 有序) ----
    dbg("inner: symbolic(ESC) begin\n");
    int *d_ub; CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    count_intermediates_kernel<<<(A_rows + 255) / 256, 256>>>(d_rp, d_ci, A_rows, d_ub);
    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total; CHECK_CUDA(cudaMemcpy(&total, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    unsigned long long *d_key; float *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total * sizeof(float)));
    expand_intermediates_kernel<<<A_rows, 256>>>(d_rp, d_ci, d_v, A_rows, d_off, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    int *c_col; float *c_val; int *c_rp;
    int Cnnz = esc_merge(d_key, d_val, total, A_rows, &c_col, &c_val, &c_rp);  // c_val 暂存,会被覆盖
    dbg("inner: structure C_nnz=%d, numeric(merge) begin\n", Cnnz);
    cudaFree(d_ub); cudaFree(d_off); cudaFree(d_key); cudaFree(d_val);

    // ---- 数值阶段(内积本体):逐元素归并 row_i 与 col_j,重算 c_val ----
    inner_numeric_kernel<<<A_rows, 256>>>(d_rp, d_ci, d_v,
                                          d_csc_cp, d_csc_ri, d_csc_v,
                                          A_rows, c_rp, c_col, c_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("inner: numeric done\n");

    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;

    cudaFree(dA); cudaFree(d_csc_cp); cudaFree(d_csc_ri); cudaFree(d_csc_v);
    cudaFree(c_col); cudaFree(c_val); cudaFree(c_rp);
}
