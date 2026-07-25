#include "spgemm.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

// profiling:当前方法标签(各 host 入口设置),共享 helper 用它打统一 [tag] phase 桩。
// parser 按相邻两戳相减得各阶段耗时。
static const char *g_tag = "?";

// ===================== CSR -> CSC(自包含,不依赖 cusparse 版本)=====================

__global__ void csc_count_kernel(const int *col_idx, int nnz, int *col_count)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nnz) return;
    atomicAdd(&col_count[col_idx[t]], 1);
}

// 逐行把 (i, col_idx[p], val[p]) 散到 CSC:用 tmp_off[col] 作每列写指针
__global__ void csc_fill_kernel(
    const int *row_ptr, const int *col_idx, const double *val,
    int A_rows, int *tmp_off,
    int *csc_row_idx, double *csc_val)
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

static void build_csc(const int *d_row_ptr, const int *d_col_idx, const double *d_val,
                      int A_rows, int A_nnz,
                      int **col_ptr_out, int **row_idx_out, double **val_out)
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

    int *d_csc_row; double *d_csc_val;
    CHECK_CUDA(cudaMalloc(&d_csc_row, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csc_val, A_nnz * sizeof(double)));
    {
        int grid = (A_rows + 255) / 256;
        csc_fill_kernel<<<grid, 256>>>(d_row_ptr, d_col_idx, d_val, A_rows,
                                       d_tmp, d_csc_row, d_csc_val);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[%s] csc\n", g_tag);
    cudaFree(d_col_count);
    cudaFree(d_tmp);
    *col_ptr_out = d_col_ptr;
    *row_idx_out = d_csc_row;
    *val_out = d_csc_val;
}

// ===================== 共用:ESC 合并(排序 + 去重求和 → CSR)=====================

__global__ void esc_finalize_kernel(
    const unsigned long long *red_key, const double *red_val,
    int C_nnz, int *C_col_idx, double *C_val, int *C_row_nnz)
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
static int esc_merge(unsigned long long *d_key, double *d_val, int total, int A_rows,
                     int **dC_col_idx, double **dC_val, int **dC_row_ptr)
{
    thrust::sort_by_key(thrust::device_ptr<unsigned long long>(d_key),
                        thrust::device_ptr<unsigned long long>(d_key + total),
                        thrust::device_ptr<double>(d_val));
    dbg("[%s] sort\n", g_tag);
    unsigned long long *d_rk; double *d_rv;
    CHECK_CUDA(cudaMalloc(&d_rk, (size_t)total * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_rv, (size_t)total * sizeof(double)));
    thrust::pair<thrust::device_ptr<unsigned long long>,
                 thrust::device_ptr<double> > e =
        thrust::reduce_by_key(
            thrust::device_ptr<unsigned long long>(d_key),
            thrust::device_ptr<unsigned long long>(d_key + total),
            thrust::device_ptr<double>(d_val),
            thrust::device_ptr<unsigned long long>(d_rk),
            thrust::device_ptr<double>(d_rv));
    int Cnnz = (int)(e.first - thrust::device_ptr<unsigned long long>(d_rk));
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[%s] reduce\n", g_tag);

    int *c_col; double *c_val; int *c_row_nnz;
    CHECK_CUDA(cudaMalloc(&c_col, (size_t)Cnnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&c_val, (size_t)Cnnz * sizeof(double)));
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
    dbg("[%s] final\n", g_tag);
    *dC_col_idx = c_col; *dC_val = c_val; *dC_row_ptr = c_row_ptr;
    return Cnnz;
}

// 把 (row_ptr|col_idx|val) 三段打包成单块并 D2H(与 Gustavson 版一致)
static void *pack_and_download(int *d_row_ptr, int *d_col_idx, double *d_val,
                               int A_rows, int Cnnz)
{
    size_t rp = (A_rows + 1) * sizeof(int);
    size_t ci = (size_t)Cnnz * sizeof(int);
    size_t vv = (size_t)Cnnz * sizeof(double);
    size_t rp_a = (rp + 3) & ~3;
    size_t ci_a = (ci + 3) & ~3;
    size_t total = rp_a + ci_a + vv;
    void *db;
    CHECK_CUDA(cudaMalloc(&db, total));
    char *base = (char*)db;
    CHECK_CUDA(cudaMemcpy(base, d_row_ptr, rp, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(base + rp_a, d_col_idx, ci, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(base + rp_a + ci_a, d_val, vv, cudaMemcpyDeviceToDevice));
    dbg("[%s] pack\n", g_tag);
    void *hb = nullptr;
    CHECK_CUDA(pinned_d2h_alloc(&hb, total));
    CHECK_CUDA(cudaMemcpy(hb, db, total, cudaMemcpyDeviceToHost));
    dbg("[%s] d2h\n", g_tag);
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
    const int *csr_row_ptr, const int *csr_col_idx, const double *csr_val,
    const int *csc_col_ptr, const int *csc_row_idx, const double *csc_val,
    int A_rows, const int *off,
    unsigned long long *key, double *val)
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
        double a_ik = csc_val[pi];
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
    g_tag = "outer"; dbg("[outer] start\n");
    size_t rp = (A_rows + 1) * sizeof(int);
    size_t ci = A_nnz * sizeof(int);
    size_t vv = A_nnz * sizeof(double);
    size_t totalA = rp + ci + vv;
    void *dA; CHECK_CUDA(cudaMalloc(&dA, totalA));
    CHECK_CUDA(cudaMemcpy(dA, A_buffer, totalA, cudaMemcpyHostToDevice));
    dbg("[outer] h2d\n");
    char *b = (char*)dA;
    int *d_rp = (int*)b; int *d_ci = (int*)(b + rp); double *d_v = (double*)(b + rp + ci);

    int *d_csc_cp, *d_csc_ri; double *d_csc_v;
    build_csc(d_rp, d_ci, d_v, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_v);   // -> [outer] csc

    int *d_ub; CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    count_outer_kernel<<<(A_rows + 255) / 256, 256>>>(d_rp, d_csc_cp, A_rows, d_ub);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[outer] count\n");

    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total; CHECK_CUDA(cudaMemcpy(&total, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[outer] scan\n");

    unsigned long long *d_key; double *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total * sizeof(double)));
    expand_outer_kernel<<<A_rows, 256>>>(d_rp, d_ci, d_v, d_csc_cp, d_csc_ri, d_csc_v,
                                         A_rows, d_off, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[outer] expand\n");

    int *c_col; double *c_val; int *c_rp;
    int Cnnz = esc_merge(d_key, d_val, total, A_rows, &c_col, &c_val, &c_rp);   // -> sort/reduce/final

    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);        // -> pack/d2h
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
    const int *csc_col_ptr, const int *csc_row_idx, const double *csc_val,
    int A_rows, const int *off,
    unsigned long long *key, double *val)
{
    int j = blockIdx.x;
    if (j >= A_rows) return;
    __shared__ int pos;
    if (threadIdx.x == 0) pos = off[j];
    __syncthreads();

    int js = csc_col_ptr[j], je = csc_col_ptr[j + 1];   // 列 j 的 k 们
    for (int p = js + threadIdx.x; p < je; p += blockDim.x) {
        int k = csc_row_idx[p];
        double a_kj = csc_val[p];
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
    g_tag = "colw"; dbg("[colw] start\n");
    size_t rp = (A_rows + 1) * sizeof(int);
    size_t ci = A_nnz * sizeof(int);
    size_t vv = A_nnz * sizeof(double);
    size_t totalA = rp + ci + vv;
    void *dA; CHECK_CUDA(cudaMalloc(&dA, totalA));
    CHECK_CUDA(cudaMemcpy(dA, A_buffer, totalA, cudaMemcpyHostToDevice));
    dbg("[colw] h2d\n");
    char *b = (char*)dA;
    int *d_rp = (int*)b; int *d_ci = (int*)(b + rp); double *d_v = (double*)(b + rp + ci);

    int *d_csc_cp, *d_csc_ri; double *d_csc_v;
    build_csc(d_rp, d_ci, d_v, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_v);   // -> [colw] csc

    int *d_ub; CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    count_colwise_kernel<<<(A_rows + 255) / 256, 256>>>(d_csc_cp, d_csc_ri, A_rows, d_ub);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[colw] count\n");

    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total; CHECK_CUDA(cudaMemcpy(&total, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[colw] scan\n");

    unsigned long long *d_key; double *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total * sizeof(double)));
    expand_colwise_kernel<<<A_rows, 256>>>(d_csc_cp, d_csc_ri, d_csc_v, A_rows, d_off, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[colw] expand\n");

    int *c_col; double *c_val; int *c_rp;
    int Cnnz = esc_merge(d_key, d_val, total, A_rows, &c_col, &c_val, &c_rp);   // -> sort/reduce/final

    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);        // -> pack/d2h
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
                                                   const double *A_val, int A_rows,
                                                   const int *row_off,
                                                   unsigned long long *key, double *val);

// 归并点积:row_i(csr, 按 col 有序) ∩ col_j(csc, 按 row 有序)
__device__ __forceinline__ double merge_dot(
    const int *csr_ci, const double *csr_v, int rs, int re,
    const int *csc_ri, const double *csc_v, int cs, int ce)
{
    double dot = 0.0f;
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
    const int *csr_rp, const int *csr_ci, const double *csr_v,
    const int *csc_cp, const int *csc_ri, const double *csc_v,
    int A_rows, const int *C_rp, const int *C_ci, double *C_val)
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
    g_tag = "inner"; dbg("[inner] start\n");
    size_t rp = (A_rows + 1) * sizeof(int);
    size_t ci = A_nnz * sizeof(int);
    size_t vv = A_nnz * sizeof(double);
    size_t totalA = rp + ci + vv;
    void *dA; CHECK_CUDA(cudaMalloc(&dA, totalA));
    CHECK_CUDA(cudaMemcpy(dA, A_buffer, totalA, cudaMemcpyHostToDevice));
    dbg("[inner] h2d\n");
    char *b = (char*)dA;
    int *d_rp = (int*)b; int *d_ci = (int*)(b + rp); double *d_v = (double*)(b + rp + ci);

    int *d_csc_cp, *d_csc_ri; double *d_csc_v;
    build_csc(d_rp, d_ci, d_v, A_rows, A_nnz, &d_csc_cp, &d_csc_ri, &d_csc_v);   // -> [inner] csc

    // ---- 符号阶段:ESC 展开+排序+去重,只取结构 ----
    int *d_ub; CHECK_CUDA(cudaMalloc(&d_ub, A_rows * sizeof(int)));
    count_intermediates_kernel<<<(A_rows + 255) / 256, 256>>>(d_rp, d_ci, A_rows, d_ub);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[inner] count\n");
    int *d_off; CHECK_CUDA(cudaMalloc(&d_off, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_off, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(d_ub),
                           thrust::device_ptr<int>(d_ub + A_rows),
                           thrust::device_ptr<int>(d_off + 1));
    int total; CHECK_CUDA(cudaMemcpy(&total, d_off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[inner] scan\n");
    unsigned long long *d_key; double *d_val;
    CHECK_CUDA(cudaMalloc(&d_key, (size_t)total * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_val, (size_t)total * sizeof(double)));
    expand_intermediates_kernel<<<A_rows, 256>>>(d_rp, d_ci, d_v, A_rows, d_off, d_key, d_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[inner] expand\n");
    int *c_col; double *c_val; int *c_rp;
    int Cnnz = esc_merge(d_key, d_val, total, A_rows, &c_col, &c_val, &c_rp);   // -> sort/reduce/final
    cudaFree(d_ub); cudaFree(d_off); cudaFree(d_key); cudaFree(d_val);

    // ---- 数值阶段(内积本体):逐元素归并 row_i 与 col_j,重算 c_val ----
    inner_numeric_kernel<<<A_rows, 256>>>(d_rp, d_ci, d_v,
                                          d_csc_cp, d_csc_ri, d_csc_v,
                                          A_rows, c_rp, c_col, c_val);
    CHECK_CUDA(cudaDeviceSynchronize());
    dbg("[inner] numeric\n");

    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);        // -> pack/d2h
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;

    cudaFree(dA); cudaFree(d_csc_cp); cudaFree(d_csc_ri); cudaFree(d_csc_v);
    cudaFree(c_col); cudaFree(c_val); cudaFree(c_rp);
}

// ==========================================================================
//  A·Aᵀ 上三角 SpGEMM  (ESC, 只算 i≤j, 利用 C=A·Aᵀ 对称)
// --------------------------------------------------------------------------
//  C[i,j] = Σ_k A[i,k]·A[j,k]   (行 i · 行 j),   对称 ⇒ 只算上三角 i≤j。
//  4 种公式展开的是【同一批】中间项 {(i,j,k) : i≤j, A[i,k]≠0, A[j,k]≠0},
//  只是外层并行轴不同;合并阶段(sort+reduce)共用 esc_merge()。
//    · 外积    : axis = k,col_k 内 a≤b 对 → 自然 i≤j;count=c_k(c_k+1)/2 精确,无需 compact
//    · Gustavson: axis = i,filter j≥i;count 用上界 + 紧凑写 + compact
//    · 列向    : axis = j,filter i≤j;count 用上界 + 紧凑写 + compact
//    · 内积    : ESC 符号(复用 Gustavson 展开)+ 数值阶段逐元素 row_i·row_j
// ==========================================================================

// thrust device_ptr 的简写,避免每行都写一长串
static inline thrust::device_ptr<int> dpi(int *p) { return thrust::device_ptr<int>(p); }
static inline thrust::device_ptr<const int> dcpi(const int *p) {
    return thrust::device_ptr<const int>(p);
}
// 把 (row, col) 打包成 64-bit key,row 在高 32 位、col 在低 32 位
static __device__ __forceinline__ unsigned long long pack_key(int row, int col) {
    return ((unsigned long long)row << 32) | (unsigned int)col;
}
// 每个 axis 一块的 grid:一行一块
static int rows_grid(int n) { return (n + 255) / 256; }

// compact 的 device kernel:把每行紧凑写在 off[] slot(带间隙)的条目搬到连续位置
__global__ void att_compact_kernel(const int *off, const int *act, const int *coff,
                                   const unsigned long long *sk, const double *sv,
                                   unsigned long long *dk, double *dv, int A_rows) {
    int ax = blockIdx.x;
    if (ax >= A_rows) return;
    int s = off[ax], n = act[ax], d = coff[ax];
    for (int t = threadIdx.x; t < n; t += blockDim.x) { dk[d + t] = sk[s + t]; dv[d + t] = sv[s + t]; }
}

// AttCtx:一次 att 计算用到的全部 device 缓冲区 + 统一生命周期。
// host 只需 init → (count kernel) → prepare_coo → (expand kernel) → [compact] → esc_merge → free_all。
struct AttCtx {
    void  *dA = nullptr;                              // A 的单块 CSR(row_ptr|col_idx|val)
    int   *csr_rp = nullptr, *csr_ci = nullptr;
    double *csr_val = nullptr;
    int   *csc_cp = nullptr, *csc_ri = nullptr;       // A 的 CSC(build_csc 建的)
    double *csc_val = nullptr;
    int   *row_ub = nullptr;                          // 每行中间项数(上界)
    int   *off = nullptr, *act = nullptr;             // 行偏移 / 实际写出数
    unsigned long long *key = nullptr; double *val = nullptr;   // 中间项 COO
    int    A_rows = 0;

    // 上传 A(H2D)+ 建 CSC。build_csc 会发 [tag] csc 桩。
    void init(void *A_buffer, int rows, int nnz) {
        A_rows = rows;
        size_t rp = (rows + 1) * sizeof(int), ci = nnz * sizeof(int),
               vv = nnz * sizeof(double);
        size_t ci_al = ALIGN8(rp + ci);              // val 8 对齐(与 read_matrix_market 的 A 布局一致)
        size_t totalA = ci_al + vv;
        CHECK_CUDA(cudaMalloc(&dA, totalA));
        CHECK_CUDA(cudaMemcpy(dA, A_buffer, totalA, cudaMemcpyHostToDevice));
        char *b = (char*)dA;
        csr_rp  = (int*)b;
        csr_ci  = (int*)(b + rp);
        csr_val = (double*)(b + ci_al);              // 8 对齐(修 double 访问 misaligned)
        build_csc(csr_rp, csr_ci, csr_val, rows, nnz, &csc_cp, &csc_ri, &csc_val);
        CHECK_CUDA(cudaMalloc(&row_ub, rows * sizeof(int)));
    }

    // count kernel 填好 row_ub 后调用:scan 得 off、读 total_ub、申请 key/val/act。返回 total_ub。
    int prepare_coo() {
        CHECK_CUDA(cudaMalloc(&off, (A_rows + 1) * sizeof(int)));
        CHECK_CUDA(cudaMemset(off, 0, sizeof(int)));
        thrust::inclusive_scan(dpi(row_ub), dpi(row_ub + A_rows), dpi(off + 1));
        int total_ub;
        CHECK_CUDA(cudaMemcpy(&total_ub, off + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMalloc(&key, (size_t)total_ub * sizeof(unsigned long long)));
        CHECK_CUDA(cudaMalloc(&val, (size_t)total_ub * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&act, A_rows * sizeof(int)));
        return total_ub;
    }

    // expand 后调用:各行紧凑写在 off[] slot(有间隙),压成连续。返回 total_actual。
    int compact() {
        int *coff;
        CHECK_CUDA(cudaMalloc(&coff, (A_rows + 1) * sizeof(int)));
        CHECK_CUDA(cudaMemset(coff, 0, sizeof(int)));
        thrust::inclusive_scan(dcpi(act), dcpi(act + A_rows), dpi(coff + 1));
        int total;
        CHECK_CUDA(cudaMemcpy(&total, coff + A_rows, sizeof(int), cudaMemcpyDeviceToHost));
        unsigned long long *nkey; double *nval;
        CHECK_CUDA(cudaMalloc(&nkey, (size_t)total * sizeof(unsigned long long)));
        CHECK_CUDA(cudaMalloc(&nval, (size_t)total * sizeof(double)));
        att_compact_kernel<<<A_rows, 256>>>(off, act, coff, key, val, nkey, nval, A_rows);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(key, nkey, (size_t)total * sizeof(unsigned long long), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(val, nval, (size_t)total * sizeof(double), cudaMemcpyDeviceToDevice));
        cudaFree(coff); cudaFree(nkey); cudaFree(nval);
        return total;
    }

    void free_all() {
        cudaFree(dA); cudaFree(csc_cp); cudaFree(csc_ri); cudaFree(csc_val);
        cudaFree(row_ub); cudaFree(off); cudaFree(act); cudaFree(key); cudaFree(val);
    }
};

// ---- 外积 att ----
__global__ void att_outer_count(const int *csc_cp, int A_rows, int *ub) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= A_rows) return;
    long long c = csc_cp[k + 1] - csc_cp[k];
    ub[k] = (int)(c * (c + 1) / 2);
}
__global__ void att_outer_expand(const int *csc_cp, const int *csc_ri, const double *csc_v,
                                 int A_rows, const int *off,
                                 unsigned long long *key, double *val) {
    int k = blockIdx.x;
    if (k >= A_rows) return;
    __shared__ int pos;
    if (threadIdx.x == 0) pos = off[k];
    __syncthreads();
    int cs = csc_cp[k], ce = csc_cp[k + 1];
    for (int a = cs + threadIdx.x; a < ce; a += blockDim.x) {
        int i = csc_ri[a]; double va = csc_v[a];
        for (int b = a; b < ce; b++) {
            int j = csc_ri[b];
            int lo = i < j ? i : j, hi = i < j ? j : i;   // 强制上三角(CSC 列内不一定有序)
            int slot = atomicAdd(&pos, 1);
            key[slot] = pack_key(lo, hi);
            val[slot] = va * csc_v[b];
        }
    }
}

// ---- Gustavson att(axis=i,filter j>=i) ----
__global__ void att_gust_count(const int *csr_rp, const int *csr_ci, const int *csc_cp,
                               int A_rows, int *ub) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    long long s = 0;
    for (int p = csr_rp[i]; p < csr_rp[i + 1]; p++) { int k = csr_ci[p]; s += csc_cp[k + 1] - csc_cp[k]; }
    ub[i] = (int)s;
}
__global__ void att_gust_expand(const int *csr_rp, const int *csr_ci, const double *csr_v,
                                const int *csc_cp, const int *csc_ri, const double *csc_v,
                                int A_rows, const int *off, int *act,
                                unsigned long long *key, double *val) {
    int i = blockIdx.x;
    if (i >= A_rows) return;
    __shared__ int pos;
    if (threadIdx.x == 0) pos = off[i];
    __syncthreads();
    for (int p = csr_rp[i] + threadIdx.x; p < csr_rp[i + 1]; p += blockDim.x) {
        int k = csr_ci[p]; double aik = csr_v[p];
        for (int q = csc_cp[k]; q < csc_cp[k + 1]; q++) {
            int j = csc_ri[q];
            if (j >= i) {
                int slot = atomicAdd(&pos, 1);
                key[slot] = pack_key(i, j);
                val[slot] = aik * csc_v[q];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) act[i] = pos - off[i];
}

// ---- 列向 att(axis=j,filter i<=j)。固定 j,k 取 CSR row j(A[j,k]≠0) ----
__global__ void att_colw_count(const int *csr_rp, const int *csr_ci, const int *csc_cp,
                               int A_rows, int *ub) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= A_rows) return;
    long long s = 0;
    for (int p = csr_rp[j]; p < csr_rp[j + 1]; p++) { int k = csr_ci[p]; s += csc_cp[k + 1] - csc_cp[k]; }
    ub[j] = (int)s;
}
__global__ void att_colw_expand(const int *csr_rp, const int *csr_ci, const double *csr_v,
                                const int *csc_cp, const int *csc_ri, const double *csc_v,
                                int A_rows, const int *off, int *act,
                                unsigned long long *key, double *val) {
    int j = blockIdx.x;
    if (j >= A_rows) return;
    __shared__ int pos;
    if (threadIdx.x == 0) pos = off[j];
    __syncthreads();
    for (int p = csr_rp[j] + threadIdx.x; p < csr_rp[j + 1]; p += blockDim.x) {
        int k = csr_ci[p]; double ajk = csr_v[p];
        for (int q = csc_cp[k]; q < csc_cp[k + 1]; q++) {
            int i = csc_ri[q];
            if (i <= j) {
                int slot = atomicAdd(&pos, 1);
                key[slot] = pack_key(i, j);
                val[slot] = csc_v[q] * ajk;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) act[j] = pos - off[j];
}

// ---- 内积 att 数值:row_i · row_j ----
__device__ __forceinline__ double rowrow_dot(const int *ci, const double *v,
                                            int si, int ei, int sj, int ej) {
    double d = 0.0f; int p = si, q = sj;
    while (p < ei && q < ej) {
        if (ci[p] == ci[q]) { d += v[p] * v[q]; p++; q++; }
        else if (ci[p] < ci[q]) p++; else q++;
    }
    return d;
}
__global__ void att_inner_numeric(const int *csr_rp, const int *csr_ci, const double *csr_v,
                                  int A_rows, const int *C_rp, const int *C_ci, double *C_val) {
    int i = blockIdx.x;
    if (i >= A_rows) return;
    int rs = csr_rp[i], re = csr_rp[i + 1];
    for (int t = C_rp[i] + threadIdx.x; t < C_rp[i + 1]; t += blockDim.x) {
        int j = C_ci[t];
        C_val[t] = rowrow_dot(csr_ci, csr_v, rs, re, csr_rp[j], csr_rp[j + 1]);
    }
}

// (公共 H2D/CSC/scan/compact 逻辑已并入上面的 AttCtx)

// ---- 外积 att host ----
void spgemm_att_outer(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                      void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    g_tag = "atto"; dbg("[atto] start\n");
    AttCtx ctx;
    ctx.init(A_buffer, A_rows, A_nnz);                 // H2D + CSC(build_csc 发 [atto] csc)
    dbg("[atto] h2d\n");

    att_outer_count<<<rows_grid(A_rows), 256>>>(ctx.csc_cp, A_rows, ctx.row_ub);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[atto] count\n");
    int total = ctx.prepare_coo(); dbg("[atto] scan\n");
    att_outer_expand<<<A_rows, 256>>>(ctx.csc_cp, ctx.csc_ri, ctx.csc_val,
                                      A_rows, ctx.off, ctx.key, ctx.val);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[atto] expand\n");
    // 外积 count 精确(c_k(c_k+1)/2),填满 off[] 无间隙,无需 compact

    int *c_col; double *c_val; int *c_rp;
    int Cnnz = esc_merge(ctx.key, ctx.val, total, A_rows, &c_col, &c_val, &c_rp);
    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;
    ctx.free_all(); cudaFree(c_col); cudaFree(c_val); cudaFree(c_rp);
}

// ---- Gustavson att host(axis=i,filter j≥i) ----
void spgemm_att_gust(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    g_tag = "attg"; dbg("[attg] start\n");
    AttCtx ctx;
    ctx.init(A_buffer, A_rows, A_nnz);
    dbg("[attg] h2d\n");

    att_gust_count<<<rows_grid(A_rows), 256>>>(ctx.csr_rp, ctx.csr_ci, ctx.csc_cp, A_rows, ctx.row_ub);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[attg] count\n");
    ctx.prepare_coo(); dbg("[attg] scan\n");
    att_gust_expand<<<A_rows, 256>>>(ctx.csr_rp, ctx.csr_ci, ctx.csr_val,
                                     ctx.csc_cp, ctx.csc_ri, ctx.csc_val,
                                     A_rows, ctx.off, ctx.act, ctx.key, ctx.val);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[attg] expand\n");
    int total = ctx.compact(); dbg("[attg] compact\n");   // filter 后行间有间隙,压紧

    int *c_col; double *c_val; int *c_rp;
    int Cnnz = esc_merge(ctx.key, ctx.val, total, A_rows, &c_col, &c_val, &c_rp);
    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;
    ctx.free_all(); cudaFree(c_col); cudaFree(c_val); cudaFree(c_rp);
}

// ---- 列向 att host(axis=j,filter i≤j) ----
void spgemm_att_colw(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    g_tag = "attc"; dbg("[attc] start\n");
    AttCtx ctx;
    ctx.init(A_buffer, A_rows, A_nnz);
    dbg("[attc] h2d\n");

    att_colw_count<<<rows_grid(A_rows), 256>>>(ctx.csr_rp, ctx.csr_ci, ctx.csc_cp, A_rows, ctx.row_ub);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[attc] count\n");
    ctx.prepare_coo(); dbg("[attc] scan\n");
    att_colw_expand<<<A_rows, 256>>>(ctx.csr_rp, ctx.csr_ci, ctx.csr_val,
                                     ctx.csc_cp, ctx.csc_ri, ctx.csc_val,
                                     A_rows, ctx.off, ctx.act, ctx.key, ctx.val);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[attc] expand\n");
    int total = ctx.compact(); dbg("[attc] compact\n");

    int *c_col; double *c_val; int *c_rp;
    int Cnnz = esc_merge(ctx.key, ctx.val, total, A_rows, &c_col, &c_val, &c_rp);
    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;
    ctx.free_all(); cudaFree(c_col); cudaFree(c_val); cudaFree(c_rp);
}

// ---- 内积 att host(ESC 符号 + 逐元素 row_i·row_j 数值) ----
void spgemm_att_inner(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                      void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz) {
    g_tag = "atti"; dbg("[atti] start\n");
    AttCtx ctx;
    ctx.init(A_buffer, A_rows, A_nnz);
    dbg("[atti] h2d\n");

    // 符号阶段:复用 Gustavson 展开(filter i≤j)+ compact → 上三角结构 (c_col 已按 (i,j) 有序)
    att_gust_count<<<rows_grid(A_rows), 256>>>(ctx.csr_rp, ctx.csr_ci, ctx.csc_cp, A_rows, ctx.row_ub);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[atti] count\n");
    ctx.prepare_coo(); dbg("[atti] scan\n");
    att_gust_expand<<<A_rows, 256>>>(ctx.csr_rp, ctx.csr_ci, ctx.csr_val,
                                     ctx.csc_cp, ctx.csc_ri, ctx.csc_val,
                                     A_rows, ctx.off, ctx.act, ctx.key, ctx.val);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[atti] expand\n");
    int total = ctx.compact(); dbg("[atti] compact\n");
    int *c_col; double *c_val; int *c_rp;
    int Cnnz = esc_merge(ctx.key, ctx.val, total, A_rows, &c_col, &c_val, &c_rp);  // c_val 丢弃
    dbg("[atti] symbolic\n");

    // 数值阶段:对每个 (i,j) 独立归并 row_i 与 row_j(C_val 覆盖)
    att_inner_numeric<<<A_rows, 256>>>(ctx.csr_rp, ctx.csr_ci, ctx.csr_val,
                                       A_rows, c_rp, c_col, c_val);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[atti] numeric\n");

    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;
    ctx.free_all(); cudaFree(c_col); cudaFree(c_val); cudaFree(c_rp);
}

// ==========================================================================
//  ATT hash SPA:C = A·Aᵀ 上三角(对称,只算 j≥i)+ 对称 mirror。
// --------------------------------------------------------------------------
//  仿 Gustavson att(axis=i):row i 的贡献来自 A 的【列 k】(CSC)——对每个 k∈A[i,:],
//  列 k 给出 (j, A[j][k]);累加 A[i][k]·A[j][k] 到 C[i][j]。filter j≥i = 上三角。
//  与 ESC att 的区别:不 expand+全局 sort,而用【SMEM hash 累加器】(atomicCAS 插 j + atomicAdd
//  累值)O(1) 去重;行内并行(256 线程)。提取无序 → 全局 sort → 上三角 CSR → mirror 成全对称。
//  长行 hash(本 kernel)/ 短行小表:host 按 flop_ub 分 bin,小行小表(高 occupancy)。
// ==========================================================================
#define ATT_HT_CAP 16384          // 单行 hash 表最大槽位(→ col+val = 196KB,需 opt-in)
#define ATT_HASH_BLOCK 256

// (row,bucket) 一块:SMEM hash 累加 j≥i 的 (A[i][k]·A[j][k])。提取无序 (i,j,val) 到 off[i] 区。
__global__ void att_hash_spa_kernel(
    const int *csr_rp, const int *csr_ci, const double *csr_v,
    const int *csc_cp, const int *csc_ri, const double *csc_v,
    int A_rows, const int *bucket_rows, int n_in_bucket, int ht_size,
    const int *off, unsigned long long *key, double *val, int *act, int *overflow_flag)
{
    int idx = blockIdx.x;
    if (idx >= n_in_bucket) return;
    int i = bucket_rows[idx];
    int tid = threadIdx.x;
    int mask = ht_size - 1;

    extern __shared__ __align__(8) int smem[];
    int   *sh_col = smem;                         // [ht_size]
    double *sh_val = (double*)(smem + ht_size);   // [ht_size]
    for (int s = tid; s < ht_size; s += ATT_HASH_BLOCK) { sh_col[s] = -1; sh_val[s] = 0.0; }
    __syncthreads();

    int rs = csr_rp[i], re = csr_rp[i + 1];
    for (int p = rs + tid; p < re; p += ATT_HASH_BLOCK) {     // 并行 k(每线程 own k)
        int k = csr_ci[p];
        double a_ik = csr_v[p];
        int cs = csc_cp[k], ce = csc_cp[k + 1];               // 列 k 的 (j, A[j][k])
        for (int q = cs; q < ce; q++) {                       // j 串行(MVP;可改 group 并行)
            int j = csc_ri[q];
            if (j < i) continue;                              // 上三角 filter
            double v = a_ik * csc_v[q];
            unsigned slot = ((unsigned)(j * 2654435761u)) & mask;   // Knuth 乘法 hash
            int probes = 0;
            while (true) {
                int old = atomicCAS(&sh_col[slot], -1, j);
                if (old == -1 || old == j) { atomicAdd(&sh_val[slot], v); break; }
                slot = (slot + 1) & mask;
                if (++probes >= ht_size) { atomicExch(overflow_flag, 1); break; }   // 溢出
            }
        }
    }
    __syncthreads();

    // extract 无序项 → key/val(off[i] 起,act[i] 计数)
    __shared__ int cnt;
    if (tid == 0) cnt = 0;
    __syncthreads();
    int base = off[i];
    for (int s = tid; s < ht_size; s += ATT_HASH_BLOCK) {
        int j = sh_col[s];
        if (j >= 0) {
            int pos = base + atomicAdd(&cnt, 1);
            key[pos] = ((unsigned long long)i << 32) | (unsigned int)j;
            val[pos] = sh_val[s];
        }
    }
    __syncthreads();
    if (tid == 0) act[i] = cnt;
}

// 上三角 CSR → 全对称 COO:每条 (i,j) i<j 复制成 (i,j)+(j,i);对角 (i,i) 一份。fcount 计实际。
__global__ void att_mirror_coo_kernel(const int *u_rp, const int *u_ci, const double *u_v,
    int n, unsigned long long *fkey, double *fval, int *fcount)
{
    int i = blockIdx.x;
    if (i >= n) return;
    for (int p = u_rp[i] + threadIdx.x; p < u_rp[i + 1]; p += blockDim.x) {
        int j = u_ci[p]; double v = u_v[p];
        int pos = atomicAdd(fcount, 1);
        fkey[pos] = ((unsigned long long)i << 32) | (unsigned int)j;
        fval[pos] = v;
        if (i != j) {                                          // 非对角:补 (j,i)
            int pos2 = atomicAdd(fcount, 1);
            fkey[pos2] = ((unsigned long long)j << 32) | (unsigned int)i;
            fval[pos2] = v;
        }
    }
}

// 从排序后的 key(row<<32|col)建 CSR row_ptr:每 entry atomicAdd 到其 row 的计数,后 scan。
__global__ void att_build_rowptr_count_kernel(const unsigned long long *key, int total, int n, int *row_cnt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) return;
    int r = (int)(key[k] >> 32);
    atomicAdd(&row_cnt[r], 1);
}

// 拆 key 低 32 位 → col(thrust::transform 的 device-lambda 替代,免 --extended-lambda)
__global__ void att_unpack_col_kernel(const unsigned long long *key, int *col, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    col[k] = (int)(unsigned int)key[k];
}

// ---- ATT hash host ----
void spgemm_att_hash(void *A_buffer, int A_rows, int A_cols, int A_nnz,
                     void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    g_tag = "atth"; dbg("[atth] start\n");
    AttCtx ctx;
    ctx.init(A_buffer, A_rows, A_nnz);                 // H2D + CSC
    dbg("[atth] h2d\n");

    // ① 每行 flop_ub(= Σ_k nnz(列k),上界 distinct j≥i)
    att_gust_count<<<(A_rows + 255) / 256, 256>>>(ctx.csr_rp, ctx.csr_ci, ctx.csc_cp, A_rows, ctx.row_ub);
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[atth] count\n");
    int total_ub = ctx.prepare_coo();                  // scan row_ub → off + total_ub;申请 key/val/act
    dbg("[atth] scan (total_ub=%d)\n", total_ub);

    // ② host 端按 ht 分 bin:每行 ht = next_pow2(row_ub) clamp [16, ATT_HT_CAP];>CAP → CAP(可能溢出)
    std::vector<int> h_row_ub(A_rows);
    CHECK_CUDA(cudaMemcpy(h_row_ub.data(), ctx.row_ub, A_rows * sizeof(int), cudaMemcpyDeviceToHost));
    int *d_overflow; CHECK_CUDA(cudaMalloc(&d_overflow, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_overflow, 0, sizeof(int)));
    CHECK_CUDA(cudaFuncSetAttribute(att_hash_spa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    ATT_HT_CAP * (int)(sizeof(int) + sizeof(double))));
    auto row_ht = [&](int ub) -> int {                       // next_pow2(ub) clamp [16, ATT_HT_CAP]
        if (ub <= 16) return 16;
        if (ub > ATT_HT_CAP) return ATT_HT_CAP;
        int p = ub - 1; p |= p >> 1; p |= p >> 2; p |= p >> 4; p |= p >> 8; p |= p >> 16;
        int np = p + 1; return np < 16 ? 16 : np;
    };
    // 11 个 ht 档(16..16384);每档收集行号,H2D,launch
    for (int bi = 0; bi < 11; bi++) {
        int ht = 16 << bi;
        std::vector<int> rows;
        for (int i = 0; i < A_rows; i++)
            if (h_row_ub[i] > 0 && row_ht(h_row_ub[i]) == ht) rows.push_back(i);
        if (rows.empty()) continue;
        int *d_rows; CHECK_CUDA(cudaMalloc(&d_rows, rows.size() * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_rows, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice));
        size_t smem = (size_t)ht * (sizeof(int) + sizeof(double));
        att_hash_spa_kernel<<<(int)rows.size(), ATT_HASH_BLOCK, smem>>>(
            ctx.csr_rp, ctx.csr_ci, ctx.csr_val, ctx.csc_cp, ctx.csc_ri, ctx.csc_val,
            A_rows, d_rows, (int)rows.size(), ht, ctx.off, ctx.key, ctx.val, ctx.act, d_overflow);
        CHECK_CUDA(cudaGetLastError());
        cudaFree(d_rows);
    }
    CHECK_CUDA(cudaDeviceSynchronize()); dbg("[atth] accumulate\n");
    int overflow; CHECK_CUDA(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));

    // ③ compact(按 act 压紧)→ 连续无序 (i,j,val)→ 全局 sort → 上三角 CSR
    int total_upper = ctx.compact(); dbg("[atth] compact (upper=%d)\n", total_upper);
    thrust::sort_by_key(thrust::device_ptr<unsigned long long>(ctx.key),
                        thrust::device_ptr<unsigned long long>(ctx.key + total_upper),
                        thrust::device_ptr<double>(ctx.val));
    dbg("[atth] sort\n");
    // 上三角 row_ptr(独立 count 数组 → exclusive scan,免 in-place 错位)
    int *u_rp; CHECK_CUDA(cudaMalloc(&u_rp, (A_rows + 1) * sizeof(int)));
    int *u_cnt; CHECK_CUDA(cudaMalloc(&u_cnt, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(u_cnt, 0, A_rows * sizeof(int)));
    att_build_rowptr_count_kernel<<<(total_upper + 255) / 256, 256>>>(ctx.key, total_upper, A_rows, u_cnt);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(u_rp, 0, sizeof(int)));                 // u_rp[0]=0
    thrust::inclusive_scan(thrust::device_ptr<int>(u_cnt), thrust::device_ptr<int>(u_cnt + A_rows),
                           thrust::device_ptr<int>(u_rp + 1));    // exclusive scan → u_rp[1..A_rows]
    // 上三角 col/val(key 低 32 = j)
    int *u_col; double *u_val;
    CHECK_CUDA(cudaMalloc(&u_col, total_upper * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&u_val, total_upper * sizeof(double)));
    att_unpack_col_kernel<<<(total_upper + 255) / 256, 256>>>(ctx.key, u_col, total_upper);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(u_val, ctx.val, total_upper * sizeof(double), cudaMemcpyDeviceToDevice));

    // ④ mirror 上三角 → 全对称 COO(2×−diag)→ sort → full CSR
    int *d_fcount; CHECK_CUDA(cudaMalloc(&d_fcount, sizeof(int))); CHECK_CUDA(cudaMemset(d_fcount, 0, sizeof(int)));
    unsigned long long *fkey; double *fval;
    CHECK_CUDA(cudaMalloc(&fkey, (size_t)2 * total_upper * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&fval, (size_t)2 * total_upper * sizeof(double)));
    att_mirror_coo_kernel<<<A_rows, ATT_HASH_BLOCK>>>(u_rp, u_col, u_val, A_rows, fkey, fval, d_fcount);
    CHECK_CUDA(cudaDeviceSynchronize());
    int Cnnz; CHECK_CUDA(cudaMemcpy(&Cnnz, d_fcount, sizeof(int), cudaMemcpyDeviceToHost));
    dbg("[atth] mirror (full nnz=%d, overflow=%d)\n", Cnnz, overflow);
    thrust::sort_by_key(thrust::device_ptr<unsigned long long>(fkey),
                        thrust::device_ptr<unsigned long long>(fkey + Cnnz),
                        thrust::device_ptr<double>(fval));
    // full CSR row_ptr(独立 count → exclusive scan)
    int *c_rp; CHECK_CUDA(cudaMalloc(&c_rp, (A_rows + 1) * sizeof(int)));
    int *c_cnt; CHECK_CUDA(cudaMalloc(&c_cnt, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(c_cnt, 0, A_rows * sizeof(int)));
    att_build_rowptr_count_kernel<<<(Cnnz + 255) / 256, 256>>>(fkey, Cnnz, A_rows, c_cnt);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(c_rp, 0, sizeof(int)));
    thrust::inclusive_scan(thrust::device_ptr<int>(c_cnt), thrust::device_ptr<int>(c_cnt + A_rows),
                           thrust::device_ptr<int>(c_rp + 1));
    CHECK_CUDA(cudaFree(c_cnt));
    int *c_col; double *c_val;
    CHECK_CUDA(cudaMalloc(&c_col, Cnnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&c_val, Cnnz * sizeof(double)));
    att_unpack_col_kernel<<<(Cnnz + 255) / 256, 256>>>(fkey, c_col, Cnnz);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(c_val, fval, Cnnz * sizeof(double), cudaMemcpyDeviceToDevice));

    *C_buffer_out = pack_and_download(c_rp, c_col, c_val, A_rows, Cnnz);
    *C_rows = A_rows; *C_cols = A_cols; *C_nnz = Cnnz;
    if (overflow) dbg("[atth] ⚠ overflow flag set (some heavy row distinct>ATT_HT_CAP)\n");

    ctx.free_all(); cudaFree(d_overflow); cudaFree(u_rp); cudaFree(u_cnt); cudaFree(u_col); cudaFree(u_val);
    cudaFree(d_fcount); cudaFree(fkey); cudaFree(fval); cudaFree(c_rp); cudaFree(c_col); cudaFree(c_val);
}
