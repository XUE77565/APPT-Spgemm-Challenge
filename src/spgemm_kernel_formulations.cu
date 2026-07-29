#include "spgemm.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

// 三种 SpGEMM 公式(自乘 C=A·A)的 ESC 对照实现:外积/列向/内积,合并阶段共用 esc_merge()。

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// profiling:当前方法标签(各 host 入口设置),parser 按相邻 [tag] phase 桩相减得各阶段耗时。
static const char *g_tag = "?";

// CSR -> CSC(自包含,不依赖 cusparse 版本)

__global__ void csc_count_kernel(const int *col_idx, int nnz, int *col_count)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nnz) return;
    atomicAdd(&col_count[col_idx[t]], 1);
}

// 逐行把 (i, col_idx[p], val[p]) 散到 CSC:用 tmp_off[col] 作每列写指针(并行散 → 列内无序)
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

// 给每个元素打 composite key = (col<<32)|row,供列内排序(lower_bound-based kernel 如 ATT merge3 需要列内有序)
__global__ void csc_make_sortkey_kernel(const int *col_ptr, int n, const int *csc_row,
                                        unsigned long long *sortkey)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    for (int p = col_ptr[c]; p < col_ptr[c + 1]; p++)
        sortkey[p] = ((unsigned long long)c << 32) | (unsigned int)csc_row[p];
}

void build_csc(const int *d_row_ptr, const int *d_col_idx, const double *d_val,
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
    // 列内排序(row_idx 升序):lower_bound-based kernel(ATT merge3)需列内有序;ESC/hash 不依赖列内序,排序无害。
    {
        unsigned long long *d_sortkey;
        CHECK_CUDA(cudaMalloc(&d_sortkey, (size_t)A_nnz * sizeof(unsigned long long)));
        csc_make_sortkey_kernel<<<(A_rows + 255) / 256, 256>>>(d_col_ptr, A_rows, d_csc_row, d_sortkey);
        thrust::sort_by_key(thrust::device_ptr<unsigned long long>(d_sortkey),
                            thrust::device_ptr<unsigned long long>(d_sortkey + A_nnz),
                            thrust::make_zip_iterator(thrust::make_tuple(
                                thrust::device_ptr<int>(d_csc_row),
                                thrust::device_ptr<double>(d_csc_val))));
        cudaFree(d_sortkey);
    }
    dbg("[%s] csc\n", g_tag);
    cudaFree(d_col_count);
    cudaFree(d_tmp);
    *col_ptr_out = d_col_ptr;
    *row_idx_out = d_csc_row;
    *val_out = d_csc_val;
}

// 共用:ESC 合并(排序 + 去重求和 → CSR)

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

// ESC 合并:输入 d_key[total]/d_val[total],输出 *dC_col_idx/*dC_val/*dC_row_ptr,返回 Cnnz。
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

// 外积 (outer product, 外层 = k)

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

// 列向 (column-wise, 外层 = j):以输出列 j 为外层并行轴,访存全走 CSC。

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

// 逐元素内积 (inner product):符号阶段借用 ESC 拿结构,数值阶段逐元素归并点积重算(比 ESC 多一遍数值归并)。

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

    // 符号阶段:ESC 展开+排序+去重,只取结构
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

    // 数值阶段(内积本体):逐元素归并 row_i 与 col_j,重算 c_val
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

// A·Aᵀ 上三角 SpGEMM(ESC,只算 i≤j,利用对称):4 种公式同一批中间项,合并阶段共用 esc_merge()。

// thrust device_ptr 的简写,避免每行都写一长串
static inline thrust::device_ptr<int> dpi(int *p) { return thrust::device_ptr<int>(p); }
static inline thrust::device_ptr<const int> dcpi(const int *p) {
    return thrust::device_ptr<const int>(p);
}
