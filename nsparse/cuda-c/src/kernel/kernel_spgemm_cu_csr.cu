#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <nsparse.h>

// NOTE: cuSPARSE 旧 SpGEMM API(cusparseXcsrgemmNnz / cusparseScsrgemm)在 CUDA 11+ 已移除。
// 本文件只保留 sample 无条件需要的 get_spgemm_flop(纯 flop 计数,不依赖 cuSPARSE);
// spgemm_kernel_cu_csr / spgemm_cu_csr(sfDEBUG 答案校验用,已 undef sfDEBUG 关闭)stub 掉。

__global__ void set_intprod_per_row(int *d_arpt, int *d_acol,
                                    const int* __restrict__ d_brpt,
                                    long long int *d_max_row_nz,
                                    int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    int nz_per_row = 0;
    int j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
        nz_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }
    d_max_row_nz[i] = nz_per_row;
}

void get_spgemm_flop(sfCSR *a, sfCSR *b,
                     int M, long long int *flop)
{
    int GS, BS;
    long long int *d_max_row_nz;

    BS = MAX_LOCAL_THREAD_NUM;
    checkCudaErrors(cudaMalloc((void **)&(d_max_row_nz), sizeof(long long int) * M));

    GS = div_round_up(M, BS);
    set_intprod_per_row<<<GS, BS>>>(a->d_rpt, a->d_col,
                                    b->d_rpt,
                                    d_max_row_nz,
                                    M);

    *flop = thrust::reduce(thrust::device, d_max_row_nz, d_max_row_nz + M);
    (*flop) *= 2;
    cudaFree(d_max_row_nz);
}

// stub: cuSPARSE 比较 kernel 已移除,sfDEBUG 关闭时不会被调用。
void spgemm_kernel_cu_csr(sfCSR *a, sfCSR *b, sfCSR *c,
                          cusparseHandle_t *cusparseHandle,
                          cusparseOperation_t *trans_a,
                          cusparseOperation_t *trans_b,
                          cusparseMatDescr_t *descr_a,
                          cusparseMatDescr_t *descr_b)
{
    (void)a; (void)b; (void)c;
    (void)cusparseHandle; (void)trans_a; (void)trans_b; (void)descr_a; (void)descr_b;
    c->nnz = 0;
}

void spgemm_cu_csr(sfCSR *a, sfCSR *b, sfCSR *c)
{
    (void)a; (void)b;
    c->nnz = 0;
}
