#include "spgemm.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    std::cerr<<"CUDA error: "<<cudaGetErrorString(_e)<<" ("<<__FILE__<<":"<<__LINE__<<")\n"; std::exit(1);} } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
    std::cerr<<"cuBLAS error "<<_s<<" ("<<__FILE__<<":"<<__LINE__<<")\n"; std::exit(1);} } while(0)

// cuBLAS dense baseline for C = A·A: densify(A) → cublasDgemm(FP64) → sparsify(C), all GPU.
// Strict FP64 via CUBLAS_PEDANTIC_MATH (disables TF32/tensor cores) — the "strong dense" reference.

// densify: scatter sparse CSR → dense row-major N×N。每 thread 一个 entry。
__global__ void densify_kernel(const int *row_ptr, const int *col_idx,
                               const double *val, double *dense, int N, int nnz) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nnz) return;
    int lo = 0, hi = N;
    while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (row_ptr[mid] <= t) lo = mid; else hi = mid - 1; }
    dense[(size_t)lo * N + col_idx[t]] = val[t];
}

__global__ void count_row_nnz_kernel(const double *dense, int *rowcnt, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const double *r = dense + (size_t)row * N;
    int c = 0;
    for (int j = 0; j < N; j++) if (fabs(r[j]) > 1e-30) c++;
    rowcnt[row] = c;
}

__global__ void sparsify_kernel(const double *dense, const int *row_ptr,
                                int *col_out, double *val_out, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const double *r = dense + (size_t)row * N;
    int pos = row_ptr[row];
    for (int j = 0; j < N; j++) { double v = r[j]; if (fabs(v) > 1e-30) { col_out[pos]=j; val_out[pos]=v; pos++; } }
}

static bool read_host_csr(const char *path, std::vector<int> &row_ptr,
                          std::vector<int> &col_idx, std::vector<double> &val,
                          int &rows, int &cols, int &nnz) {
    void *buf=nullptr; int *rp=nullptr,*ci=nullptr; double *vv=nullptr;
    if (!read_matrix_market(path,&buf,&rp,&ci,&vv,&rows,&cols,&nnz)) return false;
    row_ptr.assign(rp,rp+rows+1); col_idx.assign(ci,ci+nnz); val.assign(vv,vv+nnz);
    std::free(buf);
    std::cout<<"Read "<<path<<": "<<rows<<" x "<<cols<<", nnz = "<<nnz<<'\n';
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) { std::cerr<<"Usage: "<<argv[0]<<" A.mtx [output.mtx]\n"; return 1; }
    const char *A_path = argv[1];
    const char *out_path = (argc>=3)?argv[2]:"cublas_result.mtx";
    std::vector<int> h_rp,h_ci; std::vector<double> h_val;
    int N=0,Nc=0,nnz=0;
    if (!read_host_csr(A_path,h_rp,h_ci,h_val,N,Nc,nnz)) return 1;
    if (N!=Nc) { std::cerr<<"needs square A\n"; return 1; }

    size_t dense_bytes=(size_t)N*N*sizeof(double);
    int *d_rp,*d_ci; double *d_val,*dA,*dC;
    CHECK_CUDA(cudaMalloc(&d_rp,(N+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ci,(size_t)nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_val,(size_t)nnz*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dA,dense_bytes));
    CHECK_CUDA(cudaMalloc(&dC,dense_bytes));
    CHECK_CUDA(cudaMemcpy(d_rp,h_rp.data(),(N+1)*sizeof(int),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ci,h_ci.data(),(size_t)nnz*sizeof(int),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val,h_val.data(),(size_t)nnz*sizeof(double),cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    // 强制纯 FP64:CUBLAS_PEDANTIC_MATH 关掉 TF32/Tensor Core(慢 ~2×,严格无 TC)。
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    int TPB=256;
    int densify_grid=(nnz+TPB-1)/TPB;
    int row_grid=(N+TPB-1)/TPB;

    cudaEvent_t start,stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // warmup
    { CHECK_CUDA(cudaMemsetAsync(dA,0,dense_bytes));
      densify_kernel<<<densify_grid,TPB>>>(d_rp,d_ci,d_val,dA,N,nnz);
      double alpha=1.0,beta=0.0;
      CHECK_CUBLAS(cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,dA,N,dA,N,&beta,dC,N));
      CHECK_CUDA(cudaDeviceSynchronize()); }

    // timed compute-only: densify + dgemm + sparsify(transfers excluded)
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemsetAsync(dA,0,dense_bytes));
    densify_kernel<<<densify_grid,TPB>>>(d_rp,d_ci,d_val,dA,N,nnz);
    double alpha=1.0,beta=0.0;
    // C = A·A:dA 行-major 两次喂 cublas(列-major 视角下 = Â·Â),输出 dC 行-major 读出即 A·A
    CHECK_CUBLAS(cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&alpha,dA,N,dA,N,&beta,dC,N));

    int *d_rowcnt,*d_Crp;
    CHECK_CUDA(cudaMalloc(&d_rowcnt,(N+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_Crp,(N+1)*sizeof(int)));
    count_row_nnz_kernel<<<row_grid,TPB>>>(dC,d_rowcnt,N);
    int C_nnz=0;
    { std::vector<int> rc(N); CHECK_CUDA(cudaMemcpy(rc.data(),d_rowcnt,N*sizeof(int),cudaMemcpyDeviceToHost));
      for(int x:rc) C_nnz+=x; }
    thrust::exclusive_scan(thrust::device_ptr<int>(d_rowcnt),thrust::device_ptr<int>(d_rowcnt+N),
                           thrust::device_ptr<int>(d_Crp));
    CHECK_CUDA(cudaMemcpy(d_Crp+N,&C_nnz,sizeof(int),cudaMemcpyHostToDevice));
    int *d_Cci; double *d_Cval;
    CHECK_CUDA(cudaMalloc(&d_Cci,(size_t)C_nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_Cval,(size_t)C_nnz*sizeof(double)));
    sparsify_kernel<<<row_grid,TPB>>>(dC,d_Crp,d_Cci,d_Cval,N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,start,stop));
    std::cout<<"C: "<<N<<" x "<<N<<", nnz = "<<C_nnz<<'\n'<<"Kernel time: "<<ms<<" ms\n";

    // d2h+write(outside timed)
    std::vector<int> h_Crp(N+1),h_Cci(C_nnz); std::vector<double> h_Cval(C_nnz);
    CHECK_CUDA(cudaMemcpy(h_Crp.data(),d_Crp,(N+1)*sizeof(int),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_Cci.data(),d_Cci,(size_t)C_nnz*sizeof(int),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_Cval.data(),d_Cval,(size_t)C_nnz*sizeof(double),cudaMemcpyDeviceToHost));
    write_matrix_market(out_path,h_Crp.data(),h_Cci.data(),h_Cval.data(),N,N,C_nnz);

    cublasDestroy(handle);
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_rp)); CHECK_CUDA(cudaFree(d_ci)); CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(d_rowcnt)); CHECK_CUDA(cudaFree(d_Crp));
    CHECK_CUDA(cudaFree(d_Cci)); CHECK_CUDA(cudaFree(d_Cval));
    return 0;
}
