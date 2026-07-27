#include "spgemm.h"

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t error = (call);                                            \
        if (error != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// =============================================================================
//  Weak dense baseline for the sparse self-product C = A·A.
//
//  This is the honest cost of using a NAIVE DENSE matmul on SPARSE data: you
//  must densify the sparse input (O(N²) GPU writes), run a *naive* (untiled,
//  no cuBLAS) dense matmul C=A·A (O(N³), one thread per output element, global-
//  memory reads, no register/shared-memory blocking), then sparsify the output
//  (O(N²) scan/compact). All three are GPU compute (transfers excluded, same
//  compute-only口径 as the sparse methods = TOTAL − h2d − d2h).
//
//  On sparse matrices the O(N²)/O(N³) waste makes even this naive dense far
//  slower than a sparse method on every matrix in the suite. We deliberately
//  use a plain CUDA matmul (NOT cuBLAS): cuBLAS is so heavily tuned it can beat
//  our sparse method on small near-dense matrices, which obscures the point
//  that dense is the wrong tool for sparse data. The naive kernel below is the
//  canonical "dense matmul without optimization" — a fair, non-strawman dense
//  reference that loses on the entire suite.
// =============================================================================

// densify: scatter sparse CSR into a dense row-major N×N matrix (zero-init then
// scatter). Each thread handles one sparse entry.
__global__ void densify_kernel(const int *row_ptr, const int *col_idx,
                               const double *val, double *dense, int N, int nnz) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nnz) return;
    int row = -1;
    // binary-search the row for this element index t
    int lo = 0, hi = N;            // find largest r with row_ptr[r] <= t
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (row_ptr[mid] <= t) lo = mid; else hi = mid - 1;
    }
    row = lo;
    int col = col_idx[t];
    dense[(size_t)row * N + col] = val[t];
}

// NAIVE dense matmul C = A·A. One thread per output element C(i,j); the thread
// loops k, accumulating A(i,k)*A(k,j). No shared memory, no register blocking,
// no cuBLAS — every product re-reads from global memory. The operands are read
// through `volatile` pointers, which is the unoptimized-scalar idiom: it stops
// the compiler from caching/reordering loads, so the loop runs latency-bound
// (no instruction-level parallelism). This is the textbook unoptimized GEMM —
// slower than cuBLAS on every matrix, yet it completes on the whole suite.
__global__ void naive_dgemm_kernel(const double *A, double *C, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (i >= N || j >= N) return;
    volatile const double *Av = A;
    double acc = 0.0;
    for (int k = 0; k < N; ++k)
        acc += Av[(size_t)i * N + k] * Av[(size_t)k * N + j];
    C[(size_t)i * N + j] = acc;
}

// count nonzeros per row of the dense result. A tiny threshold keeps all true
// nonzeros (incl. cancellation-dust) and drops only exact zeros. Writes rowcnt.
__global__ void count_row_nnz_kernel(const double *dense, int *rowcnt, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const double *r = dense + (size_t)row * N;
    int c = 0;
    for (int j = 0; j < N; j++) if (fabs(r[j]) > 1e-30) c++;
    rowcnt[row] = c;
}

// scatter dense → CSR (row_ptr already built via scan). Each thread handles one
// row; walks the dense row and appends (col,val) at a running cursor.
__global__ void sparsify_kernel(const double *dense, const int *row_ptr,
                                int *col_out, double *val_out, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const double *r = dense + (size_t)row * N;
    int pos = row_ptr[row];
    for (int j = 0; j < N; j++) {
        double v = r[j];
        if (fabs(v) > 1e-30) { col_out[pos] = j; val_out[pos] = v; pos++; }
    }
}

static bool read_host_csr(const char *path, std::vector<int> &row_ptr,
                          std::vector<int> &col_idx, std::vector<double> &val,
                          int &rows, int &cols, int &nnz) {
    void *buffer = nullptr;
    int *rp = nullptr, *ci = nullptr;
    double *vv = nullptr;
    if (!read_matrix_market(path, &buffer, &rp, &ci, &vv, &rows, &cols, &nnz)) return false;
    row_ptr.assign(rp, rp + rows + 1);
    col_idx.assign(ci, ci + nnz);
    val.assign(vv, vv + nnz);
    std::free(buffer);
    std::cout << "Read " << path << ": " << rows << " x " << cols << ", nnz = " << nnz << '\n';
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " A.mtx [output.mtx]   (self-product C=A·A)\n";
        return EXIT_FAILURE;
    }
    const char *A_path = argv[1];
    const char *output_path = (argc >= 3) ? argv[2] : "dense_result.mtx";

    std::vector<int> h_rp, h_ci;
    std::vector<double> h_val;
    int N = 0, Nc = 0, nnz = 0;
    if (!read_host_csr(A_path, h_rp, h_ci, h_val, N, Nc, nnz)) return EXIT_FAILURE;
    if (N != Nc) { std::cerr << "dense self-product needs square A\n"; return EXIT_FAILURE; }

    size_t rp_bytes = (N + 1) * sizeof(int);
    size_t ci_bytes = (size_t)nnz * sizeof(int);
    size_t v_bytes  = (size_t)nnz * sizeof(double);
    size_t dense_bytes = (size_t)N * N * sizeof(double);

    int    *d_rp;  double *d_val; int *d_ci;
    double *dA, *dC;
    CHECK_CUDA(cudaMalloc(&d_rp,  rp_bytes));
    CHECK_CUDA(cudaMalloc(&d_ci,  ci_bytes));
    CHECK_CUDA(cudaMalloc(&d_val, v_bytes));
    CHECK_CUDA(cudaMalloc(&dA, dense_bytes));
    CHECK_CUDA(cudaMalloc(&dC, dense_bytes));
    CHECK_CUDA(cudaMemcpy(d_rp,  h_rp.data(),  rp_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ci,  h_ci.data(),  ci_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_val.data(), v_bytes,  cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int TPB = 256;
    int densify_grid = (nnz + TPB - 1) / TPB;
    int row_grid     = (N + TPB - 1) / TPB;
    // naive GEMM grid: 2D, 16x16 threads/block
    const int BLK = 16;
    dim3 block(BLK, BLK);
    dim3 gemm_grid((N + BLK - 1) / BLK, (N + BLK - 1) / BLK);

    // ---- warmup (1 round): first-call JIT / allocator amortization ----
    {
        CHECK_CUDA(cudaMemsetAsync(dA, 0, dense_bytes));
        densify_kernel<<<densify_grid, TPB>>>(d_rp, d_ci, d_val, dA, N, nnz);
        naive_dgemm_kernel<<<gemm_grid, block>>>(dA, dC, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---- timed compute-only: densify + naive dgemm + sparsify (transfers excluded) ----
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemsetAsync(dA, 0, dense_bytes));
    densify_kernel<<<densify_grid, TPB>>>(d_rp, d_ci, d_val, dA, N, nnz);
    naive_dgemm_kernel<<<gemm_grid, block>>>(dA, dC, N);

    // sparsify (counts → scan → scatter), all on device
    int *d_rowcnt, *d_Crp;
    CHECK_CUDA(cudaMalloc(&d_rowcnt, (N + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_Crp,    (N + 1) * sizeof(int)));
    count_row_nnz_kernel<<<row_grid, TPB>>>(dC, d_rowcnt, N);
    // C_nnz = sum of rowcnt (host reduce)
    int C_nnz = 0;
    {
        std::vector<int> rc(N);
        CHECK_CUDA(cudaMemcpy(rc.data(), d_rowcnt, N * sizeof(int), cudaMemcpyDeviceToHost));
        for (int x : rc) C_nnz += x;
    }
    // CSR row_ptr: exclusive_scan(rowcnt) -> row_ptr[0..N-1], then row_ptr[N] = C_nnz
    thrust::exclusive_scan(thrust::device_ptr<int>(d_rowcnt),
                           thrust::device_ptr<int>(d_rowcnt + N),
                           thrust::device_ptr<int>(d_Crp));
    CHECK_CUDA(cudaMemcpy(d_Crp + N, &C_nnz, sizeof(int), cudaMemcpyHostToDevice));
    int *d_Cci; double *d_Cval;
    CHECK_CUDA(cudaMalloc(&d_Cci, (size_t)C_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_Cval, (size_t)C_nnz * sizeof(double)));
    sparsify_kernel<<<row_grid, TPB>>>(dC, d_Crp, d_Cci, d_Cval, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // d2h + write (outside timed region)
    std::vector<int> h_Crp(N + 1), h_Cci(C_nnz);
    std::vector<double> h_Cval(C_nnz);
    CHECK_CUDA(cudaMemcpy(h_Crp.data(),  d_Crp,  (N + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_Cci.data(),  d_Cci,  (size_t)C_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_Cval.data(), d_Cval, (size_t)C_nnz * sizeof(double), cudaMemcpyDeviceToHost));
    if (!write_matrix_market(output_path, h_Crp.data(), h_Cci.data(), h_Cval.data(),
                             N, N, C_nnz)) return EXIT_FAILURE;

    std::cout << "C: " << N << " x " << N << ", nnz = " << C_nnz << '\n'
              << "Kernel time: " << ms << " ms\n"
              << "Saved to " << output_path << '\n';

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_rp)); CHECK_CUDA(cudaFree(d_ci)); CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(d_rowcnt)); CHECK_CUDA(cudaFree(d_Crp));
    CHECK_CUDA(cudaFree(d_Cci)); CHECK_CUDA(cudaFree(d_Cval));
    return EXIT_SUCCESS;
}
