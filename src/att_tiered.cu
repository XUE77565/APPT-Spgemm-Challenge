// src/att_tiered.cu
//  C = A * A^T : three-tier (light/medium/heavy) workload-driven SpGEMM, self-contained.
#include "att_tiered.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cmath>
#include <algorithm>

// error / timing helpers
#define ATT_CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
        fprintf(stderr, "[ATT] CUDA error %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return AttStatus::KernelError; } } while (0)
#define ATT_CUDA_CHECK2(st, lbl) do { cudaError_t _e = cudaGetLastError(); if (_e != cudaSuccess) { \
        fprintf(stderr, "[ATT] launch %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        st = AttStatus::KernelError; lbl; } } while (0)

struct EventPair {
    cudaEvent_t s, e;
    inline void init() { cudaEventCreate(&s); cudaEventCreate(&e); }
    inline void start() { cudaEventRecord(s); }
    inline float stop() { cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0.f; cudaEventElapsedTime(&ms, s, e); return ms; }
    inline void destroy() { cudaEventDestroy(s); cudaEventDestroy(e); }
};

// device helpers
__device__ __forceinline__
int dev_lower_bound(const int* a, int n, int v) {
    int lo = 0, hi = n;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
    return lo;
}

// (i << 32) | j — global sort key for heavy-row compact pairs.
__device__ __forceinline__
uint64_t make_key(int i, int j) { return ((uint64_t)(uint32_t)i << 32) | (uint32_t)j; }
__device__ __forceinline__ int key_row(uint64_t k) { return (int)(uint32_t)(k >> 32); }
__device__ __forceinline__ int key_col(uint64_t k) { return (int)(uint32_t)(k & 0xFFFFFFFFu); }

// open-addressing hash probe (mix avoids pathological strides).
__device__ __forceinline__
uint32_t hash_slot(uint32_t j, uint32_t cap_pow2) {
    uint32_t h = j;
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    return h & (cap_pow2 - 1);
}

// heavy-row task descriptor (used by k_build_tasks and k_heavy)
struct HeavyTask { int row; int64_t begin_work; int64_t end_work; };

// P0 : CSR -> CSC

// thread-per-row: tag each nonzero with its row index
__global__ void k_fill_row_id(const int* row_ptr, int m, int* row_id) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) row_id[p] = i;
}

// build sort keys (col<<32)|row and a value copy
__global__ void k_fill_csc_keys(const int* col_idx, const int* row_id,
                                const float* val, uint64_t* keys, float* copied,
                                int64_t nnz) {
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz) return;
    keys[p]  = make_key(col_idx[p], row_id[p]);   // (col, row)
    copied[p] = val[p];
}

// derive csc_row from the sorted keys (row is the low 32 bits)
__global__ void k_extract_csc_row(const uint64_t* keys, int* csc_row, int64_t nnz) {
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz) return;
    csc_row[p] = key_col(keys[p]);   // low 32 bits = row (sorted within col)
}

// thread-per-nonzero column-degree histogram
__global__ void k_col_counts(const int* col_idx, int64_t nnz, int* counts) {
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz) return;
    atomicAdd(&counts[col_idx[p]], 1);
}

// P1 : workload w_i + per-nonzeros segment jump map
__global__ void k_workload_and_seg(const int* row_ptr, const int* col_idx,
                                   const int* col_ptr, const int* csc_row,
                                   int m,
                                   int64_t* w_i, int* csc_pos, int64_t* seg_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    for (int p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
        int k  = col_idx[p];
        int lo = col_ptr[k], hi = col_ptr[k + 1];
        int off = dev_lower_bound(csc_row + lo, hi - lo, i);
        int lb  = lo + off;            // first index in csc col k with row >= i
        csc_pos[p] = lb;
        int64_t L = hi - lb;
        seg_len[p] = L;
        atomicAdd(reinterpret_cast<unsigned long long*>(&w_i[i]),
                  (unsigned long long)L);
    }
}

// sum w_i -> F_upper  (single-block reduction; w_i has m entries)
__global__ void k_sum_w(const int64_t* w_i, int m, int64_t* F_upper,
                        int64_t* w_max, double* w_sum) {
    extern __shared__ int64_t smem[];
    int64_t* ps = smem;                  // [0..n-1] partial sums
    int64_t* pm = smem + blockDim.x;     // [0..n-1] partial max
    int tid = threadIdx.x;
    int n = blockDim.x;
    int64_t s = 0, mx = 0;
    for (int i = tid; i < m; i += n) { int64_t v = w_i[i]; s += v; if (v > mx) mx = v; }
    ps[tid] = s; pm[tid] = mx;
    __syncthreads();
    for (int step = n >> 1; step > 0; step >>= 1) {
        if (tid < step) { ps[tid] += ps[tid + step];
                          if (pm[tid + step] > pm[tid]) pm[tid] = pm[tid + step]; }
        __syncthreads();
    }
    if (tid == 0) { *F_upper = ps[0]; *w_max = pm[0]; *w_sum = (double)ps[0]; }
}

// remaining kernels & host orchestration appended below

// shared hash-accumulator device functions (cap pow2, sentinel -1)
__device__ __forceinline__
void hash_clear(int* sh_key, float* sh_val, int cap, int tid, int nthreads) {
    for (int s = tid; s < cap; s += nthreads) { sh_key[s] = -1; sh_val[s] = 0.f; }
}
// numeric insert-or-accumulate; probe capped at cap, overflow sets *ovf (safety net).
__device__ __forceinline__
void hash_accum(int j, float v, int cap, int* sh_key, float* sh_val, int* ovf) {
    int mask = cap - 1;
    int s = (int)hash_slot((uint32_t)j, (uint32_t)cap);
    for (int probe = 0; probe < cap; ++probe) {
        int old = atomicCAS(&sh_key[s], -1, j);
        if (old == -1 || old == j) { atomicAdd(&sh_val[s], v); return; }
        s = (s + 1) & mask;
    }
    if (ovf) atomicExch(ovf, 1);   // overflow: signal host, do NOT silently drop
}
// symbolic: claim slot; returns true iff this thread newly created it
__device__ __forceinline__
bool hash_mark(int j, int cap, int* sh_key, int* ovf) {
    int mask = cap - 1;
    int s = (int)hash_slot((uint32_t)j, (uint32_t)cap);
    for (int probe = 0; probe < cap; ++probe) {
        int old = atomicCAS(&sh_key[s], -1, j);
        if (old == -1) return true;
        if (old == j)  return false;
        s = (s + 1) & mask;
    }
    if (ovf) atomicExch(ovf, 1);   // overflow
    return false;
}
__device__ __forceinline__ int warp_sum(int v) {
    for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
    return v;
}

// P2 : classify rows into tiers and compact into tier arrays
__global__ void k_classify_compact(const int64_t* __restrict__ w_i, int m,
                                   int64_t light_thr, int64_t heavy_thr,
                                   int* tier,
                                   int* light_rows,  int* n_light,
                                   int* medium_rows, int* n_medium,
                                   int* heavy_rows,  int* n_heavy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    int64_t w = w_i[i];
    int t = (w <= light_thr) ? 0 : (w > heavy_thr ? 2 : 1);
    tier[i] = t;
    int* cnt = (t == 0) ? n_light : (t == 1 ? n_medium : n_heavy);
    int idx = atomicAdd(cnt, 1);
    if (t == 0) light_rows[idx]  = i;
    else if (t == 1) medium_rows[idx] = i;
    else heavy_rows[idx]  = i;
}

__global__ void k_gather_w(const int* __restrict__ rows, const int64_t* __restrict__ w_i,
                           int L, int64_t* w_out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= L) return;
    w_out[k] = w_i[rows[k]];
}

// light bin-pack: cta_of[k] = prefix_work[k] / budget   (which CTA owns light row k)
__global__ void k_cta_of(const int64_t* __restrict__ prefix, int L,
                         int64_t budget, int* cta_of) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= L) return;
    cta_of[k] = (int)(prefix[k] / budget);
}
// cta_start[c] = first k with cta_of[k] == c  (cta_of is non-decreasing)
__global__ void k_cta_start(const int* __restrict__ cta_of, int L,
                            int num_ctas, int* cta_start) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_ctas) return;
    int lo = 0, hi = L;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (cta_of[mid] < c) lo = mid + 1; else hi = mid; }
    cta_start[c] = lo;
}

// heavy tasks: ntasks[r] = ceil(w_i[heavy_rows[r]] / target)
__global__ void k_ntasks(const int* __restrict__ heavy_rows,
                         const int64_t* __restrict__ w_i, int Hn,
                         int64_t target, int64_t* ntasks) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Hn) return;
    int64_t w = w_i[heavy_rows[r]];
    ntasks[r] = (w + target - 1) / target;     // ceil
}
// build tasks: thread-per-task, map t -> (row, begin_work, end_work)
__global__ void k_build_tasks(const int* __restrict__ heavy_rows,
                              const int64_t* __restrict__ w_i,
                              const int64_t* __restrict__ task_base, int Hn,
                              int64_t target, HeavyTask* tasks) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    // locate owning heavy row: largest r with task_base[r] <= t  (task_base monotonic)
    int lo = 0, hi = Hn;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (task_base[mid] <= t) lo = mid + 1; else hi = mid; }
    int r = lo - 1;
    int64_t local = t - task_base[r];
    int row = heavy_rows[r];
    int64_t w = w_i[row];
    int64_t begin = local * target;
    int64_t end = begin + target; if (end > w) end = w;
    tasks[t].row = row;
    tasks[t].begin_work = begin;
    tasks[t].end_work = end;
}

// P5/P6 glue : combine symbolic nnz, init per-row emit cursors
__global__ void k_init_cursor(const int* __restrict__ row_ptr, int m, int* cursor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) cursor[i] = row_ptr[i];
}

// P7 : construct full symmetric CSR from sorted upper (naturally col-sorted output)
__global__ void k_full_count(const uint64_t* __restrict__ ukey,
                             const float* __restrict__ uval,
                             int64_t upper_nnz, float eps, int* full_nnz) {
    int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= upper_nnz) return;
    float v = uval[q];
    if (eps > 0.f && fabsf(v) <= eps) return;     // pruned (default eps=0 -> keep all)
    uint64_t k = ukey[q];
    int i = key_row(k), j = key_col(k);
    atomicAdd(&full_nnz[i], 1);
    if (i != j) atomicAdd(&full_nnz[j], 1);       // mirror off-diagonal
}
__global__ void k_full_scatter(const uint64_t* __restrict__ ukey,
                               const float* __restrict__ uval,
                               int64_t upper_nnz, float eps,
                               const int* __restrict__ C_row_ptr, int* cursor,
                               int* C_col, float* C_val) {
    int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= upper_nnz) return;
    float v = uval[q];
    if (eps > 0.f && fabsf(v) <= eps) return;
    uint64_t k = ukey[q];
    int i = key_row(k), j = key_col(k);
    int pi = atomicAdd(&cursor[i], 1);
    C_col[C_row_ptr[i] + pi] = j;
    C_val[C_row_ptr[i] + pi] = v;
    if (i != j) {
        int pj = atomicAdd(&cursor[j], 1);
        C_col[C_row_ptr[j] + pj] = i;
        C_val[C_row_ptr[j] + pj] = v;
    }
}
// reduce-by-key counter -> per-row sym_nnz for heavy rows from reduced pairs
__global__ void k_count_reduced_per_row(const uint64_t* __restrict__ rkey,
                                        int64_t rn, const int* __restrict__ heavy_rows,
                                        int Hn, int* sym_nnz) {
    int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= rn) return;
    // rkey sorted by (row,col); count first-of-row via predecessor compare
    uint64_t k = rkey[q];
    uint64_t kp = (q > 0) ? rkey[q - 1] : (uint64_t)-1;
    int row = key_row(k);
    int prow = (kp == (uint64_t)-1) ? -1 : key_row(kp);
    if (row != prow) atomicAdd(&sym_nnz[row], 1);   // one per distinct row
}

// HEAVY rows: one row split into tasks (each a CTA), compact emit -> sort + reduce-by-key

// numeric insert/accum that also reports whether THIS call claimed the slot
__device__ __forceinline__
bool hash_accum_claim(int j, float v, int cap, int* sh_key, float* sh_val, int* ovf) {
    int mask = cap - 1;
    int s = (int)hash_slot((uint32_t)j, (uint32_t)cap);
    for (int probe = 0; probe < cap; ++probe) {
        int old = atomicCAS(&sh_key[s], -1, j);
        if (old == -1) { atomicAdd(&sh_val[s], v); return true; }
        if (old == j)  { atomicAdd(&sh_val[s], v); return false; }
        s = (s + 1) & mask;
    }
    if (ovf) atomicExch(ovf, 1);   // overflow
    return false;
}
// largest p in [0,nseg) with seg_base[base+p] <= g  (seg_base monotonic in row)
__device__ __forceinline__
int seg_of(const int64_t* seg_base, int base, int nseg, int64_t g) {
    int lo = 0, hi = nseg;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (seg_base[base + mid] <= g) lo = mid + 1; else hi = mid; }
    return lo - 1;
}

// cap sized so distinct always fits; COUNT=true counts distinct, COUNT=false emits compact pairs.
template<bool COUNT>
__global__ void k_heavy(const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                        const float* __restrict__ A_val,
                        const int* __restrict__ col_ptr, const int* __restrict__ csc_row,
                        const float* __restrict__ csc_val,
                        const int* __restrict__ csc_pos, const int64_t* __restrict__ seg_len,
                        const int64_t* __restrict__ seg_base,
                        const HeavyTask* __restrict__ tasks, int cap,
                        int64_t* __restrict__ task_distinct,
                        const int64_t* __restrict__ emit_off,
                        uint64_t* __restrict__ heavy_key, float* __restrict__ heavy_val,
                        int* __restrict__ overflow_flag) {
    extern __shared__ char smem_raw[];
    int*   sh_key = (int*)  smem_raw;
    float* sh_val = (float*)(sh_key + cap);
    const int tid = threadIdx.x, n = blockDim.x;
    const HeavyTask t = tasks[blockIdx.x];
    const int    r    = t.row;
    const int    base = row_ptr[r];
    const int    nseg = row_ptr[r + 1] - base;
    const int64_t bw = t.begin_work, ew = t.end_work;

    hash_clear(sh_key, sh_val, cap, tid, n);
    __syncthreads();

    int64_t g = bw + tid;
    int     p = -1, local = 0;
    if (g < ew) { p = seg_of(seg_base, base, nseg, g); local = (int)(g - seg_base[base + p]); }
    int claims = 0;
    while (g < ew) {
        while (local >= seg_len[base + p]) { local -= (int)seg_len[base + p]; ++p; }
        const int   pp = base + p;
        const int   j  = csc_row[csc_pos[pp] + local];
        const float v  = A_val[pp] * csc_val[csc_pos[pp] + local];
        if (hash_accum_claim(j, v, cap, sh_key, sh_val, overflow_flag)) ++claims;
        local += n;
        g     += n;
    }
    __syncthreads();

    // CTA-wide sum of claims
    __shared__ int sred[32];
    int w = warp_sum(claims);
    int lid = tid >> 5;
    if ((tid & 31) == 0) sred[lid] = w;
    __syncthreads();
    int nwarps_b = blockDim.x >> 5;
    __shared__ int sh_distinct;
    if (lid == 0) {
        int td = 0; for (int i = 0; i < nwarps_b; ++i) td += sred[i];
        if (tid == 0) {
            sh_distinct = td;
            if (COUNT) task_distinct[blockIdx.x] = td;
        }
    }
    __syncthreads();

    if (COUNT) return;

    // EMIT: scan task-local hash slots, write compact (key,partial_sum)
    int64_t ebase = emit_off[blockIdx.x];
    __shared__ int sh_ec;
    if (tid == 0) sh_ec = 0;
    __syncthreads();
    for (int s = tid; s < cap; s += n) {
        int j = sh_key[s];
        if (j >= 0) {
            int64_t pos = ebase + atomicAdd(&sh_ec, 1);
            heavy_key[pos] = make_key(r, j);
            heavy_val[pos] = sh_val[s];
        }
    }
}
// k_light: SYMBOLIC=true writes sym_nnz; SYMBOLIC=false emits (key,val) into upper buffer.
template<bool SYMBOLIC>
__global__ void k_light(const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                        const float* __restrict__ A_val,
                        const int* __restrict__ col_ptr, const int* __restrict__ csc_row,
                        const float* __restrict__ csc_val, const int* __restrict__ csc_pos,
                        const int* __restrict__ light_rows,
                        const int* __restrict__ cta_start, int cap,
                        int* sym_nnz,
                        const int* upper_row_ptr, int* upper_cursor,
                        uint64_t* upper_key, float* upper_val,
                        int* overflow_flag) {
    extern __shared__ char smem_raw[];
    const int tid   = threadIdx.x;
    const int wid   = tid >> 5;          // warp id within CTA
    const int lane  = tid & 31;
    const int nwarps = blockDim.x >> 5;
    int*   sh_key = (int*)  (smem_raw + (size_t)wid * cap * (sizeof(int) + sizeof(float)));
    float* sh_val = (float*)(sh_key + cap);

    const int cta   = blockIdx.x;
    const int r0    = cta_start[cta];
    const int r1    = cta_start[cta + 1];

    // round-robin rows across warps; each warp fully owns its rows
    for (int idx = r0 + wid; idx < r1; idx += nwarps) {
        const int r = light_rows[idx];
        hash_clear(sh_key, sh_val, cap, lane, 32);
        __syncwarp();
        int newly = 0;
        for (int p = row_ptr[r]; p < row_ptr[r + 1]; ++p) {
            const int   k  = col_idx[p];
            const float a  = A_val[p];
            const int   lb = csc_pos[p];
            const int   hi = col_ptr[k + 1];
            for (int q = lb + lane; q < hi; q += 32) {
                const int j = csc_row[q];
                if (SYMBOLIC) { if (hash_mark(j, cap, sh_key, overflow_flag)) ++newly; }
                else          { hash_accum(j, a * csc_val[q], cap, sh_key, sh_val, overflow_flag); }
            }
        }
        __syncwarp();
        if (SYMBOLIC) {
            int cnt = warp_sum(newly);
            if (lane == 0) sym_nnz[r] = cnt;
        } else {
            // emit distinct (key,val) via per-row atomic cursor
            for (int s = lane; s < cap; s += 32) {
                int j = sh_key[s];
                if (j >= 0) {
                    int pos = atomicAdd(&upper_cursor[r], 1);
                    upper_key[pos] = make_key(r, j);
                    upper_val[pos] = sh_val[s];
                }
            }
            __syncwarp();
        }
    }
}

// MEDIUM rows: one CTA per row, CTA-shared hash, flattened cursor walk
template<bool SYMBOLIC>
__global__ void k_medium(const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                         const float* __restrict__ A_val,
                         const int* __restrict__ col_ptr, const int* __restrict__ csc_row,
                         const float* __restrict__ csc_val,
                         const int* __restrict__ csc_pos, const int64_t* __restrict__ seg_len,
                         const int* __restrict__ medium_rows, int cap,
                         int* sym_nnz,
                         const int* upper_row_ptr, int* upper_cursor,
                         uint64_t* upper_key, float* upper_val,
                         int* overflow_flag) {
    extern __shared__ char smem_raw[];
    int*   sh_key = (int*)  smem_raw;
    float* sh_val = (float*)(sh_key + cap);
    const int tid = threadIdx.x;
    const int n   = blockDim.x;
    const int r   = medium_rows[blockIdx.x];
    const int base = row_ptr[r];
    const int nseg = row_ptr[r + 1] - base;

    hash_clear(sh_key, sh_val, cap, tid, n);
    __syncthreads();

    // --- init cursor: thread tid -> first (seg, local) covering global index tid ---
    int seg = 0, local = tid;
    while (seg < nseg && local >= seg_len[base + seg]) { local -= (int)seg_len[base + seg]; ++seg; }
    int my_cnt = 0;

    while (seg < nseg) {
        const int   p  = base + seg;
        const int   k  = col_idx[p];
        const float a  = A_val[p];
        const int   lb = csc_pos[p];
        const int   j  = csc_row[lb + local];
        if (SYMBOLIC) { if (hash_mark(j, cap, sh_key, overflow_flag)) ++my_cnt; }
        else          { hash_accum(j, a * csc_val[lb + local], cap, sh_key, sh_val, overflow_flag); }
        // advance by blockDim, crossing segment boundaries
        local += n;
        while (seg < nseg && local >= seg_len[base + seg]) { local -= (int)seg_len[base + seg]; ++seg; }
    }
    __syncthreads();

    if (SYMBOLIC) {
        // manual CTA-wide sum (TPB-agnostic)
        __shared__ int sred[32];
        int w = warp_sum(my_cnt);
        int lid = tid >> 5;
        if ((tid & 31) == 0) sred[lid] = w;
        __syncthreads();
        int nwarps_b = blockDim.x >> 5;
        if (lid == 0) {
            int t = 0;
            for (int i = 0; i < nwarps_b; ++i) t += sred[i];
            if (tid == 0) sym_nnz[r] = t;
        }
    } else {
        for (int s = tid; s < cap; s += n) {
            int j = sh_key[s];
            if (j >= 0) {
                int pos = atomicAdd(&upper_cursor[r], 1);
                upper_key[pos] = make_key(r, j);
                upper_val[pos] = sh_val[s];
            }
        }
    }
}

// heavy-reduce -> upper scatter helpers
// reset emit cursor for heavy rows (L/M rows already emitted; tiers disjoint)
__global__ void k_reset_heavy_cursor(const int* __restrict__ heavy_rows, int Hn,
                                     const int* __restrict__ upper_row_ptr,
                                     int* upper_cur) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Hn) return;
    int row = heavy_rows[r];
    upper_cur[row] = upper_row_ptr[row];
}
// scatter key-sorted reduced heavy pairs into the upper buffer via per-row cursor
__global__ void k_scatter_reduced(const uint64_t* __restrict__ rkey,
                                  const float* __restrict__ rval, int64_t rn,
                                  int* upper_cur,
                                  uint64_t* upper_key, float* upper_val) {
    int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= rn) return;
    int row = key_row(rkey[q]);
    int pos = atomicAdd(&upper_cur[row], 1);
    upper_key[pos] = rkey[q];
    upper_val[pos] = rval[q];
}
// fill task-work array (end_work - begin_work) for stats
__global__ void k_task_work(const HeavyTask* __restrict__ tasks, int64_t Tn,
                            int64_t* task_work) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Tn) return;
    task_work[t] = tasks[t].end_work - tasks[t].begin_work;
}

// host helper
static inline int att_next_pow2(int v) {
    if (v < 1) v = 1; --v; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}
#define ATT_ALLOC(p, bytes) do { cudaError_t _e = cudaMalloc((void**)&(p), (bytes)); \
    if (_e != cudaSuccess) { fprintf(stderr, "[ATT] alloc %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        status = AttStatus::AllocError; goto att_cleanup; } } while (0)
#define ATT_FREEP(p) do { if (p) { cudaFree(p); (p) = nullptr; } } while (0)

// att_aat_tiered : host orchestration
AttStatus att_aat_tiered(
        const int*   A_row_ptr, const int* A_col_idx, const float* A_val,
        int m, int n, int64_t nnz,
        int**   C_row_ptr_out, int** C_col_idx_out, float** C_val_out,
        int64_t* C_nnz_out,
        const AttTieredConfig& cfg, AttTiming* timing, AttStats* stats) {

    AttTiming T{};
    AttStats S{};
    S.m = m; S.n = n; S.nnz_A = nnz;
    AttStatus status = AttStatus::OK;

    // all device scratch (null-init for safe cleanup)
    int*     d_row_id=nullptr;     uint64_t* d_keys=nullptr;
    float*   d_csc_val=nullptr;    int*  d_col_counts=nullptr;
    int*     d_col_ptr=nullptr;    int*  d_csc_row=nullptr;
    int64_t* d_w=nullptr;          int*  d_csc_pos=nullptr;
    int64_t* d_seg_len=nullptr;    int64_t* d_seg_base=nullptr;
    int64_t* d_Fupper=nullptr;     int64_t* d_wmax=nullptr; double* d_wsum=nullptr;
    int*     d_tier=nullptr;       int*  d_light=nullptr; int* d_med=nullptr; int* d_heavy=nullptr;
    int*     d_n_light=nullptr;    int*  d_n_med=nullptr;  int* d_n_heavy=nullptr;
    int64_t* d_w_light=nullptr;    int64_t* d_pref_light=nullptr;
    int*     d_cta_of=nullptr;     int*  d_cta_start=nullptr;
    int64_t* d_ntasks=nullptr;     int64_t* d_task_base=nullptr;  HeavyTask* d_tasks=nullptr;
    int64_t* d_task_distinct=nullptr; int64_t* d_emit_off=nullptr;
    uint64_t* d_hkey=nullptr;      float* d_hval=nullptr;
    uint64_t* d_rkey=nullptr;      float* d_rval=nullptr;  int* d_rn=nullptr;
    int*     d_sym_nnz=nullptr;    int*  d_upper_row_ptr=nullptr; int* d_upper_cur=nullptr;
    int*     d_overflow_flag=nullptr;
    uint64_t* d_ukey=nullptr;      float* d_uval=nullptr;
    int*     d_full_nnz=nullptr;  int* d_full_cur=nullptr;
    int64_t* d_task_work=nullptr;  int64_t* d_wsort=nullptr; int64_t* d_twsort=nullptr;

    int      Ln=0, Mn=0, Hn=0, num_light_ctas=0;
    int64_t  total_tasks=0, heavy_emit_total=0; int heavy_rn=0;
    int      upper_nnz=0, C_nnz=0;

    // outputs
    int*   d_C_row_ptr=nullptr; int* d_C_col=nullptr; float* d_C_val=nullptr;

    const int  Tpb = 256;
    auto gridN = [](int64_t N)->int { return (int)((N + Tpb - 1) / Tpb); };
    auto gridM = [&](void)->int { return gridN(m); };

    // handle empty matrix
    ATT_ALLOC(d_C_row_ptr, (size_t)(m + 1) * sizeof(int));
    if (m == 0) { cudaMemset(d_C_row_ptr, 0, sizeof(int));
        *C_row_ptr_out = d_C_row_ptr; *C_col_idx_out = nullptr; *C_val_out = nullptr;
        *C_nnz_out = 0; if (timing)*timing = T; if (stats)*stats = S; return AttStatus::OK; }

    // P0 : CSR -> CSC
    { EventPair e; e.init(); e.start();
      ATT_ALLOC(d_row_id,    (size_t)nnz * sizeof(int));
      ATT_ALLOC(d_keys,      (size_t)nnz * sizeof(uint64_t));
      ATT_ALLOC(d_csc_val,   (size_t)nnz * sizeof(float));
      ATT_ALLOC(d_col_counts,(size_t)(n+1) * sizeof(int));
      ATT_ALLOC(d_col_ptr,   (size_t)(n + 1) * sizeof(int));
      ATT_ALLOC(d_csc_row,   (size_t)nnz * sizeof(int));
      cudaMemsetAsync(d_col_counts, 0, (size_t)(n+1) * sizeof(int));
      k_fill_row_id<<<gridM(), Tpb>>>(A_row_ptr, m, d_row_id);
      k_fill_csc_keys<<<gridN(nnz), Tpb>>>(A_col_idx, d_row_id, A_val, d_keys, d_csc_val, nnz);
      ATT_CUDA_CHECK2(status, goto att_cleanup);
      thrust::sort_by_key(thrust::device, d_keys, d_keys + nnz, d_csc_val);
      k_col_counts<<<gridN(nnz), Tpb>>>(A_col_idx, nnz, d_col_counts);
      { size_t tmp=0; void* dt=nullptr;                          // scan n+1 -> col_ptr[0..n], [n]=total
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_col_counts, d_col_ptr, n + 1);
        cudaMalloc(&dt, tmp);
        cub::DeviceScan::ExclusiveSum(dt, tmp, d_col_counts, d_col_ptr, n + 1);
        cudaFree(dt); }
      k_extract_csc_row<<<gridN(nnz), Tpb>>>(d_keys, d_csc_row, nnz);
      ATT_CUDA_CHECK2(status, goto att_cleanup);
      cudaFree(d_keys); d_keys = nullptr;
      T.csr2csc = e.stop(); e.destroy(); }

    // P1 : workload w_i + segment jump map
    { EventPair e; e.init(); e.start();
      ATT_ALLOC(d_w,        (size_t)m   * sizeof(int64_t));
      ATT_ALLOC(d_csc_pos,  (size_t)nnz * sizeof(int));
      ATT_ALLOC(d_seg_len,  (size_t)nnz * sizeof(int64_t));
      ATT_ALLOC(d_seg_base, (size_t)nnz * sizeof(int64_t));
      cudaMemsetAsync(d_w, 0, (size_t)m * sizeof(int64_t));
      k_workload_and_seg<<<gridM(), Tpb>>>(A_row_ptr, A_col_idx, d_col_ptr, d_csc_row,
                                           m, d_w, d_csc_pos, d_seg_len);
      thrust::exclusive_scan_by_key(thrust::device, d_row_id, d_row_id + nnz,
                                    d_seg_len, d_seg_base, (int64_t)0);
      ATT_ALLOC(d_Fupper, sizeof(int64_t)); ATT_ALLOC(d_wmax, sizeof(int64_t));
      ATT_ALLOC(d_wsum, sizeof(double));
      { size_t _sm = (size_t)Tpb * 2 * sizeof(int64_t);
        k_sum_w<<<1, Tpb, _sm>>>(d_w, m, d_Fupper, d_wmax, d_wsum); }
      int64_t Fu=0, wm=0; double ws=0;
      cudaMemcpy(&Fu, d_Fupper, 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(&wm, d_wmax, 8, cudaMemcpyDeviceToHost);
      cudaMemcpy(&ws, d_wsum, 8, cudaMemcpyDeviceToHost);
      S.F_upper = Fu; S.w_max = wm; S.w_avg = m ? ws / m : 0.0;
      ATT_CUDA_CHECK2(status, goto att_cleanup);
      T.workload = e.stop(); e.destroy(); }

    // P2 : classify + compact + light bin-pack + heavy task build
    { EventPair e; e.init(); e.start();
      ATT_ALLOC(d_tier,    (size_t)m * sizeof(int));
      ATT_ALLOC(d_light,   (size_t)m * sizeof(int));
      ATT_ALLOC(d_med,     (size_t)m * sizeof(int));
      ATT_ALLOC(d_heavy,   (size_t)m * sizeof(int));
      ATT_ALLOC(d_n_light, sizeof(int)); ATT_ALLOC(d_n_med, sizeof(int));
      ATT_ALLOC(d_n_heavy, sizeof(int));
      cudaMemsetAsync(d_n_light, 0, 4); cudaMemsetAsync(d_n_med, 0, 4);
      cudaMemsetAsync(d_n_heavy, 0, 4);
      k_classify_compact<<<gridM(), Tpb>>>(d_w, m, cfg.light_work_threshold,
                                           cfg.heavy_work_threshold, d_tier,
                                           d_light, d_n_light, d_med, d_n_med,
                                           d_heavy, d_n_heavy);
      cudaMemcpy(&Ln, d_n_light, 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(&Mn, d_n_med,   4, cudaMemcpyDeviceToHost);
      cudaMemcpy(&Hn, d_n_heavy, 4, cudaMemcpyDeviceToHost);
      S.n_light = Ln; S.n_medium = Mn; S.n_heavy = Hn;

      // light bin-pack
      const int64_t budget = cfg.light_cta_work_budget > 0 ? cfg.light_cta_work_budget : 1;
      if (Ln > 0) {
          ATT_ALLOC(d_w_light,   (size_t)Ln * sizeof(int64_t));
          ATT_ALLOC(d_pref_light,(size_t)Ln * sizeof(int64_t));
          k_gather_w<<<gridN(Ln), Tpb>>>(d_light, d_w, Ln, d_w_light);
          { size_t tmp=0; void* dt=nullptr;
            cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_w_light, d_pref_light, Ln);
            cudaMalloc(&dt, tmp);
            cub::DeviceScan::ExclusiveSum(dt, tmp, d_w_light, d_pref_light, Ln);
            cudaFree(dt); }
          int64_t pl=0, wl=0;
          cudaMemcpy(&pl, d_pref_light + (Ln - 1), 8, cudaMemcpyDeviceToHost);
          cudaMemcpy(&wl, d_w_light    + (Ln - 1), 8, cudaMemcpyDeviceToHost);
          int64_t total_light_work = pl + wl;
          num_light_ctas = (int)((total_light_work + budget - 1) / budget);
          if (num_light_ctas < 1) num_light_ctas = 1;
          ATT_ALLOC(d_cta_of,    (size_t)Ln * sizeof(int));
          ATT_ALLOC(d_cta_start, (size_t)(num_light_ctas + 1) * sizeof(int));
          k_cta_of<<<gridN(Ln), Tpb>>>(d_pref_light, Ln, budget, d_cta_of);
          k_cta_start<<<gridN(num_light_ctas), Tpb>>>(d_cta_of, Ln, num_light_ctas, d_cta_start);
          cudaMemcpy(d_cta_start + num_light_ctas, &Ln, sizeof(int), cudaMemcpyHostToDevice);
      }
      S.n_light_ctas = num_light_ctas;

      // heavy tasks
      if (Hn > 0) {
          ATT_ALLOC(d_ntasks,   (size_t)(Hn+1) * sizeof(int64_t));
          ATT_ALLOC(d_task_base,(size_t)(Hn + 1) * sizeof(int64_t));
          cudaMemsetAsync(d_ntasks, 0, (size_t)(Hn+1) * sizeof(int64_t));
          k_ntasks<<<gridN(Hn), Tpb>>>(d_heavy, d_w, Hn, cfg.heavy_task_target, d_ntasks);
          thrust::exclusive_scan(thrust::device, d_ntasks, d_ntasks + (Hn + 1),  // [0..Hn] -> task_base[0..Hn]
                                 d_task_base, (int64_t)0);                        // task_base[Hn] = total
          cudaMemcpy(&total_tasks, d_task_base + Hn, 8, cudaMemcpyDeviceToHost);
          if (total_tasks > 0) {
              ATT_ALLOC(d_tasks, (size_t)total_tasks * sizeof(HeavyTask));
              k_build_tasks<<<gridN(total_tasks), Tpb>>>(d_heavy, d_w, d_task_base, Hn,
                                                        cfg.heavy_task_target, d_tasks);
          }
      }
      S.n_heavy_tasks = total_tasks;
      ATT_CUDA_CHECK2(status, goto att_cleanup);
      T.task_build = e.stop(); e.destroy(); }

    // sym_nnz shared by all tiers (heavy fills heavy rows below; L/M fill theirs)
    ATT_ALLOC(d_sym_nnz, (size_t)(m + 1) * sizeof(int));
    cudaMemsetAsync(d_sym_nnz, 0, (size_t)(m + 1) * sizeof(int));
    // hash overflow flag: any tier hash exhausting its cap sets this -> host hard-errors.
    ATT_ALLOC(d_overflow_flag, sizeof(int));
    cudaMemsetAsync(d_overflow_flag, 0, sizeof(int));

    // P3 : heavy compute (count -> scan -> emit -> sort -> reduce-by-key)
    if (Hn > 0 && total_tasks > 0) {
        const int heavy_cap = att_next_pow2((int)ceil((double)cfg.heavy_task_target /
                                                      (double)cfg.hash_load_factor));
        const size_t heavy_smem = (size_t)heavy_cap * (sizeof(int) + sizeof(float));
        if (heavy_smem > 49152) {
            cudaFuncSetAttribute((void*)k_heavy<true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, heavy_smem);
            cudaFuncSetAttribute((void*)k_heavy<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, heavy_smem);
        }
        { EventPair e; e.init(); e.start();
          ATT_ALLOC(d_task_distinct, (size_t)total_tasks * sizeof(int64_t));
          k_heavy<true><<<(int)total_tasks, cfg.medium_cta_threads, heavy_smem>>>(
              A_row_ptr, A_col_idx, A_val, d_col_ptr, d_csc_row, d_csc_val,
              d_csc_pos, d_seg_len, d_seg_base, d_tasks, heavy_cap,
              d_task_distinct, nullptr, nullptr, nullptr, d_overflow_flag);
          ATT_CUDA_CHECK2(status, goto att_cleanup);
          ATT_ALLOC(d_emit_off, (size_t)total_tasks * sizeof(int64_t));
          thrust::exclusive_scan(thrust::device, d_task_distinct, d_task_distinct + total_tasks,
                                 d_emit_off, (int64_t)0);
          int64_t el=0, ed=0;
          cudaMemcpy(&el, d_emit_off      + (total_tasks - 1), 8, cudaMemcpyDeviceToHost);
          cudaMemcpy(&ed, d_task_distinct + (total_tasks - 1), 8, cudaMemcpyDeviceToHost);
          heavy_emit_total = el + ed;
          S.heavy_emit_pairs = heavy_emit_total;
          if (heavy_emit_total > 0) {
              ATT_ALLOC(d_hkey, (size_t)heavy_emit_total * sizeof(uint64_t));
              ATT_ALLOC(d_hval, (size_t)heavy_emit_total * sizeof(float));
              k_heavy<false><<<(int)total_tasks, cfg.medium_cta_threads, heavy_smem>>>(
                  A_row_ptr, A_col_idx, A_val, d_col_ptr, d_csc_row, d_csc_val,
                  d_csc_pos, d_seg_len, d_seg_base, d_tasks, heavy_cap,
                  d_task_distinct, d_emit_off, d_hkey, d_hval, d_overflow_flag);
              ATT_CUDA_CHECK2(status, goto att_cleanup);
          }
          T.heavy_compute = e.stop(); e.destroy(); }

        { EventPair e; e.init(); e.start();
          if (heavy_emit_total > 0) {
              thrust::sort_by_key(thrust::device, d_hkey, d_hkey + heavy_emit_total, d_hval);
              ATT_ALLOC(d_rkey, (size_t)heavy_emit_total * sizeof(uint64_t));
              ATT_ALLOC(d_rval, (size_t)heavy_emit_total * sizeof(float));
              ATT_ALLOC(d_rn,   sizeof(int));
              { size_t tmp=0; void* dt=nullptr;
                cub::DeviceReduce::ReduceByKey(nullptr, tmp, d_hkey, d_rkey, d_hval, d_rval,
                                               d_rn, cub::Sum(), (int)heavy_emit_total);
                cudaMalloc(&dt, tmp);
                cub::DeviceReduce::ReduceByKey(dt, tmp, d_hkey, d_rkey, d_hval, d_rval,
                                               d_rn, cub::Sum(), (int)heavy_emit_total);
                cudaFree(dt); }
              cudaMemcpy(&heavy_rn, d_rn, 4, cudaMemcpyDeviceToHost);
              // per-row sym_nnz for heavy rows
              k_count_reduced_per_row<<<gridN(heavy_rn), Tpb>>>(d_rkey, heavy_rn, d_heavy, Hn, d_sym_nnz);
          }
          ATT_CUDA_CHECK2(status, goto att_cleanup);
          T.heavy_reduce = e.stop(); e.destroy(); }
    }

    // P4 : light/medium symbolic  -> sym_nnz
    { EventPair e; e.init(); e.start();
      const int light_cap  = att_next_pow2(std::max(cfg.hash_cap_light,
              (int)ceil((double)cfg.light_work_threshold / (double)cfg.hash_load_factor)));
      const int light_tpb  = cfg.light_cta_threads;
      const int nwl        = light_tpb / 32;
      const size_t light_smem = (size_t)nwl * light_cap * (sizeof(int) + sizeof(float));
      if (num_light_ctas > 0)
          k_light<true><<<num_light_ctas, light_tpb, light_smem>>>(
              A_row_ptr, A_col_idx, A_val, d_col_ptr, d_csc_row, d_csc_val, d_csc_pos,
              d_light, d_cta_start, light_cap, d_sym_nnz, nullptr, nullptr, nullptr, nullptr, d_overflow_flag);

      const int med_cap  = cfg.medium_cap_override > 0
          ? cfg.medium_cap_override
          : att_next_pow2((int)ceil((double)cfg.heavy_work_threshold / (double)cfg.hash_load_factor));
      const size_t med_smem = (size_t)med_cap * (sizeof(int) + sizeof(float));
      if (med_smem > 49152) {
          cudaFuncSetAttribute((void*)k_medium<true>,  cudaFuncAttributeMaxDynamicSharedMemorySize, med_smem);
          cudaFuncSetAttribute((void*)k_medium<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, med_smem);
      }
      if (Mn > 0)
          k_medium<true><<<Mn, cfg.medium_cta_threads, med_smem>>>(
              A_row_ptr, A_col_idx, A_val, d_col_ptr, d_csc_row, d_csc_val, d_csc_pos,
              d_seg_len, d_med, med_cap, d_sym_nnz, nullptr, nullptr, nullptr, nullptr, d_overflow_flag);
      ATT_CUDA_CHECK2(status, goto att_cleanup);
      T.sym_lightmed = e.stop(); e.destroy();

      // P5 : scan sym_nnz -> upper_row_ptr
      e.init(); e.start();
      ATT_ALLOC(d_upper_row_ptr, (size_t)(m + 1) * sizeof(int));
      { size_t tmp=0; void* dt=nullptr;                              // scan m+1 -> upper_row_ptr[0..m]
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_sym_nnz, d_upper_row_ptr, m + 1);
        cudaMalloc(&dt, tmp);
        cub::DeviceScan::ExclusiveSum(dt, tmp, d_sym_nnz, d_upper_row_ptr, m + 1);
        cudaFree(dt); }
      cudaMemcpy(&upper_nnz, d_upper_row_ptr + m, sizeof(int), cudaMemcpyDeviceToHost);
      if (cfg.allow_int32_output && upper_nnz < 0) {                 // overflow (signed)
          status = AttStatus::Int32Overflow; goto att_cleanup; }
      S.C_nnz = 0; // filled after mirror
      T.assemble_ptr = e.stop(); e.destroy(); }

    // P6 : numeric emit (light/medium) + heavy scatter
    { EventPair e; e.init(); e.start();
      ATT_ALLOC(d_ukey, (size_t)upper_nnz * sizeof(uint64_t));
      ATT_ALLOC(d_uval, (size_t)upper_nnz * sizeof(float));
      ATT_ALLOC(d_upper_cur, (size_t)m * sizeof(int));
      k_init_cursor<<<gridM(), Tpb>>>(d_upper_row_ptr, m, d_upper_cur);

      const int light_cap  = att_next_pow2(std::max(cfg.hash_cap_light,
              (int)ceil((double)cfg.light_work_threshold / (double)cfg.hash_load_factor)));
      const int light_tpb  = cfg.light_cta_threads;
      const int nwl        = light_tpb / 32;
      const size_t light_smem = (size_t)nwl * light_cap * (sizeof(int) + sizeof(float));
      if (num_light_ctas > 0)
          k_light<false><<<num_light_ctas, light_tpb, light_smem>>>(
              A_row_ptr, A_col_idx, A_val, d_col_ptr, d_csc_row, d_csc_val, d_csc_pos,
              d_light, d_cta_start, light_cap, nullptr, d_upper_row_ptr, d_upper_cur, d_ukey, d_uval, d_overflow_flag);

      const int med_cap  = cfg.medium_cap_override > 0
          ? cfg.medium_cap_override
          : att_next_pow2((int)ceil((double)cfg.heavy_work_threshold / (double)cfg.hash_load_factor));
      const size_t med_smem = (size_t)med_cap * (sizeof(int) + sizeof(float));
      if (Mn > 0)
          k_medium<false><<<Mn, cfg.medium_cta_threads, med_smem>>>(
              A_row_ptr, A_col_idx, A_val, d_col_ptr, d_csc_row, d_csc_val, d_csc_pos,
              d_seg_len, d_med, med_cap, nullptr, d_upper_row_ptr, d_upper_cur, d_ukey, d_uval, d_overflow_flag);
      ATT_CUDA_CHECK2(status, goto att_cleanup);
      T.num_lightmed = e.stop(); e.destroy();

      // heavy scatter (reduced pairs -> upper buffer)
      e.init(); e.start();
      if (Hn > 0 && heavy_rn > 0) {
          k_reset_heavy_cursor<<<gridN(Hn), Tpb>>>(d_heavy, Hn, d_upper_row_ptr, d_upper_cur);
          k_scatter_reduced<<<gridN(heavy_rn), Tpb>>>(d_rkey, d_rval, heavy_rn, d_upper_cur, d_ukey, d_uval);
          ATT_CUDA_CHECK2(status, goto att_cleanup);
      }
      T.heavy_scatter = e.stop(); e.destroy(); }

    // overflow check: any tier hash exhausted its cap?
    { int ovf = 0; cudaMemcpy(&ovf, d_overflow_flag, sizeof(int), cudaMemcpyDeviceToHost);
      S.hash_overflow_fallbacks = ovf;
      if (ovf) { fprintf(stderr, "[ATT] hash overflow detected (cap too small) — aborting, not returning bad output\n");
                 status = AttStatus::KernelError; goto att_cleanup; } }

    // P7 : global sort of upper + full symmetric CSR construction
    { EventPair e; e.init(); e.start();
      if (upper_nnz > 0)
          thrust::sort_by_key(thrust::device, d_ukey, d_ukey + upper_nnz, d_uval);
      ATT_ALLOC(d_full_nnz, (size_t)(m + 1) * sizeof(int));
      cudaMemsetAsync(d_full_nnz, 0, (size_t)(m + 1) * sizeof(int));
      if (upper_nnz > 0)
          k_full_count<<<gridN(upper_nnz), Tpb>>>(d_ukey, d_uval, upper_nnz, cfg.zero_epsilon, d_full_nnz);
      { size_t tmp=0; void* dt=nullptr;                              // scan m+1 -> C_row_ptr[0..m]
        cub::DeviceScan::ExclusiveSum(nullptr, tmp, d_full_nnz, d_C_row_ptr, m + 1);
        cudaMalloc(&dt, tmp);
        cub::DeviceScan::ExclusiveSum(dt, tmp, d_full_nnz, d_C_row_ptr, m + 1);
        cudaFree(dt); }
      cudaMemcpy(&C_nnz, d_C_row_ptr + m, sizeof(int), cudaMemcpyDeviceToHost);
      if (cfg.allow_int32_output && C_nnz < 0) {
          status = AttStatus::Int32Overflow; goto att_cleanup; }
      if (C_nnz > 0) {
          ATT_ALLOC(d_C_col, (size_t)C_nnz * sizeof(int));
          ATT_ALLOC(d_C_val, (size_t)C_nnz * sizeof(float));
          ATT_ALLOC(d_full_cur, (size_t)m * sizeof(int));
          k_init_cursor<<<gridM(), Tpb>>>(d_C_row_ptr, m, d_full_cur);
          k_full_scatter<<<gridN(upper_nnz), Tpb>>>(d_ukey, d_uval, upper_nnz, cfg.zero_epsilon,
                                                   d_C_row_ptr, d_full_cur, d_C_col, d_C_val);
          ATT_CUDA_CHECK2(status, goto att_cleanup);
      }
      S.C_nnz = C_nnz;
      T.sym_csr = e.stop(); e.destroy(); }

    // stats : w_p99, task workload max/avg/p99 (not counted in T.total)
    if (m > 0) {
        ATT_ALLOC(d_wsort, (size_t)m * sizeof(int64_t));
        cudaMemcpy(d_wsort, d_w, (size_t)m * sizeof(int64_t), cudaMemcpyDeviceToDevice);
        thrust::sort(thrust::device, d_wsort, d_wsort + m);
        int64_t pv = 0;
        int idx = (int)((double)m * 0.99); if (idx >= m) idx = m - 1;
        cudaMemcpy(&pv, d_wsort + idx, 8, cudaMemcpyDeviceToHost);
        S.w_p99 = (double)pv;
    }
    if (total_tasks > 0) {
        ATT_ALLOC(d_task_work, (size_t)total_tasks * sizeof(int64_t));
        k_task_work<<<gridN(total_tasks), Tpb>>>(d_tasks, total_tasks, d_task_work);
        int64_t twmax=0; double twsum=0;
        { int64_t* h = (int64_t*)malloc(total_tasks * sizeof(int64_t));
          cudaMemcpy(h, d_task_work, total_tasks * sizeof(int64_t), cudaMemcpyDeviceToHost);
          for (int64_t i=0;i<total_tasks;++i){ if(h[i]>twmax)twmax=h[i]; twsum+=h[i]; }
          thrust::sort(thrust::device, d_task_work, d_task_work + total_tasks);
          int64_t pv=0; int tidx=(int)((double)total_tasks*0.99); if(tidx>=total_tasks) tidx=total_tasks-1;
          cudaMemcpy(&pv, d_task_work + tidx, 8, cudaMemcpyDeviceToHost);
          S.task_work_max = twmax;
          S.task_work_avg = twsum / (double)total_tasks;
          S.task_work_p99 = (double)pv;
          free(h); }
    }

    T.total = T.csr2csc + T.workload + T.task_build + T.heavy_compute + T.heavy_reduce
            + T.sym_lightmed + T.assemble_ptr + T.num_lightmed + T.heavy_scatter + T.sym_csr;

    // success: hand outputs to caller
    *C_row_ptr_out = d_C_row_ptr; d_C_row_ptr = nullptr;
    *C_col_idx_out = d_C_col;     d_C_col = nullptr;
    *C_val_out     = d_C_val;     d_C_val = nullptr;
    *C_nnz_out     = C_nnz;
    if (timing) *timing = T;
    if (stats)  *stats  = S;

att_cleanup:
    // free all temporaries (outputs already nulled on success path)
    ATT_FREEP(d_row_id);     ATT_FREEP(d_keys);      ATT_FREEP(d_csc_val);
    ATT_FREEP(d_col_counts); ATT_FREEP(d_col_ptr);   ATT_FREEP(d_csc_row);
    ATT_FREEP(d_w);          ATT_FREEP(d_csc_pos);   ATT_FREEP(d_seg_len);
    ATT_FREEP(d_seg_base);   ATT_FREEP(d_Fupper);    ATT_FREEP(d_wmax);
    ATT_FREEP(d_wsum);
    ATT_FREEP(d_tier);       ATT_FREEP(d_light);     ATT_FREEP(d_med);     ATT_FREEP(d_heavy);
    ATT_FREEP(d_n_light);    ATT_FREEP(d_n_med);     ATT_FREEP(d_n_heavy);
    ATT_FREEP(d_w_light);    ATT_FREEP(d_pref_light);ATT_FREEP(d_cta_of);  ATT_FREEP(d_cta_start);
    ATT_FREEP(d_ntasks);     ATT_FREEP(d_task_base); ATT_FREEP(d_tasks);
    ATT_FREEP(d_task_distinct); ATT_FREEP(d_emit_off);
    ATT_FREEP(d_hkey);       ATT_FREEP(d_hval);      ATT_FREEP(d_rkey);    ATT_FREEP(d_rval);
    ATT_FREEP(d_rn);
    ATT_FREEP(d_sym_nnz);    ATT_FREEP(d_upper_row_ptr); ATT_FREEP(d_upper_cur);
    ATT_FREEP(d_overflow_flag);
    ATT_FREEP(d_ukey);       ATT_FREEP(d_uval);      ATT_FREEP(d_full_nnz); ATT_FREEP(d_full_cur);
    ATT_FREEP(d_task_work);  ATT_FREEP(d_wsort);     ATT_FREEP(d_twsort);
    ATT_FREEP(d_C_row_ptr);  ATT_FREEP(d_C_col);     ATT_FREEP(d_C_val);
    return status;
}
