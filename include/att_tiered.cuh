#ifndef ATT_TIERED_CUH
#define ATT_TIERED_CUH
//
//  C = A * A^T : three-tier (light/medium/heavy) workload-driven SpGEMM, self-contained.
//
#include <cstdint>

struct AttTieredConfig {
    // CTA geometry
    int   medium_cta_threads = 256;   // threads per medium-row CTA
    int   light_cta_threads  = 128;   // threads per light-row-group CTA
    int   warp_threads       = 32;

    // hash accumulators (open addressing, linear probe, key=col j)
    int   hash_cap_light       = 128;    // slots, warp-private (light)   -> ~1 KB/warp
    int   hash_cap_medium_max  = 8192;   // max slots, CTA-shared (medium)-> ~64 KB
    int   hash_cap_heavy_task  = 2048;   // slots, task-local (heavy)     -> ~16 KB
    float hash_load_factor     = 0.6f;   // fallback when fill > LF
    int   medium_cap_override  = 0;      // >0 = force this medium-hash cap (baseline: all-medium, one CTA/row)

    // tier thresholds (workload w_i, int64 upper-triangle partial products)
    int64_t light_work_threshold  = 256;    // w_i <=  -> light
    int64_t heavy_work_threshold  = 8192;   // w_i >   -> heavy  (else medium)
    int64_t heavy_task_target     = 2048;   // partial products per heavy task
    int     light_cta_work_budget = 1024;   // bin-pack light rows up to this cumulative work

    // numerics
    float  zero_epsilon      = 0.0f;  // drop |val|<=eps AFTER full accumulate; 0 = keep all
    bool   allow_int32_output= true;  // false -> error on any int32 overflow (int64 path not built)

    // caps / fallbacks
    int64_t heavy_emit_cap_per_task = 0;    // 0 = use each task workload as the loose emit cap
};

struct AttTiming {   // milliseconds, measured with CUDA events
    float csr2csc = 0, workload = 0, task_build = 0;
    float heavy_compute = 0, heavy_reduce = 0;
    float sym_lightmed = 0, assemble_ptr = 0;
    float num_lightmed = 0, heavy_scatter = 0;
    float sym_csr = 0, total = 0;
};

struct AttStats {
    int     m = 0, n = 0;
    int64_t nnz_A = 0;
    int64_t F_upper = 0;          // = sum_k d_k(d_k+1)/2
    int     n_light = 0, n_medium = 0, n_heavy = 0;
    int64_t n_light_ctas = 0, n_heavy_tasks = 0;
    int64_t heavy_emit_pairs = 0; // compact (key,sum) pairs emitted before reduce
    int64_t C_nnz = 0;            // full symmetric output nnz
    int64_t w_max = 0;     double w_avg = 0, w_p99 = 0;
    int64_t task_work_max = 0; double task_work_avg = 0, task_work_p99 = 0;
    int64_t hash_overflow_fallbacks = 0;
};

enum class AttStatus {
    OK,
    Int32Overflow,        // C_nnz or a row_ptr entry would exceed INT32_MAX
    HeavyEmitExceeded,    // a heavy task's compact emit exceeded its cap
    KernelError,
    AllocError,
};

// Inputs reside in device memory; outputs are cudaMalloc'd (caller frees).
AttStatus att_aat_tiered(
    const int*   A_row_ptr, const int* A_col_idx, const float* A_val,
    int m, int n, int64_t nnz,
    int**   C_row_ptr,     // length m+1   (device)
    int**   C_col_idx,     // length C_nnz (device)
    float** C_val,         // length C_nnz (device)
    int64_t* C_nnz_out,
    const AttTieredConfig& cfg = {},
    AttTiming* timing = nullptr, AttStats* stats = nullptr);

#endif  // ATT_TIERED_CUH
