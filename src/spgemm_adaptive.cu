#include "spgemm.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// METHOD 环境变量单方法门控:未设/空/"all" → 跑全部;否则只跑 == METHOD(大小写不敏感)的块
bool should_run_method(const char *key) {
    const char *m = std::getenv("METHOD");
    if (!m || !*m || !std::strcmp(m, "all")) return true;
    // 大小写不敏感比较
    for (int i = 0; ; i++) {
        char a = m[i], b = key[i];
        if (a >= 'A' && a <= 'Z') a += 32;
        if (b >= 'A' && b <= 'Z') b += 32;
        if (a != b) return false;
        if (a == 0) return true;
    }
}

// ==========================================================================
//  自适应 dispatcher(集成进 src):C = A·A 的完整数据流。
//  指标:flop_proxy = A_nnz² / A_rows  (C=A·A 中间积数的 O(1) 估计,≈ Σ row_nnz²)。
//  flop_proxy > FLOP_THR → hash SPA(大/高工作量,hash 主场);否则 merge3(小/稀疏)。
//  hash 溢出(distinct>HASH_CAP)→ 自动回退 merge3。
// ==========================================================================

// env 读取(分流阈值可被环境变量覆盖)
static int env_int(const char *k, int def) {
    const char *e = std::getenv(k);
    if (e && *e) { int v = std::atoi(e); if (v > 0) return v; }
    return def;
}
static double env_double(const char *k, double def) {
    const char *e = std::getenv(k);
    if (e && *e) { double v = std::atof(e); if (v > 0) return v; }
    return def;
}

#define USE_HASH_ROW_NUM 0

void spgemm_self_product_adaptive(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    // A_buffer 布局:[row_ptr (A_rows+1) | col_idx (A_nnz) | val (A_nnz)]
    // 分流依据:① 规模 n(中小→merge)② 重行不均(极不平均的重行→hash)
    //   hash = 大阵(n > ADAPTIVE_SIZE_THR) 或 重行(max_row_nnz > ADAPTIVE_HEAVY_THR 或 skew > ADAPTIVE_SKEW_THR)
    //   否则 → merge3(中小 + 均衡)
    const int *row_ptr = (const int *)A_buffer;
    int max_row_nnz = 0;
    for (int i = 0; i < A_rows; i++) {                    // O(n) host 扫描,极廉价
        int rnz = row_ptr[i + 1] - row_ptr[i];
        if (rnz > max_row_nnz) max_row_nnz = rnz;
    }
    double avg  = A_rows > 0 ? (double)A_nnz / A_rows : 0.0;
    double skew = avg > 0.0 ? (double)max_row_nnz / avg : 0.0;

    int    size_thr  = env_int("ADAPTIVE_SIZE_THR", 10000);    // 大阵:n > 此值(默认 1 万,taxonomy 的 L 线)
    int    heavy_thr = env_int("ADAPTIVE_HEAVY_THR", 128);     // 重行:max_row_nnz > 此值(bp_* ≈300)
    double skew_thr  = env_double("ADAPTIVE_SKEW_THR", 12.0);  // 极不平均:skew = max/avg > 此值(bp_* ≈60)
    int large = (A_rows > size_thr);
    int heavy = (max_row_nnz > heavy_thr) || (skew > skew_thr);
    int use_hash = large || heavy;
    const char *why = large ? (heavy ? "large+heavy" : "large") : (heavy ? "heavy" : "-");
    dbg("[adapt] n=%d maxrow=%d skew=%.1f → %s (%s)\n", A_rows, max_row_nnz, skew, use_hash ? "hash" : "merge3", why);
    printf("[adapt] n=%d maxrow=%d skew=%.1f → %s (%s)\n", A_rows, max_row_nnz, skew, use_hash ? "hash" : "merge3", why);

    if (use_hash) {
        spgemm_self_product_hash(A_buffer, A_rows, A_cols, A_nnz,
                                 C_buffer_out, C_rows, C_cols, C_nnz);
        if (*C_nnz < 0) {                                 // hash 溢出 → 回退 merge3
            dbg("[adapt] hash overflow → fallback merge3\n");
            printf("[adapt] hash overflow → fallback merge3\n");
            spgemm_self_product_merge3(A_buffer, A_rows, A_cols, A_nnz,
                                       C_buffer_out, C_rows, C_cols, C_nnz);
        }
    } else {
        spgemm_self_product_merge3(A_buffer, A_rows, A_cols, A_nnz,
                                   C_buffer_out, C_rows, C_cols, C_nnz);
    }
}
