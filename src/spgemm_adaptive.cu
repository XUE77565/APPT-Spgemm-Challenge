#include "spgemm.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

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
//
//  多变量调度公式(100 阵实测拟合,R²=0.824,准确率 93%):
//    score = -1.31×log10(flop_proxy) + 1.21×log10(n) + 1.98×log10(max_row_nnz)
//            - 2.43×log10(skew) + 1.29
//    hash iff score < 0
//  其中 flop_proxy = A_nnz²/n, skew = max_row_nnz / (A_nnz/n)
//
//  公式含义:flop 高(中间积多,hash dedup 省)+ 重行(SMEM hash 并行) -> hash;
//           大 n(行多但每行轻)+ 高 skew(straggler) -> merge3。
//  hash 溢出(distinct>HASH_CAP) -> 自动回退 merge3。
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

    // ── 多变量调度公式(100 阵拟合,R²=0.824,准确率 93%) ──
    double flop_proxy = (double)A_nnz * A_nnz / std::max(A_rows, 1);
    double lfp = log10(std::max(flop_proxy, 1.0));
    double ln  = log10((double)A_rows);
    double lmr = log10((double)std::max(max_row_nnz, 1));
    double lsk = log10(std::max(skew, 1.0));
    double score = -1.3085 * lfp + 1.2131 * ln + 1.9815 * lmr - 2.4331 * lsk + 1.2943;
    // score < 0 → hash(flop 高 / 重行 / 均匀);score ≥ 0 → merge3(小阵 / 不均匀 straggler)
    const char *force = std::getenv("ADAPTIVE_FORCE");
    int use_hash;
    if (force && force[0]) {
        use_hash = (force[0] == 'h' || force[0] == 'H') ? 1 : 0;   // 环境变量强制覆盖(测试用)
    } else {
        use_hash = (score < 0.0) ? 1 : 0;
    }
    char why_buf[128];
    snprintf(why_buf, sizeof(why_buf), "score=%.2f fp=%.0f mr=%d sk=%.1f", score, flop_proxy, max_row_nnz, skew);
    const char *why = why_buf;
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

// ==========================================================================
//  ATT 自适应 dispatcher:C = A·Aᵀ 上三角。同 AA Auto 的分流逻辑(score 公式),
//  调 att_hash / att_merge3(两者均为 AA 的忠实拷贝,B=Aᵀ(CSC)+ j≥i)。
//  hash 溢出(distinct>HASH_CAP)→ 自动回退 att_merge3。
// ==========================================================================
void spgemm_att_adaptive(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    const int *row_ptr = (const int *)A_buffer;
    int max_row_nnz = 0;
    for (int i = 0; i < A_rows; i++) {
        int rnz = row_ptr[i + 1] - row_ptr[i];
        if (rnz > max_row_nnz) max_row_nnz = rnz;
    }
    double avg  = A_rows > 0 ? (double)A_nnz / A_rows : 0.0;
    double skew = avg > 0.0 ? (double)max_row_nnz / avg : 0.0;
    double flop_proxy = (double)A_nnz * A_nnz / std::max(A_rows, 1);
    double lfp = log10(std::max(flop_proxy, 1.0));
    double ln  = log10((double)A_rows);
    double lmr = log10((double)std::max(max_row_nnz, 1));
    double lsk = log10(std::max(skew, 1.0));
    double score = -1.3085 * lfp + 1.2131 * ln + 1.9815 * lmr - 2.4331 * lsk + 1.2943;
    const char *force = std::getenv("ADAPTIVE_FORCE");
    int use_hash;
    if (force && force[0]) use_hash = (force[0] == 'h' || force[0] == 'H') ? 1 : 0;
    else use_hash = (score < 0.0) ? 1 : 0;
    dbg("[att-adapt] n=%d maxrow=%d skew=%.1f → %s (score=%.2f fp=%.0f)\n",
        A_rows, max_row_nnz, skew, use_hash ? "att_hash" : "att_merge3", score, flop_proxy);

    if (use_hash) {
        spgemm_att_hash(A_buffer, A_rows, A_cols, A_nnz, C_buffer_out, C_rows, C_cols, C_nnz);
        if (*C_nnz < 0) {                                 // hash 溢出 → 回退 att_merge3
            dbg("[att-adapt] att_hash overflow → fallback att_merge3\n");
            spgemm_att_merge3(A_buffer, A_rows, A_cols, A_nnz, C_buffer_out, C_rows, C_cols, C_nnz);
        }
    } else {
        spgemm_att_merge3(A_buffer, A_rows, A_cols, A_nnz, C_buffer_out, C_rows, C_cols, C_nnz);
    }
}
