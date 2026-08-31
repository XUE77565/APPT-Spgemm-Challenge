#!/usr/bin/env python3
"""复验队列(docs/65):把所有"环境灾变中断/噪声下未定论"的结论在净窗重验。
并入 loop 工作流:净窗优先排空本队列,再开新实验。

用法:.venv/bin/python scripts/reverify_queue.py [--force] [--only mat1 mat2 ...]
  --force  跳过负载检查(仅在明确知道自己在干什么时)
输出:compare/reverify_<date>.log + 对 v28 CSV 的补丁(交替中位数)+ 差异报告。"""
import os, sys, csv, subprocess, re, statistics, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_methods as CM

DIR = "data/ocean/square"
V28 = "compare/ocean337/methods_cmp_v28_reverify.csv"
LOAD_MAX = 6.0        # 净窗判据:1min load < 6(历史干净段 ~1-3,gem5 期 11-34)

# ── 复验清单(全部带"为什么在这里"的出处)────────────────────────────
QUEUE = {
    # A. ADAPT_EXPAND 受害者(v27 救回值测于 load11,docs/60 §5)
    "v27_victims": ["c-big", "Flan_1565", "inline_1", "F2", "TSOPF_RS_b678_c2", "rajat28",
                    "F1", "dielFilterV2real", "dielFilterV3clx", "bmwcra_1", "nd12k",
                    "cage15", "bone010"],
    # B. 路由窄化三赢(测于 load11-34,docs/63 §6)
    "routing_wins": ["c-64", "c-64b", "TSOPF_FS_b39_c7", "3Dspectralwave", "3Dspectralwave2"],
    # C. v26 翻赢复核(v26 窗口基本净但个别边界,docs/60 §3)
    "flips": ["pkustk01", "olafu", "brainpc2", "c-53", "bbmat", "rajat25"],
    # D. 回归守卫(必须保持,任何净窗复验都带上)
    "guards": ["bcsstk30", "pwtk", "c-62", "c-62ghs", "Ga3As3H12", "mult_dcop_03", "Cube_Coup_dt0"],
    # E. 高方差阵(refresh 单值永不可信,交替中位数为准,docs/62 方法学)
    "highvar": ["rajat16", "rajat18", "rajat20", "a0nsdsil", "fp"],
    # F. F1 专项:v27 的 ex1.4 误选(harness 被噪声骗,需在干净环境重跑双 expand)
}
F1_SPECIAL = "F1"    # 已在 victims 里,双 expand 复验时重点看选择

def load_ok():
    try:
        la = float(open("/proc/loadavg").read().split()[0])
        return la, la < LOAD_MAX
    except Exception:
        return -1, False

def run_matrix(m):
    """与 compare_methods 生产路径完全同口径(含 harness 双 expand 取优)。"""
    p = os.path.join(DIR, m + ".mtx")
    return CM.run_spgemm_method(p, "hash", exp_cnnz=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    la, ok = load_ok()
    if not ok and not args.force:
        print(f"✗ 负载 {la:.1f} ≥ {LOAD_MAX}(非净窗)——复验只在净窗跑,--force 可强制作业")
        sys.exit(2)
    print(f"✓ 净窗确认(load {la:.1f}),开始复验 {datetime.datetime.now():%F %T}")

    mats = args.only or [m for grp in QUEUE.values() for m in grp]
    mats = list(dict.fromkeys(mats))          # 去重保序
    # v28 基底:从 v27 拷贝(其后逐阵覆写)
    if not os.path.exists(V28):
        import shutil
        shutil.copy("compare/ocean337/methods_cmp_v27_harness.csv", V28)

    rows = {r["matrix"]: r for r in csv.DictReader(open(V28))}
    report = []
    for m in mats:
        ts, best = [], None
        for _ in range(args.reps):
            r = run_matrix(m)
            if r: ts.append(r[0]); best = r
        if not ts:
            print(f"  {m:18s} FAIL"); continue
        med = round(statistics.median(ts), 3)
        spread = f"[{min(ts):.1f},{max(ts):.1f}]"
        old = float(rows[m]["Auto"]) if rows.get(m, {}).get("Auto") else None
        d = f"{(med/old-1)*100:+6.1f}%" if old else "  new"
        # nnz 红旗:复验值 vs CSV 记录
        nz_csv = rows.get(m, {}).get("cnnz")
        flag = "" if (not nz_csv or int(float(nz_csv)) == best[2]) else f" ⚠nnz {best[2]} vs {nz_csv}"
        if rows.get(m): rows[m]["Auto"] = str(med)
        report.append((m, old, med, d, spread, flag))
        print(f"  {m:18s} {str(old):>8s} → {med:8.2f} {d} {spread}{flag}", flush=True)

    with open(V28, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys()))
        w.writeheader(); w.writerows(rows.values())
    print(f"\n已写回 {V28};合计 {len(report)} 阵")

if __name__ == "__main__":
    main()
