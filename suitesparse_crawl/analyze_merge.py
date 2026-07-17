#!/usr/bin/env python3
# 分析 k-way merge 的两版 vs ESC sort 的 benchmark 结果。
#   merge(ser) = serial v1(每行 thread0)
#   merge(par) = 并行 v2(warp-perrow 协作)
# 读 profile_aa_summary.csv,聚焦:① merge2(par) vs ESC 的交叉点;② v2 相对 serial 的加速。
#
# 用法:
#   .venv/bin/python suitesparse_crawl/analyze_merge.py [summary.csv]

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from profile_aa import (
    SURFACE, INK_PRI, INK_SEC, INK_MUTED, GRIDLINE, BASELINE,
    C_CU, C_MAN, CLASS_COLOR, CLASS_ORDER,
)

C_GOOD, C_BAD, C_TIE = "#1baf7a", "#e34948", "#898781"   # 变快绿 / 变慢红 / 持平灰

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SUMMARY = Path(sys.argv[1]) if len(sys.argv) > 1 else (HERE / "profile_aa_summary.csv")
# 图表输出到 compare/<folder>/:每次运行新建一个文件夹,不同 compare 对象互不覆盖。
# 可用 argv[2] 或环境变量 MERGE_COMPARE_DIR 指定文件夹名(默认带时间戳)。
import shutil
from datetime import datetime
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = (Path(sys.argv[2]) if len(sys.argv) > 2
           else Path(os.environ.get("MERGE_COMPARE_DIR",
                                    REPO / f"compare/merge_vs_esc_{_ts}")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIN, TIE = 1.0, 1.05   # ratio<WIN=更快;WIN..TIE 算持平


def load():
    df = pd.read_csv(SUMMARY)
    d = df.dropna(subset=["merge2_compute_t", "man_compute_t"]).copy()
    # compute-only(各阶段求和去 h2d/d2h):与 cuBLAS(kernel)公平对比;
    # 映射成代码内统一列名 man_time/merge_time/merge2_time/cu_time(现均为 compute-only)
    d["man_time"]   = d["man_compute_t"].astype(float)
    d["merge_time"] = d["merge_compute_t"].astype(float)
    d["merge2_time"]= d["merge2_compute_t"].astype(float)
    d["cu_time"]    = d["cu_compute_t"].astype(float) if "cu_compute_t" in d.columns else np.nan
    # cuBLAS 稠密 GEMM baseline(新主 baseline)。profile_aa_summary.csv 已含 cublas_ms
    # (profile_aa.py 写入);若旧 summary 没有,再从 baseline_cublas.csv 补 merge。
    if "cublas_ms" not in d.columns:
        cub_path = HERE / "baseline_cublas.csv"
        if cub_path.exists():
            d = d.merge(pd.read_csv(cub_path)[["name", "cublas_ms"]], on="name", how="left")
        else:
            d["cublas_ms"] = np.nan
    d["cublas_ms"] = d["cublas_ms"].astype(float)
    # Ocean SpGEMM baseline(compute-only = GPU 阶段求和)
    if "ocean_ms" not in d.columns:
        oce_path = HERE / "baseline_ocean.csv"
        if oce_path.exists():
            d = d.merge(pd.read_csv(oce_path)[["name", "ocean_ms"]], on="name", how="left")
        else:
            d["ocean_ms"] = np.nan
    d["ocean_ms"] = d["ocean_ms"].astype(float)
    d["ratio_par_cublas"] = d["merge2_time"] / d["cublas_ms"]  # 并行 vs cuBLAS(主 baseline)
    d["ratio_par_ocean"]  = d["merge2_time"] / d["ocean_ms"]   # 并行 vs Ocean
    d["ratio_par"] = d["merge2_time"] / d["man_time"]   # 并行 vs ESC
    d["ratio_par_cu"] = d["merge2_time"] / d["cu_time"] # 并行 vs cuSPARSE(旧 baseline)
    d["ratio_ser"] = d["merge_time"] / d["man_time"]    # 串行 vs ESC
    d["speedup"] = d["merge_time"] / d["merge2_time"]   # 并行 vs 串行(>1=并行更快)
    d["heavy"] = d["name"].str.startswith("bp_")
    return d


def print_summary(d):
    n = len(d)

    def tally(r):
        w = (r < WIN).sum(); t = ((r >= WIN) & (r <= TIE)).sum(); l = (r > TIE).sum()
        return w, t, l, np.exp(np.log(r).mean())

    wp, tp, lp, geop = tally(d["ratio_par"])
    wpc, tpc, lpc, geopc = tally(d["ratio_par_cu"])
    wcb, tcb, lcb, geocb = tally(d["ratio_par_cublas"].dropna())
    wo, to_, lo, geoo = tally(d["ratio_par_ocean"].dropna())
    ws, ts, ls, geos = tally(d["ratio_ser"])
    print("=" * 70)
    print(f"merge 对比(compute-only,不含 h2d/d2h;cuBLAS=sgemm kernel;Ocean=GPU 阶段求和),共 {n} 矩阵   [{os.path.basename(SUMMARY)}]")
    print("-" * 70)
    print(f"  {'':<18}{'赢':>5}{'持平':>6}{'输':>5}{'几何均值':>10}")
    print(f"  {'par vs Ocean':<18}{wo:>5}{to_:>6}{lo:>5}{geoo:>9.3f}×   ← hash SpGEMM 对照")
    print(f"  {'par vs cuBLAS':<18}{wcb:>5}{tcb:>6}{lcb:>5}{geocb:>9.3f}×   ← 主 baseline(稠密 GEMM)")
    print(f"  {'par vs cuSPARSE':<18}{wpc:>5}{tpc:>6}{lpc:>5}{geopc:>9.3f}×   (旧 baseline)")
    print(f"  {'par vs gust(ESC)':<18}{wp:>5}{tp:>6}{lp:>5}{geop:>9.3f}×")
    print(f"  {'ser vs gust(ESC)':<18}{ws:>5}{ts:>6}{ls:>5}{geos:>9.3f}×")
    sp = np.exp(np.log(d["speedup"]).mean())
    faster = (d["speedup"] > 1.0).sum()
    print(f"\n  并行 vs 串行:几何均值加速 {sp:.2f}×;并行更快的矩阵 {faster}/{n}")
    print()
    print(f"{'class':<18}{'#':>4}{'cuBLAS':>9}{'Ocean':>8}{'gust':>8}{'merge(par)':>11}"
          f"{'par/cuBL':>9}{'par/Oce':>8}{'par/gust':>9}")
    print("-" * 90)
    for c in CLASS_ORDER:
        s = d[d["class"] == c]
        if len(s) == 0:
            continue
        pg = np.exp(np.log(s["ratio_par"]).mean())
        rcb = s["ratio_par_cublas"].replace(0, np.nan).dropna()
        pcb = np.exp(np.log(rcb).mean()) if len(rcb) else np.nan
        ro = s["ratio_par_ocean"].replace(0, np.nan).dropna()
        poc = np.exp(np.log(ro).mean()) if len(ro) else np.nan
        print(f"{c:<18}{len(s):>4}{s['cublas_ms'].mean():>9.2f}{s['ocean_ms'].mean():>8.2f}{s['man_time'].mean():>8.2f}"
              f"{s['merge2_time'].mean():>11.2f}"
              f"{(f'{pcb:.2f}×' if not np.isnan(pcb) else '--'):>9}"
              f"{(f'{poc:.2f}×' if not np.isnan(poc) else '--'):>8}{pg:>8.2f}×")


def _class_scatter(ax, d, x, y, **kw):
    """按 class 分组画 scatter(固定色序);bp_* 重行族叠一层醒目标记。"""
    for c in CLASS_ORDER:
        s = d[(d["class"] == c) & (~d["heavy"])]
        if len(s):
            ax.scatter(s[x], s[y], s=34, c=CLASS_COLOR[c], alpha=0.78,
                       edgecolors="none", label=c, zorder=3, **kw)
    h = d[d["heavy"]]
    if len(h):
        ax.scatter(h[x], h[y], marker="X", s=70, c=C_BAD, alpha=0.9,
                   edgecolors="white", linewidths=0.6,
                   label="bp_* 重行族", zorder=4)


# ---- 图0(主):merge2(par) vs cuBLAS 稠密 GEMM(log-log)----
# 新主 baseline:小矩阵 cuBLAS 极快(赢),大矩阵 cuBLAS O(n³) 惨输 → 戏剧性交叉点
def chart_scatter_cublas(d):
    if d["cublas_ms"].isna().all():
        print("  (跳过 cuBLAS 散点:无 baseline_cublas.csv)")
        return
    dd = d.dropna(subset=["cublas_ms", "merge2_time"])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    _class_scatter(ax, dd, "cublas_ms", "merge2_time")
    lo = min(dd["cublas_ms"].min(), dd["merge2_time"].min()) * 0.5
    hi = max(dd["cublas_ms"].max(), dd["merge2_time"].max()) * 2.0
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=BASELINE, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.text(hi * 0.5, lo * 1.3, "merge(par) 更快", color=C_GOOD, fontsize=9, ha="right")
    ax.text(lo * 1.5, hi * 0.55, "cuBLAS 更快", color=C_BAD, fontsize=9)
    for name in ["bcsstk30", "bp_200", "1138_bus"]:
        sel = dd[dd["name"] == name]
        if len(sel):
            r = sel.iloc[0]
            ax.annotate(f"{name} ({r['ratio_par_cublas']:.2g}×)",
                        (r["cublas_ms"], r["merge2_time"]),
                        fontsize=8, color=INK_SEC, xytext=(8, 6), textcoords="offset points")
    ax.set_xlabel("cuBLAS sgemm kernel (ms)"); ax.set_ylabel("merge(par) compute (ms)")
    ax.set_title("并行 merge vs cuBLAS(compute-only):虚线下方=merge 赢")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, which="both", color=GRIDLINE, linewidth=0.6)
    fig.tight_layout()
    out = OUT_DIR / "merge2_vs_cublas_scatter.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  图: {out.relative_to(REPO)}")


# ---- 柱状图:各类别 cuBLAS / gust(ESC) / merge(par) 均值(log y),直观看 crossover ----
def chart_bar(d):
    methods = [("cuBLAS", "cublas_ms", C_CU), ("Ocean", "ocean_ms", "#4a3aa7"),
               ("gust(ESC)", "man_time", C_MAN), ("merge(par)", "merge2_time", C_GOOD)]
    classes = [c for c in CLASS_ORDER if len(d[d["class"] == c])]
    means = {}
    for lab, col, _ in methods:
        s = d.dropna(subset=[col])
        means[lab] = [s[s["class"] == c][col].mean() if len(s[s["class"] == c]) else np.nan for c in classes]
    x = np.arange(len(classes))
    w = 0.20
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, (lab, _, color) in enumerate(methods):
        vals = np.array([v if not np.isnan(v) else 1e-3 for v in means[lab]])
        bars = ax.bar(x + (i - 1.5) * w, vals, width=w, color=color, label=lab, zorder=3)
        for b, v in zip(bars, means[lab]):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK_SEC, rotation=0)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n({len(d[d['class']==c])})" for c in classes])
    ax.set_ylabel("耗时 (ms, 对数, compute-only)")
    ax.set_title("各方法 compute-only 对照(按稀疏类别):cuBLAS 小矩阵赢、merge(par) 大稀疏矩阵赢")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    ax.grid(axis="y", which="both", color=GRIDLINE, linewidth=0.6)
    fig.tight_layout()
    out = OUT_DIR / "methods_bar_by_class.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  图: {out.relative_to(REPO)}")


# ---- 图1:merge2(par) vs gust(ESC)(log-log),虚线下方=并行 merge 赢 ----
def chart_scatter(d):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    _class_scatter(ax, d, "man_time", "merge2_time")
    lo = min(d["man_time"].min(), d["merge2_time"].min()) * 0.7
    hi = max(d["man_time"].max(), d["merge2_time"].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color=BASELINE, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.text(hi * 0.5, lo * 1.3, "merge(par) 更快", color=C_GOOD, fontsize=9, ha="right")
    ax.text(lo * 1.5, hi * 0.55, "ESC 更快", color=C_BAD, fontsize=9)
    for name in ["bcsstk30", "bp_200"]:
        sel = d[d["name"] == name]
        if len(sel):
            r = sel.iloc[0]
            ax.annotate(f"{name} ({r['ratio_par']:.2f}×)", (r["man_time"], r["merge2_time"]),
                        fontsize=8, color=INK_SEC, xytext=(8, 6), textcoords="offset points")
    ax.set_xlabel("gust(ESC) 耗时 (ms)"); ax.set_ylabel("merge(par) 耗时 (ms)")
    ax.set_title("并行 merge vs ESC:每矩阵总耗时(对数,虚线下方=并行 merge 赢)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, which="both", color=GRIDLINE, linewidth=0.6)
    fig.tight_layout()
    out = OUT_DIR / "merge2_vs_esc_scatter.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  图: {out.relative_to(REPO)}")


# ---- 图2:并行相对串行的加速 vs C_nnz(>1=并行更快)----
def chart_speedup(d):
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    _class_scatter(ax, d, "cu_cnnz", "speedup")
    ax.axhline(1.0, ls="--", lw=1.2, color=BASELINE, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(d["speedup"].min() * 0.8, d["speedup"].max() * 1.2)
    ax.text(d["cu_cnnz"].max(), 1.05, "并行更快", color=C_GOOD, fontsize=9, ha="right", va="bottom")
    ax.text(d["cu_cnnz"].max(), 0.96, "串行更快", color=C_BAD, fontsize=9, ha="right", va="top")
    for name in ["bcsstk30", "bp_200"]:
        sel = d[d["name"] == name]
        if len(sel):
            r = sel.iloc[0]
            ax.annotate(f"{name} ({r['speedup']:.1f}×)", (r["cu_cnnz"], r["speedup"]),
                        fontsize=8, color=INK_SEC, xytext=(8, 6), textcoords="offset points")
    ax.set_xlabel("C 输出 nnz(对数)"); ax.set_ylabel("串行 / 并行 耗时比(对数,>1=并行更快)")
    ax.set_title("并行化的收益:merge(par) 相对 merge(ser) 的加速")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, which="both", color=GRIDLINE, linewidth=0.6)
    fig.tight_layout()
    out = OUT_DIR / "merge2_speedup_over_serial.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  图: {out.relative_to(REPO)}")


# ---- 图3:merge(par) vs ESC 各类别 赢/平/输 ----
def chart_winloss(d):
    rows = []
    for c in CLASS_ORDER:
        s = d[d["class"] == c]
        if len(s) == 0:
            continue
        r = s["ratio_par"]
        rows.append({"class": c, "n": len(s),
                     "赢": int((r < WIN).sum()),
                     "持平": int(((r >= WIN) & (r <= TIE)).sum()),
                     "输": int((r > TIE).sum())})
    g = pd.DataFrame(rows)
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = np.arange(len(g))
    left = np.zeros(len(g))
    for col, color in [("赢", C_GOOD), ("持平", C_TIE), ("输", C_BAD)]:
        vals = g[col].values
        ax.barh(y, vals, left=left, color=color, height=0.6, label=col, zorder=3)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        left += vals
    ax.set_yticks(y); ax.set_yticklabels([f"{r['class']}\n({r['n']})" for _, r in g.iterrows()])
    ax.invert_yaxis()
    ax.set_xlabel("矩阵数"); ax.set_title("并行 merge vs ESC:各类别胜负分布")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right")
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.6); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out = OUT_DIR / "merge2_winloss_by_class.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  图: {out.relative_to(REPO)}")


def main():
    d = load()
    print_summary(d)
    # 自包含:把源数据 CSV 也拷进输出文件夹
    try:
        shutil.copy(SUMMARY, OUT_DIR / SUMMARY.name)
        shutil.copy(HERE / "baseline_cublas.csv", OUT_DIR / "baseline_cublas.csv")
        shutil.copy(HERE / "baseline_ocean.csv", OUT_DIR / "baseline_ocean.csv")
    except Exception:
        pass
    print(f"\n生成图表(输出到 {OUT_DIR.relative_to(REPO)}/):")
    chart_scatter_cublas(d)
    chart_bar(d)
    chart_scatter(d)
    chart_speedup(d)
    chart_winloss(d)
    print(f"\n数据 + 图表均在:{OUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
