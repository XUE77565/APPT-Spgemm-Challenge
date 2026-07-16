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
    d = df.dropna(subset=["merge2_time", "man_time"]).copy()
    for c in ("man_time", "merge_time", "merge2_time", "cu_time"):
        d[c] = d[c].astype(float)
    d["ratio_par"] = d["merge2_time"] / d["man_time"]   # 并行 vs ESC
    d["ratio_par_cu"] = d["merge2_time"] / d["cu_time"] # 并行 vs cuSPARSE
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
    ws, ts, ls, geos = tally(d["ratio_ser"])
    print("=" * 70)
    print(f"merge 对比,共 {n} 矩阵   [{os.path.basename(SUMMARY)}]")
    print("-" * 70)
    print(f"  {'':<16}{'赢':>5}{'持平':>6}{'输':>5}{'几何均值':>10}")
    print(f"  {'par vs cuSPARSE':<16}{wpc:>5}{tpc:>6}{lpc:>5}{geopc:>9.3f}×")
    print(f"  {'par vs gust(ESC)':<16}{wp:>5}{tp:>6}{lp:>5}{geop:>9.3f}×")
    print(f"  {'ser vs gust(ESC)':<16}{ws:>5}{ts:>6}{ls:>5}{geos:>9.3f}×")
    sp = np.exp(np.log(d["speedup"]).mean())
    faster = (d["speedup"] > 1.0).sum()
    print(f"\n  并行 vs 串行:几何均值加速 {sp:.2f}×;并行更快的矩阵 {faster}/{n}")
    print()
    print(f"{'class':<18}{'#':>4}{'cu':>7}{'gust':>8}{'merge(ser)':>11}{'merge(par)':>11}"
          f"{'par/gust':>9}{'par/cu':>8}{'par/ser':>9}")
    print("-" * 70)
    for c in CLASS_ORDER:
        s = d[d["class"] == c]
        if len(s) == 0:
            continue
        pg = np.exp(np.log(s["ratio_par"]).mean())
        pc = np.exp(np.log(s["ratio_par_cu"]).mean())
        ps = np.exp(np.log(s["speedup"]).mean())
        print(f"{c:<18}{len(s):>4}{s['cu_time'].mean():>7.2f}{s['man_time'].mean():>8.2f}"
              f"{s['merge_time'].mean():>11.2f}{s['merge2_time'].mean():>11.2f}"
              f"{pg:>8.2f}×{pc:>7.2f}×{ps:>8.2f}×")


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
    except Exception:
        pass
    print(f"\n生成图表(输出到 {OUT_DIR.relative_to(REPO)}/):")
    chart_scatter(d)
    chart_speedup(d)
    chart_winloss(d)
    print(f"\n数据 + 图表均在:{OUT_DIR.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
