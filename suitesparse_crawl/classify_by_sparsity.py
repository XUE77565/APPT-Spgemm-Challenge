#!/usr/bin/env python3
"""
按【密度】对 SuiteSparse 全部方阵分类(参考 lookup/sort.png 的 D/MS/HS 方案),
并从【全集合】(非仅本地)为每个类别挑选代表性矩阵,供后续 per-class profiling 找瓶颈。

类别(密度 = nnz / (rows × cols),百分比):
  Dense (D)        : density >= 10%
  Mildly sparse(MS): 1% <= density < 10%
  Highly sparse(HS): density < 1%

代表矩阵挑选规则:
  - 优先在“可 profiling”尺寸内挑选(n <= CAP_N 且 nnz <= CAP_NNZ),
    避免挑到 2 亿维的巨型矩阵(跑不动/OOM);
  - 在该类的密度区间上【均匀取分位点】,保证密度跨度覆盖;
  - 若该类可 profiling 子集太小,自动放宽尺寸上限。
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ---- 字体(中文) ----
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)

# ---- 调色板(dataviz 校验过的参考调色板) ----
SURFACE, INK_PRI, INK_SEC, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE = "#e1e0d9", "#c3c2b7"
C_DENSE  = "#2a78d6"   # 蓝
C_MILD   = "#eb6834"   # 橙(slot8)
C_HIGH   = "#008300"   # 绿(slot4)
C_EXTR   = "#4a3aa7"   # 紫(slot5)—— 第 4 类 Extremely sparse
CLASS_COLOR = {"Dense": C_DENSE, "Mildly sparse": C_MILD,
               "Highly sparse": C_HIGH, "Extremely sparse": C_EXTR}
CLASS_ORDER = ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK_PRI, "axes.labelcolor": INK_SEC, "axes.titlecolor": INK_PRI,
    "axes.edgecolor": BASELINE, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 130,
})

# 分类边界(密度百分比)
D_HI, MS_HI, MS_LO, HS_LO = 100.0, 10.0, 1.0, 0.1


def classify(density_pct):
    if density_pct >= MS_HI:
        return "Dense"
    if density_pct >= MS_LO:
        return "Mildly sparse"
    if density_pct >= HS_LO:
        return "Highly sparse"
    return "Extremely sparse"


def pick_representatives(sub):
    """在一个类里联合【尺寸(nnz)分层 + 密度极值】挑代表(去重)。

    尺寸分层:取 nnz 分位 0.10/0.50/0.90(小/中/大),覆盖 profiling 的
              “启动开销主导 → 带宽/工作量主导”谱段;
    密度极值:取密度分位 0.05/0.95,补足类内密度跨度。
    """
    if len(sub) == 0:
        return pd.DataFrame()
    seen, picks = set(), []

    def add(frame):
        for _, r in frame.iterrows():
            if int(r["id"]) not in seen:
                seen.add(int(r["id"]))
                picks.append(r)

    bysize = sub.sort_values("nnz")
    m = len(bysize)
    for q in (0.10, 0.50, 0.90):
        i = min(m - 1, max(0, int(round(q * (m - 1)))))
        add(bysize.iloc[[i]])

    bydens = sub.sort_values("density_pct")
    m = len(bydens)
    for q in (0.05, 0.95):
        i = min(m - 1, max(0, int(round(q * (m - 1)))))
        add(bydens.iloc[[i]])

    return pd.DataFrame(picks)


def main():
    here = Path(__file__).resolve().parent
    csv_path = here / "suitesparse_metadata_square.csv"
    if not csv_path.exists():
        print(f"找不到 {csv_path};先运行 "
              f"scrape_suitesparse.py --square-only --out {csv_path.name}",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df[df["is_square"]].copy()
    df["density_pct"] = df["density"] * 100.0
    df["class"] = df["density_pct"].apply(classify)
    df["n"] = df["rows"].astype(np.int64)
    n = len(df)

    # 本地 data/ 里已有的矩阵名(仅作标注,不参与挑选)
    local_dir = here.parent / "data"
    local_names = set()
    if local_dir.is_dir():
        local_names = {p.name for p in local_dir.iterdir() if p.is_dir()}
    df["in_local_data"] = df["name"].isin(local_names)

    print(f"\n=== 全集合方阵分类(共 {n:,} 个) ===")
    print(f"{'类别':<16}{'数量':>8}{'占比':>8}   密度范围")
    summary_rows = []
    for c in CLASS_ORDER:
        sub = df[df["class"] == c]
        cnt = len(sub)
        lo, hi = (sub["density_pct"].min(), sub["density_pct"].max()) if cnt else (0, 0)
        print(f"  {c:<16}{cnt:>7,}{cnt/n*100:>7.1f}%   "
              f"[{lo:.2e}%, {hi:.4f}%]")
        summary_rows.append((c, cnt, lo, hi))

    # ---- 挑代表:可 profiling 尺寸 + 密度跨度 ----
    CAP_N_TIGHT, CAP_NNZ_TIGHT = 500_000, 10_000_000      # 舒适档(放宽)
    CAP_N_LOOSE, CAP_NNZ_LOOSE = 2_000_000, 50_000_000    # 放宽档

    reps_all = []
    print("\n=== 各类代表矩阵(从全集合挑选,优先可 profiling 尺寸) ===")
    for c in CLASS_ORDER:
        cls_df = df[df["class"] == c]
        tier = cls_df[(cls_df["n"] <= CAP_N_TIGHT) & (cls_df["nnz"] <= CAP_NNZ_TIGHT)]
        tier_label = f"n≤{CAP_N_TIGHT:,} & nnz≤{CAP_NNZ_TIGHT:,}"
        if len(tier) < 5:  # 该类舒适档不够,放宽
            tier = cls_df[(cls_df["n"] <= CAP_N_LOOSE) & (cls_df["nnz"] <= CAP_NNZ_LOOSE)]
            tier_label = f"n≤{CAP_N_LOOSE:,} & nnz≤{CAP_NNZ_LOOSE:,} (放宽)"
        reps = pick_representatives(tier)
        print(f"\n[{c}]  池子 {len(cls_df):,} 个;挑 {len(reps)} 个代表(尺寸档:{tier_label})")
        print(f"  {'name':<34}{'group':<14}{'n':>12}{'nnz':>13}{'density%':>11}{'本地':>5}")
        for _, r in reps.iterrows():
            flag = "✓" if r["in_local_data"] else " "
            print(f"  {r['name']:<34}{r['group']:<14}{int(r['n']):>12,}"
                  f"{int(r['nnz']):>13,}{r['density_pct']:>11.3e}{flag:>5}")
        reps_all.append(reps)

    reps_df = pd.concat(reps_all, ignore_index=True) if reps_all else pd.DataFrame()
    keep = ["class", "id", "name", "group", "n", "nnz", "density_pct",
            "sparsity_pct", "in_local_data", "url_mm"]
    reps_df = reps_df[[c for c in keep if c in reps_df.columns]]
    reps_df.to_csv(here / "representatives.csv", index=False)

    # 全量分类结果
    out_cols = ["id", "name", "group", "n", "nnz", "density_pct",
                "sparsity_pct", "class", "in_local_data", "url_mm"]
    df[out_cols].sort_values(["class", "density_pct"], ascending=[True, False]) \
        .to_csv(here / "classification_square.csv", index=False)
    print(f"\n写出: classification_square.csv(全量 {n:,} 行带类别)")
    print(f"写出: representatives.csv(各类代表)")

    # ---- 图:复刻参考图(密度轴 + D/MS/HS 色带 + 真实点 + 代表高亮) ----
    draw_chart(df, reps_df, here / "charts" / "sparsity_classes.png")
    print("写出: charts/sparsity_classes.png")


def draw_chart(df, reps_df, outpath):
    outpath.parent.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    dmin = max(df["density_pct"].min() * 0.5, 1e-7)

    # 背景色带 D / MS / HS / ES
    ax.axvspan(MS_HI, 100.0 * 1.3, color=C_DENSE, alpha=0.10, zorder=0)
    ax.axvspan(MS_LO, MS_HI, color=C_MILD, alpha=0.12, zorder=0)
    ax.axvspan(HS_LO, MS_LO, color=C_HIGH, alpha=0.10, zorder=0)
    ax.axvspan(dmin, HS_LO, color=C_EXTR, alpha=0.10, zorder=0)

    # 真实矩阵点(strip,纵向抖动)
    for c in CLASS_ORDER:
        sub = df[df["class"] == c]
        y = rng.uniform(0.15, 0.85, size=len(sub))
        ax.scatter(sub["density_pct"], y, s=10, alpha=0.4, edgecolors="none",
                   color=CLASS_COLOR[c], zorder=3)
    # 分界线
    for x in (MS_HI, MS_LO, HS_LO):
        ax.axvline(x, color=BASELINE, linewidth=1.0, linestyle="--", zorder=2)
    # 类标签
    band_y = 0.96
    ax.text(np.sqrt(MS_HI * 100.0), band_y, "Dense (D)", ha="center", va="top",
            color=C_DENSE, fontsize=11, fontweight="bold")
    ax.text(np.sqrt(MS_LO * MS_HI), band_y, "Mildly sparse (MS)", ha="center", va="top",
            color=C_MILD, fontsize=11, fontweight="bold")
    ax.text(np.sqrt(HS_LO * MS_LO), band_y, "Highly sparse (HS)", ha="center", va="top",
            color=C_HIGH, fontsize=11, fontweight="bold")
    ax.text(np.sqrt(dmin * HS_LO), band_y, "Extremely sparse (ES)", ha="center", va="top",
            color=C_EXTR, fontsize=11, fontweight="bold")

    # 代表矩阵高亮
    if len(reps_df):
        yr = rng.uniform(0.15, 0.85, size=len(reps_df))
        ax.scatter(reps_df["density_pct"], yr, s=60, marker="D", facecolors="none",
                   edgecolors=INK_PRI, linewidths=1.2, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(dmin, 100.0 * 1.3)
    ticks = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    ax.set_xticks([t for t in ticks if dmin <= t <= 130])
    ax.set_xticklabels([f"{t:g}%" for t in [t for t in ticks if dmin <= t <= 130]])
    ax.set_yticks([])
    ax.set_xlabel("密度 = nnz / (rows × cols)")
    ax.set_title("SuiteSparse 全方阵按密度分类(D / MS / HS / ES)  ◆ = 各类代表")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
