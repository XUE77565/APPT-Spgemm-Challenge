#!/usr/bin/env python3
"""
分析 SuiteSparse Matrix Collection 的【方阵】元数据,聚焦稀疏度与大小。
(已过滤掉所有长方阵;不涉及 group / kind / shape / year。)

图表(写入 ./charts/):
  density_hist.png       – 稀疏度:log10(密度) 分布              [稀疏度]
  nnz_per_row_hist.png   – 每行平均非零数 分布                    [稀疏度]
  size_hist.png          – 维度 n 与 非零数 分布(双面板)         [大小]
  rows_vs_nnz.png        – 维度 n vs 非零数(对数-对数散点)        [大小]

配色采用 dataviz 技能中校验过的参考调色板(CVD 安全,固定顺序分类色),浅色模式。
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# 注册系统中文字体,避免中文显示成方框(豆腐字)
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)

# ---- 调色板(校验过的参考调色板,浅色模式) ---------------------------------
SURFACE    = "#fcfcfb"
INK_PRI    = "#0b0b0b"
INK_SEC    = "#52514e"
INK_MUTED  = "#898781"
GRIDLINE   = "#e1e0d9"
BASELINE   = "#c3c2b7"
C_SLOT1    = "#2a78d6"   # 蓝 – 单一系列默认色
C_SLOT6    = "#e34948"   # 红 – 中位数参考线

# ---- matplotlib 全局样式 -----------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor":   SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color":       INK_PRI,
    "axes.labelcolor":  INK_SEC,
    "axes.titlecolor":  INK_PRI,
    "axes.edgecolor":   BASELINE,
    "xtick.color":      INK_MUTED,
    "ytick.color":      INK_MUTED,
    "axes.grid":        True,
    "grid.color":       GRIDLINE,
    "grid.linewidth":   0.8,
    "axes.linewidth":   0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":      ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "figure.dpi":       130,
})


def style_axis(ax):
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)


def save(fig, name, outdir):
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  写出 {path}")


def median_label(ax, x, text):
    """画红色虚线中位数标线 + 带底色的文字标注(避免压在直方图上)。"""
    ax.axvline(x, color=C_SLOT6, linewidth=1.4, linestyle="--", zorder=4)
    ax.text(x * 1.02, ax.get_ylim()[1] * 0.92, text,
            color=C_SLOT6, fontsize=8, ha="left", va="top",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.9))


def main():
    here = Path(__file__).resolve().parent
    csv_path = here / "suitesparse_metadata_square.csv"
    outdir = here / "charts"
    outdir.mkdir(exist_ok=True)
    if not csv_path.exists():
        print(f"找不到 {csv_path};请先运行 "
              f"scrape_suitesparse.py --square-only --out {csv_path.name}",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df[df["is_square"]].copy()      # 防御性:只保留方阵
    df["n"] = df["rows"].astype(np.int64)   # 方阵维度 n(rows == cols)
    n = len(df)
    print(f"\n=== SuiteSparse Matrix Collection · 方阵子集 = {n:,} 个 ===\n")

    # ---- 数值摘要 ----
    total_nnz = int(df["nnz"].sum())
    print(f"方阵总数   : {n:,}")
    print(f"总非零数   : {total_nnz:,}")
    for col in ["n", "nnz", "nnz_per_row"]:
        s = df[col]
        print(f"  {col:12s}: 最小 {s.min():,.1f}  中位数 {s.median():,.1f}  "
              f"均值 {s.mean():,.0f}  最大 {s.max():,.0f}")
    sp = df["sparsity_pct"]
    print(f"  {'sparsity%':12s}: 最小 {sp.min():.4f}  中位数 {sp.median():.4f}  "
          f"均值 {sp.mean():.4f}  最大 {sp.max():.4f}")
    den = df["density"]
    print(f"  密度中位数 : {den.median():.3e}  "
          f"(几何均值 {np.exp(np.log(den[den > 0]).mean()):.3e})")

    print("\n最大的 5 个方阵(按维度 n):")
    for _, r in df.nlargest(5, "n").iterrows():
        print(f"  n={r['n']:>11,}  nnz={r['nnz']:>13,}  "
              f"稀疏度={r['sparsity_pct']:.4f}%  {r['name']} ({r['group']})")

    print("\n非零数最多的 5 个方阵:")
    for _, r in df.nlargest(5, "nnz").iterrows():
        print(f"  n={r['n']:>11,}  nnz={r['nnz']:>13,}  "
              f"稀疏度={r['sparsity_pct']:.4f}%  {r['name']} ({r['group']})")

    # ---- 图表 ----
    print("\n渲染图表:")

    # 1. 稀疏度:log10(密度)
    d = df["density"].replace(0, np.nan).dropna()
    logd = np.log10(d)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.hist(logd, bins=60, color=C_SLOT1, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.set_xlabel("密度 = nnz / (rows × cols)   [log₁₀]")
    ax.set_ylabel("方阵数量")
    ax.set_title("稀疏度分布(密度越低越稀疏)")
    style_axis(ax)
    secx = ax.secondary_xaxis("top")
    xt = np.arange(int(np.floor(logd.min())), int(np.ceil(logd.max())) + 1)
    ax.set_xticks(xt)
    secx.set_xticks(xt)
    secx.set_xticklabels([f"{10.0 ** v * 100:.2g}%" for v in xt],
                         fontsize=8, color=INK_MUTED)
    secx.tick_params(length=0)
    secx.set_xlabel("密度 %", color=INK_MUTED, fontsize=8)
    save(fig, "density_hist.png", outdir)

    # 2. 每行平均非零数
    npr = df["nnz_per_row"]
    npr = npr[npr > 0]
    lognpr = np.log10(npr)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.hist(lognpr, bins=60, color=C_SLOT1, edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.set_xlabel("每行平均非零数 = nnz / n   [log₁₀]")
    ax.set_ylabel("方阵数量")
    ax.set_title("每行非零密度分布")
    style_axis(ax)
    median_label(ax, np.log10(npr.median()), f"中位数 {npr.median():.1f}")
    save(fig, "nnz_per_row_hist.png", outdir)

    # 3. 维度 n 与 非零数 分布(双面板)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, title in [
        (axes[0], "n",   "方阵维度 n 分布"),
        (axes[1], "nnz", "非零数 nnz 分布"),
    ]:
        vals = np.log10(df[col])
        ax.hist(vals, bins=60, color=C_SLOT1, edgecolor=SURFACE,
                linewidth=0.6, zorder=3)
        ax.set_xlabel(f"log₁₀({col})")
        ax.set_ylabel("方阵数量")
        ax.set_title(title)
        style_axis(ax)
        med = df[col].median()
        median_label(ax, np.log10(med), f"中位数 {med:,.0f}")
    save(fig, "size_hist.png", outdir)

    # 4. 维度 n vs 非零数(对数-对数散点,单系列蓝色)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(df["n"], df["nnz"], s=12, alpha=0.45,
               edgecolors="none", color=C_SLOT1, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    # 先固定坐标范围为数据范围,避免参考线把范围撑爆
    ax.set_xlim(df["n"].min() * 0.8, df["n"].max() * 1.3)
    ax.set_ylim(df["nnz"].min() * 0.5, df["nnz"].max() * 2)
    # 参考线:每行非零数等值线 nnz = c·n(平行对角线 = 稀疏密度带)
    xs = np.logspace(np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1]), 50)
    for c in (1, 10, 100, 1000):
        ax.plot(xs, c * xs, color=INK_MUTED, linewidth=0.9, linestyle="--",
                zorder=2, label=f"{c} 非零/行")
    ax.set_xlabel("维度 n (log)")
    ax.set_ylabel("非零数 nnz (log)")
    ax.set_title("方阵维度 vs 非零数(虚线 = 每行非零数等值线)")
    ax.legend(frameon=False, fontsize=8, title="密度带", title_fontsize=8)
    style_axis(ax)
    save(fig, "rows_vs_nnz.png", outdir)

    print(f"\n完成。图表在 {outdir}")


if __name__ == "__main__":
    main()
