#!/usr/bin/env python3
"""
对比 A/B 两份 profile_aa_<tag>.csv(分阶段明细)的 d2h / h2d / 合计。
产出:
  1) 对比表 → 打印到屏幕 + 追加写日志(默认与第一份 CSV 同目录的 ab_compare.log)
  2) 对比图 → charts/ab_compare.png(legacy vs pool 并排柱 + 百分比标注)

用法:
  python compare_ab.py <legacy.csv> <pool.csv> [log_path]
"""
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _HAS_MPL = True
except Exception as e:                      # matplotlib 不可用时退化为只出表
    _HAS_MPL = False
    _MPL_ERR = e

NAME = {"cu": "cuSPARSE", "gust": "Gustavson", "outer": "外积", "colw": "列向", "inner": "内积"}
TAGS = ["cu", "gust", "outer", "colw", "inner"]
ALLP = ["h2d", "csc", "count", "scan", "expand", "workest", "compute", "copy",
        "sort", "reduce", "final", "numeric", "pack", "d2h"]

# ---- 画图风格(与 profile_aa.py / profile_att.py 一致)----
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if _HAS_MPL and os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
C_LEG, C_POOL = "#898781", "#1baf7a"   # legacy 灰 / pool 绿(更快)
C_GOOD, C_BAD = "#1baf7a", "#e34948"   # 变快 绿 / 变慢 红


def stat(df, tag):
    """返回每指标的 Series(便于取 median/mean 等任意统计量)。"""
    s = df[df.tag == tag]
    return {
        "h2d": s["h2d"],
        "d2h": s["d2h"],
        "total": s[ALLP].sum(axis=1, min_count=1),
    }


def cell(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return f"{b:9.3f}"
    return f"{b:7.3f} ({(b-a)/a*100:+5.0f}%)"


def plot(stats_leg, stats_pool, out_path):
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "axes.titlecolor": "#0b0b0b",
        "axes.edgecolor": "#c3c2b7", "xtick.color": "#898781", "ytick.color": "#898781",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 130,
    })
    metrics = [("d2h", "d2h(下载 C)"), ("h2d", "h2d(上传 A)"), ("total", "合计")]
    names = [NAME[t] for t in TAGS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    x = np.arange(len(TAGS))
    w = 0.38
    for ax, (key, title) in zip(axes, metrics):
        leg = [stats_leg[t][key] for t in TAGS]
        pool = [stats_pool[t][key] for t in TAGS]
        ax.bar(x - w / 2, leg, w, color=C_LEG, label="legacy", zorder=3)
        ax.bar(x + w / 2, pool, w, color=C_POOL, label="pool", zorder=3)
        for xi, a, b in zip(x, leg, pool):
            if a and not pd.isna(a) and b and not pd.isna(b):
                pct = (b - a) / a * 100
                ax.annotate(f"{pct:+.0f}%", (xi + w / 2, b), ha="center", va="bottom",
                            fontsize=8.5, fontweight="bold",
                            color=(C_GOOD if pct < 0 else C_BAD),
                            xytext=(0, 2), textcoords="offset points")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("ms")
        ax.grid(axis="x", visible=False)
    axes[-1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Pinned 内存池 A/B:legacy vs pool(100 矩阵均值)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    leg_path, pool_path = sys.argv[1], sys.argv[2]
    base_dir = os.path.dirname(os.path.abspath(leg_path))
    log_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(base_dir, "ab_compare.log")
    chart_path = os.path.join(base_dir, "charts", "ab_compare.png")

    leg = pd.read_csv(leg_path, na_values=[""])
    pool = pd.read_csv(pool_path, na_values=[""])

    def agg(series, fn):
        s = series.dropna()
        return fn(s) if s.size else np.nan

    # 中位数(鲁棒于个别大矩阵的 cudaMalloc 抖动)为主;均值仅作参考
    sleg, spool = {t: stat(leg, t) for t in TAGS}, {t: stat(pool, t) for t in TAGS}
    med_leg  = {t: {k: agg(sleg[t][k], np.median)   for k in ("h2d", "d2h", "total")} for t in TAGS}
    med_pool = {t: {k: agg(spool[t][k], np.median)  for k in ("h2d", "d2h", "total")} for t in TAGS}
    mean_leg  = {t: {k: agg(sleg[t][k], np.mean)    for k in ("h2d", "d2h", "total")} for t in TAGS}
    mean_pool = {t: {k: agg(spool[t][k], np.mean)   for k in ("h2d", "d2h", "total")} for t in TAGS}

    lines = []

    def out(s=""):
        lines.append(s)

    def emit_table(a_dict, b_dict):
        out(f"{'方法':<12}{'d2h':>22}{'h2d':>22}{'合计':>16}")
        out(f"{'':<12}{'legacy → pool':>22}{'legacy → pool':>22}{'legacy → pool':>16}")
        for t in TAGS:
            a, b = a_dict[t], b_dict[t]
            out(f"{NAME[t]:<12}{cell(a['d2h'], b['d2h']):>22}"
                f"{cell(a['h2d'], b['h2d']):>22}{cell(a['total'], b['total']):>16}")

    out(datetime.now().strftime("# %Y-%m-%d %H:%M:%S  A/B compare"))
    out(f"legacy: {leg_path} ({len(leg[leg.tag=='cu'])} matrices/method)")
    out(f"pool  : {pool_path}")
    out("=" * 74)
    out("【均值 mean】聚合视图 —— 含大矩阵真实的 h2d 副作用(负% = pool 更快)")
    emit_table(mean_leg, mean_pool)
    out("=" * 74)
    out("【中位数 median】典型矩阵视图 —— 多为小矩阵,h2d 副作用不显")
    emit_table(med_leg, med_pool)
    out("=" * 74)
    out("说明:h2d 在【均值】里偏高是【真实副作用,不是噪声】:锁大块 pinned 拖慢了大")
    out("     矩阵(bcsstk30/32)A 的 H2D memcpy;典型小矩阵不受影响,故【中位数】里 h2d≈不变。")
    out("     d2h 在两个视图里都是大幅下降(pool 的目标,成立)。")

    report = "\n".join(lines)
    print(report)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(report + "\n\n")
    print(f"\n[已追加到日志] {log_path}")

    if _HAS_MPL:
        plot(mean_leg, mean_pool, chart_path)
        print(f"[对比图] {chart_path}")
    else:
        print(f"[跳过对比图] matplotlib 不可用({_MPL_ERR})")


if __name__ == "__main__":
    main()
