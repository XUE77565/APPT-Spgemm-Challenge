#!/usr/bin/env python3
"""
把两份 profile_aa.py 的 stdout 日志(各含一张"分阶段构成"表)画成一张
"每个方法两根柱(有无 arena)并排"的对比图。10 根堆叠柱在同一坐标系。

用法:
  python compare_pinned.py [no-arena.log] [arena.log] [out.png]
默认:
  no-arena.log = compare/pinned-and-arena/no-pinned+arena.log
  arena.log   = compare/pinned-and-arena/pinned+arena.log
  out.png     = compare/pinned-and-arena/per_method_compare.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "axes.titlecolor": "#0b0b0b",
    "axes.edgecolor": "#c3c2b7", "xtick.color": "#898781", "ytick.color": "#898781",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 10, "axes.titlesize": 12.5, "axes.titleweight": "bold", "figure.dpi": 130,
})

GROUPS = ["h2d", "d2h", "符号", "计算", "合并", "数值归并", "打包"]
COLOR = {"h2d": "#b8b8b0", "d2h": "#6f6f68", "符号": "#1baf7a", "计算": "#2a78d6",
         "合并": "#eda100", "数值归并": "#e34948", "打包": "#4a3aa7"}
ORDER = ["cuSPARSE", "Gustavson", "外积", "列向", "内积"]
MSET = set(ORDER)


def parse(logpath):
    """从 profile_aa.py 的 stdout 里抓"分阶段构成"表 -> {method: {group: ms}}。"""
    data, grab = {}, False
    for line in open(logpath, encoding="utf-8", errors="replace"):
        s = line.strip()
        if "分阶段构成" in s:
            grab = True
            continue
        if not grab or not s or s[0] in "=-" or s.startswith("方法"):
            continue
        toks = s.split()
        if len(toks) >= 8 and toks[0] in MSET:
            vals = toks[1:8]
            data[toks[0]] = {g: (0.0 if v == "--" else float(v)) for g, v in zip(GROUPS, vals)}
    return data


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    cd = os.path.join(root, "compare", "pinned-and-arena")
    no_log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(cd, "no-pinned+arena.log")
    yes_log = sys.argv[2] if len(sys.argv) > 2 else os.path.join(cd, "pinned+arena.log")
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(cd, "per_method_compare.png")

    nopin = parse(no_log)    # 无 arena(USE_MEMPOOL=0)
    pin = parse(yes_log)     # 有 arena(USE_MEMPOOL=1)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    x = np.arange(len(ORDER))
    w = 0.40
    for vi, (data, alpha) in enumerate([(nopin, 0.45), (pin, 1.0)]):
        offs = -w / 2 if vi == 0 else w / 2
        bottoms = np.zeros(len(ORDER))
        for g in GROUPS:
            vals = np.array([data[m][g] for m in ORDER])
            if vals.sum() == 0:
                continue
            ax.bar(x + offs, vals, w, bottom=bottoms, color=COLOR[g], alpha=alpha,
                   edgecolor="white", linewidth=0.5, zorder=3)
            bottoms += vals
        for xi, m in enumerate(ORDER):           # 柱顶标合计
            tot = sum(data[m].values())
            ax.annotate(f"{tot:.2f}", (xi + offs, tot), ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
    for xi, m in enumerate(ORDER):               # 每个方法上方标降幅
        t0, t1 = sum(nopin[m].values()), sum(pin[m].values())
        red = (t1 - t0) / t0 * 100
        ax.annotate(f"{red:+.0f}%", (xi, t0 * 1.04), ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#1baf7a" if red < 0 else "#e34948")

    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=11)
    ax.set_ylabel("耗时 (ms, 各矩阵均值)")
    ax.set_ylim(0, max(sum(nopin[m].values()) for m in ORDER) * 1.18)
    ax.set_title("有无 pinned+arena 对比 —— 每个方法两根柱:浅色(半透明)=无 arena,实色=有 arena;上方为合计降幅")
    ax.grid(axis="x", visible=False)

    ph = [Patch(facecolor=COLOR[g], edgecolor="none", label=g) for g in GROUPS]
    ph += [Patch(facecolor="#bbbbbb", alpha=0.45, edgecolor="none", label="无 arena (USE_MEMPOOL=0)"),
           Patch(facecolor="#bbbbbb", alpha=1.0, edgecolor="none", label="有 arena (USE_MEMPOOL=1)")]
    ax.legend(handles=ph, frameon=False, fontsize=8.5, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_pinned] {no_log}\n              vs {yes_log}\n  -> {out}")


if __name__ == "__main__":
    main()
