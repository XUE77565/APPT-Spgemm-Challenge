#!/usr/bin/env python3
"""
把两份 profile_aa.py 的 stdout 日志(各含一张"分阶段构成"表)画成一张
"每个方法两根柱(有无 arena)并排"的对比图。10 根堆叠柱在同一坐标系。

【动态解析】直接读表头那行得到 group 名,因此 profile_aa.py 的分组怎么变
(h2d/d2h 拆分、排序/去重/收尾 拆分…)都能自适应,不用改本脚本。

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

# 已知 group 的颜色(覆盖 h2d/d2h 拆分、排序/去重/收尾 拆分、压缩 等);未知 group 用灰。
COLOR = {
    "h2d": "#b8b8b0", "d2h": "#6f6f68",
    "符号": "#1baf7a", "计算": "#2a78d6",
    "排序": "#eda100", "去重": "#c98a1e", "收尾": "#a86620", "合并": "#eda100",
    "压缩": "#d4b48a",
    "数值归并": "#e34948", "打包": "#4a3aa7",
}
_FALLBACK = ["#7f7f7f", "#9467bd", "#8c564b", "#e377c2"]
ORDER = ["cuSPARSE", "Gustavson", "外积", "列向", "内积"]
MSET = set(ORDER)


def parse(logpath):
    """从 profile_aa.py 的 stdout 抓"分阶段构成"表。
    返回 (data, groups):data={method:{group:ms}}, groups=表头读到的 group 名列表(动态)。"""
    data, groups, grab = {}, None, False
    for line in open(logpath, encoding="utf-8", errors="replace"):
        s = line.strip()
        if "分阶段构成" in s:
            grab = True
            continue
        if not grab or not s or s[0] in "=-":
            continue
        toks = s.split()
        if toks[0] == "方法":                       # 表头:方法 g1 g2 … 合计
            groups = toks[1:-1]                     # 去掉首(方法)尾(合计)
            continue
        if groups and toks[0] in MSET:
            vals = toks[1:1 + len(groups)]
            data[toks[0]] = {g: (0.0 if v == "--" else float(v)) for g, v in zip(groups, vals)}
    return data, (groups or [])


def color_for(groups):
    """给每个 group 一个颜色:已知用 COLOR,未知的按 _FALLBACK 轮询。"""
    out, fi = {}, 0
    for g in groups:
        if g in COLOR:
            out[g] = COLOR[g]
        else:
            out[g] = _FALLBACK[fi % len(_FALLBACK)]; fi += 1
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    cd = os.path.join(root, "compare", "pinned-and-arena")
    no_log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(cd, "no-pinned+arena.log")
    yes_log = sys.argv[2] if len(sys.argv) > 2 else os.path.join(cd, "pinned+arena.log")
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(cd, "per_method_compare.png")

    nopin, g1 = parse(no_log)
    pin, g2 = parse(yes_log)
    groups = g1 or g2                            # 两份表头一致,取其一
    COL = color_for(groups)

    fig, ax = plt.subplots(figsize=(13, 6.4))
    x = np.arange(len(ORDER))
    w = 0.40
    for vi, (data, alpha) in enumerate([(nopin, 0.45), (pin, 1.0)]):
        offs = -w / 2 if vi == 0 else w / 2
        bottoms = np.zeros(len(ORDER))
        for g in groups:
            vals = np.array([data.get(m, {}).get(g, 0.0) for m in ORDER])
            if vals.sum() == 0:
                continue
            ax.bar(x + offs, vals, w, bottom=bottoms, color=COL[g], alpha=alpha,
                   edgecolor="white", linewidth=0.5, zorder=3)
            bottoms += vals
        for xi, m in enumerate(ORDER):
            tot = sum(data.get(m, {}).values())
            ax.annotate(f"{tot:.2f}", (xi + offs, tot), ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")
    for xi, m in enumerate(ORDER):
        t0 = sum(nopin.get(m, {}).values())
        t1 = sum(pin.get(m, {}).values())
        if t0 > 0:
            red = (t1 - t0) / t0 * 100
            ax.annotate(f"{red:+.0f}%", (xi, t0 * 1.04), ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="#1baf7a" if red < 0 else "#e34948")

    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=11)
    ax.set_ylabel("耗时 (ms, 各矩阵均值)")
    ymax = max((sum(nopin.get(m, {}).values()) for m in ORDER), default=0.0)
    ax.set_ylim(0, max(ymax, 0.1) * 1.18)
    ax.set_title("有无 pinned+arena 对比 —— 每个方法两根柱:浅色(半透明)=无 arena,实色=有 arena;上方为合计降幅")
    ax.grid(axis="x", visible=False)

    ph = [Patch(facecolor=COL[g], edgecolor="none", label=g) for g in groups]
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
