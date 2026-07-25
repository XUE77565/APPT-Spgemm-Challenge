#!/usr/bin/env python3
"""主图(论文用):大阵档对比 + small/large geomean 揭示套件假象。
  Panel A(左):7 个结构大阵(n>=1e4,排除退化对角阵 bcsstm25)× 6 方法 grouped bar,log y。
              故事:Auto 与 Ocean 竞争(bcsstk30/32 略输、17/18/29/31 赢),全面碾压 opSparse/HSMU/merge3。
  Panel B(右):small(n<1e4)vs large(n>=1e4)geomean → 揭示全 100 阵 geomean 被 78% 小阵 launch 开销主导
              (spECK 瘦管道小阵最强 → 全量 geomean 看似最优是假象),大阵档才是真实比较。
  口径:compute-only ms(double,H100)。配色 = dataviz 校验通过的 categorical 固定序(Auto=蓝 hero)。
用法:plot_main_figure.py <methods_cmp.csv> [out_png]
"""
import os, sys, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np

plt.rcParams.update({"font.family": ["DejaVu Sans"], "axes.unicode_minus": False})

# dataviz 校验通过的 categorical 固定序(PASS all hard checks, light mode)
# Auto(ours)= slot1 蓝 hero;Ocean= slot2 橙(主 SOTA 对手);其后基线
METHODS = [
    ("Auto",     "#2a78d6"),
    ("Ocean",    "#eb6834"),
    ("spECK",    "#1baf7a"),
    ("opSparse", "#eda100"),
    ("HSMU",     "#e87ba4"),
    ("merge3",   "#008300"),
]
SURFACE = "#fcfcfb"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"
GRID    = "#e6e5df"
EXCLUDE = {"bcsstm25"}   # 退化对角质量阵(cnnz=n), muddy 大阵故事

def fval(r, col):
    v = r.get(col, "")
    return float(v) if v not in (None, "", "None") else None

def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "compare/method_cmp_20260724_145121/methods_cmp.csv"
    out_png  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(csv_path), "main_figure_large.png")
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["_n"] = int(r["n"]) if r.get("n") else 0

    # ============ Panel A:大阵(n>=1e4,排除退化)grouped bar ============
    large = [r for r in rows if r["_n"] >= 10000 and r["matrix"] not in EXCLUDE]
    large.sort(key=lambda r: r["_n"])
    mats = [r["matrix"] for r in large]
    nx = np.arange(len(mats))
    nm = len(METHODS)
    w = 0.125                      # 薄柱;组内相邻柱间留 surface gap(slot=略宽于 w)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [2.7, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (axA, axB):
        ax.set_facecolor(SURFACE)

    for i, (m, color) in enumerate(METHODS):
        vals = [fval(r, m) for r in large]
        ys = [v if v else 1e-3 for v in vals]
        offset = (i - (nm - 1) / 2) * (w + 0.006)   # 相邻柱间 ~0.006 surface gap
        bars = axA.bar(nx + offset, ys, width=w, color=color, label=m, zorder=3,
                       edgecolor=SURFACE, linewidth=0.6)
        # 选择性直标:Auto(ours)+ Ocean(对手)标 ms 值(其余靠图例+位置)
        if m in ("Auto", "Ocean"):
            for j, v in enumerate(vals):
                if v:
                    axA.text(nx[j] + offset, v * 1.07, f"{v:.2f}", ha="center", va="bottom",
                             fontsize=6.3, color=INK_SEC, rotation=90)
    axA.set_yscale("log")
    axA.set_xticks(nx)
    axA.set_xticklabels([f"{m}\n(n={r['_n']})" for m, r in zip(mats, large)], fontsize=8)
    axA.set_ylabel("compute-only time (ms, log)", color=INK_SEC, fontsize=9)
    axA.set_title("(A)  Large matrices  (n ≥ 10⁴, structural; Auto all dispatch hash)",
                  color=INK_PRI, fontsize=10, loc="left", pad=8)
    axA.grid(axis="y", which="both", color=GRID, linewidth=0.6, zorder=0)
    axA.set_axisbelow(True)
    for s in ("top", "right"): axA.spines[s].set_visible(False)
    for s in ("left", "bottom"): axA.spines[s].set_color(GRID)
    axA.tick_params(colors=INK_SEC, labelsize=8)

    # ============ Panel B:small vs large geomean(揭示套件假象)============
    small = [r for r in rows if r["_n"] < 10000]
    large_all = [r for r in rows if r["_n"] >= 10000 and r["matrix"] not in EXCLUDE]
    groups = [("small\n(n<10⁴, %d mats)" % len(small), small),
              ("large\n(n≥10⁴, %d mats)" % len(large_all), large_all)]
    gx = np.arange(len(groups))
    for i, (m, color) in enumerate(METHODS):
        ys = [geomean([fval(r, m) for r in sub]) for _, sub in groups]
        ys = [y if y else 1e-3 for y in ys]
        offset = (i - (nm - 1) / 2) * (w + 0.006)
        axB.bar(gx + offset, ys, width=w, color=color, label=m, zorder=3,
                edgecolor=SURFACE, linewidth=0.6)
        # 标 large 组的 geomean 值(关键:大阵档真实数字)
        for j, (_, sub) in enumerate(groups):
            g = geomean([fval(r, m) for r in sub])
            if g and j == 1:   # 只标 large
                axB.text(gx[j] + offset, g * 1.08, f"{g:.2f}", ha="center", va="bottom",
                         fontsize=6.2, color=INK_SEC, rotation=90)
    axB.set_yscale("log")
    axB.set_xticks(gx)
    axB.set_xticklabels([g[0] for g in groups], fontsize=8)
    axB.set_ylabel("geomean compute-only (ms, log)", color=INK_SEC, fontsize=9)
    axB.set_title("(B)  Suite artifact: small-matrix launch cost\n dominates the all-100 geomean",
                  color=INK_PRI, fontsize=10, loc="left", pad=8)
    axB.grid(axis="y", which="both", color=GRID, linewidth=0.6, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"): axB.spines[s].set_visible(False)
    for s in ("left", "bottom"): axB.spines[s].set_color(GRID)
    axB.tick_params(colors=INK_SEC, labelsize=8)

    # 共用图例(固定序,identity 不靠颜色单)
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=nm,
               frameon=False, fontsize=9, columnspacing=1.6, handletextpad=0.5)
    fig.suptitle("SpGEMM C=A·A on H100 (double, compute-only): large-matrix tier is the honest comparison",
                 color=INK_PRI, fontsize=11, y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, bbox_inches="tight", facecolor=SURFACE, dpi=160)
    plt.close(fig)
    print(f"主图: {out_png}")
    # 打印大阵档数字(便于核验/入表)
    print("\nPanel A 大阵(compute-only ms):")
    print("matrix        n      " + "".join(f"{m:>10}" for m, _ in METHODS))
    for r in large:
        print(f"{r['matrix']:<12} {r['_n']:<7}" + "".join(
            (f"{fval(r,m):>10.3f}" if fval(r,m) else f"{'DNF':>10}") for m, _ in METHODS))
    print("\nPanel B geomean:")
    for gname, sub in groups:
        cells = []
        for m, _ in METHODS:
            gm = geomean([fval(r, m) for r in sub])
            cells.append(f"{gm:.3f}" if gm else "-")
        print(f"  {gname.replace(chr(10),' '):<24}" + "".join(f"{m}={c}  " for (m, _), c in zip(METHODS, cells)))

if __name__ == "__main__":
    main()
