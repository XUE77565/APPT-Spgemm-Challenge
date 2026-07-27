#!/usr/bin/env python3
"""读 compare_methods.py 产出的 methods_cmp.csv → 仿 profile_methods_bar.png 出柱状图 + 类别聚合表。
按 A 密度分 4 类(Dense/Mildly/Highly/Extremely),每类一组柱(Ocean/HSMU/cu/serial/merge3/Auto),log y。
用法:plot_methods_cmp.py <methods_cmp.csv> [out_png]
"""
import os, sys, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)
plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})

CLASS_ORDER = ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]

def classify(d):
    try: d = float(d)
    except: return "?"
    if d >= 10.0: return "Dense"
    if d >= 1.0:  return "Mildly sparse"
    if d >= 0.1:  return "Highly sparse"
    return "Extremely sparse"

def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "compare/methods_cmp.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(csv_path)[0] + "_bar.png"
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["_class"] = classify(r.get("density_pct", ""))

    methods = [("cu", "#2a78d6"), ("opSparse", "#eb6834"), ("HSMU", "#1baf7a"),
               ("dense", "#eda100"), ("Auto", "#1485A4")]
    colmap = {"cu": "cu", "opSparse": "opSparse", "HSMU": "HSMU", "dense": "dense", "Auto": "Auto"}

    classes = [c for c in CLASS_ORDER if any(r["_class"] == c for r in rows)]

    # ---- 类别聚合表(几何均值 ms)----
    print("=" * 92)
    print("类别聚合(几何均值 ms,空=无数据)")
    print("-" * 92)
    hdr = f"{'class(n)':<22}" + "".join(f"{m:>14}" for m, _ in methods)
    print(hdr)
    for c in classes:
        sub = [r for r in rows if r["_class"] == c]
        cells = []
        for m, _ in methods:
            col = colmap[m]
            vals = [float(r[col]) for r in sub if r.get(col) not in (None, "", "None")]
            cells.append(f"{geomean(vals):>14.3f}" if vals else f"{'-':>14}")
        print(f"{c+'('+str(len(sub))+')':<22}" + "".join(cells))
    # 总体几何均值
    allc = []
    for m, _ in methods:
        col = colmap[m]
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "None")]
        allc.append(f"{geomean(vals):>14.3f}" if vals else f"{'-':>14}")
    print(f"{'ALL('+str(len(rows))+')':<22}" + "".join(allc))
    # Auto 选择统计
    n_hash = sum(1 for r in rows if r.get("Auto_choice") == "hash")
    n_m3 = sum(1 for r in rows if r.get("Auto_choice", "").startswith("merge3"))
    print(f"\nAuto 选择:hash {n_hash} / merge3 {n_m3}")
    print("=" * 92)

    # ---- 柱状图(按类别分组,log y)----
    means = {}
    for m, _ in methods:
        col = colmap[m]
        means[m] = []
        for c in classes:
            sub = [r for r in rows if r["_class"] == c]
            vals = [float(r[col]) for r in sub if r.get(col) not in (None, "", "None")]
            means[m].append(geomean(vals) if vals else np.nan)

    x = np.arange(len(classes))
    nm = len(methods)
    w = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (lab, color) in enumerate(methods):
        vals = np.array([v if not np.isnan(v) else 1e-3 for v in means[lab]])
        ax.bar(x + (i - (nm - 1) / 2) * w, vals, width=w, color=color, label=lab, zorder=3,
               edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({sum(1 for r in rows if r['_class']==c)})" for c in classes])
    ax.set_ylabel("耗时 (ms, 对数, 几何均值)")
    ax.set_title("5 方法对照(按 A 密度类别):cuSPARSE / opSparse / HSMU / dense / Auto(ours)")
    ax.legend(frameon=False, fontsize=8.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.00))
    ax.grid(axis="y", which="both", color="#e1e0d9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图: {out_png}")

if __name__ == "__main__":
    main()
