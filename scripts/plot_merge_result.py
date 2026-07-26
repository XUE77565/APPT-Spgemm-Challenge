#!/usr/bin/env python3
"""Merge Design2 — result figure: estimation-based (flop_ub) sizing vs exact count-merge.

Panel A: compute-only (cudaEvent, double, H100 PCIe) grouped bar across 6 structural
         matrices. exact-count (red, the eliminated double-merge) vs flop_ub (teal, ours).
         Annotate per-matrix savings.
Panel B: phase breakdown for bcsstk30 (representative) — count-merge (red) is replaced by
         a cheap lower_bound+sum sizing pass; the merge core (teal) is unchanged.

Data: measured this session via `METHOD=merge3 MRG3_FLOP_UB={0,1} ... | grep mrg3-prof`,
      compute-only = TOTAL(GPU) - h2d - d2h. See worklog adaptive_hash_2026-07-21.md §32.
Palette: deck (teal #1485A4 ours / red #C00000 bottleneck / gray #6B7280 overhead).
"""
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TEAL, RED, GRAY = "#1485A4", "#C00000", "#6B7280"
INK, MUTED = "#1F2933", "#5C6773"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
})

# (matrix, exact_count_compute, flop_ub_compute, count_phase)
DATA = [
    ("bcsstk08", 1.118, 0.855, 0.362),
    ("bcsstk17", 1.575, 1.188, 0.586),
    ("bcsstk29", 1.863, 1.413, 0.702),
    ("bcsstk31", 4.069, 3.108, 1.548),
    ("bcsstk32", 7.531, 5.682, 2.520),
    ("bcsstk30", 16.624, 11.209, 6.156),
]
# bcsstk30 phase breakdown (flop_ub path), ms
EXACT = {"count": 6.156, "scan": 0.043, "merge": 10.425}
FLOP  = {"sizing": 0.837, "merge": 10.372}   # sizing = flop+fscan+rscan+compact


def draw_result(axA, axB):
    # ---- Panel A: grouped compute-only bar ----
    mats = [d[0] for d in DATA]
    exact = np.array([d[1] for d in DATA])
    flop = np.array([d[2] for d in DATA])
    x = np.arange(len(mats)); w = 0.38
    axA.bar(x - w/2, exact, w, color=RED,  label="exact count-merge", zorder=3)
    axA.bar(x + w/2, flop,  w, color=TEAL, label="flop_ub sizing (ours)", zorder=3)
    axA.set_yscale("log")
    axA.set_ylim(0.5, 40)
    axA.set_xticks(x); axA.set_xticklabels(mats, rotation=30, ha="right")
    axA.set_ylabel("compute-only time (ms)")
    for i in range(len(mats)):
        save = 100 * (exact[i] - flop[i]) / exact[i]
        axA.annotate(f"−{save:.0f}%", (x[i], flop[i]),
                     textcoords="offset points", xytext=(0, -14),
                     ha="center", color=TEAL, fontsize=9.5, fontweight="bold")
    axA.set_title("(A)  Estimation sizing replaces the double-merge count pass",
                  fontsize=11.5, color=INK, pad=8, loc="left")
    axA.legend(frameon=False, loc="upper left", fontsize=9.5)
    axA.grid(axis="y", which="both", color="#E5E7EB", linewidth=0.6, zorder=0)
    axA.set_axisbelow(True)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- Panel B: phase breakdown, bcsstk30 ----
    labels = ["exact count-merge", "flop_ub sizing\n(ours)"]
    # stack order bottom->top: merge (teal), then symbolic stage
    merge_h = [EXACT["merge"], FLOP["merge"]]
    sym_h   = [EXACT["count"] + EXACT["scan"], FLOP["sizing"]]
    xb = np.array([0, 1])
    axB.bar(xb, merge_h, 0.5, color=TEAL, zorder=3, label="warp merge (core)")
    axB.bar(xb, sym_h, 0.5, bottom=merge_h, color=[RED, GRAY], zorder=3)
    for i in range(2):
        tot = merge_h[i] + sym_h[i]
        axB.annotate(f"{tot:.2f} ms", (xb[i], tot), textcoords="offset points",
                     xytext=(0, 6), ha="center", color=INK, fontsize=10.5, fontweight="bold")
        # label the symbolic segment
        axB.annotate(f"{sym_h[i]:.2f}", (xb[i], merge_h[i] + sym_h[i]/2),
                     textcoords="offset points", xytext=(0, 0), ha="center",
                     color="white", fontsize=9, fontweight="bold")
        axB.annotate(f"{merge_h[i]:.2f}", (xb[i], merge_h[i]/2),
                     textcoords="offset points", xytext=(0, 0), ha="center",
                     color="white", fontsize=9, fontweight="bold")
    axB.set_xticks(xb); axB.set_xticklabels(labels, fontsize=10)
    axB.set_ylim(0, 19)
    axB.set_ylabel("compute-only time (ms)")
    axB.set_title("(B)  bcsstk30 phase breakdown  —  count 6.16 ms → sizing 0.84 ms",
                  fontsize=11.5, color=INK, pad=8, loc="left")
    axB.annotate("merge core\nunchanged", xy=(0.25, 10.4), xytext=(0.62, 14.5),
                 color=TEAL, fontsize=9, ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))
    axB.annotate("count-merge\neliminated", xy=(0.0, 13.7), xytext=(0.30, 5.5),
                 color=RED, fontsize=9, ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    axB.legend([Patch(color=TEAL), Patch(color=GRAY)],
               ["warp merge (core)", "symbolic sizing"],
               frameon=False, loc="upper right", fontsize=9.5)
    axB.grid(axis="y", color="#E5E7EB", linewidth=0.6, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    draw_result(axA, axB)
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/merge_result.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    print("wrote fig/merge_result.{png,pdf}")


if __name__ == "__main__":
    main()
