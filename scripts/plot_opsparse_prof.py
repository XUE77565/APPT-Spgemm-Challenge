#!/usr/bin/env python3
"""opSparse runtime profiling — symbolic vs accumulation (numeric) vs overhead.

opSparse (Liu 2022, exact hash-count symbolic) stage breakdown, cudaEvent-free
host timing reported by opSparse itself, double, H100. Stacked absolute-time bars
show both the time (height) and the share (%) inside each segment.

  symbolic     = symbolic + symbolic_binning  (exact-count sizing; our MinHash is 3.3x faster)
  accumulation = numeric + numeric_binning + reduce  (hash insert/accumulate)
  overhead     = setup + prefix + allocate + cleanup

Output: fig/opsparse_prof.{png,pdf}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

AMBER, GRAY, LIGHT = "#E0A458", "#5C6773", "#D7DCE1"
INK, MUTED, TEAL = "#1F2933", "#9AA6B2", "#1485A4"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none",
                     "axes.labelcolor": MUTED, "xtick.color": INK, "ytick.color": MUTED})

MATS = ["bcsstk30", "bcsstk32", "bcsstk31", "bcsstk29", "bcsstk17"]
SYM = np.array([0.816, 0.650, 0.429, 0.251, 0.184])
ACC = np.array([1.575, 0.990, 0.689, 0.414, 0.302])
OH  = np.array([0.654, 0.781, 1.001, 0.886, 0.596])
TOT = SYM + ACC + OH


def main():
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    x = np.arange(len(MATS))
    b1 = ax.bar(x, SYM, 0.55, color=AMBER, edgecolor="white", lw=0.6, label="symbolic  (exact-count)")
    b2 = ax.bar(x, ACC, 0.55, bottom=SYM, color=GRAY, edgecolor="white", lw=0.6, label="accumulation  (numeric)")
    b3 = ax.bar(x, OH,  0.55, bottom=SYM + ACC, color=LIGHT, edgecolor="white", lw=0.6, label="overhead  (setup / alloc)")
    ax.set_ylim(0, 3.5)
    ax.set_xticks(x); ax.set_xticklabels(MATS, fontsize=10)
    ax.set_ylabel("opSparse runtime (ms)")

    def lbl(rects, vals, color):
        for r, v in zip(rects, vals):
            h = r.get_height()
            yc = r.get_y() + h / 2
            if h > 0.18:                       # only label segments tall enough
                ax.text(r.get_x() + r.get_width() / 2, yc, f"{v / TOT[list(rects).index(r)] * 100:.0f}%",
                        ha="center", va="center", color=color, fontsize=9, fontweight="bold")
    lbl(b1, SYM, INK); lbl(b2, ACC, "white"); lbl(b3, OH, INK)
    # total time on top
    for i, t in enumerate(TOT):
        ax.text(x[i], t + 0.08, f"{t:.2f} ms", ha="center", va="bottom", color=INK, fontsize=9)

    ax.set_title("opSparse runtime — symbolic + accumulation dominate on large matrices",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left")
    ax.text(0.0, 1.04, "symbolic = exact-count sizing  (our MinHash is 3.3× faster on bcsstk30)",
            transform=ax.transAxes, fontsize=8.8, color=TEAL, style="italic")
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#ECEFF2", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/opsparse_prof.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/opsparse_prof.{png,pdf}")


if __name__ == "__main__":
    main()
