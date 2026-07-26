#!/usr/bin/env python3
"""merge3 (column-domain) vs merge2 (1 warp/row) — clean grouped bar, small-matrix wins."""
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TEAL, AMBER = "#1485A4", "#E0A458"   # ours / baseline (validated pair)
INK, MUTED = "#1F2A33", "#9AA6B2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK})

# (matrix, n, merge2_compute, merge3_compute) — compute-only, double, H100, this session
DATA = [
    ("can_256",   256, 0.313, 0.176),
    ("bp_0",      822, 0.933, 0.377),
    ("bcsstk08", 1074, 2.374, 0.870),
]


def draw(ax):
    mats = [d[0] for d in DATA]
    ns   = [d[1] for d in DATA]
    m2   = np.array([d[2] for d in DATA])
    m3   = np.array([d[3] for d in DATA])
    x = np.arange(len(mats)); w = 0.36
    ax.bar(x - w/2, m2, w, color=AMBER, zorder=3)
    ax.bar(x + w/2, m3, w, color=TEAL,  zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\nn={n:,}" for m, n in zip(mats, ns)], fontsize=10.5)
    ax.set_ylabel("compute-only time (ms)")
    for i in range(len(mats)):
        ax.annotate(f"{m2[i]:.2f}", (x[i]-w/2, m2[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", color=AMBER, fontsize=9.5, fontweight="bold")
        ax.annotate(f"{m3[i]:.2f}", (x[i]+w/2, m3[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", color=TEAL, fontsize=9.5, fontweight="bold")
        sp = m2[i] / m3[i]
        # place the speedup directly above the taller (merge2) bar, clear of the legend
        ax.annotate(f"{sp:.1f}×", (x[i], m2[i]), xytext=(0, 22), textcoords="offset points",
                    ha="center", color=TEAL, fontsize=12, fontweight="bold")
    ax.legend([Patch(facecolor=AMBER), Patch(facecolor=TEAL)],
              ["merge2  ·  1 warp / row", "merge3  ·  column-domain (ours)"],
              frameon=False, loc="upper left", fontsize=10, ncol=1)
    ax.grid(axis="y", which="major", color="#ECEFF2", linewidth=0.8, zorder=0)
    ax.grid(axis="y", which="minor", color="#F4F6F8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    draw(ax)
    fig.suptitle("On small matrices, column-domain bucketing (merge3) beats 1-warp merge (merge2)",
                 fontsize=12.5, color=INK, y=0.98, x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, -0.02, "compute-only (cudaEvent), double, H100 PCIe.  "
                          "Matrices with compute > 0.3 ms (above launch-overhead noise).",
             ha="left", color=MUTED, fontsize=8.3)
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/merge2_vs_merge3.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/merge2_vs_merge3.{png,pdf}")


if __name__ == "__main__":
    main()
