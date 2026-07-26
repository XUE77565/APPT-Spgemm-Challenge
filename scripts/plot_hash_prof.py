#!/usr/bin/env python3
"""Hash Design3 — profiling figures for the two innovation columns.

  symbolic_prof  — MinHash sizing vs HLL (ablation) vs opSparse exact-count
                   (1.6-3.3x faster than exact-count; matches HLL).
  accum_prof     — in-place hash accumulation vs sort-based ESC accumulation
                   (expand+sort+reduce), 11-16x cheaper.

Source: cudaEvent compute-only, H100 PCIe, double (this session). ESC phases from
the esc-prof run (also in plot_esc_share.py); hash phases from METHOD=hash prof.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TEAL, GRAY, AMBER, RED = "#1485A4", "#9AA6B2", "#E0A458", "#C00000"
INK, MUTED = "#1F2933", "#5C6773"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.8,
                     "axes.labelcolor": MUTED, "xtick.color": INK, "ytick.color": MUTED})

MATS = ["bcsstk30", "bcsstk32", "bcsstk31", "bcsstk29", "bcsstk17"]
# symbolic stage (ms)
OPS = [0.768, 0.608, 0.327, 0.209, 0.148]   # opSparse exact-count
HLL = [0.201, 0.219, 0.157, 0.108, 0.095]   # HLL (ablation)
MH  = [0.235, 0.238, 0.166, 0.113, 0.095]   # MinHash (ours)
# ESC accumulation = expand + sort + reduce ; hash accumulate (MinHash)
ESC_ACC = [34.72, 19.86, 9.22, 5.97, 3.85]
HACC    = [2.224, 1.380, 0.849, 0.486, 0.354]


def symbolic():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(MATS)); w = 0.26
    ax.bar(x - w, OPS, w, color=AMBER, edgecolor="white", lw=0.5, label="opSparse · exact-count")
    ax.bar(x,     HLL, w, color=GRAY,  edgecolor="white", lw=0.5, label="HLL (Ocean-style)")
    ax.bar(x + w, MH,  w, color=TEAL,  edgecolor="white", lw=0.5, label="MinHash (ours)")
    ax.set_yscale("log"); ax.set_ylim(0.05, 1.3)
    ax.set_xticks(x); ax.set_xticklabels(MATS, fontsize=10)
    ax.set_ylabel("symbolic stage time (ms)")
    for i in range(len(MATS)):
        r = OPS[i] / MH[i]
        ax.annotate(f"{r:.1f}×", (x[i] - w, OPS[i]), xytext=(0, 2),
                    textcoords="offset points", ha="center", color=RED,
                    fontsize=9, fontweight="bold")
    ax.text(0.985, 0.96, "opSparse → MinHash speedup", transform=ax.transAxes,
            ha="right", va="top", color=RED, fontsize=8.6)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_title("Symbolic stage — MinHash ≈ HLL, 1.6–3.3× faster than exact-count",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#ECEFF2", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_prof_symbolic.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/hash_prof_symbolic.{png,pdf}")


def accum():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(MATS)); w = 0.36
    ax.bar(x - w / 2, ESC_ACC, w, color=RED, edgecolor="white", lw=0.5,
           label="ESC · expand + sort + reduce")
    ax.bar(x + w / 2, HACC,    w, color=TEAL, edgecolor="white", lw=0.5,
           label="Hash SPA · in-place accumulate (ours)")
    ax.set_yscale("log"); ax.set_ylim(0.2, 60)
    ax.set_xticks(x); ax.set_xticklabels(MATS, fontsize=10)
    ax.set_ylabel("accumulation time (ms)")
    for i in range(len(MATS)):
        r = ESC_ACC[i] / HACC[i]
        ax.annotate(f"{r:.0f}×", (x[i], ESC_ACC[i]),
                    xytext=(0, 2), textcoords="offset points", ha="center", color=TEAL,
                    fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Accumulation — in-place hash dedup is 11–16× cheaper than sort-based ESC",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#ECEFF2", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_prof_accum.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/hash_prof_accum.{png,pdf}")


if __name__ == "__main__":
    symbolic()
    accum()
