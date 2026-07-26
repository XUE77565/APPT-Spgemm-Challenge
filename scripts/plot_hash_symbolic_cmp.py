#!/usr/bin/env python3
"""Symbolic-stage timing: MinHash vs exact-count (opSparse) — ONLY symbolic, nothing else.

Grouped bar, 5 structural matrices, two series:
  exact-count (opSparse, amber)  vs  MinHash (ours, teal).
log y; speedup annotated above each pair.

Source: cudaEvent compute-only (MinHash) / opSparse host timing (exact-count),
double, H100. Numbers from verified runs (compare/hll_vs_minhash + opSparse profile).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TEAL, AMBER = "#1485A4", "#E0A458"
INK, MUTED = "#1F2933", "#5C6773"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.linewidth": 0.8,
                     "axes.labelcolor": MUTED, "xtick.color": INK, "ytick.color": MUTED})

MATS = ["bcsstk30", "bcsstk32", "bcsstk31", "bcsstk29", "bcsstk17"]
EXACT = [0.768, 0.608, 0.327, 0.209, 0.148]   # opSparse exact-count symbolic (ms)
MH    = [0.235, 0.238, 0.166, 0.113, 0.095]   # MinHash symbolic (ms)


def main():
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    x = np.arange(len(MATS)); w = 0.38
    ax.bar(x - w / 2, EXACT, w, color=AMBER, edgecolor="white", lw=0.6, label="exact-count  (opSparse)")
    ax.bar(x + w / 2, MH,    w, color=TEAL,  edgecolor="white", lw=0.6, label="MinHash  (ours)")
    ax.set_yscale("log"); ax.set_ylim(0.05, 1.3)
    ax.set_xticks(x); ax.set_xticklabels(MATS, fontsize=10.5)
    ax.set_ylabel("symbolic stage time (ms)", fontsize=10.5)
    for i in range(len(MATS)):
        ax.text(x[i] - w / 2, EXACT[i] * 1.10, f"{EXACT[i]:.2f}", ha="center", va="bottom",
                color=MUTED, fontsize=8)
        ax.text(x[i] + w / 2, MH[i] * 1.10, f"{MH[i]:.2f}", ha="center", va="bottom",
                color=TEAL, fontsize=8, fontweight="bold")
        sp = EXACT[i] / MH[i]
        ax.text(x[i], EXACT[i] * 1.20, f"{sp:.1f}×", ha="center", va="center",
                color=TEAL, fontsize=10, fontweight="bold")
    ax.set_title("Symbolic stage — MinHash vs exact-count",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#ECEFF2", lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_symbolic_cmp.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/hash_symbolic_cmp.{png,pdf}")


if __name__ == "__main__":
    main()
