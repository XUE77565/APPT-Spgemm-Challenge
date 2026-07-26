#!/usr/bin/env python3
"""Hash Design3 — DATA: MinHash sizing vs HLL (ablation) and vs exact-count (opSparse).

Panel A — honest trade-off vs HLL: compute-only (ms, log) on the estimator-active
          large matrices, with over-alloc annotated. MinHash pays an 8–19% premium
          entirely from a slightly looser over-alloc (3.0–3.14x vs 2.88–2.99x).
Panel B — structural win vs exact-count: the symbolic (sizing) stage on bcsstk30 —
          MinHash 0.23 ms vs opSparse's exact hash-count 0.82 ms (~3.5x faster),
          matching HLL throughput.

Source: cudaEvent compute-only, H100 PCIe, double (compare/hll_vs_minhash_*.csv;
inno/hash_innovation_kmv.md §4–5). Numbers hardcoded from verified runs.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, GRAY = "#1F2933", "#5C6773", "#C9D1D9"
TEAL, TEAL_LT, RED = "#1485A4", "#9FCFD8", "#C00000"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none",
                     "axes.labelcolor": MUTED, "xtick.color": INK, "ytick.color": MUTED})

# ---- data (compute-only ms, double, verified) ----
MATS = ["bcsstk30", "bcsstk32", "bcsstk33"]
HLL_C = [2.71, 1.93, 0.96]
MH_C = [3.26, 2.06, 1.08]
HLL_O = [2.88, 2.99, 2.90]
MH_O = [3.14, 3.00, 2.96]
# symbolic stage (bcsstk30)
SYM = {"opSparse": 0.82, "HLL": 0.21, "MinHash": 0.23}


def panel_a(ax):
    x = np.arange(len(MATS))
    w = 0.36
    b1 = ax.bar(x - w / 2, HLL_C, w, color=GRAY, edgecolor="white", lw=0.6, label="HLL (ablation)")
    b2 = ax.bar(x + w / 2, MH_C, w, color=TEAL, edgecolor="white", lw=0.6, label="MinHash (ours)")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 6)
    ax.set_xticks(x); ax.set_xticklabels(MATS)
    ax.set_ylabel("compute-only (ms)")
    ax.set_title("A · MinHash vs HLL — honest cost", fontsize=11, color=INK, fontweight="bold", loc="left")
    # over-alloc annotations above each pair
    for i in range(len(MATS)):
        ax.text(x[i] - w / 2, HLL_C[i] * 1.10, f"{HLL_O[i]:.2f}×", ha="center", va="bottom",
                color=MUTED, fontsize=8)
        ax.text(x[i] + w / 2, MH_C[i] * 1.10, f"{MH_O[i]:.2f}×", ha="center", va="bottom",
                color=TEAL, fontsize=8, fontweight="bold")
    ax.text(0.985, 0.94, "over-alloc factor", transform=ax.transAxes, ha="right", va="top",
            color=MUTED, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.98, 0.80))
    ax.text(0.02, 0.06, "+8–19% compute, all from looser over-alloc",
            transform=ax.transAxes, color=INK, fontsize=8.8, style="italic")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#ECEEF1", lw=0.8)


def panel_b(ax):
    items = list(SYM.items())
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [RED, GRAY, TEAL]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", lw=0.6, width=0.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("symbolic stage (ms)")
    ax.set_title("B · vs exact-count (opSparse) — 3.5× faster", fontsize=11, color=INK,
                 fontweight="bold", loc="left")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                color=INK, fontsize=9.5, fontweight="bold")
    # 3.5x bracket between opSparse and MinHash
    ax.annotate("", xy=(2, 0.30), xytext=(0, 0.95),
                arrowprops=dict(arrowstyle="-", color=RED, lw=1.0))
    ax.text(1.0, 0.97, "≈3.5× faster", ha="center", va="bottom", color=RED,
            fontsize=9.5, fontweight="bold")
    ax.text(0.02, 0.06, "bcsstk30 · sizing stage; MinHash keeps the hash family's edge over exact count",
            transform=ax.transAxes, color=INK, fontsize=8.5, style="italic")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color="#ECEEF1", lw=0.8)


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 4.7),
                                 gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.28})
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.16)
    panel_a(a1)
    panel_b(a2)
    fig.suptitle("MinHash sizing — orthogonal to HLL, 3.5× faster than exact symbolic count",
                 fontsize=13.5, color=INK, y=0.965, fontweight="bold")
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_result.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/hash_result.{png,pdf}")


if __name__ == "__main__":
    main()
