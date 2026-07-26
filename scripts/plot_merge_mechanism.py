#!/usr/bin/env python3
"""Merge Design2 — DESIGN: column-domain bucketing mechanism.

Left-to-right pipeline of spgemm_self_product_merge3 for one output row i:
  (1) k sorted column chains  {A[k,:] : k in A[i,:]}  (CSR ⇒ each chain column-sorted)
  (2) split the column range [0,n) into K disjoint buckets; per chain, an O(log nnz)
      lower_bound locates the chain's sub-range inside each bucket (no full scan)
  (3) one warp per (row, bucket) merges that bucket's sub-ranges (warp-shfl min + sum)
  (4) buckets are disjoint & ordered ⇒ concatenation is column-sorted CSR (no global sort)

Lower inset contrasts Gustavson (1 warp serially scans all k heads → heavy-row straggler).
Pure illustration.
"""
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

INK, MUTED, GRAY = "#1F2933", "#5C6773", "#C2CAD3"
TEAL, RED = "#1485A4", "#C00000"
SHADES = ["#D6ECF0", "#B0DCE3", "#7FC4CF", "#3FA0B2", "#1485A4"]  # K buckets, light→dark
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def _stage_label(ax, x, y, text, sub=None, color=INK):
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)
    if sub:
        ax.text(x, y - 0.055, sub, ha="center", va="center", fontsize=8.2, color=MUTED)


def draw_mechanism(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # ---- Stage 1: k sorted chains (right of stage 1 label) ----
    chains = [[1, 3, 7, 9, 14], [2, 4, 8, 12], [5, 6, 10, 11, 13], [0, 15]]
    x0, x1 = 1.0, 3.2
    y_top, y_bot = 4.7, 2.7
    ys = np.linspace(y_top, y_bot, len(chains))
    for ci, (ch, y) in enumerate(zip(chains, ys)):
        ax.text(x0 - 0.12, y, f"k{ci}", ha="right", va="center", fontsize=8.5, color=MUTED)
        xs = np.linspace(x0, x1, len(ch))
        for v, xv in zip(ch, xs):
            ax.add_patch(Rectangle((xv - 0.10, y - 0.12), 0.20, 0.24, fc="white",
                                   ec=GRAY, lw=0.9, zorder=3))
            ax.text(xv, y, str(v), ha="center", va="center", fontsize=7.2, color=INK, zorder=4)
    _stage_label(ax, 2.1, 5.35, "(1) k sorted chains", "row i reads A[k,:] · CSR ⇒ col-sorted")

    # K bucket vertical bands drawn over the chains region
    K = 5
    # column domain [0,16); bucket edges
    edges = np.array([0, 3.2, 6.4, 9.6, 12.8, 16.0])
    # map column value -> x position (same scale as chain xs)
    def vx(v):
        return x0 + (v / 16.0) * (x1 - x0)
    for k in range(K):
        xa, xb = vx(edges[k]), vx(edges[k + 1])
        ax.add_patch(Rectangle((xa, y_bot - 0.35), xb - xa, (y_top - y_bot) + 0.70,
                               fc=SHADES[k], ec="none", alpha=0.55, zorder=1))
        ax.text((xa + xb) / 2, y_top + 0.30, f"B{k}", ha="center", va="center",
                fontsize=8.2, color=SHADES[k], fontweight="bold")
    # dashed bucket dividers
    for e in edges[1:-1]:
        xv = vx(e)
        ax.add_line(Line2D([xv, xv], [y_bot - 0.35, y_top + 0.35], color="white", lw=1.4, zorder=2))

    # ---- Stage 2/3: K warps (one per bucket) ----
    wx0, wx1 = 4.4, 6.6
    for k in range(K):
        wy = np.linspace(4.55, 2.85, K)[k]
        ax.add_patch(FancyBboxPatch((wx0, wy - 0.20), wx1 - wx0, 0.40,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=SHADES[k], ec="white", lw=1.0, zorder=3))
        ax.text((wx0 + wx1) / 2, wy, f"warp {k}:  lower_bound + shfl-merge",
                ha="center", va="center", fontsize=7.6, color="white", fontweight="bold", zorder=4)
    _stage_label(ax, 5.5, 5.35, "(2-3) K warps, one per bucket",
                 "O(log nnz) sub-range · warp-shfl k-way merge")

    # arrow chains→warps region
    ax.add_patch(FancyArrow(3.35, 3.7, 0.95, 0, width=0.012, head_width=0.18, head_length=0.12,
                            color=MUTED, length_includes_head=True, zorder=2))

    # ---- Stage 4: concatenated sorted output ----
    out_vals = sorted([v for ch in chains for v in ch])
    ox0, ox1 = 7.7, 9.7
    xs = np.linspace(ox0, ox1, len(out_vals))
    oy = 3.7
    for v, xv in zip(out_vals, xs):
        ax.add_patch(Rectangle((xv - 0.10, oy - 0.13), 0.20, 0.26, fc="#EAF4F6",
                               ec=TEAL, lw=1.0, zorder=3))
        ax.text(xv, oy, str(v), ha="center", va="center", fontsize=7.2, color=TEAL, zorder=4)
    ax.text((ox0 + ox1) / 2, oy + 0.42, "B0  B1  B2  B3  B4  (already ordered)",
            ha="center", fontsize=8.0, color=MUTED)
    _stage_label(ax, 8.7, 5.35, "(4) concatenate = sorted CSR", "disjoint & ordered ⇒ no global sort")
    # arrow warps→output
    ax.add_patch(FancyArrow(6.75, 3.7, 0.85, 0, width=0.012, head_width=0.18, head_length=0.12,
                            color=MUTED, length_includes_head=True, zorder=2))

    # ---- Lower inset: Gustavson contrast ----
    iy = 0.95
    ax.add_patch(FancyBboxPatch((0.5, iy - 0.62), 9.0, 1.18,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc="#FBF1F1", ec="#E6C9C9", lw=1.0, zorder=1))
    ax.text(0.75, iy + 0.32, "Gustavson (classical):", fontsize=9.2, fontweight="bold", color=RED)
    ax.text(0.75, iy + 0.02,
            "1 warp / row serially scans all k chain heads (compare → min → advance).",
            fontsize=8.6, color=INK)
    ax.text(0.75, iy - 0.26,
            "Heavy row (large k, long chains) ⇒ one unit does all the work ⇒ straggler.",
            fontsize=8.6, color=INK)
    ax.text(0.75, iy - 0.48,
            "Column-domain bucketing gives the row K warps ⇒ K× intra-row parallelism, "
            "output still sorted.",
            fontsize=8.6, color=TEAL, fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    draw_mechanism(ax)
    fig.suptitle("Column-domain bucketing: split a row’s column range, merge each bucket in parallel",
                 fontsize=12.5, color=INK, y=0.98, fontweight="bold")
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/merge_mechanism.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/merge_mechanism.{png,pdf}")


if __name__ == "__main__":
    main()
