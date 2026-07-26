#!/usr/bin/env python3
"""Hash Design3 — INSIGHT: the accumulation-method landscape of SpGEMM.

Three ways to turn a row's intermediate products into C[i,:]:
  (1) Sort-based (ESC / Gustavson) — materialize all O(flop) products, sort, reduce.
  (2) Merge-based (column-domain, Design 2) — bucket-merge all O(flop) products.
  (3) Hash SPA (ours) — a per-row SMEM hash table collapses duplicates IN PLACE
      during accumulation; only the distinct C[i,:] entries are ever written.
  (4) MinHash sizing (ours) — a per-row distinct upper bound right-sizes the table,
      orthogonal to Ocean's HLL.

2x2 iconic schematic: panels 1-2 neutral gray (prior), panels 3-4 teal (ours),
panel 1 carries the red "accumulation bottleneck" marker (the motivation).
Pure illustration (measured numbers cited as labels only).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch

INK, MUTED, GRAY = "#1F2933", "#5C6773", "#9AA6B2"
TEAL, RED = "#1485A4", "#C00000"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none"})


def _dots(ax, xs, y, r, color):
    for x in xs:
        ax.add_patch(plt.Circle((x, y), r, color=color, zorder=3))


def _arrow(ax, x, y, dx, dy, color, lw=1.4):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.006, head_width=0.035,
                            head_length=0.022, color=color, lw=lw,
                            length_includes_head=True, zorder=4))


def _title(ax, title, sub, ours=False):
    ax.text(0.02, 0.965, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=TEAL if ours else INK, va="top")
    ax.text(0.02, 0.905, sub, transform=ax.transAxes, fontsize=8.6, color=MUTED, va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])


# ----------------------------------------------------------------- panel 1: sort-based
def panel_sort(ax, title, sub):
    # row i (a few k-nodes) -> expand (dense O(flop)) -> sort -> reduce -> C
    _dots(ax, np.linspace(0.16, 0.34, 3), 0.79, 0.022, MUTED)
    ax.text(0.10, 0.79, "row i", ha="right", va="center", color=INK, fontsize=8.5)
    _arrow(ax, 0.36, 0.79, 0.06, -0.05, GRAY)
    # expand: dense strip (the 19x blowup)
    _dots(ax, np.linspace(0.14, 0.86, 30), 0.70, 0.013, GRAY)
    ax.text(0.50, 0.76, "expand  O(flop)", ha="center", color=MUTED, fontsize=8.6)
    _arrow(ax, 0.50, 0.675, 0.0, -0.07, GRAY)
    # sort: grouped dots
    for g, x0 in zip([6, 5, 4, 6, 5], np.linspace(0.16, 0.80, 5)):
        _dots(ax, np.linspace(x0, x0 + 0.06, g), 0.54, 0.012, "#7E8B9A")
    ax.text(0.50, 0.60, "sort", ha="center", color=MUTED, fontsize=8.6)
    _arrow(ax, 0.50, 0.515, 0.0, -0.07, GRAY)
    # reduce -> C (distinct)
    _dots(ax, np.linspace(0.20, 0.80, 9), 0.36, 0.02, MUTED)
    ax.text(0.50, 0.30, "C[i,:]  (distinct)", ha="center", color=INK, fontsize=8.6)
    # red bottleneck bracket on expand + sort
    ax.add_patch(FancyBboxPatch((0.11, 0.49), 0.78, 0.30,
                                boxstyle="round,pad=0.01", fill=False, ec=RED, lw=1.8, zorder=5))
    ax.text(0.50, 0.17, "accumulation = 72% of runtime  ·  19× duplicates",
            ha="center", color=RED, fontsize=9, fontweight="bold")
    _title(ax, title, sub)


# ----------------------------------------------------------------- panel 2: merge-based
def panel_merge(ax, title, sub):
    # K gray column buckets under the row, merging down to C
    _dots(ax, np.linspace(0.16, 0.34, 3), 0.79, 0.022, MUTED)
    ax.text(0.10, 0.79, "row i", ha="right", va="center", color=INK, fontsize=8.5)
    K = 5
    seg = np.linspace(0.12, 0.88, K + 1)
    for k in range(K):
        ax.add_patch(Rectangle((seg[k], 0.54), seg[k + 1] - seg[k], 0.12,
                               fc="#C9D1D9", ec="white", lw=1.0, zorder=3))
        cx = (seg[k] + seg[k + 1]) / 2
        _arrow(ax, cx, 0.82, 0.0, -0.14, GRAY)
    ax.text(0.50, 0.72, "column buckets", ha="center", color=MUTED, fontsize=8.6)
    _arrow(ax, 0.50, 0.52, 0.0, -0.10, GRAY)
    _dots(ax, np.linspace(0.20, 0.80, 9), 0.34, 0.02, MUTED)
    ax.text(0.50, 0.26, "merge → C[i,:]  (sorted)", ha="center", color=INK, fontsize=8.6)
    ax.text(0.50, 0.13, "still scans all O(flop) products", ha="center",
            color=MUTED, fontsize=8.8, style="italic")
    _title(ax, title, sub)


# ----------------------------------------------------------------- panel 3: Hash SPA (ours)
def panel_hash(ax, title, sub, ours=True):
    edge = TEAL
    _dots(ax, np.linspace(0.16, 0.34, 3), 0.79, 0.022, MUTED)
    ax.text(0.10, 0.79, "row i", ha="right", va="center", color=INK, fontsize=8.5)
    # SMEM hash table: grid of slots
    nx, ny = 7, 3
    x0, y0, w, h = 0.20, 0.40, 0.60, 0.26
    sx, sy = w / nx, h / ny
    filled = [(0, 2), (3, 2), (5, 2), (1, 1), (4, 1), (2, 0), (6, 0)]
    for ix in range(nx):
        for iy in range(ny):
            fc = "#CFE8EE" if (ix, iy) in filled else "#F4FAFB"
            ax.add_patch(Rectangle((x0 + ix * sx, y0 + iy * sy), sx * 0.9, sy * 0.82,
                                   fc=fc, ec=edge, lw=0.9, zorder=3))
    ax.text(0.50, 0.70, "SMEM hash table", ha="center", color=edge, fontsize=8.8, fontweight="bold")
    ax.text(0.07, 0.55, "atomicCAS\n+ atomicAdd", ha="center", va="center",
            color=edge, fontsize=7.8, fontweight="bold")
    _arrow(ax, 0.36, 0.79, 0.0, -0.18, edge)
    # distinct out
    _arrow(ax, 0.50, 0.39, 0.0, -0.07, edge)
    _dots(ax, np.linspace(0.22, 0.78, 7), 0.28, 0.022, edge)
    ax.text(0.50, 0.20, "C[i,:]  (distinct, deduped in place)", ha="center",
            color=edge, fontsize=8.4, fontweight="bold")
    _title(ax, title, sub, ours=ours)


# ----------------------------------------------------------------- panel 4: MinHash sizing (ours)
def panel_minhash(ax, title, sub, ours=True):
    edge = TEAL
    # m partition cells, each "min h"
    m = 8
    xs = np.linspace(0.12, 0.88, m)
    for x in xs:
        ax.add_patch(Rectangle((x - 0.035, 0.60), 0.07, 0.12,
                               fc="#CFE8EE", ec=edge, lw=0.9, zorder=3))
        ax.text(x, 0.66, "min h", ha="center", va="center", color=edge, fontsize=6.6)
    ax.text(0.50, 0.78, "per-partition MinHash sketch", ha="center", color=edge,
            fontsize=8.6, fontweight="bold")
    _arrow(ax, 0.50, 0.59, 0.0, -0.10, edge)
    # formula box
    ax.add_patch(FancyBboxPatch((0.18, 0.36), 0.64, 0.13,
                                boxstyle="round,pad=0.02", fc="#F4FAFB", ec=edge, lw=1.0, zorder=3))
    ax.text(0.50, 0.425, r"$\hat n = 2^{32}\cdot\sum_j(1/\mathrm{min}_j)\,-\,m$",
            ha="center", va="center", color=INK, fontsize=9.5)
    _arrow(ax, 0.50, 0.35, 0.0, -0.08, edge)
    ax.text(0.50, 0.22, "distinct upper bound  →  hash-table size", ha="center",
            color=edge, fontsize=8.6, fontweight="bold")
    ax.text(0.50, 0.12, "orthogonal to Ocean's HLL (min-value vs max-leading-zero)",
            ha="center", color=MUTED, fontsize=7.8, style="italic")
    _title(ax, title, sub, ours=ours)


def main():
    fig = plt.figure(figsize=(12.2, 6.4))
    specs = [
        (panel_sort,     "(1) Sort-based  ·  ESC / Gustavson ’78", "materialize O(flop), then sort + reduce"),
        (panel_merge,    "(2) Merge-based  ·  column-domain (D2)", "bucket-merge still scans O(flop) products"),
        (panel_hash,     "(3) Hash SPA  ·  Ours", "in-place atomic dedup · writes only distinct"),
        (panel_minhash,  "(4) MinHash sizing  ·  Ours", "per-row distinct UB · orthogonal to HLL"),
    ]
    for i, (fn, t, s) in enumerate(specs):
        ax = fig.add_subplot(2, 2, i + 1)
        ours = i >= 2
        fn(ax, t, s) if i < 2 else fn(ax, t, s, ours=True)
        if ours:
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(TEAL); sp.set_linewidth(1.8)
            ax.patch.set_facecolor("#F4FAFB")

    fig.suptitle("Accumulation has three families — hash deduplicates in place, "
                 "not after a 19× blow-up", fontsize=13, color=INK, y=0.995, fontweight="bold")
    fig.text(0.5, 0.015,
             "Sort/merge materialize O(flop) (dup 19×) before dedup.   "
             "Hash SPA (ours): a per-row table collapses duplicates as they accumulate.",
             ha="center", color=RED, fontsize=10.5, fontweight="bold")
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_landscape.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/hash_landscape.{png,pdf}")


if __name__ == "__main__":
    main()
