#!/usr/bin/env python3
"""Merge Design2 — INSIGHT: the parallel-axis landscape of merge-based SpGEMM.

Four orthogonal axes along which a row-merge can be parallelized. Prior work uses three
(cross-row Gustavson; cross-row load balance by row length bhSparse; K-chain merge tree
SpArch). None partitions a row's COLUMN range — exactly where the heavy-row straggler lives
and the only cut that keeps output sorted. We add the 4th axis (column-domain bucketing).

2x2 iconic schematic: panels 1-3 neutral gray (prior work), panel 4 teal (ours) with a red
"straggler" marker on the serialized axis. Pure illustration (no measured data).
"""
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

INK, MUTED, GRAY = "#1F2933", "#5C6773", "#9AA6B2"
TEAL, RED = "#1485A4", "#C00000"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none"})


def _lane(ax, y, n, color, r=0.10):
    """a horizontal lane of n dots at height y"""
    xs = np.linspace(0.12, 0.88, n)
    for x in xs:
        c = plt.Circle((x, y), r, color=color, zorder=3)
        ax.add_patch(c)
    return xs


def panel_crossrow(ax, title, sub, ours=False):
    edge = TEAL if ours else GRAY
    for i, y in enumerate(np.linspace(0.80, 0.20, 4)):
        _lane(ax, y, 9, edge, r=0.055)
        ax.add_patch(FancyArrow(0.92, y, 0.0, 0.0, color=edge, lw=0))  # placeholder
    # "1 unit per row" arrows on the left
    for y in np.linspace(0.80, 0.20, 4):
        ax.add_patch(FancyArrow(0.04, y, 0.06, 0.0, width=0.012,
                                head_width=0.05, head_length=0.02,
                                color=edge, length_includes_head=True, zorder=4))
    # serialized column range (the straggler) marked on the heaviest row
    ax.add_patch(Rectangle((0.12, 0.74), 0.76, 0.05, fill=False, ec=RED, lw=1.6, zorder=5))
    ax.text(0.50, 0.86, "column range\nserialized", ha="center", va="bottom",
            color=RED, fontsize=8.5, fontweight="bold")
    _title(ax, title, sub)


def panel_balance(ax, title, sub):
    # rows of different lengths grouped into 3 length-buckets
    lens = [4, 5, 4, 7, 8, 7, 11, 12, 11]
    buckets = [("short", GRAY), ("mid", "#7E8B9A"), ("long", "#5C6773")]
    ys = np.linspace(0.82, 0.18, len(lens))
    for y, n in zip(ys, lens):
        b = 0 if n <= 5 else (1 if n <= 8 else 2)
        _lane(ax, y, n, buckets[b][1], r=0.05)
    ax.text(0.97, 0.50, "group rows\nby length\n(cross-row\nbalance)", ha="right", va="center",
            color=MUTED, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D7DCE1", lw=0.7))
    _title(ax, title, sub)


def panel_tree(ax, title, sub):
    # binary merge tree: 5 leaf chains merging to a root
    leaves = np.linspace(0.14, 0.86, 5)
    yl = 0.22
    for x in leaves:
        ax.add_patch(Rectangle((x-0.05, yl-0.03), 0.10, 0.06, fc=GRAY, ec="none", zorder=3))
    # layer 1
    l1 = [(leaves[0]+leaves[1])/2, (leaves[2]+leaves[3])/2, leaves[4]]
    y1 = 0.45
    # layer 2
    l2 = [(l1[0]+l1[1])/2, l1[2]]
    y2 = 0.62
    root = np.mean(l2); yr = 0.78
    def node(x, y, c):
        ax.add_patch(plt.Circle((x, y), 0.045, color=c, zorder=4))
    for x in l1: node(x, y1, "#7E8B9A")
    for x in l2: node(x, y2, "#5C6773")
    node(root, yr, MUTED)
    def edge(a, b):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=1.1, zorder=2))
    for i in range(5):
        tgt = l1[i//2] if i < 4 else l1[2]
        edge((leaves[i], yl+0.03), (tgt, y1-0.045))
    edge((l1[0], y1+0.045), (l2[0], y2-0.045))
    edge((l1[1], y1+0.045), (l2[0], y2-0.045))
    edge((l1[2], y1+0.045), (l2[1], y2-0.045))
    for x in l2: edge((x, y2+0.045), (root, yr-0.045))
    ax.text(0.50, 0.10, "K chains → O(log K) compare per element", ha="center",
            color=MUTED, fontsize=8.5)
    _title(ax, title, sub)


def panel_column(ax, title, sub, ours=True):
    edge = TEAL
    K = 5
    seg_x = np.linspace(0.10, 0.90, K + 1)
    shades = ["#9FCFD8", "#5FB0BE", "#2E91A4", TEAL, "#0E6E83"]
    for k in range(K):
        ax.add_patch(Rectangle((seg_x[k], 0.40), seg_x[k+1]-seg_x[k], 0.16,
                               fc=shades[k], ec="white", lw=1.2, zorder=3))
    # one warp arrow per bucket
    for k in range(K):
        cx = (seg_x[k]+seg_x[k+1])/2
        ax.add_patch(FancyArrow(cx, 0.74, 0.0, -0.14, width=0.010,
                                head_width=0.05, head_length=0.03,
                                color=TEAL, length_includes_head=True, zorder=4))
        ax.text(cx, 0.80, f"w{k}", ha="center", color=TEAL, fontsize=8, fontweight="bold")
    # concatenated sorted output
    ax.add_patch(Rectangle((0.10, 0.20), 0.80, 0.07, fc="#EAF4F6", ec=TEAL, lw=1.0, zorder=3))
    ax.text(0.50, 0.235, "concatenate buckets → column-sorted CSR (no global sort)",
            ha="center", va="center", color=TEAL, fontsize=8.2, fontweight="bold")
    ax.text(0.50, 0.34, "1 row, K disjoint column buckets", ha="center", color=INK, fontsize=8.5)
    _title(ax, title, sub, ours=ours)


def _title(ax, title, sub, ours=False):
    ax.text(0.02, 0.965, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            color=TEAL if ours else INK, va="top")
    ax.text(0.02, 0.905, sub, transform=ax.transAxes, fontsize=8.6, color=MUTED, va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])


def draw_landscape(fig):
    specs = [
        (panel_crossrow, "(1) Cross-row  ·  Gustavson ’78", "rows independent · 1 unit / row"),
        (panel_balance,  "(2) Row-length balance  ·  bhSparse ’14", "cross-row load balance by row length"),
        (panel_tree,     "(3) K-chain merge tree  ·  SpArch ’19", "cut per-element compare, not the column range"),
        (panel_column,   "(4) Column-domain bucketing  ·  Ours", "intra-row column parallel · order-preserving"),
    ]
    for i, (fn, t, s) in enumerate(specs):
        ax = fig.add_subplot(2, 2, i + 1)
        ours = (i == 3)
        if fn is panel_column:
            fn(ax, t, s, ours=True)
        elif fn is panel_crossrow:
            fn(ax, t, s)
        else:
            fn(ax, t, s)
        if ours:
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_edgecolor(TEAL); spine.set_linewidth(1.8)
            ax.patch.set_facecolor("#F4FAFB")


def main():
    fig = plt.figure(figsize=(12.2, 6.4))
    draw_landscape(fig)
    fig.suptitle("Merge-based SpGEMM has four parallel axes — prior work uses three; "
                 "none partitions the row’s column range",
                 fontsize=13, color=INK, y=0.995, fontweight="bold")
    fig.text(0.5, 0.015,
             "Axes 1–3 leave the column range serialized ⟹ heavy rows straggle.   "
             "Axis 4 (ours): disjoint ordered buckets concatenate to sorted CSR.",
             ha="center", color=RED, fontsize=10.5, fontweight="bold")
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/merge_landscape.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/merge_landscape.{png,pdf}")


if __name__ == "__main__":
    main()
