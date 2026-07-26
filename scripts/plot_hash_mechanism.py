#!/usr/bin/env python3
"""Hash Design3 — DESIGN: the Hash-SPA pipeline.

Left-to-right pipeline of one output row C[i,:]:
  row products (i,k)·B[k,:]  →  SMEM hash table (atomicCAS insert / atomicAdd
  accumulate — duplicates collapse in place)  →  extract distinct (col,val)  →
  compact + sort  →  column-sorted CSR.
A two-phase MinHash sketch (ours) feeds the table size from above. Bottom band
contrasts with sort-based ESC (no O(flop) materialization).
Pure illustration (measured numbers cited as labels only).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch

INK, MUTED, GRAY = "#1F2933", "#5C6773", "#9AA6B2"
TEAL, RED = "#1485A4", "#C00000"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none"})


def _box(ax, x, y, w, h, fc, ec, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw, zorder=3))


def _arrow(ax, x, y, dx, dy, color, lw=1.8):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.006, head_width=0.03,
                            head_length=0.018, color=color, lw=lw,
                            length_includes_head=True, zorder=4))


def _dots(ax, xs, y, r, color):
    for x in xs:
        ax.add_patch(plt.Circle((x, y), r, color=color, zorder=5))


def main():
    fig, ax = plt.subplots(figsize=(12.6, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    CY = 0.42  # pipeline centerline

    # ---- stage 1: row products
    _box(ax, 0.015, 0.32, 0.155, 0.22, "#F4F5F7", GRAY)
    _dots(ax, np.linspace(0.04, 0.14, 3), 0.45, 0.014, MUTED)
    _dots(ax, np.linspace(0.035, 0.145, 7), 0.37, 0.010, GRAY)
    ax.text(0.092, 0.30, "row products\n(i,k)·B[k,:]", ha="center", va="top",
            color=INK, fontsize=9)
    _arrow(ax, 0.175, CY, 0.05, 0, MUTED)

    # ---- stage 2: SMEM hash table (core, teal)
    _box(ax, 0.235, 0.30, 0.30, 0.26, "#F4FAFB", TEAL, lw=1.8)
    nx, ny = 6, 3
    hx, hy, hw, hh = 0.255, 0.345, 0.26, 0.135
    sx, sy = hw / nx, hh / ny
    filled = {(0, 2), (2, 2), (4, 2), (5, 2), (1, 1), (3, 1), (0, 0), (3, 0), (5, 0)}
    for ix in range(nx):
        for iy in range(ny):
            fc = "#2E91A4" if (ix, iy) in filled else "white"
            ax.add_patch(Rectangle((hx + ix * sx, hy + iy * sy), sx * 0.88, sy * 0.80,
                                   fc=fc, ec=TEAL, lw=0.8, zorder=4))
    ax.text(0.385, 0.575, "SMEM hash table", ha="center", color=TEAL, fontsize=9.5, fontweight="bold")
    ax.text(0.385, 0.30, "atomicCAS  insert  ·  atomicAdd  accumulate", ha="center", va="top",
            color=TEAL, fontsize=8.2)
    _arrow(ax, 0.540, CY, 0.05, 0, MUTED)

    # ---- stage 3: extract distinct
    _box(ax, 0.60, 0.32, 0.135, 0.22, "#F4F5F7", GRAY)
    _dots(ax, np.linspace(0.615, 0.72, 5), CY, 0.018, TEAL)
    ax.text(0.667, 0.30, "extract\ndistinct (col,val)", ha="center", va="top",
            color=INK, fontsize=8.8)
    _arrow(ax, 0.740, CY, 0.05, 0, MUTED)

    # ---- stage 4: compact + sort -> CSR
    _box(ax, 0.795, 0.32, 0.185, 0.22, "#F4FAFB", TEAL, lw=1.2)
    # sorted dots (ascending)
    _dots(ax, np.linspace(0.81, 0.965, 8), CY, 0.015, TEAL)
    ax.text(0.887, 0.30, "compact + sort\n→ column-sorted CSR", ha="center", va="top",
            color=TEAL, fontsize=8.8, fontweight="bold")

    # ---- MinHash sizing feeding the table (ours, novelty)
    _box(ax, 0.255, 0.74, 0.26, 0.16, "#EAF4F6", TEAL, lw=1.4)
    # sketch cells
    m = 6
    xs = np.linspace(0.275, 0.495, m)
    for x in xs:
        ax.add_patch(Rectangle((x - 0.012, 0.785), 0.024, 0.05, fc="#CFE8EE", ec=TEAL, lw=0.7, zorder=4))
    ax.text(0.385, 0.855, "MinHash sizing  ·  ours", ha="center", color=TEAL,
            fontsize=9, fontweight="bold")
    ax.text(0.385, 0.755, r"$\hat n = 2^{32}\sum_j(1/\mathrm{min}_j)\,-\,m$",
            ha="center", va="top", color=INK, fontsize=8.6)
    _arrow(ax, 0.385, 0.735, 0.0, -0.155, TEAL)
    ax.text(0.40, 0.665, "table size", ha="left", va="center", color=TEAL, fontsize=8, style="italic")

    # ---- bottom contrast band (vs sort-based)
    _box(ax, 0.015, 0.05, 0.965, 0.13, "#FCEFEF", RED, lw=1.3)
    ax.text(0.50, 0.115,
            "vs sort-based (ESC):  no O(flop) intermediate is materialized —",
            ha="center", color=RED, fontsize=10.5, fontweight="bold")
    ax.text(0.50, 0.075,
            "duplicates (dup factor up to 19×) collapse as they arrive, so only the "
            "distinct C[i,:] is ever written and sorted.",
            ha="center", color=INK, fontsize=9.8)

    fig.suptitle("Hash SPA — deduplicate in place during accumulation",
                 fontsize=13.5, color=INK, y=0.985, fontweight="bold")
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_mechanism.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/hash_mechanism.{png,pdf}")


if __name__ == "__main__":
    main()
