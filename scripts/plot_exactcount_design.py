#!/usr/bin/env python3
"""Exact-count symbolic (opSparse / nsparse baseline) — flow design diagram.

Shows WHY exact-count symbolic is slow: it enumerates ALL O(flop) products
(including 19× duplicates) just to count distinct columns — the SAME work as
the numeric stage, duplicated for sizing.  This is the "insight" that motivates
MinHash (which sketches rows in O(nnz), never enumerating products).

Output: fig/exactcount_design.{png,pdf}
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow

TEAL, RED, AMBER, GRAY = "#1485A4", "#C00000", "#E0A458", "#C9D1D9"
TEAL_LT, TEAL_D, INK, MUTED = "#CFE8EE", "#0E6E83", "#1F2933", "#5C6773"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none"})


def chip(ax, x, y, w, h, text, fc, ec, tc, fs=8.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=4)


def arr(ax, x, y, dx, dy, color=MUTED, lw=1.6):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.006, head_width=0.022,
                            head_length=0.014, color=color, lw=lw,
                            length_includes_head=True, zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.965, "Exact-count symbolic — enumerates ALL products just to count",
            ha="center", fontsize=13, color=INK, fontweight="bold")
    ax.text(0.5, 0.925, "(opSparse / nsparse baseline)  ·  O(flop): the SAME work as numeric accumulation, duplicated for sizing",
            ha="center", fontsize=8.8, color=MUTED)

    # ---- stage 1: output row references A-rows ----
    ax.text(0.085, 0.86, "output row i", ha="center", fontsize=8.5, color=MUTED)
    chip(ax, 0.035, 0.73, 0.10, 0.075, "row i", "white", GRAY, INK, fs=8.5)
    arr(ax, 0.085, 0.73, -0.02, -0.05, MUTED)
    arr(ax, 0.085, 0.73, 0.02, -0.05, MUTED)
    chip(ax, 0.005, 0.61, 0.07, 0.055, "k₁:{5,8}", "#F4F5F7", GRAY, INK, fs=7.0)
    chip(ax, 0.095, 0.61, 0.07, 0.055, "k₂:{12,5}", "#F4F5F7", GRAY, INK, fs=7.0)

    arr(ax, 0.17, 0.635, 0.04, 0, MUTED, lw=2.0)

    # ---- stage 2: enumerate ALL products (the blowup) ----
    ax.text(0.34, 0.86, "enumerate ALL products  (i, j)", ha="center",
            fontsize=8.8, color=RED, fontweight="bold")
    # product chips (with dup)
    prods = [("j=5", False), ("j=8", False), ("j=12", False), ("j=5", True)]
    py = 0.76
    for v, dup in prods:
        chip(ax, 0.28, py - 0.03, 0.12, 0.05, v,
             "#F6D98E" if dup else "white", AMBER if dup else GRAY,
             INK, fs=7.8, bold=dup)
        py -= 0.062
    # red bracket + annotation
    ax.add_patch(FancyBboxPatch((0.265, 0.45), 0.15, 0.40,
                                boxstyle="round,pad=0.008", fill=False, ec=RED, lw=1.6, zorder=2))
    ax.text(0.34, 0.43, "4 products\n(1 duplicate)", ha="center", fontsize=7.0, color=RED)
    ax.text(0.34, 0.355, "bcsstk30: 173M products\n→ only 8.95M distinct",
            ha="center", fontsize=7.2, color=RED, fontweight="bold")

    arr(ax, 0.42, 0.635, 0.04, 0, MUTED, lw=2.0)

    # ---- stage 3: hash each j → mark in temporary hash set ----
    ax.text(0.58, 0.86, "hash each j → mark distinct", ha="center",
            fontsize=8.8, color=TEAL_D, fontweight="bold")
    # mini hash set (3 slots)
    for i, (v, dup_hit) in enumerate([("5", False), ("8", False), ("12", False)]):
        sy = 0.72 - i * 0.075
        chip(ax, 0.52, sy, 0.12, 0.055, f"slot: {v}", TEAL_LT, TEAL, TEAL_D, fs=7.4, bold=True)
    # arrows from products to set
    arr(ax, 0.42, 0.73, 0.09, 0, MUTED)     # j=5 → slot 5
    arr(ax, 0.42, 0.668, 0.09, 0.02, MUTED)  # j=8 → slot 8
    arr(ax, 0.42, 0.606, 0.09, 0.04, MUTED)  # j=12 → slot 12
    arr(ax, 0.42, 0.544, 0.09, -0.09, AMBER) # j=5 dup → hits slot 5 (already there)
    ax.text(0.58, 0.50, "dup j=5 → already marked\n→ skipped", ha="center",
            fontsize=6.6, color=AMBER, style="italic")

    arr(ax, 0.65, 0.635, 0.04, 0, TEAL, lw=2.0)

    # ---- stage 4: count → size buffer ----
    chip(ax, 0.71, 0.60, 0.13, 0.075, "distinct = 3", TEAL, TEAL, "white", fs=8.2, bold=True)
    ax.text(0.775, 0.56, "= C[i,:] nnz", ha="center", fontsize=7.2, color=TEAL_D)
    arr(ax, 0.775, 0.545, 0, -0.05, TEAL)
    chip(ax, 0.71, 0.45, 0.13, 0.06, "→ size buffer", "#EAF4F6", TEAL, TEAL_D, fs=7.6, bold=True)

    # ---- right callout: cost ----
    chip(ax, 0.87, 0.52, 0.12, 0.18, "", "#FCEFEF", RED, INK)
    ax.text(0.93, 0.645, "cost", ha="center", fontsize=7.8, color=RED, fontweight="bold")
    ax.text(0.93, 0.605, "O(flop)", ha="center", fontsize=9, color=RED, fontweight="bold")
    ax.text(0.93, 0.565, "= same as\nnumeric", ha="center", fontsize=6.6, color=RED)

    # ---- bottom red band: the insight ----
    ax.add_patch(FancyBboxPatch((0.01, 0.03), 0.98, 0.12, boxstyle="round,pad=0.010",
                                fc="#FCEFEF", ec=RED, lw=1.2, zorder=2))
    ax.text(0.5, 0.105,
            "Exact-count symbolic touches ALL O(flop) products (19× duplicates) —",
            ha="center", color=RED, fontsize=9.2, fontweight="bold", zorder=3)
    ax.text(0.5, 0.065,
            "the SAME work as numeric accumulation, duplicated just to count distinct.   "
            "MinHash replaces it with an O(nnz) sketch — no enumeration.",
            ha="center", color=INK, fontsize=8.6, zorder=3)

    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/exactcount_design.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/exactcount_design.{png,pdf}")


if __name__ == "__main__":
    main()
