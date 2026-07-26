#!/usr/bin/env python3
"""Hash Design3 — detailed mechanism diagrams for the two innovations.

  minhash_design — per-partition MinHash sizing: hash columns → m partitions each
                   keeps the MIN hash (atomicMin) → estimate distinct without counting.
                   Concrete example (3 distinct columns → estimate ≈ 3).
  dedup_design   — in-place dedup: atomicCAS inserts each column once, atomicAdd
                   accumulates; duplicates collapse. Example (3 products, 1 dup → 2 distinct).

Output: fig/hash_minhash_design.{png,pdf}, fig/hash_dedup_design.{png,pdf}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow

TEAL, RED, AMBER, GRAY = "#1485A4", "#C00000", "#E0A458", "#C9D1D9"
TEAL_LT, TEAL_D, INK, MUTED = "#CFE8EE", "#0E6E83", "#1F2933", "#5C6773"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none"})


def _chip(ax, x, y, w, h, text, fc, ec, tc, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=4)


def _arrow(ax, x, y, dx, dy, color=MUTED, lw=1.5):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.006, head_width=0.022,
                            head_length=0.014, color=color, lw=lw,
                            length_includes_head=True, zorder=4))


# ====================================================================== MinHash
def minhash():
    fig, ax = plt.subplots(figsize=(14.0, 6.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "MinHash sizing  —  symbolic stage, BEFORE numeric accumulation",
            ha="center", fontsize=13, color=INK, fontweight="bold")
    ax.text(0.5, 0.925, "estimate each row's distinct columns → size the hash table   (no multiply — just hashing column indices)",
            ha="center", fontsize=9.2, color=MUTED)

    def hdr(c, txt):
        ax.text(c, 0.865, txt, ha="center", fontsize=9.2, color=TEAL_D, fontweight="bold")

    # ---- stage 1: column source ----
    hdr(0.072, "①  columns  j")
    ax.text(0.072, 0.82, "from A's structure", ha="center", fontsize=7.0, color=MUTED)
    cols = [("5", False), ("12", False), ("5", True), ("8", False), ("12", True)]
    cy = 0.76
    for v, dup in cols:
        fc = "#F6D98E" if dup else "white"
        _chip(ax, 0.032, cy - 0.026, 0.08, 0.048, f"j={v}", fc, AMBER if dup else GRAY,
              INK, fs=7.8, bold=dup)
        cy -= 0.060
    ax.text(0.072, 0.43, "amber = duplicate", ha="center", fontsize=6.6, color=AMBER, style="italic")
    ax.text(0.072, 0.34, "no arithmetic —\njust column indices", ha="center", fontsize=6.6, color=MUTED)
    _arrow(ax, 0.118, 0.60, 0.028, 0, MUTED, lw=1.8)

    # ---- stage 2: hash(j) ----
    hdr(0.235, "②  hash(j)  ·  MurmurHash3")
    _chip(ax, 0.150, 0.62, 0.17, 0.155, "", "#F4FAFB", TEAL, INK)
    for i, (j, h) in enumerate([("5", "0xD4B2A100"), ("12", "0xB8F10401"), ("8", "0x7F3D9E20")]):
        y = 0.735 - i * 0.050
        ax.text(0.168, y, f"j={j}", ha="left", va="center", fontsize=7.8, color=INK)
        ax.text(0.218, y, "→", ha="left", va="center", fontsize=7.8, color=MUTED)
        ax.text(0.238, y, h, ha="left", va="center", fontsize=7.8, color=TEAL_D, fontweight="bold")
    ax.text(0.235, 0.575, "deterministic:", ha="center", fontsize=6.6, color=MUTED)
    ax.text(0.235, 0.535, "same j → same hash", ha="center", fontsize=6.8, color=INK, style="italic")
    ax.text(0.235, 0.495, "(j=5 → 0xD4B2A100)", ha="center", fontsize=6.4, color=AMBER)
    _arrow(ax, 0.322, 0.60, 0.024, 0, MUTED, lw=1.8)

    # ---- stage 3: partition + keep MIN ----
    hdr(0.478, "③  partition (low bits)  ·  keep the MIN  (atomicMin)")
    # P0 detail (3 values -> min)
    ax.text(0.478, 0.775, "bucket P0 received:", ha="center", fontsize=7.0, color=MUTED)
    for i, (v, ismin) in enumerate([("0xD4B2A100", False), ("0x7F3D9E20", True), ("0xD4B2A100", False)]):
        vx = 0.350 + i * 0.078
        fc = TEAL if ismin else "#E6E9ED"
        _chip(ax, vx, 0.715, 0.066, 0.042, v, fc, TEAL if ismin else GRAY,
              "white" if ismin else MUTED, fs=5.4, bold=ismin)
    ax.text(0.461, 0.765, "MIN", ha="center", fontsize=6.6, color=TEAL_D, fontweight="bold")
    # the 4 partition mins
    parts = [("P0", "0x7F3D9E20", True, "{5,8}"), ("P1", "0xB8F10401", True, "{12}"),
             ("P2", "∅", False, ""), ("P3", "∅", False, "")]
    for i, (name, mn, filled, saw) in enumerate(parts):
        px = 0.350 + i * 0.066
        fc = TEAL_LT if filled else "#F0F2F4"
        _chip(ax, px, 0.555, 0.060, 0.105, "", fc, TEAL if filled else GRAY, INK)
        ax.text(px + 0.030, 0.635, name, ha="center", fontsize=6.8, color=TEAL_D, fontweight="bold")
        ax.text(px + 0.030, 0.600, mn, ha="center", fontsize=5.4,
                color=INK if filled else MUTED, fontweight="bold")
        if filled:
            ax.text(px + 0.030, 0.572, saw, ha="center", fontsize=5.4, color=MUTED)
        else:
            ax.text(px + 0.030, 0.585, "empty", ha="center", fontsize=5.8, color=MUTED)
    ax.text(0.478, 0.515, "each bucket keeps only its smallest hash", ha="center",
            fontsize=7.0, color=TEAL_D, style="italic")
    _arrow(ax, 0.620, 0.60, 0.036, 0, MUTED, lw=1.8)

    # ---- stage 4: estimate ----
    hdr(0.738, "④  estimate distinct")
    _chip(ax, 0.665, 0.66, 0.150, 0.085, "", "#EAF4F6", TEAL, INK)
    ax.text(0.740, 0.715, r"$\hat n = 2^{32}\sum_j(1/min_j)\!-\!m$", ha="center", fontsize=7.4, color=INK)
    ax.text(0.740, 0.675, "(all buckets full)", ha="center", fontsize=6.0, color=MUTED)
    ax.text(0.740, 0.625, "here V=2 < m=4 →", ha="center", fontsize=6.8, color=MUTED)
    ax.text(0.740, 0.590, r"linear: $-m\ln(1\!-\!V/m)$", ha="center", fontsize=7.0, color=INK)
    ax.text(0.740, 0.550, "= −4·ln(0.5) ≈ 2.77", ha="center", fontsize=7.2, color=TEAL_D, fontweight="bold")
    ax.text(0.740, 0.495, "true distinct = 3  ✓", ha="center", fontsize=8.0, color=TEAL_D, fontweight="bold")
    _arrow(ax, 0.820, 0.60, 0.036, 0, MUTED, lw=1.8)

    # ---- stage 5: table size -> accumulation ----
    hdr(0.915, "⑤  table size")
    _chip(ax, 0.872, 0.585, 0.086, 0.115, "right-size\nhash table", TEAL, TEAL, "white", fs=7.4, bold=True)
    _arrow(ax, 0.915, 0.58, 0, -0.06, TEAL, lw=1.8)
    ax.text(0.915, 0.485, "→ accumulation\n(next stage)", ha="center", fontsize=6.8, color=TEAL_D)

    ax.text(0.5, 0.045, "O(nnz): Phase 1 sketches each A-row's columns; Phase 2 merges per output row.   "
            "Never enumerates the 19× product stream — orthogonal to Ocean's HLL.",
            ha="center", fontsize=8.4, color=TEAL_D, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_minhash_design.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig/hash_minhash_design.{png,pdf}")


# ====================================================================== dedup
def dedup():
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "In-place dedup — atomicCAS inserts once, atomicAdd accumulates",
            ha="center", fontsize=12.5, color=INK, fontweight="bold")

    # ---- left: products (j, v) with a dup ----
    ax.text(0.085, 0.89, "products (i, j, v)", ha="center", fontsize=9.5, color=MUTED)
    prods = [("5, 2", False), ("5, 3", True), ("8, 1", False)]
    cy = 0.78
    ys = []
    for v, dup in prods:
        fc = "#F6E2C8" if dup else "white"
        ec = AMBER if dup else GRAY
        _chip(ax, 0.035, cy - 0.035, 0.10, 0.06, f"j,v = {v}", fc, ec, INK, fs=8.0, bold=dup)
        ys.append(cy); cy -= 0.085
    ax.text(0.085, 0.46, "(amber = same j, duplicate)", ha="center", fontsize=7.4, color=AMBER, style="italic")

    # ---- center: hash table (slots) ----
    ax.text(0.43, 0.89, "SMEM hash table  ·  hash(j) → slot", ha="center",
            fontsize=9.5, color=TEAL_D, fontweight="bold")
    # slot for j=5 (two arrows in), slot for j=8
    _chip(ax, 0.32, 0.66, 0.22, 0.13, "", TEAL_LT, TEAL, INK)
    ax.text(0.43, 0.755, "slot[ j=5 ]", ha="center", fontsize=8.0, color=TEAL_D, fontweight="bold")
    ax.text(0.43, 0.705, "col = 5   val = 5", ha="center", fontsize=8.4, color=INK, fontweight="bold")
    _chip(ax, 0.32, 0.45, 0.22, 0.13, "", TEAL_LT, TEAL, INK)
    ax.text(0.43, 0.545, "slot[ j=8 ]", ha="center", fontsize=8.0, color=TEAL_D, fontweight="bold")
    ax.text(0.43, 0.495, "col = 8   val = 1", ha="center", fontsize=8.4, color=INK, fontweight="bold")
    # arrows: (5,2)->slot5, (5,3)->slot5 (offset landings), (8,1)->slot8
    _arrow(ax, 0.14, ys[0], 0.17, 0.755 - ys[0], AMBER, lw=1.8)   # j5 v2 -> slot5 (upper)
    _arrow(ax, 0.14, ys[1], 0.17, 0.700 - ys[1], AMBER, lw=1.8)   # j5 v3 -> slot5 (lower)
    _arrow(ax, 0.14, ys[2], 0.17, 0.515 - ys[2], MUTED, lw=1.8)   # j8 -> slot8
    # ops annotation
    ax.text(0.575, 0.725, "atomicCAS:  5 already in\n→ skip insert", ha="left", fontsize=7.4, color=RED)
    ax.text(0.575, 0.68, "atomicAdd:  2 + 3 = 5", ha="left", fontsize=7.4, color=TEAL_D, fontweight="bold")
    ax.text(0.575, 0.515, "atomicCAS:  insert 8\natomicAdd:  +1", ha="left", fontsize=7.4, color=TEAL_D)

    _arrow(ax, 0.78, 0.60, 0.05, 0, MUTED, lw=2.0)

    # ---- right: distinct output ----
    _chip(ax, 0.84, 0.50, 0.14, 0.20, "", "#EAF4F6", TEAL, INK)
    ax.text(0.91, 0.665, "distinct", ha="center", fontsize=8.4, color=TEAL_D, fontweight="bold")
    ax.text(0.91, 0.625, "(5, 5)", ha="center", fontsize=8.6, color=INK, fontweight="bold")
    ax.text(0.91, 0.585, "(8, 1)", ha="center", fontsize=8.6, color=INK, fontweight="bold")
    ax.text(0.91, 0.535, "→ C[i,:]", ha="center", fontsize=8.0, color=TEAL_D)
    ax.text(0.91, 0.45, "3 in → 2 out", ha="center", fontsize=9.2, color=TEAL_D, fontweight="bold")

    ax.text(0.5, 0.06, "Duplicates collapse on arrival — only distinct (j, Σv) is written.  "
            "Real dup factor up to 19× (bcsstk30).",
            ha="center", fontsize=9, color=TEAL_D, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_dedup_design.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig/hash_dedup_design.{png,pdf}")


if __name__ == "__main__":
    minhash()
    dedup()
