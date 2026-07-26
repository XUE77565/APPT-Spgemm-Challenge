#!/usr/bin/env python3
"""Design 3 (Hash) slide v2 — clearer MinHash + in-place dedup.

  ① Insight     — opSparse baseline: symbolic (teal) + accumulation (red) = 79%
  ② Symbolic    — MinHash: per-partition MIN-hash → estimate; 3.3x faster than
                  exact-count, orthogonal to HLL
  ③ Accumulation — in-place dedup: atomicCAS / atomicAdd collapse duplicates;
                  11-16x cheaper than sort-based ESC

Single 16:9 composite. Output: fig/hash_design_slide.{png,pdf}.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow

TEAL, RED, GRAY = "#1485A4", "#C00000", "#C9D1D9"
TEAL_LT, TEAL_D, INK, MUTED, YEL = "#CFE8EE", "#0E6E83", "#1F2933", "#5C6773", "#FFF6D5"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none"})


def header(ax, tag, title):
    ax.add_patch(Rectangle((0, 0.905), 1, 0.095, fc=TEAL, ec="none", zorder=2))
    ax.text(0.04, 0.952, tag, transform=ax.transAxes, fontsize=12.5, color="white",
            fontweight="bold", va="center")
    ax.text(0.13, 0.952, title, transform=ax.transAxes, fontsize=11, color="white",
            fontweight="bold", va="center")
    ax.add_patch(Rectangle((0, 0), 1, 0.905, fc="#FBFDFE", ec="#D7E2E6", lw=0.8, zorder=1))


def step(ax, y, label, fc, ec=GRAY, tc=INK, h=0.085, fs=8.4, bold=False, w=0.86):
    ax.add_patch(FancyBboxPatch(((1 - w) / 2, y), w, h, boxstyle="round,pad=0.010",
                                fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(0.50, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=4)


def vflow(ax, boxes, y_top=0.84, y_bot=0.36):
    """Stack boxes top→bottom with downward arrows in the gaps."""
    n = len(boxes)
    total_h = sum(b["h"] for b in boxes)
    gap = (y_top - y_bot - total_h) / (n - 1) if n > 1 else 0.0
    y = y_top
    for i, b in enumerate(boxes):
        y -= b["h"]
        step(ax, y, b["label"], b["fc"], ec=b.get("ec", GRAY), tc=b.get("tc", INK),
             h=b["h"], fs=b.get("fs", 8.4), bold=b.get("bold", False))
        if i < n - 1:
            ax.add_patch(FancyArrow(0.50, y - 0.004, 0.0, -(gap - 0.010), width=0.006,
                                    head_width=0.024, head_length=min(0.014, gap / 2),
                                    color=TEAL, lw=1.4, length_includes_head=True, zorder=4))
        y -= gap


def callout(ax, y, h, big, small):
    ax.add_patch(FancyBboxPatch((0.08, y), 0.84, h, boxstyle="round,pad=0.012",
                                fc=YEL, ec=TEAL, lw=1.0, zorder=3))
    ax.text(0.50, y + h * 0.64, big, ha="center", fontsize=10, color=TEAL_D,
            fontweight="bold", zorder=4)
    ax.text(0.50, y + h * 0.26, small, ha="center", fontsize=7.4, color=INK, zorder=4)


def caption(ax, text, color, y=0.045):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=8.3, color=color, fontweight="bold")


# ---------------------------------------------------------- column 1: opSparse baseline
def col_baseline(ax):
    header(ax, "①", "Insight · opSparse baseline")
    ACC, SYM, OH = 51.7, 26.8, 21.5
    y0, H = 0.16, 0.66
    ya = y0 + H * ACC / 100
    ys = ya + H * SYM / 100
    ax.add_patch(Rectangle((0.30, y0), 0.40, H * ACC / 100, fc=RED, ec="white", lw=1.0, zorder=3))
    ax.add_patch(Rectangle((0.30, ya), 0.40, H * SYM / 100, fc=TEAL, ec="white", lw=1.0, zorder=3))
    ax.add_patch(Rectangle((0.30, ys), 0.40, H * OH / 100, fc=GRAY, ec="white", lw=1.0, zorder=3))
    ax.text(0.50, y0 + H * ACC / 200, f"accumulation\n{ACC:.0f}%", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold", zorder=4)
    ax.text(0.50, ya + H * SYM / 200, f"symbolic\n{SYM:.0f}%", ha="center", va="center",
            color="white", fontsize=8.4, fontweight="bold", zorder=4)
    ax.text(0.50, ys + H * OH / 200, "overhead", ha="center", va="center",
            color=MUTED, fontsize=7.3, zorder=4)
    ax.text(0.50, 0.875, "opSparse · bcsstk30 · 3.04 ms total", ha="center",
            fontsize=7.7, color=MUTED)
    caption(ax, "Symbolic + accumulation = 79% —\nthe two stages we accelerate.", RED)


# ---------------------------------------------------------- column 2: MinHash
def col_minhash(ax):
    header(ax, "②", "Symbolic · MinHash sizing")
    ax.text(0.5, 0.875, "estimate distinct columns without counting", ha="center",
            fontsize=7.6, color=MUTED)
    vflow(ax, [
        dict(label="hash each column index", fc="white", h=0.070),
        dict(label="split into m partitions —\nkeep the MIN hash in each",
             fc=TEAL_LT, ec=TEAL, tc=TEAL_D, h=0.095, bold=True, fs=8.1),
        dict(label=r"$\hat n = 2^{32}\sum_j(1/min_j)\,-\,m$", fc="#EAF4F6", ec=TEAL, h=0.070, fs=9),
        dict(label="→ right-size the hash table", fc=TEAL, ec=TEAL, tc="white", h=0.070, bold=True),
    ])
    callout(ax, 0.15, 0.15, "3.3× faster symbolic", "0.23 ms (ours)  vs  0.82 ms (exact-count)")
    caption(ax, "Per-partition MinHash — orthogonal\nto Ocean's HLL (min vs max-leading-zero).", TEAL_D)


# ---------------------------------------------------------- column 3: in-place dedup
def col_dedup(ax):
    header(ax, "③", "Accumulation · in-place dedup")
    ax.text(0.5, 0.875, "collapse duplicates as they arrive", ha="center",
            fontsize=7.6, color=MUTED)
    vflow(ax, [
        dict(label="products (i, j, v) stream in\n(j repeats — dup factor up to 19×)",
             fc="white", h=0.095, fs=8.0),
        dict(label="hash(j) → slot\natomicCAS · insert j once\natomicAdd · accumulate v",
             fc=TEAL_LT, ec=TEAL, tc=TEAL_D, h=0.120, bold=True, fs=8.0),
        dict(label="distinct (j, Σv) —\nduplicates collapsed",
             fc=TEAL, ec=TEAL, tc="white", h=0.095, bold=True, fs=8.2),
    ])
    callout(ax, 0.15, 0.15, "11–16× cheaper accumulation", "in-place dedup vs ESC expand+sort+reduce")
    caption(ax, "No O(flop) expand/sort —\nonly distinct C[i,:] is written.", TEAL_D)


def main():
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    at = fig.add_axes([0, 0.93, 1, 0.06]); at.axis("off")
    at.add_patch(Rectangle((0, 0), 1, 1, fc=TEAL, ec="none", transform=at.transAxes))
    at.text(0.5, 0.5, "Design 3 · Hash SPA  —  MinHash symbolic + in-place accumulation",
            transform=at.transAxes, ha="center", va="center", fontsize=14.5,
            color="white", fontweight="bold")
    for ax, fn in zip(
            [fig.add_axes([0.022 + i * 0.326, 0.045, 0.305, 0.86]) for i in range(3)],
            [col_baseline, col_minhash, col_dedup]):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        fn(ax)
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_design_slide.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/hash_design_slide.{png,pdf}")


if __name__ == "__main__":
    main()
