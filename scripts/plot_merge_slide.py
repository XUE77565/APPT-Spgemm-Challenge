#!/usr/bin/env python3
"""Assemble two 16:9 slide composites for the Design2 (Merge) page(s).

Page A (insight):  4-axis parallel landscape  +  per-row workload imbalance
Page B (design):   column-domain bucketing mechanism  +  flop_ub sizing result

The user builds the real PPT; these composites are a layout reference / drop-in.
Embeds the standalone fig/*.png produced by plot_merge_{landscape,skew,mechanism,result}.py.
"""
import os
import matplotlib.pyplot as plt
from PIL import Image

INK, TEAL = "#1F2933", "#1485A4"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def _place(fig, rect, png, pad=0.006):
    ax = fig.add_axes(rect)
    ax.axis("off")
    im = Image.open(png)
    ax.imshow(im)


def _title(fig, x, y, text, sub=None, color=INK):
    fig.text(x, y, text, fontsize=15, fontweight="bold", color=color, va="top")
    if sub:
        fig.text(x, y - 0.045, sub, fontsize=9.5, color="#5C6773", va="top")


def build_insight():
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")  # 16:9
    _title(fig, 0.5/12.8, 0.995, "Design 2 · Merge — the missing parallel axis",
           "Why merge-based SpGEMM was abandoned on GPUs, and the axis we add")
    # landscape (top, full width)
    _place(fig, [0.012, 0.40, 0.976, 0.50], "fig/merge_landscape.png")
    # skew (bottom-left)
    _place(fig, [0.012, 0.035, 0.50, 0.345], "fig/merge_skew.png")
    # insight takeaways (bottom-right)
    ax = fig.add_axes([0.535, 0.045, 0.44, 0.32]); ax.axis("off")
    takeaways = (
        "• Prior merge SpGEMM uses 3 parallel axes:\n"
        "   cross-row (Gustavson), row-length balancing\n"
        "   (bhSparse), K-chain merge tree (SpArch).\n\n"
        "• None partitions a row's column range — so the\n"
        "   heavy-row straggler is unavoidable, and the field\n"
        "   defected to hash / ESC accumulators.\n\n"
        "• Workload is imbalanced (top 20% of rows hold\n"
        "   27–44% of intermediate products), gating any\n"
        "   single-unit-per-row merge.\n\n"
        "• Column-domain bucketing is a 4th, order-preserving\n"
        "   axis: disjoint buckets concatenate to sorted CSR."
    )
    ax.text(0.0, 1.0, takeaways, fontsize=9.6, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F4FAFB", ec=TEAL, lw=1.2))
    fig.savefig("fig/merge_slide_insight.png", dpi=170, facecolor="white",
                bbox_inches="tight")
    fig.savefig("fig/merge_slide_insight.pdf", facecolor="white", bbox_inches="tight")
    print("wrote fig/merge_slide_insight.{png,pdf}")


def build_design():
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")  # 16:9
    _title(fig, 0.5/12.8, 0.995, "Design 2 · Merge — column-domain bucketing",
           "Split the row's column range; one warp per bucket; concatenate ⇒ sorted", color=TEAL)
    # mechanism (top, full width)
    _place(fig, [0.012, 0.50, 0.976, 0.42], "fig/merge_mechanism.png")
    # result (bottom, full width)
    _place(fig, [0.012, 0.045, 0.976, 0.42], "fig/merge_result.png")
    fig.savefig("fig/merge_slide_design.png", dpi=170, facecolor="white",
                bbox_inches="tight")
    fig.savefig("fig/merge_slide_design.pdf", facecolor="white", bbox_inches="tight")
    print("wrote fig/merge_slide_design.{png,pdf}")


if __name__ == "__main__":
    build_insight()
    build_design()
