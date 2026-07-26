#!/usr/bin/env python3
"""Assemble two 16:9 slide composites for the Design3 (Hash) page(s).

Page A (insight):  accumulation-method landscape  +  ESC accumulation-bottleneck
                   (motivation)  +  takeaways
Page B (design):   hash-SPA mechanism pipeline  +  MinHash sizing data

The user builds the real PPT; these composites are a layout reference / drop-in.
Embeds fig/*.png produced by plot_hash_landscape / plot_hash_mechanism /
plot_hash_result and the earlier fig/esc_runtime_share.png.
"""
import os
import matplotlib.pyplot as plt
from PIL import Image

INK, TEAL = "#1F2933", "#1485A4"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def _place(fig, rect, png):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.imshow(Image.open(png))


def _title(fig, x, y, text, sub=None, color=INK):
    fig.text(x, y, text, fontsize=15, fontweight="bold", color=color, va="top")
    if sub:
        fig.text(x, y - 0.045, sub, fontsize=9.5, color="#5C6773", va="top")


def build_insight():
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    _title(fig, 0.5 / 12.8, 0.995,
           "Design 3 · Hash — deduplicate in place, not after a 19× blow-up",
           "Three accumulation families; hash SPA collapses duplicates as they arrive")
    _place(fig, [0.012, 0.40, 0.976, 0.50], "fig/hash_landscape.png")
    _place(fig, [0.012, 0.035, 0.415, 0.345], "fig/esc_runtime_share.png")
    ax = fig.add_axes([0.455, 0.045, 0.52, 0.32]); ax.axis("off")
    takeaways = (
        "• Sort-based (ESC) materializes O(flop) — duplicates up to\n"
        "   19× — before dedup; accumulation = 72% of runtime.\n\n"
        "• Hash SPA: a per-row SMEM hash table does atomicCAS /\n"
        "   atomicAdd, so duplicates collapse on arrival. Only the\n"
        "   distinct C[i,:] is ever written or sorted.\n\n"
        "• MinHash sizing (ours): per-row distinct upper bound,\n"
        "   orthogonal to Ocean's HLL (min-value vs max-leading-zero).\n\n"
        "• Cost vs HLL: +8–19% compute (looser over-alloc 3.0–3.14×\n"
        "   vs 2.88×); vs exact-count (opSparse): 3.5× faster symbolic."
    )
    ax.text(0.0, 1.0, takeaways, fontsize=9.6, va="top", color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#F4FAFB", ec=TEAL, lw=1.2))
    fig.savefig("fig/hash_slide_insight.png", dpi=170, facecolor="white", bbox_inches="tight")
    fig.savefig("fig/hash_slide_insight.pdf", facecolor="white", bbox_inches="tight")
    print("wrote fig/hash_slide_insight.{png,pdf}")


def build_design():
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    _title(fig, 0.5 / 12.8, 0.995,
           "Design 3 · Hash — SPA pipeline + MinHash sizing",
           "In-place dedup + an HLL-orthogonal estimator for the table size", color=TEAL)
    _place(fig, [0.012, 0.50, 0.976, 0.42], "fig/hash_mechanism.png")
    _place(fig, [0.012, 0.045, 0.976, 0.42], "fig/hash_result.png")
    fig.savefig("fig/hash_slide_design.png", dpi=170, facecolor="white", bbox_inches="tight")
    fig.savefig("fig/hash_slide_design.pdf", facecolor="white", bbox_inches="tight")
    print("wrote fig/hash_slide_design.{png,pdf}")


if __name__ == "__main__":
    build_insight()
    build_design()
