#!/usr/bin/env python3
"""MinHash sizing — two-phase flow design diagram.

  Phase 1 — sketch each A-row ONCE: scan its column indices, hash(j) → partition
            (low bits) → atomicMin. Produces one sketch (array of m bucket-mins) per A-row.
  Phase 2 — per output row i: merge the sketches of the A-rows it references
            (per-bucket min; NO re-hash) → the merged sketch is C[i,:]'s sketch →
            apply  n̂ = 2^32·Σ(1/min) − m  → C[i,:] nnz → size the hash table.

Concrete toy example: A row1={5,8}, row2={12,5}, row3={9}; output row i refs {1,2}
→ C[i,:] distinct = {5,8,12} = 3 → estimate ≈ 3.

Output: fig/hash_minhash_flow.{png,pdf}
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow

TEAL, GRAY = "#1485A4", "#C9D1D9"
TEAL_LT, TEAL_D, INK, MUTED = "#CFE8EE", "#0E6E83", "#1F2933", "#5C6773"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "none"})


def bucket(ax, x, y, w, h, val, faded=False):
    filled = val != "∅"
    fc = ("#F0F2F4" if faded else TEAL_LT) if filled else ("#F7F8F9" if faded else "white")
    ec = GRAY if (faded or not filled) else TEAL
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8, zorder=3))
    ax.text(x + w / 2, y + h / 2, val, ha="center", va="center", fontsize=5.7,
            color=INK if (filled and not faded) else MUTED, zorder=4)


def sketch_row(ax, x0, y, label, vals, faded=False, w=0.052, h=0.042):
    col = MUTED if faded else TEAL_D
    ax.text(x0 - 0.008, y + h / 2, label, ha="right", va="center", fontsize=7.4,
            color=col, fontweight="bold")
    for i, v in enumerate(vals):
        bucket(ax, x0 + i * w, y, w * 0.92, h, v, faded=faded)


def arr(ax, x, y, dx, dy, color=MUTED, lw=1.6):
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.006, head_width=0.022,
                            head_length=0.014, color=color, lw=lw,
                            length_includes_head=True, zorder=4))


def main():
    fig, ax = plt.subplots(figsize=(13.8, 7.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "MinHash sizing — two-phase flow",
            ha="center", fontsize=14, color=INK, fontweight="bold")
    ax.text(0.5, 0.932, "Phase 1 sketches each A-row once (hash once)  ·  Phase 2 merges per output row (no re-hash) → estimate",
            ha="center", fontsize=9.2, color=MUTED)

    # ============================ PHASE 1 ============================
    ax.add_patch(FancyBboxPatch((0.01, 0.54), 0.98, 0.37, boxstyle="round,pad=0.008",
                                fc="#FBFDFE", ec=TEAL, lw=1.2, zorder=1))
    ax.text(0.02, 0.885, "Phase 1 · sketch each A-row   (hash + partition + atomicMin · O(nnz))",
            fontsize=10, color=TEAL_D, fontweight="bold")

    # A operand
    ax.add_patch(FancyBboxPatch((0.025, 0.60), 0.135, 0.235, boxstyle="round,pad=0.01",
                                fc="white", ec=GRAY, lw=0.9, zorder=2))
    ax.text(0.0925, 0.815, "A (operand)", ha="center", fontsize=8.2, color=INK, fontweight="bold")
    for i, (r, cols, f) in enumerate([("row 1", "{5, 8}", False), ("row 2", "{12, 5}", False),
                                      ("row 3", "{9}", True)]):
        ax.text(0.035, 0.775 - i * 0.052, f"{r} :", ha="left", va="center",
                fontsize=7.6, color=MUTED if f else INK)
        ax.text(0.075, 0.775 - i * 0.052, cols, ha="left", va="center",
                fontsize=7.6, color=MUTED if f else INK)
    ax.text(0.0925, 0.58, "(…more rows)", ha="center", fontsize=6.6, color=MUTED)

    arr(ax, 0.165, 0.72, 0.05, 0)
    ax.text(0.225, 0.755, "per row:", ha="center", fontsize=7.0, color=MUTED)
    ax.text(0.225, 0.725, "hash(j) →", ha="center", fontsize=7.0, color=MUTED)
    ax.text(0.225, 0.695, "bucket h&127", ha="center", fontsize=7.0, color=MUTED)
    ax.text(0.225, 0.665, "→ atomicMin", ha="center", fontsize=7.0, color=TEAL_D, fontweight="bold")
    arr(ax, 0.285, 0.72, 0.05, 0)

    # sketches (m=128, show 4) — each distinct column lands in its own bucket
    sketch_row(ax, 0.36, 0.80, "sketch(row 1) =", ["0x3F2AB2A0", "0x7C199C44", "∅", "∅"])
    sketch_row(ax, 0.36, 0.70, "sketch(row 2) =", ["0x3F2AB2A0", "∅", "0x4F19", "∅"])
    sketch_row(ax, 0.36, 0.60, "sketch(row 3) =", ["∅", "∅", "∅", "0x6A22"], faded=True)
    ax.text(0.36, 0.568, "5→b0 · 8→b1 · 12→b2   (deterministic hash ⇒ shared cols share a bucket)",
            fontsize=5.9, color=MUTED, style="italic")
    ax.text(0.36, 0.548, "(m = 128 buckets; 4 shown)", fontsize=6.6, color=MUTED, style="italic")
    # bucket header
    for i, b in enumerate(["b0", "b1", "b2", "b3"]):
        ax.text(0.36 + i * 0.052 + 0.024, 0.855, b, ha="center", fontsize=6.2, color=MUTED)

    # ============================ PHASE 2 ============================
    ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.45, boxstyle="round,pad=0.008",
                                fc="#FBFDFE", ec=TEAL, lw=1.2, zorder=1))
    ax.text(0.02, 0.465, "Phase 2 · per output row i: merge referenced sketches (per-bucket min · no re-hash) → estimate",
            fontsize=10, color=TEAL_D, fontweight="bold")
    ax.text(0.02, 0.435, "output row i  references  A-rows {1, 2}   →   C[i,:] columns = {5,8} ∪ {12,5} = {5, 8, 12}",
            fontsize=8.0, color=INK)

    # merge: sketch1 + sketch2 -> merged (per-bucket min)
    mx = 0.10
    sketch_row(ax, mx, 0.36, "sketch(row 1) =", ["0x3F2AB2A0", "0x7C199C44", "∅", "∅"])
    sketch_row(ax, mx, 0.29, "sketch(row 2) =", ["0x3F2AB2A0", "∅", "0x4F19", "∅"])
    # per-bucket min arrows
    for i in range(4):
        arr(ax, mx + i * 0.052 + 0.024, 0.285, 0, -0.035, TEAL, lw=1.8)
    ax.text(0.05, 0.255, "min per bucket  (col 5 shared ⇒ b0 kept once)", fontsize=6.3, color=TEAL_D, fontweight="bold")
    sketch_row(ax, mx, 0.19, "merged  =", ["0x3F2AB2A0", "0x7C199C44", "0x4F19", "∅"])
    ax.text(mx + 4 * 0.052 + 0.01, 0.21, "= sketch of C[i,:]", ha="left", va="center",
            fontsize=7.4, color=TEAL_D, fontweight="bold")

    arr(ax, 0.40, 0.21, 0.05, 0)

    # estimate
    ax.add_patch(FancyBboxPatch((0.45, 0.135), 0.30, 0.175, boxstyle="round,pad=0.012",
                                fc="#EAF4F6", ec=TEAL, lw=1.0, zorder=3))
    ax.text(0.60, 0.285, "estimate distinct", ha="center", fontsize=8.4, color=TEAL_D, fontweight="bold")
    ax.text(0.60, 0.250, r"$\hat n = -m\,\ln(1 - V/m)$", ha="center", fontsize=8.4, color=INK)
    ax.text(0.60, 0.215, "V = 3 filled of m = 128", ha="center", fontsize=6.8, color=MUTED)
    ax.text(0.60, 0.180, "≈ 3.0", ha="center", fontsize=8.4, color=TEAL_D, fontweight="bold")
    ax.text(0.60, 0.145, r"dense regime: $2^{32}\Sigma(1/min)-m$", ha="center", fontsize=5.8, color=MUTED, style="italic")

    arr(ax, 0.745, 0.22, 0.05, 0)
    ax.add_patch(FancyBboxPatch((0.80, 0.14), 0.18, 0.085, boxstyle="round,pad=0.012",
                                fc=TEAL, ec=TEAL, lw=1.0, zorder=3))
    ax.text(0.89, 0.20, "C[i,:] nnz ≈ 3\n→ size hash table", ha="center", va="center",
            fontsize=7.8, color="white", fontweight="bold", zorder=4)

    ax.text(0.5, 0.015, "Total O(nnz): each A nonzero is hashed exactly once (Phase 1); Phase 2 only min-merges → never enumerates the 19× products.",
            ha="center", fontsize=8.4, color=TEAL_D, fontweight="bold")

    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/hash_minhash_flow.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/hash_minhash_flow.{png,pdf}")


if __name__ == "__main__":
    main()
