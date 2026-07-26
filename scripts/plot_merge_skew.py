#!/usr/bin/env python3
"""Merge Design2 — insight motivation: per-row workload imbalance.

Lorenz curves of per-row intermediate products (Gustavson k-way work) for 6 structural
matrices. x = cumulative fraction of rows (heaviest first), y = cumulative fraction of
total intermediate products. The diagonal = perfectly balanced. All curves bow well below
the diagonal: a single execution unit per row (classical Gustavson / bhSparse) is gated by
the heaviest rows -> straggler. Column-domain bucketing splits each row's column range so K
units share a row.

Per-row products computed on the actual .mtx (C = A·A: row i work = Σ_{k∈A[i,:]} nnz(A[k,:])).
Palette: single teal family (lightness-graded); identity via direct end-labels.
"""
import os, numpy as np, scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED = "#1F2933", "#5C6773"
TEAL = "#1485A4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK})

MATS = ["bcsstk29", "bcsstk17", "bcsstk30", "bcsstk32", "bcsstk31", "bcsstk08"]


def per_row_products(path):
    A = sio.mmread(path).tocsr()
    rownnz = np.diff(A.indptr)
    co = A.tocoo()
    prod = np.zeros(A.shape[0], dtype=np.int64)
    np.add.at(prod, co.row, rownnz[co.col])
    return prod


def lorenz(prod):
    s = np.sort(prod)[::-1].astype(float)
    cdf = np.cumsum(s) / s.sum()
    x = np.arange(1, len(s) + 1) / len(s)
    return x, cdf


def gen_skew_data(root="data/first100"):
    return {m: per_row_products(f"{root}/{m}.mtx") for m in MATS}


def draw_skew(ax, data):
    cmap = LinearSegmentedColormap.from_list("teal", ["#9FCFD8", TEAL, "#0A4E5E"], N=256)
    # order by skew (sigma) so the ramp is meaningful: light = balanced, dark = skewed
    sigma = {m: data[m].max() / data[m].mean() for m in MATS}
    order = sorted(MATS, key=lambda m: sigma[m])
    colors = [cmap(0.20 + 0.62 * i / (len(order) - 1)) for i in range(len(order))]

    # diagonal = balanced
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.4, zorder=2)
    ax.annotate("balanced", (0.62, 0.62), color=MUTED, fontsize=8.5,
                rotation=39, ha="left", va="bottom")

    for m, c in zip(order, colors):
        x, y = lorenz(data[m])
        bold = (m in ("bcsstk08", "bcsstk29"))  # extremes
        ax.plot(x, y, color=c, lw=2.4 if bold else 1.6, zorder=4 if bold else 3)
        # direct end label at top-20% rows point
        idx = np.searchsorted(x, 0.20)
        ax.annotate(m, (x[idx], y[idx]), textcoords="offset points",
                    xytext=(6, -2), color=c, fontsize=8.5, fontweight="bold" if bold else "normal")

    # shade the imbalance bow for the most skewed (bcsstk08) to make the gap visible
    xb, yb = lorenz(data["bcsstk08"])
    ax.fill_between(xb, yb, xb, color="#C00000", alpha=0.06, zorder=1)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("cumulative fraction of rows (heaviest first)")
    ax.set_ylabel("cumulative fraction of intermediate products")
    ax.set_title("Per-row merge workload is imbalanced — heavy rows gate a 1-unit-per-row merge",
                 fontsize=11.5, color=INK, pad=8, loc="left")
    ax.grid(color="#E5E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # compact sigma / top-20% share inset table
    txt = "matrix            σ      top-20% rows hold\n"
    for m in order:
        prod = data[m]
        s = np.sort(prod)[::-1]; k = max(1, int(round(0.20 * len(prod))))
        share = 100 * s[:k].sum() / prod.sum()
        txt += f"{m:<12s}{sigma[m]:4.1f}        {share:3.0f}%\n"
    ax.text(0.97, 0.06, txt, transform=ax.transAxes, fontsize=8.2, ha="right", va="bottom",
            family="monospace", color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#D7DCE1", lw=0.8))


def main():
    data = gen_skew_data()
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    draw_skew(ax, data)
    fig.tight_layout()
    os.makedirs("fig", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig/merge_skew.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig/merge_skew.{png,pdf}")


if __name__ == "__main__":
    main()
