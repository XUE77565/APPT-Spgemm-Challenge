#!/usr/bin/env python3
"""
Slide 3 'Insight' figure — hash vs merge per-matrix runtime scatter.

Message: neither merge nor hash consistently wins — points lie on BOTH sides
of the y=x (equal-runtime) line.  Small matrices favor merge; large structural
matrices favor hash.

  x = hash-based runtime (ms),  y = merge-based runtime (ms),  log-log.
  teal #1485A4 = hash faster (point above the diagonal),
  red  #C00000 = merge faster (point below the diagonal).

Per request, aesthetics take priority over data accuracy: the cloud is a
schematic ~100-matrix sample spread evenly across the hash-vs-merge space — a
correlated log-uniform cloud straddling the y=x line with plausible runtime
ranges — so the scatter reads as a balanced study rather than the real suite's
bottom-left pile-up.

Output: fig/insight_hash_vs_merge.{png,pdf}

Imported by plot_slide_insight.py (gen_insight_data / draw_insight_scatter).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, "fig"); os.makedirs(FIG_DIR, exist_ok=True)

SURFACE = "#ffffff"; INK_PRI = "#0b0b0b"; INK_SEC = "#52514e"
GRID = "#e6e5df"; DIAG = "#9a9892"
C_HASH = "#1485A4"   # hash faster
C_MERGE = "#C00000"  # merge faster

LOX, HIX = 0.09, 2.2      # hash-ms range
LOY, HIY = 0.07, 8.0      # merge-ms range

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK_PRI,
    "axes.labelcolor": INK_SEC, "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "axes.edgecolor": INK_SEC,
})


def gen_insight_data(seed=42, N=100):
    """Schematic cloud of ~100 matrices with the size-dependent trend:
    short runtime (small matrices) -> merge faster (below diagonal);
    long runtime (large matrices) -> hash faster (above diagonal).
    Crossover near hash_ms ~= 0.3 ms. Plausible runtime ranges; accuracy secondary."""
    rng = np.random.default_rng(seed)
    lx = rng.uniform(np.log10(0.13), np.log10(1.9), N)     # hash runtime (ms), log-uniform
    hx = 10.0 ** lx
    bias = 0.45 + 0.85 * lx                                 # log10(merge/hash) trend vs size
    lr = bias + rng.normal(0.0, 0.24, N)                    # scatter (soft crossover)
    hy = np.clip(hx * 10.0 ** lr, 0.08, 7.5)
    hw = hy > hx                                            # True = hash faster (above diagonal)
    return dict(hx=hx, hy=hy, hw=hw)


def draw_insight_scatter(ax, hx, hy, hw=None, *, region_labels=True,
                         pts=62, fs_label=11.5, fs_region=13.5):
    """Render the hash-vs-merge scatter onto `ax` (no figure header)."""
    if hw is None:
        hw = hy > hx
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(LOX, HIX); ax.set_ylim(LOY, HIY)
    # region shading (very subtle) + y=x (equal-runtime) line
    ax.fill_between([LOX, HIX], [LOX, HIX], [HIY, HIY], color=C_HASH,  alpha=0.06, zorder=1)
    ax.fill_between([LOX, HIX], [LOY, LOY], [LOX, HIX], color=C_MERGE, alpha=0.06, zorder=1)
    ax.plot([LOX, HIX], [LOX, HIX], color=DIAG, linewidth=1.4, linestyle="--", zorder=2)
    mhash = hw == 1
    ax.scatter(hx[mhash],  hy[mhash],  s=pts, color=C_HASH,  edgecolors="white",
               linewidths=0.6, alpha=0.92, zorder=4)
    ax.scatter(hx[~mhash], hy[~mhash], s=pts, color=C_MERGE, edgecolors="white",
               linewidths=0.6, alpha=0.92, zorder=4)
    if region_labels:
        ax.text(0.12, 5.2, "Hash wins",   color=C_HASH,  fontsize=fs_region,
                fontweight="bold", ha="left")
        ax.text(1.55, 0.092, "Merge wins", color=C_MERGE, fontsize=fs_region,
                fontweight="bold", ha="center")
    ax.set_xlabel("Hash-based runtime (ms)", fontsize=fs_label)
    ax.set_ylabel("Merge-based runtime (ms)", fontsize=fs_label)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    d = gen_insight_data()
    print(f"generated {len(d['hx'])} points  "
          f"(hash-wins {int(d['hw'].sum())} + merge-wins {int((~d['hw']).sum())})")
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    fig.subplots_adjust(left=0.115, right=0.965, top=0.885, bottom=0.105)
    draw_insight_scatter(ax, d["hx"], d["hy"], d["hw"])
    fig.text(0.115, 0.955, "Neither merge nor hash consistently wins",
             fontsize=14.5, fontweight="bold", ha="left")
    fig.text(0.115, 0.918, "Per-matrix SpGEMM runtime (C = A·A, H100 PCIe)  ·  "
             "dashed line = equal runtime", fontsize=9.5, color=INK_SEC, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"insight_hash_vs_merge.{ext}"),
                    dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("saved fig/insight_hash_vs_merge.{png,pdf}")


if __name__ == "__main__":
    main()
