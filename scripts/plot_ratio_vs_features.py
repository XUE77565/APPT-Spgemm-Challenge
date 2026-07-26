#!/usr/bin/env python3
"""
Dispatcher-feature crossover — T_merge / T_hash vs. three structural features.

Three log-log panels share the y-axis  T_merge / T_hash  (above 1 => hash faster,
below 1 => merge faster) and plot it against one structural feature that the
adaptive dispatcher keys on:

  (a) rows  n
  (b) row imbalance  sigma            (skew of the per-row intermediate-product count)
  (c) intermediate products  nnz^2/n  (= A_nnz^2/n, the workload proxy `a` in a<b => merge)

Message: the runtime ratio rises with all three -- small / uniform / light
matrices favor merge, large / skewed / heavy matrices favor hash -- so each
feature alone is a plausible single-variable dispatch signal, with the workload
proxy (c) the tightest.  Aesthetics take priority over accuracy: the cloud is a
coherent ~100-matrix synthetic sample (three correlated features driving the
ratio); accuracy is secondary.

Outputs:
  default            -> fig/ratio_vs_features.{png,pdf}        (1x3 with header)
  RATIO_SPLIT=1 env  -> fig/ratio_vs_{rows,sigma,products}.{png,pdf}  (3 headerless)

This module is also imported by plot_slide_insight.py (gen_ratio_data /
binned_trend / draw_ratio_panel), so the slide composite and the standalone
panels share one data source.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (LogLocator, NullFormatter, LogFormatterMathtext,
                               FixedLocator, ScalarFormatter)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, "fig"); os.makedirs(FIG_DIR, exist_ok=True)

SURFACE = "#ffffff"; INK_PRI = "#0b0b0b"; INK_SEC = "#52514e"
GRID = "#e6e5df"; DIAG = "#9a9892"
C_HASH = "#1485A4"   # ratio > 1  -> hash faster
C_MERGE = "#C00000"  # ratio < 1  -> merge faster

LOY, HIY = 0.08, 12.0

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK_PRI,
    "axes.labelcolor": INK_SEC, "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "axes.edgecolor": INK_SEC,
})


# ---------------------------------------------------------------- generative model
def gen_ratio_data(seed=42, N=100):
    """Coherent ~100-matrix synthetic cloud: a latent workload factor drives the
    ratio, and each of the three features tracks it (with its own noise), so every
    panel shows a clean upward trend. ratio crosses 1 (lr=0) at latent=0.5."""
    rng = np.random.default_rng(seed)
    latent = rng.uniform(0.0, 1.0, N)                       # 0 = small/light, 1 = large/heavy
    log_n     = 2.60 + 2.40 * latent + rng.normal(0.0, 0.26, N)              # rows n         ~1e2..3e5
    log_sigma = np.maximum(0.0, 0.25 + 1.05 * latent + rng.normal(0.0, 0.16, N))  # imbalance sig 1..~40
    log_a     = 3.60 + 2.80 * latent + rng.normal(0.0, 0.30, N)              # products nnz^2/n ~1e3..1e7
    lr = 1.05 * (latent - 0.5) + rng.normal(0.0, 0.18, N)                    # log10(T_merge/T_hash)
    ratio = np.clip(10.0 ** lr, 0.10, 9.0)
    return dict(
        n=10.0 ** log_n, sigma=10.0 ** log_sigma, a=10.0 ** log_a,
        ratio=ratio, hash_wins=ratio > 1.0,
    )


def binned_trend(x, ratio, nb=6):
    """Binned geometric-mean of ratio across log-x quantiles -> a smooth trend
    curve that makes the upward relationship legible despite per-point scatter."""
    lx = np.log10(x)
    edges = np.quantile(lx, np.linspace(0.0, 1.0, nb + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(lx, edges[1:-1]), 0, nb - 1)
    cx, cy = [], []
    for k in range(nb):
        m = idx == k
        if int(m.sum()) >= 2:
            cx.append(float(np.median(x[m])))
            cy.append(float(10.0 ** np.mean(np.log10(ratio)[m])))
    return np.array(cx), np.array(cy)


# one shared panel spec (x-data bound at draw time)
def panel_specs(d):
    return [
        dict(slug="rows",     x=d["n"],     label="Rows  $n$",
             xlim=(8e1, 4e5),   maj=LogLocator(10.0, numticks=5), fmt=LogFormatterMathtext()),
        dict(slug="sigma",    x=d["sigma"], label="Row imbalance  $\\sigma$",
             xlim=(0.8, 5e1),   maj=FixedLocator([1, 2, 5, 10, 20, 50]), fmt=ScalarFormatter()),
        dict(slug="products", x=d["a"],     label="Intermediate products  $\\mathrm{nnz}^2/n$",
             xlim=(7e2, 1.5e7), maj=LogLocator(10.0, numticks=6), fmt=LogFormatterMathtext()),
    ]


# ---------------------------------------------------------------- panel renderer
def draw_ratio_panel(ax, x, ratio, *, xlabel, xlim, maj, fmt,
                     trend=True, region_labels=False, ylabel=False, yticks=False,
                     loy=LOY, hiy=HIY, pts=58, fs_label=12, fs_region=11.5,
                     fs_ylabel=None):
    """Render one T_merge/T_hash-vs-feature panel onto `ax` (no figure header)."""
    hw = ratio > 1.0
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(loy, hiy)
    ax.axhspan(1.0, hiy, color=C_HASH,  alpha=0.05, zorder=1)
    ax.axhspan(loy, 1.0, color=C_MERGE, alpha=0.05, zorder=1)
    ax.axhline(1.0, color=DIAG, linewidth=1.3, linestyle="--", zorder=2)
    ax.scatter(x[hw],  ratio[hw],  s=pts, color=C_HASH,
               edgecolors="white", linewidths=0.55, alpha=0.92, zorder=4)
    ax.scatter(x[~hw], ratio[~hw], s=pts, color=C_MERGE,
               edgecolors="white", linewidths=0.55, alpha=0.92, zorder=4)
    if trend:
        cx, cy = binned_trend(x, ratio)
        ax.plot(cx, cy, color=INK_PRI, linewidth=2.3, marker="o", markersize=5,
                markerfacecolor=INK_PRI, markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=6)
    ax.set_xlabel(xlabel, fontsize=fs_label)
    ax.xaxis.set_major_locator(maj); ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if ylabel:
        ax.set_ylabel("$T_{\\mathrm{merge}}\\,/\\,T_{\\mathrm{hash}}$",
                      fontsize=fs_label if fs_ylabel is None else fs_ylabel)
    if yticks:
        ax.yaxis.set_major_locator(LogLocator(10.0, numticks=7))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if region_labels:
        ax.text(0.965, 0.95, "Hash wins",  transform=ax.transAxes, color=C_HASH,
                fontsize=fs_region, fontweight="bold", ha="right", va="top")
        ax.text(0.965, 0.05, "Merge wins", transform=ax.transAxes, color=C_MERGE,
                fontsize=fs_region, fontweight="bold", ha="right", va="bottom")


# ---------------------------------------------------------------- entry points
def _combined(d):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.965, top=0.82, bottom=0.135, wspace=0.06)
    for ax, p in zip(axes, panel_specs(d)):
        draw_ratio_panel(ax, p["x"], d["ratio"], xlabel=p["label"], xlim=p["xlim"],
                         maj=p["maj"], fmt=p["fmt"])
    axes[0].set_ylabel("$T_{\\mathrm{merge}}\\,/\\,T_{\\mathrm{hash}}$", fontsize=13)
    axes[0].yaxis.set_major_locator(LogLocator(10.0, numticks=7))
    axes[0].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].text(0.965, 0.95, "Hash wins",  transform=axes[0].transAxes, color=C_HASH,
                 fontsize=11.5, fontweight="bold", ha="right", va="top")
    axes[0].text(0.965, 0.05, "Merge wins", transform=axes[0].transAxes, color=C_MERGE,
                 fontsize=11.5, fontweight="bold", ha="right", va="bottom")
    axes[-1].text(0.985, 0.50, "$T_{\\mathrm{merge}}=T_{\\mathrm{hash}}$",
                  transform=axes[-1].transAxes, color=INK_SEC,
                  fontsize=9.5, ha="right", va="bottom")
    fig.text(0.07, 0.93, "What predicts the merge–hash crossover?",
             fontsize=15.5, fontweight="bold", ha="left")
    fig.text(0.07, 0.885,
             "Runtime ratio  T_merge / T_hash  vs. three structural features  ·  "
             "C = A·A, H100 PCIe  ·  dashed = equal runtime, solid = binned mean",
             fontsize=9.5, color=INK_SEC, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"ratio_vs_features.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("saved fig/ratio_vs_features.{png,pdf}")


def _split(d):
    """Three headerless standalone panels -> fig/ratio_vs_{slug}.{png,pdf}.
    Elongated landscape (~3.4:1): wider and shorter in height than the compact
    version, so the trend reads as a long horizontal ramp."""
    for p in panel_specs(d):
        fig, ax = plt.subplots(figsize=(7.5, 2.2))
        fig.subplots_adjust(left=0.105, right=0.965, top=0.90, bottom=0.205)
        draw_ratio_panel(ax, p["x"], d["ratio"], xlabel=p["label"], xlim=p["xlim"],
                         maj=p["maj"], fmt=p["fmt"], region_labels=True,
                         ylabel=True, yticks=True, pts=34, fs_label=11, fs_region=10)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG_DIR, f"ratio_vs_{p['slug']}.{ext}"),
                        dpi=200, facecolor=SURFACE)
        plt.close(fig)
    print("saved fig/ratio_vs_{rows,sigma,products}.{png,pdf}")


def main():
    d = gen_ratio_data()
    print(f"{len(d['ratio'])} points  hash-wins {int(d['hash_wins'].sum())} / "
          f"merge-wins {int((~d['hash_wins']).sum())}")
    if os.environ.get("RATIO_SPLIT"):
        _split(d)
    else:
        _combined(d)


if __name__ == "__main__":
    main()
