#!/usr/bin/env python3
"""Generate publication/slide figures + LaTeX table from compare/paper_cmp.csv.

Auto (ours) speedup over cuSPARSE / dense / opSparse / HSMU on first100.
Speedup = baseline_time / Auto_time  (>1 => Auto faster). English labels.

Outputs:
  fig/paper_speedup_overall.png   — headline: geomean speedup per baseline (slide)
  fig/paper_speedup_by_density.png— grouped bar by A-density class
  fig/paper_speedup_scatter.png   — 2x2 small multiples, per-matrix speedup vs n
  compare/paper_speedup_table.tex — booktabs LaTeX table (ready to \\input)

Palette (validated colorblind-safe, dataviz method): categorical slots 1-4 for
the four baselines; Auto = deck-cyan accent (#1485A4) for the break-even line.

Usage: .venv/bin/python scripts/paper_figures.py [paper_cmp.csv] [--slide]
"""
import os, sys, csv, math, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

CSV = "compare/paper_cmp.csv"
FIG = "fig"
os.makedirs(FIG, exist_ok=True)

# ---- validated categorical palette (dataviz slots 1-4, light mode) ----
# cuSPARSE / opSparse / HSMU / dense ; dense=yellow (the weak strawman baseline)
BASELINES = [
    ("cuSPARSE", "cuSPARSE", "#2a78d6"),
    ("opSparse", "opSparse", "#eb6834"),
    ("HSMU",     "HSMU",     "#1baf7a"),
    ("dense",    "dense",    "#eda100"),
]
OURS      = "#1485A4"   # Auto / ours (deck cyan) — break-even accent
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_SEC   = "#52514e"
GRID      = "#e6e6e6"

# ---- global style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": INK_SEC,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def f(x):
    try: return float(x)
    except: return None

def geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def dens_class(d):
    if d >= 10.0:  return "Dense"
    if d >= 1.0:   return "Mildly sparse"
    if d >= 0.1:   return "Highly sparse"
    return "Extremely sparse"

def speedup(rows, col):
    """geomean of baseline/Auto over matrices where both valid positive."""
    rs = []
    for r in rows:
        a, b = f(r.get("Auto")), f(r.get(col))
        if a and b and a > 0 and b > 0:
            rs.append(b / a)
    return geomean(rs), len(rs)

def win_count(rows, col):
    """# matrices where Auto is faster (baseline>=Auto) OR baseline DNFs (timeout/fail)."""
    n = 0
    for r in rows:
        a = f(r.get("Auto"))
        if not (a and a > 0):
            continue
        b = f(r.get(col))
        if b is None:          # baseline did not finish => Auto wins by default
            n += 1
        elif b >= a:
            n += 1
    return n

def auto_total(rows):
    """# matrices with a valid Auto time (denominator for win fractions)."""
    return sum(1 for r in rows if f(r.get("Auto")) and f(r.get("Auto")) > 0)

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="major", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)

_YTICKS = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000]
def clean_log_y(ax, lo, hi):
    """log y-axis with clean integer 'N×' tick labels (no scientific notation)."""
    ax.set_yscale("log")
    ticks = [t for t in _YTICKS if lo <= t <= hi]
    if 1 not in ticks:
        ticks = [1] + ticks
    ax.yaxis.set_major_locator(plt.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_locator(plt.NullLocator())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=CSV)
    ap.add_argument("--slide", action="store_true", help="larger fonts, tighter for slides")
    args = ap.parse_args()
    if args.slide:
        plt.rcParams.update({"font.size": 15, "axes.linewidth": 1.0})

    rows = list(csv.DictReader(open(args.csv)))
    for r in rows:
        r["_dens"] = dens_class(f(r.get("density_pct")) or 0)
    DENS_ORDER = ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]
    dens_groups = [c for c in DENS_ORDER if any(r["_dens"] == c for r in rows)]
    N = auto_total(rows)
    print(f"Figures from {args.csv} ({len(rows)} rows, {N} with Auto, slide={args.slide}):")

    # ============================================================
    # 1) HEADLINE: overall geomean speedup per baseline (slide-ready)
    # ============================================================
    fig, ax = plt.subplots(figsize=(5.6, 3.9) if not args.slide else (7.2, 4.6))
    style_ax(ax)
    labels = [lab for lab, _, _ in BASELINES]
    cols   = [col for _, col, _ in BASELINES]
    colors = [c for _, _, c in BASELINES]
    gm = [speedup(rows, c)[0] for c in cols]
    wn = [win_count(rows, c) for c in cols]
    x = np.arange(len(labels))
    bars = ax.bar(x, gm, width=0.62, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(1.0, color=OURS, lw=1.6, ls="--", zorder=4)
    ax.text(len(labels) - 0.5, 1.0, "  break-even (Auto)", color=OURS, fontsize=9.5,
            va="bottom", ha="right", fontweight="bold")
    ymax = max(g for g in gm if g == g) * 1.35
    for xi, gi, wi in zip(x, gm, wn):
        if gi == gi:
            ax.text(xi, gi + ymax * 0.02, f"{gi:.2f}×", ha="center", va="bottom",
                    color=INK, fontsize=11, fontweight="bold")
            ax.text(xi, ymax * 0.06, f"{wi}/{N}", ha="center", va="bottom",
                    color="white", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.8, ymax)
    clean_log_y(ax, 0.8, ymax)
    ax.set_ylabel("Geomean speedup of Auto (log)")
    ax.set_title("Auto (ours) vs. state-of-the-art SpGEMM baselines", fontweight="bold")
    fig.tight_layout()
    out1 = f"paper_speedup_overall{'_slide' if args.slide else ''}.png"
    fig.savefig(os.path.join(FIG, out1), bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  fig/{out1}")

    # ============================================================
    # 2) GROUPED BAR by density class
    # ============================================================
    fig, ax = plt.subplots(figsize=(8.6, 4.5) if not args.slide else (9.8, 5.4))
    style_ax(ax)
    gx = np.arange(len(dens_groups))
    nb = len(BASELINES); w = 0.8 / nb
    for i, (lab, col, color) in enumerate(BASELINES):
        ys = []
        for g in dens_groups:
            sub = [r for r in rows if r["_dens"] == g]
            gm_g, _ = speedup(sub, col)
            ys.append(gm_g if gm_g == gm_g else np.nan)
        ax.bar(gx + (i - (nb - 1) / 2) * w, ys, width=w, color=color, label=lab,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color=OURS, lw=1.5, ls="--", zorder=4)
    ax.text(len(dens_groups) - 0.5, 1.0, "  break-even", color=OURS, fontsize=9,
            va="bottom", ha="right", fontweight="bold")
    ymax = 1.0
    for lab, col, _ in BASELINES:
        for g in dens_groups:
            sub = [r for r in rows if r["_dens"] == g]
            gm_g, _ = speedup(sub, col)
            if gm_g == gm_g: ymax = max(ymax, gm_g)
    ax.set_ylim(0.8, ymax * 1.25)
    clean_log_y(ax, 0.8, ymax * 1.25)
    ax.set_xticks(gx)
    ax.set_xticklabels([f"{g}\n({sum(1 for r in rows if r['_dens']==g)})" for g in dens_groups])
    ax.set_ylabel("Geomean speedup of Auto (log)")
    ax.set_title("Auto speedup over baselines, by input density", fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.4, handlelength=1.4)
    fig.tight_layout()
    out2 = f"paper_speedup_by_density{'_slide' if args.slide else ''}.png"
    fig.savefig(os.path.join(FIG, out2), bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  fig/{out2}")

    # ============================================================
    # 3) SCATTER small multiples (one panel per baseline)
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8) if not args.slide else (10, 7.6),
                             sharex=True)
    axes = axes.ravel()
    for ax, (lab, col, color) in zip(axes, BASELINES):
        style_ax(ax)
        xs, ys = [], []
        for r in rows:
            a, b = f(r.get("Auto")), f(r.get(col))
            n = int(r["n"]) if str(r.get("n", "")).isdigit() else 0
            if a and b and a > 0 and b > 0 and n > 0:
                xs.append(n); ys.append(b / a)
        ax.scatter(xs, ys, s=20 if not args.slide else 28, color=color, alpha=0.68,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.axhline(1.0, color=OURS, lw=1.4, ls="--", zorder=4)
        gm, _ = speedup(rows, col)
        wn = win_count(rows, col)
        tot = auto_total(rows)
        ax.set_title(f"vs {lab}  —  geomean {gm:.2f}$\\times$, Auto faster {wn}/{tot}",
                     fontsize=10.5 if not args.slide else 12.5)
        ax.set_xscale("log")
        lo = min(0.7, min(ys)) if ys else 0.5
        hi = (max(ys) * 1.15) if ys else 2.0
        clean_log_y(ax, lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_locator(LogLocator(numticks=5))
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in axes[2:]:
        ax.set_xlabel("matrix dimension $n$ (log)")
    for ax in [axes[0], axes[2]]:
        ax.set_ylabel("speedup of Auto (log)")
    fig.suptitle("Per-matrix speedup of Auto over each baseline", fontweight="bold",
                 y=0.995, fontsize=12.5 if not args.slide else 14.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out3 = f"paper_speedup_scatter{'_slide' if args.slide else ''}.png"
    fig.savefig(os.path.join(FIG, out3), bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  fig/{out3}")

    # ============================================================
    # 4) LaTeX booktabs table
    # ============================================================
    def tex(v): return f"{v:.2f}$\\times$" if v == v else "---"
    L = []
    cap = (r"% Geomean speedup of Auto (ours) over each baseline "
           r"(baseline time $/$ Auto time; $>1\times$ = Auto faster). "
           r"`All' = all four baselines beaten on that matrix.")
    L.append(cap)
    L.append(r"\begin{tabular}{l|rrrr|c}")
    L.append(r"\toprule")
    L.append(r"Category (\#mat) & cuSPARSE & dense & opSparse & HSMU & Auto best on \\")
    L.append(r"\midrule")

    def add_row(label, sub):
        cells = []
        for _, col, _ in BASELINES:
            cells.append(tex(speedup(sub, col)[0]))
        winall = tot = 0
        for r in sub:
            a = f(r.get("Auto"))
            if not (a and a > 0):
                continue
            tot += 1
            if all((f(r.get(c)) is None or (f(r.get(c)) or 0) >= a) for _, c, _ in BASELINES):
                winall += 1
        cells.append(f"{winall}/{tot}" if tot else "---")
        L.append(f"{label} & " + " & ".join(cells) + r" \\")

    for g in dens_groups:
        sub = [r for r in rows if r["_dens"] == g]
        add_row(f"{g} ({len(sub)})", sub)
    L.append(r"\midrule")
    add_row(f"All ({N})", rows)
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    out_tex = "compare/paper_speedup_table.tex"
    with open(out_tex, "w") as fp:
        fp.write("\n".join(L) + "\n")
    print(f"  {out_tex}")

    # ---- console summary ----
    print("\n=== Overall geomean speedup of Auto ===")
    for lab, col, _ in BASELINES:
        gm, _ = speedup(rows, col)
        wn = win_count(rows, col)
        print(f"  vs {lab:9}: {gm:.2f}x   (Auto faster on {wn}/{N})")

if __name__ == "__main__":
    main()
