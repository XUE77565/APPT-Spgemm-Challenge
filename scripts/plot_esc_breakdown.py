#!/usr/bin/env python3
"""
ESC Gustavson (C = A·A) phase-breakdown figure.

Proves that ACCUMULATION (expand + sort + reduce_by_key over the dup-bloated
intermediate set) is the dominant cost of the traditional Expand-Sort-Combine
Gustavson, on H100 PCIe, double precision.

  Panel A — detailed phase breakdown of one large matrix (bcsstk30):
            9 phases, log-scale horizontal bars, coloured by pipeline role.
  Panel B  — composition (share of total) across 5 structural matrices:
            accumulation (blue) dominates everywhere (65-81%).

Data: cudaEvent phase profiler (tag "esc-prof") in spgemm_kernel_manual.cu,
compute-only (cudaMalloc excluded), METHOD=manual, DBG=1, stable run.

Output: fig/esc_accumulation_breakdown.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------- paths / style
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")
os.makedirs(FIG_DIR, exist_ok=True)

SURFACE  = "#fcfcfb"
INK_PRI  = "#0b0b0b"
INK_SEC  = "#52514e"
GRID     = "#e4e3df"

# Validated 4-hue categorical palette (dataviz validate_palette.js, light, all PASS)
ROLE_COLOR = {
    "accumulation": "#2a78d6",   # blue  — the story / hero
    "transfer":     "#eb6834",   # orange
    "output":       "#1baf7a",   # aqua
    "symbolic":     "#eda100",   # yellow
}
ROLE_ORDER = ["accumulation", "transfer", "output", "symbolic"]  # CVD-safe adjacent order

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor":   SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family":      "DejaVu Sans",
    "text.color":       INK_PRI,
    "axes.labelcolor":  INK_SEC,
    "xtick.color":      INK_SEC,
    "ytick.color":      INK_PRI,
    "axes.edgecolor":   INK_SEC,
    "axes.linewidth":   0.8,
})

# ----------------------------------------------------------------- data
# pipeline phase order (host function spgemm_self_product_manual)
PHASES = ["h2d", "count", "scan", "expand", "sort", "reduce", "final", "pack", "d2h"]
PHASE_ROLE = {
    "h2d": "transfer", "d2h": "transfer",
    "count": "symbolic", "scan": "symbolic",
    "expand": "accumulation", "sort": "accumulation", "reduce": "accumulation",
    "final": "output", "pack": "output",
}
PHASE_LABEL = {
    "h2d": "H2D (A)", "count": "count", "scan": "scan",
    "expand": "expand", "sort": "sort", "reduce": "reduce_by_key",
    "final": "finalize", "pack": "pack", "d2h": "D2H (C)",
}

# cudaEvent ms (stable run). order = PHASES
DATA = {
    "bcsstk30": [2.888, 0.041, 0.035, 2.625, 29.830, 2.265, 0.994, 0.163, 3.899],
    "bcsstk32": [2.969, 0.045, 0.035, 1.579, 16.927, 1.353, 0.482, 0.125, 3.006],
    "bcsstk29": [1.416, 0.026, 0.036, 0.397, 5.132, 0.445, 0.159, 0.043, 1.044],
    "bcsstk31": [1.874, 0.034, 0.035, 0.598, 7.940, 0.679, 0.315, 0.090, 2.489],
    "bcsstk17": [0.713, 0.026, 0.035, 0.248, 3.307, 0.296, 0.115, 0.031, 0.635],
}
DUP = {"bcsstk30": 19.4, "bcsstk32": 14.9, "bcsstk29": 15.7, "bcsstk31": 10.1, "bcsstk17": 13.8}
# matrix order for Panel B (by total time desc)
MAT_ORDER = ["bcsstk30", "bcsstk32", "bcsstk31", "bcsstk29", "bcsstk17"]

PRIMARY = "bcsstk30"
N_INTER = 173_481_412   # bcsstk30 expanded intermediates
N_OUT   = 8_946_070     # bcsstk30 C_nnz

def role_ms(name):
    d = dict(zip(PHASES, DATA[name]))
    out = {r: 0.0 for r in ROLE_ORDER}
    for p, v in d.items():
        out[PHASE_ROLE[p]] += v
    return out, sum(d.values())

# ============================================================== figure
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(12.4, 5.7), gridspec_kw={"width_ratios": [1.32, 1.0]})
fig.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.115, wspace=0.34)

# ----------------------------------------------------- Panel A: detail
dA = dict(zip(PHASES, DATA[PRIMARY]))
totalA = sum(dA.values())
order = sorted(PHASES, key=lambda p: dA[p])           # ascending (barh draws bottom-up)
ys = np.arange(len(order))
vals = np.array([dA[p] for p in order])

bars = axA.barh(ys, vals, color=[ROLE_COLOR[PHASE_ROLE[p]] for p in order],
                height=0.72, zorder=3, edgecolor=SURFACE, linewidth=0.6)
axA.set_yticks(ys)
axA.set_yticklabels([PHASE_LABEL[p] for p in order], fontsize=9.5)
axA.set_xscale("log")
axA.set_xlim(0.022, 60)
axA.set_xlabel("phase time (ms, log scale)", fontsize=9.5)
axA.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=6))
axA.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
axA.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

accum_set = {"expand", "sort", "reduce"}
for y, p, v in zip(ys, order, vals):
    pct = 100.0 * v / totalA
    # value + percent just past the bar end
    axA.text(v * 1.12, y, f"{v:.2f} ms  ({pct:.1f}%)",
             va="center", ha="left", fontsize=8.6, color=INK_PRI,
             fontweight="bold" if p in accum_set else "normal")

accum_ms = sum(dA[p] for p in accum_set)
axA.text(0.0, 1.155,
         "(A)  ESC Gustavson phase breakdown — bcsstk30",
         transform=axA.transAxes, fontsize=11.5, fontweight="bold", color=INK_PRI)
axA.text(0.0, 1.085,
         f"Accumulation (expand + sort + reduce) = {accum_ms:.1f} ms = "
         f"{100*accum_ms/totalA:.0f}% of runtime",
         transform=axA.transAxes, fontsize=9, color=INK_SEC)
axA.text(0.0, 1.035,
         f"{N_INTER/1e6:.1f}M intermediate products → {N_OUT/1e6:.2f}M outputs "
         f"(redundancy {DUP[PRIMARY]:.1f}×)",
         transform=axA.transAxes, fontsize=9, color=INK_SEC)

# ----------------------------------------------------- Panel B: across matrices
# normalised 100% stacked horizontal bars, role segments
role_mat = {r: [] for r in ROLE_ORDER}
for m in MAT_ORDER:
    rms, tot = role_ms(m)
    for r in ROLE_ORDER:
        role_mat[r].append(100.0 * rms[r] / tot)

yb = np.arange(len(MAT_ORDER))
left = np.zeros(len(MAT_ORDER))
for r in ROLE_ORDER:
    share = np.array(role_mat[r])
    axB.barh(yb, share, left=left, color=ROLE_COLOR[r], height=0.66,
             edgecolor=SURFACE, linewidth=0.8, zorder=3, label=r)
    # label accumulation % inside its segment
    if r == "accumulation":
        for yi, s, l in zip(yb, share, left):
            axB.text(l + s / 2, yi, f"{s:.0f}%", va="center", ha="center",
                     fontsize=9, color="white", fontweight="bold")
    left += share

axB.set_yticks(yb)
axB.set_yticklabels([f"{m}\n(dup {DUP[m]:.1f}×)" for m in MAT_ORDER], fontsize=9.5)
axB.set_xlim(0, 100)
axB.set_xlabel("share of runtime (%)", fontsize=9.5)
axB.set_xticks([0, 25, 50, 75, 100])
axB.tick_params(axis="x", labelsize=8.5)
for s in ("top", "right", "left"):
    axB.spines[s].set_visible(False)
axB.tick_params(axis="y", length=0)
axB.text(0.0, 1.155, "(B)  Accumulation dominates across matrices",
         transform=axB.transAxes, fontsize=11.5, fontweight="bold", color=INK_PRI)
axB.text(0.0, 1.075, "share of ESC runtime per matrix (dup factor = intermediates / outputs)",
         transform=axB.transAxes, fontsize=9, color=INK_SEC)

# ----------------------------------------------------- shared legend (roles)
handles = [Patch(facecolor=ROLE_COLOR[r], edgecolor="none",
                 label={"accumulation": "Accumulation (expand + sort + reduce)",
                        "transfer": "Transfer (H2D + D2H)",
                        "output": "Output (finalize + pack)",
                        "symbolic": "Symbolic (count + scan)"}[r])
           for r in ROLE_ORDER]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=9.2, bbox_to_anchor=(0.515, 0.005), handletextpad=0.5,
           columnspacing=1.4)

fig.suptitle("Traditional ESC Gustavson SpGEMM is accumulation-bound  "
             "(H100 PCIe, C = A·A, double)",
             fontsize=13.5, fontweight="bold", color=INK_PRI, x=0.075, y=0.955, ha="left")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"esc_accumulation_breakdown.{ext}"),
                dpi=200, bbox_inches="tight", facecolor=SURFACE)
print("saved:", os.path.join(FIG_DIR, "esc_accumulation_breakdown.{png,pdf}"))
