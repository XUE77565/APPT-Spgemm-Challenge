#!/usr/bin/env python3
"""
ESC Gustavson runtime share — single stacked bar.

One 100% bar showing the mean runtime share (across 5 structural matrices)
of each pipeline role. expand + sort + reduce are merged into ACCUMULATION.

  symbolic   = count + scan
  accumulation = expand + sort + reduce_by_key     <- the bottleneck
  output     = finalize + pack
  transfer   = H2D + D2H

Share = mean over matrices of (role_ms / total_ms) * 100.

Output: fig/esc_runtime_share.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")
os.makedirs(FIG_DIR, exist_ok=True)

SURFACE = "#fcfcfb"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"

# palette matched to the deck (pre(1).pptx): brand teal-blue accent (#1485A4)
# + blue family (#0D77C3/#0A4996); red emphasis (#C00000) marks the bottleneck
ROLE_COLOR = {"accumulation": "#C00000", "transfer": "#1485A4",
              "output": "#0D77C3", "symbolic": "#0A4996"}
# segment order = magnitude desc AND CVD-safe adjacent order
ROLE_ORDER = ["accumulation", "transfer", "output", "symbolic"]
ROLE_FULL = {
    "accumulation": "Accumulation (expand + sort + reduce)",
    "transfer":     "Transfer (H2D + D2H)",
    "output":       "Output (finalize + pack)",
    "symbolic":     "Symbolic (count + scan)",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK_PRI,
    "axes.labelcolor": INK_SEC, "xtick.color": INK_SEC, "ytick.color": INK_PRI,
})

# cudaEvent ms, pipeline order: h2d,count,scan,expand,sort,reduce,final,pack,d2h
PH = ["h2d", "count", "scan", "expand", "sort", "reduce", "final", "pack", "d2h"]
DATA = {
    "bcsstk30": [2.888, 0.041, 0.035, 2.625, 29.830, 2.265, 0.994, 0.163, 3.899],
    "bcsstk32": [2.969, 0.045, 0.035, 1.579, 16.927, 1.353, 0.482, 0.125, 3.006],
    "bcsstk29": [1.416, 0.026, 0.036, 0.397, 5.132, 0.445, 0.159, 0.043, 1.044],
    "bcsstk31": [1.874, 0.034, 0.035, 0.598, 7.940, 0.679, 0.315, 0.090, 2.489],
    "bcsstk17": [0.713, 0.026, 0.035, 0.248, 3.307, 0.296, 0.115, 0.031, 0.635],
}
ROLE_PHASE = {"h2d": "transfer", "d2h": "transfer", "count": "symbolic", "scan": "symbolic",
              "expand": "accumulation", "sort": "accumulation", "reduce": "accumulation",
              "final": "output", "pack": "output"}

# per-matrix role share %
shares = {r: [] for r in ROLE_ORDER}
for m, v in DATA.items():
    d = dict(zip(PH, v))
    tot = sum(v)
    rms = {r: 0.0 for r in ROLE_ORDER}
    for p, x in d.items():
        rms[ROLE_PHASE[p]] += x
    for r in ROLE_ORDER:
        shares[r].append(100.0 * rms[r] / tot)

mean_share = {r: float(np.mean(shares[r])) for r in ROLE_ORDER}
minmax = {r: (min(shares[r]), max(shares[r])) for r in ROLE_ORDER}
print("mean share (%):", {r: round(mean_share[r], 1) for r in ROLE_ORDER})
print(2323)

# ============================================================== figure  (slender bar, direct labels, no legend/title)
ROLE_NAME = {"accumulation": "Accumulation", "transfer": "Transfer",
             "output": "Output", "symbolic": "Symbolic"}

fig, ax = plt.subplots(figsize=(11.0, 1.1))
fig.subplots_adjust(left=0.04, right=0.965, top=0.78, bottom=0.22)

left = 0.0
BAR_Y, BAR_H = 0.5, 0.80
for r in ROLE_ORDER:
    s = mean_share[r]
    ax.barh([BAR_Y], [s], left=left, color=ROLE_COLOR[r], height=BAR_H,
            edgecolor=SURFACE, linewidth=1.0, zorder=3)
    if r in ("accumulation", "transfer"):   # direct-label the two large segments
        ax.text(left + s / 2, BAR_Y, f"{ROLE_NAME[r]}  {s:.0f}%",
                ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", zorder=4)
    left += s

ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xticks([])
for sp in ("top", "right", "left", "bottom"):
    ax.spines[sp].set_visible(False)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"esc_runtime_share.{ext}"),
                dpi=200, facecolor=SURFACE)
print("saved fig/esc_runtime_share.{png,pdf}")
