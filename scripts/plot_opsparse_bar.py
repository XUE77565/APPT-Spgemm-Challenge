#!/usr/bin/env python3
"""opSparse bcsstk30 — single slender stacked bar (like esc_runtime_share).

Drops overhead; shows only the two compute stages, symbolic (exact-count) and
accumulation (numeric), renormalized to their sum.  bcsstk30 is the example where
accumulation is longest (1.58 ms) and overhead is smallest.

Output: fig/opsparse_bar.{png,pdf}
"""
import os
import matplotlib.pyplot as plt

TEAL, RED, GRAY, INK = "#1485A4", "#C00000", "#C9D1D9", "#1F2933"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

# opSparse bcsstk32 (host timing, double, H100) — highest symbolic share (~26%) in first100
SYM = 0.607          # symbolic + symbolic_binning  (exact-count sizing)
ACC = 0.959          # numeric + numeric_binning    (hash accumulate)
OH  = 0.743          # setup + prefix + allocate + cleanup
TOT = SYM + ACC + OH
ps, pa, po = 100 * SYM / TOT, 100 * ACC / TOT, 100 * OH / TOT

fig, ax = plt.subplots(figsize=(11.0, 1.1))
fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.20)
Y, H = 0.5, 0.80
# order: accumulation (red) | symbolic (teal) | overhead (gray, unlabeled)
ax.barh([Y], [pa], left=0.0,         color=RED,  height=H, edgecolor="white", linewidth=1.0, zorder=3)
ax.barh([Y], [ps], left=pa,          color=TEAL, height=H, edgecolor="white", linewidth=1.0, zorder=3)
ax.barh([Y], [po], left=pa + ps,     color=GRAY, height=H, edgecolor="white", linewidth=1.0, zorder=3)
ax.text(pa / 2, Y, f"accumulation  {pa:.0f}%  ({ACC:.2f} ms)", ha="center", va="center",
        color="white", fontsize=13, fontweight="bold", zorder=4)
ax.text(pa + ps / 2, Y, f"symbolic  {ps:.0f}%  ({SYM:.2f} ms)", ha="center", va="center",
        color="white", fontsize=13, fontweight="bold", zorder=4)
ax.set_xlim(0, 100); ax.set_ylim(0, 1)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

os.makedirs("fig", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"fig/opsparse_bar.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote fig/opsparse_bar.{{png,pdf}}  (accumulation {pa:.1f}% / symbolic {ps:.1f}% / overhead {po:.1f}%)")

