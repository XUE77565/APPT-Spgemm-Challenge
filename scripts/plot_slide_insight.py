#!/usr/bin/env python3
"""
Compose the Insight slide as a single 16:9 image (v2).

总 (phenomenon, left): hash-vs-merge per-matrix scatter — points on BOTH sides of
the equal-runtime line → neither consistently wins.
分 (drivers, right): three T_merge/T_hash panels stacked, each with an EXPLICIT
relationship caption placed beside it ("ratio ↑ ⇒ hash"), plus a top-right color
legend (Hash wins / Merge wins / Binned mean).

Both halves share the teal=Hash-wins / red=Merge-wins encoding. Reuses gen + draw
helpers from plot_insight_scatter.py and plot_ratio_vs_features.py (seed 42).

Output: fig/slide_insight.{png,pdf}
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_insight_scatter as IS
import plot_ratio_vs_features as RF

FIG_DIR = RF.FIG_DIR
SURFACE = IS.SURFACE; INK_PRI = IS.INK_PRI; INK_SEC = IS.INK_SEC
C_HASH = IS.C_HASH; C_MERGE = IS.C_MERGE

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK_PRI,
    "axes.labelcolor": INK_SEC, "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "axes.edgecolor": INK_SEC,
})

# explicit per-panel caption pieces (mathtext-safe arrows)
REL = "ratio $\\uparrow$ $\\Rightarrow$ hash"          # the generic explicit finding
VAR = {"rows": "with rows $n$",
       "sigma": "with $\\sigma$",
       "products": "with $\\mathrm{nnz}^2/n$"}

fig = plt.figure(figsize=(13.333, 7.5))

# ---- title row ----
fig.text(0.045, 0.945, "Insight", fontsize=22, fontweight="bold", ha="left")
fig.text(0.475, 0.948, "What predicts the winner?", fontsize=14,
         fontweight="bold", ha="left")
fig.text(0.045, 0.905,
         "Neither merge nor hash consistently wins — but the winner is predictable",
         fontsize=12.5, color=INK_SEC, ha="left")

# ---- color legend (top-right): explains both halves + the trend line ----
handles = [
    Line2D([0], [0], marker="o", linestyle="", markerfacecolor=C_HASH,
           markeredgecolor="white", markersize=10, label="Hash wins"),
    Line2D([0], [0], marker="o", linestyle="", markerfacecolor=C_MERGE,
           markeredgecolor="white", markersize=10, label="Merge wins"),
    Line2D([0], [0], color=INK_PRI, marker="o", markersize=4.5,
           markerfacecolor=INK_PRI, markeredgecolor="white", linewidth=2,
           label="Binned mean"),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.965, 0.965),
           frameon=False, fontsize=10.5, handletextpad=0.5, labelspacing=0.55,
           borderaxespad=0)

# ---- 总: big hash-vs-merge scatter (left) ----
ax_big = fig.add_axes([0.045, 0.085, 0.40, 0.76])
d_ins = IS.gen_insight_data()
IS.draw_insight_scatter(ax_big, d_ins["hx"], d_ins["hy"], d_ins["hw"],
                        region_labels=True, pts=44, fs_label=11, fs_region=11.5)
fig.text(0.045, 0.050,
         "Phenomenon — points on both sides of y = x: neither wins",
         fontsize=10.5, color=INK_SEC, ha="left", style="italic")

# ---- 分: three ratio panels with explicit captions beside each (right) ----
d_rat = RF.gen_ratio_data()
specs = RF.panel_specs(d_rat)
stack = [
    (specs[0], [0.60, 0.608, 0.365, 0.190], False),   # rows
    (specs[1], [0.60, 0.378, 0.365, 0.190], True),    # σ (carries the y-axis label)
    (specs[2], [0.60, 0.148, 0.365, 0.190], False),   # nnz²/n
]
for p, box, ylabel in stack:
    mid = box[1] + box[3] / 2.0
    ax = fig.add_axes(box)
    RF.draw_ratio_panel(ax, p["x"], d_rat["ratio"], xlabel=p["label"], xlim=p["xlim"],
                        maj=p["maj"], fmt=p["fmt"], region_labels=False,
                        ylabel=ylabel, yticks=True, pts=14, fs_label=9, fs_ylabel=9.5)
    # explicit relationship caption beside the panel (left side)
    fig.text(0.586, mid + 0.018, REL, ha="right", va="center",
             fontsize=10.5, fontweight="bold", color=C_HASH)
    fig.text(0.586, mid - 0.024, VAR[p["slug"]], ha="right", va="center",
             fontsize=9.5, color=INK_SEC)

fig.text(0.60, 0.050,
         "Drivers — each feature pushes the ratio up  ⇒  adaptive dispatch",
         fontsize=10.5, color=INK_SEC, ha="left", style="italic")

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"slide_insight.{ext}"), dpi=200, facecolor=SURFACE)
print("saved fig/slide_insight.{png,pdf}")
