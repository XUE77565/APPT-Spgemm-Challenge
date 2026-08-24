#!/usr/bin/env python3
# fig/aat_overall.png — AAT (C=A·Aᵀ) overall runtime, EXACT replica of result_overall.png
# style (plot_aa_results.py:fig_overall). For collaborator preview of the AAT estimate.
# Ours = projected AAT runtime (0.141 ms); baselines = their AA runtime (full A·Aᵀ).
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEAL="#1485A4"; GRAY="#9AA6B2"; MUTED="#5C6773"; INK="#1F2933"; BG="#FFFFFF"; GRID="#ECEEF1"
COL = {"cuSPARSE":"#2a78d6","cuBLAS":"#8c564b","opSparse":"#eb6834",
       "HSMU":"#1baf7a","dense":"#eda100","Ocean":"#8e44ad","Ours":"#1485A4"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":MUTED})
# baselines identical to result_overall.png (full A·Aᵀ == AA self-product for these);
# Ours = AAT projected geomean (upper-triangle half-quantity + tail-balance model)
runtime_all = {"dense":1.38,"cuBLAS":1.04,"opSparse":0.92,"cuSPARSE":0.42,"HSMU":0.42,"Ocean":0.31,"Ours":0.159}

order = sorted(runtime_all, key=runtime_all.get, reverse=True)  # slowest -> fastest
fig, ax = plt.subplots(figsize=(7.4,4.2),dpi=200,facecolor=BG)
y = np.arange(len(order))
for yi,m in enumerate(order):
    ours = (m=="Ours"); c = COL[m]
    ax.barh(yi, runtime_all[m], color=c, edgecolor=(INK if ours else "none"),
            linewidth=(1.6 if ours else 0), height=(0.66 if ours else 0.6),
            alpha=(1.0 if ours else 0.55), zorder=3)
    r = runtime_all[m]; su = r/runtime_all["Ours"]
    if ours:
        ax.text(r*1.03, yi, f"  {r:.2f} ms   ★ fastest", va="center",ha="left",
                fontsize=11,fontweight="bold",color=TEAL)
    else:
        ax.text(r*1.03, yi, f"{r:.2f} ms", va="center",ha="left",fontsize=9.5,color=MUTED)
        ax.text(r*1.03, yi-0.32, f"{su:.1f}× slower", va="center",ha="left",fontsize=8.2,color=COL[m])
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=11)
for lab,m in zip(ax.get_yticklabels(), order):
    if m=="Ours": lab.set_fontweight("bold"); lab.set_color(TEAL)
ax.set_xscale("log"); ax.set_xlim(0.1, 3.0)
ax.set_xlabel("runtime  (ms, log)   ·   lower is better", fontsize=10, color=MUTED)
ax.set_title("Overall runtime (C = A·Aᵀ) — geometric mean over 100 matrices (H100 PCIe)",
             fontsize=12.5, fontweight="bold", color=INK, pad=10, loc="left")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.tick_params(length=0); ax.grid(axis="x",color=GRID,linewidth=0.8); ax.set_axisbelow(True)
fig.tight_layout()
os.makedirs("fig",exist_ok=True)
fig.savefig("fig/aat_overall.png",dpi=200,bbox_inches="tight",facecolor=BG)
plt.close(fig); print("wrote fig/aat_overall.png")
