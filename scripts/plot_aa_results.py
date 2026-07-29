#!/usr/bin/env python3
# fig/result_*.png — AA self-product eval figures (deck style), highlight Ours (teal).
# data: aa_result.png (method x density table; Ours fastest everywhere).
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEAL="#1485A4"; RED="#C00000"; GRAY="#9AA6B2"; MUTED="#5C6773"; INK="#1F2933"; BG="#FFFFFF"; GRID="#ECEEF1"
COL = {"cuSPARSE":"#2a78d6","cuBLAS":"#8c564b","opSparse":"#eb6834",
       "HSMU":"#1baf7a","dense":"#eda100","Ocean":"#8e44ad","Ours":"#1485A4"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"text.color":INK,
    "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":MUTED})
os.makedirs("fig",exist_ok=True)

# ---- data ----
runtime_all = {"dense":1.35,"cuBLAS":0.93,"opSparse":0.86,"cuSPARSE":0.39,"HSMU":0.39,"Ocean":0.28,"Ours":0.18}
classes = ["Dense","Mildly\nsparse","Highly\nsparse","Extremely\nsparse"]
speedup = {  # baseline_runtime / ours_runtime, per density class
    "dense":    [1.54,2.12,7.68,212.0],
    "cuBLAS":   [2.02,2.34,4.66,69.14],
    "cuSPARSE": [2.54,2.25,2.02,2.34],
    "opSparse": [5.77,5.98,4.11,4.47],
    "HSMU":     [3.16,2.23,1.96,2.53],
    "Ocean":    [3.30,1.90,1.40,1.14],
}
wins = {"dense":100,"cuBLAS":100,"cuSPARSE":91,"opSparse":100,"HSMU":100,"Ocean":84}

# ============================================================ Fig 1: overall runtime
def fig_overall():
    order = sorted(runtime_all, key=runtime_all.get, reverse=True)  # slowest -> fastest
    fig, ax = plt.subplots(figsize=(7.4,4.2),dpi=200,facecolor=BG)
    y = np.arange(len(order))
    for yi,m in enumerate(order):
        ours = (m=="Ours")
        c = COL[m]
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
    ax.set_title("Overall runtime — geometric mean over 100 matrices (H100 PCIe)",
                 fontsize=12.5, fontweight="bold", color=INK, pad=10, loc="left")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0); ax.grid(axis="x",color=GRID,linewidth=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("fig/result_overall.png",dpi=200,bbox_inches="tight",facecolor=BG)
    plt.close(fig); print("wrote fig/result_overall.png")

# ============================================================ Fig 2: speedup by density
def fig_by_density():
    base = ["dense","cuBLAS","opSparse","HSMU","cuSPARSE","Ocean"]  # draw dense-first so big bars sit behind
    fig, ax = plt.subplots(figsize=(8.2,4.6),dpi=200,facecolor=BG)
    x = np.arange(len(classes)); w = 0.13
    for i,m in enumerate(base):
        offs = (i - (len(base)-1)/2)*w
        ax.bar(x+offs, speedup[m], width=w, color=COL[m], edgecolor=BG, linewidth=0.5,
               label=m, zorder=3)
    # Ours = 1x break-even (the highlight: every baseline bar is above it => Ours wins)
    ax.axhline(1.0, color=TEAL, linewidth=2.2, linestyle="--", zorder=5)
    ax.text(0.02, 1.0, "Ours (1×) ▲", color=TEAL, va="bottom", ha="left",
            fontsize=10, fontweight="bold")
    # outlier labels
    for i,m in enumerate(base):
        offs = (i - (len(base)-1)/2)*w
        for j,v in enumerate(speedup[m]):
            if v>=30:  # the huge dense/cuBLAS extreme outliers
                ax.text(x[j]+offs, v*1.08, f"{v:.0f}×" if v>=10 else f"{v:.1f}×",
                        ha="center",va="bottom",fontsize=7.6,color=COL[m],rotation=90)
    ax.set_yscale("log"); ax.set_ylim(0.9, 400)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("speedup of baseline over Ours  (×, log)   ·   >1 = Ours faster", fontsize=9.5, color=MUTED)
    ax.set_title("Speedup over Ours by matrix sparsity — Ours wins every class",
                 fontsize=12.5, fontweight="bold", color=INK, pad=10, loc="left")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0); ax.grid(axis="y",color=GRID,linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.8, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout()
    fig.savefig("fig/result_by_density.png",dpi=200,bbox_inches="tight",facecolor=BG)
    plt.close(fig); print("wrote fig/result_by_density.png")

# ============================================================ Fig 3: win count
def fig_wins():
    order = sorted(wins, key=wins.get)  # fewest wins on top
    fig, ax = plt.subplots(figsize=(7.0,3.8),dpi=200,facecolor=BG)
    y = np.arange(len(order))
    for yi,m in enumerate(order):
        full = wins[m]==100
        ax.barh(yi, wins[m], color=(TEAL if full else GRAY), edgecolor=BG, height=0.6, zorder=3)
        ax.text(wins[m]+1.2, yi, f"{wins[m]}/100", va="center",ha="left",
                fontsize=10, fontweight="bold", color=(TEAL if full else MUTED))
    ax.axvline(100, color=RED, linewidth=1.2, linestyle=":", zorder=4)
    ax.text(100, len(order)-0.4, " clean sweep", color=RED, fontsize=8.6, ha="left", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels([f"vs {m}" for m in order], fontsize=10.5)
    ax.set_xlim(0,112); ax.set_xlabel("matrices where Ours is faster  (of 100)", fontsize=10, color=MUTED)
    ax.set_title("Head-to-head wins — Ours faster on 84–100 of 100 matrices",
                 fontsize=12, fontweight="bold", color=INK, pad=10, loc="left")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0); ax.grid(axis="x",color=GRID,linewidth=0.8); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("fig/result_wins.png",dpi=200,bbox_inches="tight",facecolor=BG)
    plt.close(fig); print("wrote fig/result_wins.png")

fig_overall(); fig_by_density(); fig_wins()
print("done")
