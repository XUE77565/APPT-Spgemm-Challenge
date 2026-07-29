#!/usr/bin/env python3
# fig/launch_fusion_design.png — intuitive design diagram for kernel fusion.
# key visual: each launch pays a FIXED overhead (red); small-matrix compute ≈ 0,
# so runtime = stack of red overheads. Fusion merges kernels → fewer red blocks → less time.
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

TEAL="#1485A4"; RED="#C00000"; GRAY="#9AA6B2"; MUTED="#5C6773"; INK="#1F2933"
BG="#FFFFFF"; WORK="#c9d3dd"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})

fig=plt.figure(figsize=(11.4,5.4),dpi=200,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)

fig.text(0.012,0.955,"Kernel Fusion  —  fewer launches, less fixed overhead",
         fontsize=16,fontweight="bold",color=INK,ha="left",va="top")
fig.text(0.012,0.918,"each kernel launch pays a FIXED overhead; for small matrices the compute ≈ 0, so runtime = the stack of red overheads.  Fusion merges kernels → fewer red blocks.",
         fontsize=9.4,color=MUTED,ha="left",va="top")

# legend
ax.add_patch(Rectangle((0.014,0.855),0.022,0.020,fc=RED,ec="none")); ax.text(0.040,0.865,"launch overhead (fixed per launch)",fontsize=8.6,color=INK,va="center")
ax.add_patch(Rectangle((0.300,0.855),0.022,0.020,fc=WORK,ec="none")); ax.text(0.326,0.865,"actual compute work",fontsize=8.6,color=INK,va="center")

def timeline(x0, y, launches, total_w, lab, col, sub):
    """draw a row of launches; each = red overhead + thin gray work."""
    n=len(launches); unit=total_w/n
    oh=unit*0.78; wk=unit*0.16; gp=unit*0.06
    h=0.055
    for i,name in enumerate(launches):
        x=x0+i*unit
        ax.add_patch(Rectangle((x,y),oh,h,fc=RED,ec="white",lw=0.8))           # overhead
        ax.add_patch(Rectangle((x+oh,y),wk,h,fc=WORK,ec="white",lw=0.8))       # work
        if name:
            ax.text(x+oh/2,y-0.022,name,ha="center",fontsize=6.6,color=MUTED,rotation=0)
    # bracket + total label
    ax.annotate("",xy=(x0,y-0.042),xytext=(x0+total_w,y-0.042),
                arrowprops=dict(arrowstyle="-",color=col,lw=0))
    ax.text(x0+total_w+0.012,y+h/2,lab,fontsize=9.4,color=col,fontweight="bold",va="center")
    ax.text(x0+total_w+0.012,y+h/2-0.030,sub,fontsize=7.8,color=MUTED,va="center")

# ---- BEFORE ----
fig.text(0.012,0.80,"BEFORE  —  binning/sizing phase as many separate kernels",fontsize=10.5,color=RED,fontweight="bold")
before=["compute_bucket","","bucket_count","","thrust\nest_scan","","thrust\ncnnz_scan",""]
timeline(0.05,0.70,before,0.74,"≈ 8 launches",RED,"×8 red = big overhead")
ax.text(0.05,0.668,"(thrust scan ≈ 3 kernels each → counted as multiple launches)",fontsize=7.4,color=MUTED)

# ---- fusion arrow ----
ax.add_patch(FancyArrowPatch((0.42,0.625),(0.42,0.535),arrowstyle="-|>",mutation_scale=16,color=TEAL,lw=2.0))
ax.text(0.45,0.58,"fuse",fontsize=9.2,color=TEAL,fontweight="bold",va="center")

# ---- AFTER ----
fig.text(0.012,0.495,"AFTER (ours)  —  fused kernels + single-block scan",fontsize=10.5,color=TEAL,fontweight="bold")
after=["fused\nbucket+count","single-block\nest_scan","single-block\ncnnz_scan"]
timeline(0.05,0.405,after,0.30,"3 launches",TEAL,"×3 red = small overhead")
ax.text(0.05,0.373,"large matrices keep thrust scan (A_rows > 1024 gate) — no regression",fontsize=7.4,color=MUTED)

# ---- bottom: result + what-fused ----
ax.add_patch(FancyBboxPatch((0.012,0.04),0.976,0.20,boxstyle="round,pad=0.004,rounding_size=0.012",fc="#F2FAFB",ec=TEAL,lw=1.2,zorder=2))
ax.text(0.028,0.215,"What got fused",fontsize=9.2,color=MUTED,fontweight="bold")
ax.text(0.028,0.185,"•  compute_bucket + bucket_count  →  1 fused kernel (writes bucket id AND atomic-adds histogram in one pass)",fontsize=8.4,color=INK)
ax.text(0.028,0.158,"•  2 separate D2H  →  1 merged async D2H + 1 sync",fontsize=8.4,color=INK)
ax.text(0.028,0.131,"•  thrust est/cnnz scan (multi-kernel)  →  single-block Hillis–Steele scan (1 launch, for A_rows ≤ 1024)",fontsize=8.4,color=INK)
ax.text(0.5,0.072,"Result — bp* gap to cuSPARSE   0.03–0.08 ms  →  0.008 ms     ·     bcsstk30 (large) 3.24 → 3.11 ms (no regression)",
        ha="center",fontsize=9.8,color=TEAL,fontweight="bold")

fig.savefig("fig/launch_fusion_design.png",dpi=200,bbox_inches="tight",facecolor=BG)
print("wrote fig/launch_fusion_design.png")
