#!/usr/bin/env python3
# fig/h2d_d2h_pool.png — detailed pinned-memory-pool mechanism for h2d/d2h (deck style).
# before (lock every transfer) vs after (lock once, reuse) — why it is faster.
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

TEAL="#1485A4"; RED="#C00000"; GRAY="#9AA6B2"; MUTED="#5C6773"; INK="#1F2933"
BG="#FFFFFF"; TINT_T="#F2FAFB"; TINT_R="#FCF4F4"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})

fig=plt.figure(figsize=(11.6,4.8),dpi=200,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)

def box(x,y,w,h,fc,ec,lw=1.3,rad=0.012,fs=8.6,txt=None,c=INK,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.004,rounding_size={rad}",
                 fc=fc,ec=ec,lw=lw,zorder=3))
    if txt: ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,
                    color=c,fontweight=("bold" if bold else "normal"),zorder=4)
def arr(x0,y0,x1,y1,c=MUTED,lw=1.3,ls="-"):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=11,
                 color=c,lw=lw,linestyle=ls,zorder=4,shrinkA=1,shrinkB=1))

# ---- two panel backgrounds ----
ax.add_patch(FancyBboxPatch((0.008,0.10),0.485,0.80,boxstyle="round,pad=0.004,rounding_size=0.014",
             fc=TINT_R,ec=RED,lw=1.4,zorder=1))
ax.add_patch(FancyBboxPatch((0.507,0.10),0.485,0.80,boxstyle="round,pad=0.004,rounding_size=0.014",
             fc=TINT_T,ec=TEAL,lw=1.4,zorder=1))
ax.text(0.25,0.875,"① Without pool  —  page-lock EVERY transfer",ha="center",fontsize=11,color=RED,fontweight="bold")
ax.text(0.749,0.875,"② Pinned pool (ours)  —  lock ONCE, reuse",ha="center",fontsize=11,color=TEAL,fontweight="bold")

# ============ Panel A: naive ============
yA=0.50; hA=0.115
box(0.020,yA,0.092,hA,TINT_R,RED,1.3,fs=7.8,txt="cudaMallocHost\n(pin pages)",c=RED,bold=True)
box(0.135,yA,0.080,hA,"#eef0f2",GRAY,1.0,fs=8.0,txt="host buf\n(pageable)")
arr(0.112,yA+hA/2,0.135,yA+hA/2)
# PCIe hop to GPU
arr(0.215,yA+hA/2,0.300,yA+hA/2)
ax.text(0.257,yA+hA/2+0.022,"PCIe",ha="center",fontsize=7.2,color=MUTED)
box(0.300,yA,0.062,hA,"#ffffff",INK,1.0,fs=8.4,txt="GPU",bold=True)
arr(0.362,yA+hA/2,0.388,yA+hA/2)
box(0.388,yA,0.095,hA,TINT_R,RED,1.3,fs=7.8,txt="cudaFreeHost\n(unpin)",c=RED,bold=True)
# loop-back arrow (× N)
arr(0.435,yA+hA,0.066,yA+hA,RED,1.1,"--")
ax.text(0.25,yA+hA+0.030,"↻  repeated × N transfers",ha="center",fontsize=8.4,color=RED,fontweight="bold")
# why-slow caption
ax.text(0.25,0.205,"✗  every h2d/d2h pays a page-lock syscall (pin + unpin)\n     + pageable mem is internally staged → can't async / full BW",
        ha="center",va="center",fontsize=8.6,color=INK)

# ============ Panel B: pinned pool ============
# pool rack (top): allocated & locked once
ax.add_patch(FancyBboxPatch((0.560,0.74),0.18,0.055,boxstyle="round,pad=0.004,rounding_size=0.012",
             fc=TINT_T,ec=TEAL,lw=1.3,zorder=3))
ax.text(0.65,0.808,"pinned pool  (init: allocate & page-lock ONCE)",ha="center",fontsize=7.8,color=TEAL,fontweight="bold")
for k in range(5):
    ax.add_patch(Rectangle((0.572+k*0.032,0.748),0.024,0.030,fc=TEAL,ec="white",lw=0.8,zorder=4))
yB=0.50; hB=0.115
box(0.518,yB,0.090,hB,TINT_T,TEAL,1.3,fs=7.8,txt="pool.get()\nO(1) bump",c=TEAL,bold=True)
box(0.628,yB,0.080,hB,"#d9eef3",TEAL,1.0,fs=8.0,txt="pinned buf\n(locked)")
arr(0.608,yB+hB/2,0.628,yB+hB/2)
arr(0.708,yB+hB/2,0.793,yB+hB/2)
ax.text(0.750,yB+hB/2+0.022,"PCIe DMA\nasync",ha="center",fontsize=7.0,color=TEAL)
box(0.793,yB,0.062,hB,"#ffffff",INK,1.0,fs=8.4,txt="GPU",bold=True)
arr(0.855,yB+hB/2,0.880,yB+hB/2)
box(0.880,yB,0.095,hB,"#eef0f2",GRAY,1.0,fs=7.8,txt="pool.return()\n(no-op)",c=MUTED)
# arrow from pool rack down to pool.get (supply)
arr(0.60,0.748,0.563,yB+hB,TEAL,1.0,"--")
# why-fast caption
ax.text(0.749,0.205,"✓  zero page-lock syscalls in the transfer loop\n     + pinned → DMA direct over PCIe → full BW, async",
        ha="center",va="center",fontsize=8.6,color=INK)

# ============ bottom result bar ============
ax.add_patch(FancyBboxPatch((0.008,0.02),0.984,0.055,boxstyle="round,pad=0.004,rounding_size=0.012",
             fc="#FFF7F0",ec=TEAL,lw=1.2,zorder=2))
ax.text(0.5,0.066,"Result — wall-clock",ha="center",fontsize=8.8,color=MUTED)
ax.text(0.5,0.038,"58.7 → 9.2 ms  (bcsstk30)      ·      9.4 → 1.9 ms  (bcsstk17)      ·      1.84 → 0.91 ms  (bcsstk11)",
        ha="center",fontsize=10.2,color=TEAL,fontweight="bold")

fig.savefig("fig/h2d_d2h_pool.png",dpi=200,bbox_inches="tight",facecolor=BG)
fig.savefig("fig/h2d_d2h_pool.pdf",bbox_inches="tight",facecolor=BG)
print("wrote fig/h2d_d2h_pool.png")
