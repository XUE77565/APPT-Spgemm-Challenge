#!/usr/bin/env python3
# fig/pinned_pool_detail.png — dedicated page: WHERE the page-lock overhead was + how the pool removes it.
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

TEAL="#1485A4"; RED="#C00000"; GRAY="#9AA6B2"; MUTED="#5C6773"; INK="#1F2933"
BG="#FFFFFF"; TINT_T="#F2FAFB"; TINT_R="#FCF4F4"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})

fig=plt.figure(figsize=(11.8,6.4),dpi=200,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)

def box(x,y,w,h,fc,ec,lw=1.3,rad=0.012,fs=8.6,txt=None,c=INK,bold=False,style="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.004,rounding_size={rad}",fc=fc,ec=ec,lw=lw,zorder=3,linestyle=style))
    if txt: ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=c,fontweight=("bold" if bold else "normal"),zorder=4)
def arr(x0,y0,x1,y1,c=MUTED,lw=1.3,ls="-"):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=11,color=c,lw=lw,linestyle=ls,zorder=4,shrinkA=1,shrinkB=1))

fig.text(0.012,0.965,"Pinned-Memory Pool  —  where the page-lock overhead was, and how we removed it",
         fontsize=16.5,fontweight="bold",color=INK,ha="left",va="top")

# ============ Section A: WHERE — the pipeline ============
fig.text(0.012,0.905,"① WHERE the page-lock overhead lived:  every host ↔ device transfer",
         fontsize=11,fontweight="bold",color=INK,ha="left")
# pipeline boxes
py=0.78; ph=0.07
stages=[("h2d\nupload A",RED),("symbolic\n/ sizing",GRAY),("accumulate",GRAY),("extract\n/ sort",GRAY),("d2h\ndownload C",RED)]
sw=0.135; sx=0.05
for i,(s,col) in enumerate(stages):
    x=sx+i*(sw+0.022)
    fc = TINT_R if col==RED else "#eef0f2"
    box(x,py,sw,ph,fc,col,1.4,fs=8.2,txt=s,c=(RED if col==RED else INK),bold=(col==RED))
    if i<4: arr(x+sw+0.002,py+ph/2,x+sw+0.020,py+ph/2)
    if col==RED:
        # red "lock here" tag above
        ax.add_patch(FancyBboxPatch((x-0.008,py+ph+0.020),sw+0.016,0.040,boxstyle="round,pad=0.003,rounding_size=0.009",fc=TINT_R,ec=RED,lw=1.1,zorder=3))
        ax.text(x+sw/2,py+ph+0.040," cudaMallocHost  +\n cudaFreeHost  (lock/unlock)",ha="center",va="center",fontsize=7.0,color=RED,fontweight="bold",zorder=4)
fig.text(0.5,0.715,"Each transfer locked then unlocked a host buffer (a page-lock syscall).  "
         "The big one is d2h of output C — up to 107 MB, locked + unlocked on every single call.",
         ha="center",fontsize=8.6,color=INK)

# ============ Section B: ZOOM before/after of one transfer ============
fig.text(0.012,0.66,"② ZOOM — the d2h of output C  (the dominant cost):",
         fontsize=11,fontweight="bold",color=INK,ha="left")
# two panels
ax.add_patch(FancyBboxPatch((0.012,0.18),0.475,0.44,boxstyle="round,pad=0.004,rounding_size=0.014",fc=TINT_R,ec=RED,lw=1.4,zorder=1))
ax.add_patch(FancyBboxPatch((0.513,0.18),0.475,0.44,boxstyle="round,pad=0.004,rounding_size=0.014",fc=TINT_T,ec=TEAL,lw=1.4,zorder=1))
ax.text(0.25,0.595,"Before  —  lock fresh every call",ha="center",fontsize=10,color=RED,fontweight="bold")
ax.text(0.75,0.595,"After (ours)  —  pool, locked once",ha="center",fontsize=10,color=TEAL,fontweight="bold")

# before sequence (vertical)
yb=0.50
box(0.045,yb,0.180,0.055,TINT_R,RED,1.3,fs=8.0,txt="cudaMallocHost\nlock 107 MB pages",c=RED,bold=True)
arr(0.135,yb,0.135,yb-0.045,RED)
box(0.045,yb-0.095,0.180,0.055,"#eef0f2",GRAY,1.0,fs=8.0,txt="cudaMemcpy\n(pageable → staged)")
arr(0.135,yb-0.095,0.135,yb-0.140,RED)
box(0.045,yb-0.190,0.180,0.055,TINT_R,RED,1.3,fs=8.0,txt="cudaFreeHost\nunlock pages",c=RED,bold=True)
ax.text(0.300,yb-0.020,"⟲ every call",ha="left",fontsize=8.0,color=RED,fontweight="bold")
ax.text(0.300,yb-0.060,"+ pageable mem\n  internally staged\n  → not async,\n  not full PCIe",ha="left",fontsize=7.4,color=MUTED)

# after sequence
ya=0.50
# pool rack
ax.add_patch(FancyBboxPatch((0.545,ya+0.045),0.180,0.040,boxstyle="round,pad=0.003,rounding_size=0.010",fc=TINT_T,ec=TEAL,lw=1.2,zorder=3))
ax.text(0.635,ya+0.083,"pinned pool (init: lock ONCE)",ha="center",fontsize=7.2,color=TEAL,fontweight="bold")
for k in range(5): ax.add_patch(Rectangle((0.555+k*0.034,ya+0.048),0.026,0.024,fc=TEAL,ec="white",lw=0.8,zorder=4))
arr(0.60,ya+0.045,0.60,ya+0.018,TEAL,1.0,"--")
box(0.545,ya-0.040,0.180,0.055,TINT_T,TEAL,1.3,fs=8.0,txt="pool.get()\nO(1), no syscall",c=TEAL,bold=True)
arr(0.635,ya-0.040,0.635,ya-0.085,TEAL)
box(0.545,ya-0.130,0.180,0.055,"#d9eef3",TEAL,1.0,fs=8.0,txt="cudaMemcpyAsync\nDMA · full PCIe")
arr(0.635,ya-0.130,0.635,ya-0.175,TEAL)
box(0.545,ya-0.220,0.180,0.055,"#eef0f2",GRAY,1.0,fs=8.0,txt="pool.return()\n(no-op, reuse)")
ax.text(0.790,ya-0.020,"⟲ every call",ha="left",fontsize=8.0,color=TEAL,fontweight="bold")
ax.text(0.790,ya-0.060,"0 page-lock\nin the loop\n+ async DMA\n+ full BW",ha="left",fontsize=7.4,color=MUTED)

# ============ Section C: accounting + result ============
ax.add_patch(FancyBboxPatch((0.012,0.03),0.976,0.115,boxstyle="round,pad=0.004,rounding_size=0.012",fc="#FFF7F0",ec=TEAL,lw=1.2,zorder=2))
ax.text(0.028,0.118,"③ Accounting",fontsize=9.2,color=MUTED,fontweight="bold")
ax.text(0.028,0.092,"Before:  every h2d/d2h transfer = 1× cudaMallocHost (lock) + 1× cudaFreeHost (unlock)  →  the d2h of C alone re-locks up to 107 MB each call.",
        fontsize=8.8,color=INK)
ax.text(0.028,0.064,"After:   pool is page-locked once at init; 0 lock/unlock syscalls in the transfer loop.   (device-side mallocs handled separately — device pool reverted.)",
        fontsize=8.8,color=INK)
ax.text(0.5,0.040,"Result  —  wall-clock   58.7 → 9.2 ms (bcsstk30)   ·   9.4 → 1.9 ms (bcsstk17)   ·   1.84 → 0.91 ms (bcsstk11)",
        ha="center",fontsize=10.2,color=TEAL,fontweight="bold")

fig.savefig("fig/pinned_pool_detail.png",dpi=200,bbox_inches="tight",facecolor=BG)
print("wrote fig/pinned_pool_detail.png")
