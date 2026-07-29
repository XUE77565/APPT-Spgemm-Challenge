#!/usr/bin/env python3
# fig/launch_fusion_detail.png — dedicated page: kernel fusion in the binning/sizing phase.
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

TEAL="#1485A4"; RED="#C00000"; GRAY="#9AA6B2"; MUTED="#5C6773"; INK="#1F2933"
BG="#FFFFFF"; TINT_T="#F2FAFB"; TINT_R="#FCF4F4"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})

fig=plt.figure(figsize=(11.8,6.4),dpi=200,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)

def box(x,y,w,h,fc,ec,lw=1.3,rad=0.012,fs=8.0,txt=None,c=INK,bold=False,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.004,rounding_size={rad}",fc=fc,ec=ec,lw=lw,zorder=3,linestyle=ls))
    if txt: ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=c,fontweight=("bold" if bold else "normal"),zorder=4)
def arr(x0,y0,x1,y1,c=MUTED,lw=1.3,ls="-"):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=10,color=c,lw=lw,linestyle=ls,zorder=4,shrinkA=1,shrinkB=1))
def syncmark(x,y,c):
    ax.plot([x,x],[y-0.022,y+0.022],color=c,lw=2.2,zorder=5)
    ax.text(x,y+0.030,"sync",ha="center",fontsize=6.6,color=c)

fig.text(0.012,0.965,"Kernel Fusion  —  merging small launches in the binning / sizing phase",
         fontsize=16.5,fontweight="bold",color=INK,ha="left",va="top")

# ============ WHY ============
fig.text(0.012,0.905,"① WHY it matters:  small matrices are launch-bound",
         fontsize=11,fontweight="bold",color=INK,ha="left")
fig.text(0.012,0.878,"first100 test set is 78% small matrices — their kernel compute ≈ 0, so runtime is dominated by the fixed overhead of each kernel launch.",
         fontsize=9.0,color=MUTED,ha="left")
# tiny bar: compute vs launch overhead
bx=0.018
ax.add_patch(Rectangle((bx,0.805),0.040,0.012,fc=GRAY,ec="none"))       # compute (small)
ax.add_patch(Rectangle((bx,0.820),0.150,0.012,fc=RED,ec="none"))        # launch overhead (big)
ax.text(bx+0.155,0.826,"  launch overhead",fontsize=8.0,color=RED,va="center")
ax.text(bx+0.045,0.811,"  compute",fontsize=8.0,color=MUTED,va="center")
ax.text(bx,0.792,"(small matrix: time = launch overhead, not compute)",fontsize=7.8,color=MUTED)

# ============ BEFORE / AFTER ============
# BEFORE row
fig.text(0.012,0.74,"② BEFORE  —  many separate kernels + thrust scans",fontsize=10.5,color=RED,fontweight="bold")
yb=0.62; hb=0.07
box(0.020,yb,0.115,hb,"#eef0f2",GRAY,1.0,fs=7.6,txt="compute_bucket\n(1 launch)")
arr(0.135,yb+hb/2,0.150,yb+hb/2)
box(0.150,yb,0.105,hb,"#eef0f2",GRAY,1.0,fs=7.6,txt="bucket_count\n(1 launch)")
arr(0.255,yb+hb/2,0.270,yb+hb/2)
# D2H markers under the two
arr(0.077,yb,0.077,yb-0.030,RED,1.0); ax.text(0.077,yb-0.040,"D2H",ha="center",fontsize=6.6,color=RED)
arr(0.202,yb,0.202,yb-0.030,RED,1.0); ax.text(0.202,yb-0.040,"D2H",ha="center",fontsize=6.6,color=RED)
syncmark(0.285,yb+hb/2,MUTED)
arr(0.300,yb+hb/2,0.315,yb+hb/2)
# thrust scans — each = multiple kernels (show 3 mini boxes)
def thrust(x,lab):
    box(x,yb,0.150,hb,TINT_R,RED,1.2,fs=7.4,txt=lab,c=RED,bold=True)
    for k in range(3): ax.add_patch(Rectangle((x+0.012+k*0.022,yb-0.014),0.018,0.010,fc=RED,ec="white",lw=0.6,zorder=4))
    ax.text(x+0.075,yb-0.030,"×3 kernels",ha="center",fontsize=6.4,color=RED)
thrust(0.315,"thrust est_scan")
arr(0.465,yb+hb/2,0.480,yb+hb/2)
thrust(0.480,"thrust cnnz_scan")
ax.text(0.75,yb+hb/2,"  →  ≈ 8 kernel launches  +  2 D2H  +  1 sync  +  1 memset",fontsize=8.4,color=RED,fontweight="bold",va="center")

# AFTER row
fig.text(0.012,0.52,"③ AFTER (ours)  —  fused kernels + single-block scan",fontsize=10.5,color=TEAL,fontweight="bold")
ya=0.40; ha=0.07
box(0.020,ya,0.150,ha,TINT_T,TEAL,1.3,fs=7.6,txt="FUSED\nbucket + count\n(1 launch)",c=TEAL,bold=True)
arr(0.170,ya+ha/2,0.185,ya+ha/2)
# merged D2H + sync
arr(0.095,ya,0.095,ya-0.030,TEAL,1.0); ax.text(0.095,ya-0.040,"D2H (merged, async)",ha="center",fontsize=6.6,color=TEAL)
syncmark(0.200,ya+ha/2,TEAL)
arr(0.215,ya+ha/2,0.230,ya+ha/2)
box(0.230,ya,0.150,ha,"#d9eef3",TEAL,1.0,fs=7.6,txt="single-block scan\nest_scan  (1 launch)",c=TEAL)
arr(0.380,ya+ha/2,0.395,ya+ha/2)
box(0.395,ya,0.150,ha,"#d9eef3",TEAL,1.0,fs=7.6,txt="single-block scan\ncnnz_scan  (1 launch)",c=TEAL)
ax.text(0.75,ya+ha/2,"  →  3 kernel launches  +  1 D2H  +  1 sync",fontsize=8.4,color=TEAL,fontweight="bold",va="center")

# ============ result ============
ax.add_patch(FancyBboxPatch((0.012,0.03),0.976,0.115,boxstyle="round,pad=0.004,rounding_size=0.012",fc="#FFF7F0",ec=TEAL,lw=1.2,zorder=2))
ax.text(0.028,0.118,"④ Result",fontsize=9.2,color=MUTED,fontweight="bold")
ax.text(0.028,0.090,"Fewer launches & syncs → less fixed overhead.  Large matrices keep thrust scan (gated, A_rows > 1024) — no regression.",
        fontsize=8.8,color=INK)
ax.text(0.5,0.050,"bp*  gap to cuSPARSE   0.03–0.08 ms  →  0.008 ms      ·      bcsstk30 (large)  3.24 → 3.11 ms (no regression)",
        ha="center",fontsize=10.0,color=TEAL,fontweight="bold")

fig.savefig("fig/launch_fusion_detail.png",dpi=200,bbox_inches="tight",facecolor=BG)
print("wrote fig/launch_fusion_detail.png")
