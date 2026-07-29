#!/usr/bin/env python3
# fig/engopt_slide.png — engineering-optimization slide (deck style).
# compact 4-card compute pipeline (top) + detailed h2d/d2h pinned-pool figure (bottom, featured).
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import PIL.Image as Image
import numpy as np

TEAL="#1485A4"; RED="#C00000"; GRAY="#9AA6B2"; MUTED="#5C6773"; INK="#1F2933"
BG="#FFFFFF"; TINT_T="#F2FAFB"; TINT_R="#FCF4F4"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})

fig=plt.figure(figsize=(12.8,7.2),dpi=200,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)

fig.text(0.012,0.965,"Engineering Optimization — Pipeline Phases + Pinned-Memory Transfer",
         fontsize=18,fontweight="bold",color=INK,ha="left",va="top")
fig.text(0.012,0.936,"compute pipeline (sizing → accumulate → extract → launch)  +  the foundational h2d/d2h optimization",
         fontsize=10.2,color=MUTED,ha="left",va="top")

def box(x,y,w,h,fc,ec,lw=1.3,rad=0.012):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.004,rounding_size={rad}",fc=fc,ec=ec,lw=lw,zorder=2))

# ---------- compact 4-card compute pipeline ----------
cards=[("① Sizing","MinHash 1-pass + EXPAND 1.5","symbolic 3.5× > opSparse · over-alloc 3.14→2.29×",True),
       ("② Accumulate","hash SPA  (atomicCAS+Add)","ATOMIC FLOOR · 3 levers failed",False),
       ("③ Extract","count-sort (sync-free) + pack","beats bitonic/radix · pack −0.09ms",True),
       ("④ Launch","kernel fusion (binning+scan)","bp* gap-to-cu 0.03–0.08→0.008ms",True)]
cw=0.235; gap=0.012; x0=0.018; y0=0.66; ch=0.245
for i,(title,what,res,kept) in enumerate(cards):
    x=x0+i*(cw+gap); edge=TEAL if kept else RED; tint=TINT_T if kept else TINT_R
    box(x,y0,cw,ch,tint,edge,lw=1.5,rad=0.014)
    ax.add_patch(FancyBboxPatch((x,y0+ch-0.040),cw,0.040,boxstyle="round,pad=0.004,rounding_size=0.012",
                 fc=edge,ec="none",zorder=3))
    ax.text(x+cw/2,y0+ch-0.020,title,ha="center",va="center",color="white",fontsize=11,fontweight="bold",zorder=4)
    ax.text(x+0.012,y0+ch-0.058,what,ha="left",va="top",fontsize=8.8,color=INK,fontweight="bold")
    ax.text(x+0.012,y0+ch-0.110,res,ha="left",va="top",fontsize=8.2,color=(TEAL if kept else RED),fontweight="bold")
    # verdict chip
    cx=x+cw-0.080; cyp=y0+0.018
    box(cx,cyp,0.070,0.024,(TEAL if kept else RED),(TEAL if kept else RED),rad=0.010)
    ax.text(cx+0.035,cyp+0.012,("KEPT" if kept else "FLOOR"),ha="center",va="center",color="white",
            fontsize=7.8,fontweight="bold",zorder=5)
# inter-card flow arrows
for i in range(3):
    xa=x0+(i+1)*cw+i*gap; xb=x0+(i+1)*(cw+gap)
    ax.annotate("",xy=(xb-0.001,y0+ch/2),xytext=(xa+0.001,y0+ch/2),
                arrowprops=dict(arrowstyle="-|>",color=MUTED,lw=1.0),zorder=6)

# ---------- featured: detailed h2d/d2h pinned-pool figure (bottom) ----------
ax.text(0.5,0.628,"⚙  Foundation · pinned-memory pool for h2d / d2h  (the first optimization we did)",
        ha="center",va="center",fontsize=11,color=TEAL,fontweight="bold")
img=Image.open("fig/h2d_d2h_pool.png"); arr=np.asarray(img)
axI=fig.add_axes([0.060,0.012,0.880,0.590])  # aspect ~ matches image (2.36:1)
axI.imshow(arr); axI.axis("off")

os.makedirs("fig",exist_ok=True)
fig.savefig("fig/engopt_slide.png",dpi=200,bbox_inches="tight",facecolor=BG)
fig.savefig("fig/engopt_slide.pdf",bbox_inches="tight",facecolor=BG)
print("wrote fig/engopt_slide.png")
