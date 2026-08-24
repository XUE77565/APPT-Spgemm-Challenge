#!/usr/bin/env python3
"""PROJECTED AAT (C=A·Aᵀ) performance — analytical model, NOT measurement.

GPU is down so AAT can't run. This projects AAT from the measured AA (C=A·A)
compute-only times using the AA phase breakdown (inno/engiOpti.md) and the
AAT path's two optimizations:
  (1) 半量 / upper-triangle  -> halves the WORK-SCALING phases
  (2) tail-balanced partitioning -> load-balance benefit (bounded by the proven
      atomic accumulate floor) MINUS its own CPU-workload + reshape overhead.

MODEL (per matrix, compute-only):
  AA_compute T decomposes into:
    f_accel(T) : work-scaling fraction (accumulate + compact+sort + MinHash-merge)
    f_fixed    : 1 - f_accel  (mh_construct on input A, binning, scan, launches)
  f_accel(T) fit to the 5 measured phase points (inno/engiOpti.md), rising with T:
      f_accel(T) = FMIN + (FMAX-FMIN)*(1 - exp(-T/TAU))
      FMIN=0.40 (small matrices are launch/sizing-bound), FMAX=0.85, TAU=0.5 ms
  Upper-triangle halves the accelerable part; tail-balance is modeled as a small
  ADDITIVE overhead (BAL_OH) that exceeds its occupancy gain on large (saturated)
  matrices:
      AAT_compute = T * [ f_fixed + 0.5*f_accel ]  +  BAL_OH*T
                  = T * [ 1 - 0.5*f_accel + BAL_OH ]
  Baselines compute FULL A·Aᵀ (no symmetry exploit) => AAT_baseline = AA_baseline
  (exact for the symmetric structural matrices that dominate the suite).

Everything is labeled PROJECTED / pending GPU validation.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- model params (transparent; fit to inno/engiOpti.md phase data) ----
FMIN, FMAX, TAU = 0.40, 0.85, 0.5      # f_accel(T) saturating curve
HALF = 0.5                              # upper-triangle factor on accelerable work
BAL_OH = 0.05                           # tail-balance overhead (~5% of compute)

def f_accel(T):
    return FMIN + (FMAX - FMIN) * (1 - math.exp(-T / TAU))
def aat_ratio(T):           # AAT_compute / AA_compute
    return 1.0 - HALF * f_accel(T) + BAL_OH

# ---- load AA measured (compute-only) ----
b1 = {r["matrix"]: r for r in csv.DictReader(open("compare/aa(best1)/methods_cmp.csv"))}
top = {r["matrix"]: r for r in csv.DictReader(open("compare/methods_cmp.csv"))}
for m in b1: b1[m]["cublas"] = top[m]["cublas"]
rows = list(b1.values())
def num(x):
    try: return float(x)
    except: return None
def gm(x): x=[v for v in x if v and v>0 and math.isfinite(v)]; return math.exp(sum(math.log(v) for v in x)/len(x)) if x else float('nan')
def classify(d):
    d = d or 0
    if d >= 10: return "Dense"
    if d >= 1:  return "Mild"
    if d >= 0.1:return "High"
    return "Extreme"
BASES = [("dense","dense"),("cuBLAS","cublas"),("cuSPARSE","cu"),("opSparse","opSparse"),("HSMU","HSMU"),("Ocean","Ocean")]

# project per matrix
for r in rows:
    T = num(r["Auto"])
    r["_T"] = T; r["_ratio"] = aat_ratio(T); r["_aat"] = T * aat_ratio(T)
    r["_c"] = classify(num(r["density_pct"]))

# ---- report model + headline ----
print("="*64, "\nAAT PROJECTION MODEL (compute-only, PROJECTED — not measured)\n"+"="*64)
print(f"  f_accel(T) = {FMIN} -> {FMAX} as T grows (TAU={TAU} ms); HALF={HALF}; BAL_OH={BAL_OH}")
print(f"  AAT/AA ratio = 1 - 0.5*f_accel + {BAL_OH}")
print(f"  small(T=0.14): f_accel={f_accel(0.14):.2f} ratio={aat_ratio(0.14):.2f} (={1/aat_ratio(0.14):.2f}x)")
print(f"  large(T=3.6) : f_accel={f_accel(3.6):.2f} ratio={aat_ratio(3.6):.2f} (={1/aat_ratio(3.6):.2f}x)")
aa_g = gm([r["_T"] for r in rows]); aat_g = gm([r["_aat"] for r in rows])
print(f"\n  AA  Auto geomean : {aa_g:.3f} ms")
print(f"  AAT Auto geomean : {aat_g:.3f} ms   (ratio {aat_g/aa_g:.2f}, i.e. {aa_g/aat_g:.2f}x vs AA path)\n")

print(f"{'':2s}PROJECTED AAT speedup (geomean) vs each baseline   [AA speedup for reference]")
print(f"{'baseline':10s}{'AAT proj':>10s}{'AA actual':>11s}{'AAT wins':>10s}{'AA wins':>9s}")
proj_overall = {}
for disp,col in BASES:
    aat_sp = [num(r[col])/r["_aat"] for r in rows if num(r[col]) and r["_aat"]>0]
    aa_sp  = [num(r[col])/r["_T"]  for r in rows if num(r[col]) and r["_T"]>0]
    aat_win = sum(1 for r in rows if num(r[col]) and r["_aat"]<num(r[col]))
    aa_win  = sum(1 for r in rows if num(r[col]) and r["_T"]<num(r[col]))
    proj_overall[disp] = gm(aat_sp)
    print(f"{disp:10s}{gm(aat_sp):>9.2f}x{gm(aa_sp):>10.2f}x{aat_win:>7d}/100{aa_win:>6d}/100")

# ---- per-class for the table ----
CLS = ["Dense","Mild","High","Extreme"]
print("\n  PROJECTED AAT speedup by density class (geomean):")
cnt={c:sum(1 for r in rows if r["_c"]==c) for c in CLS}
print("baseline   ".ljust(10)+"".join(f"{c}({cnt[c]})".rjust(12) for c in CLS)+"ALL".rjust(10))
cell={}
for disp,col in BASES:
    line=f"{disp:10s}"
    for c in CLS:
        sub=[r for r in rows if r["_c"]==c]
        sp=[num(r[col])/r["_aat"] for r in sub if num(r[col]) and r["_aat"]>0]
        cell[(disp,c)]=gm(sp); line+=f"{gm(sp):>11.2f}x"
    line+=f"{proj_overall[disp]:>9.2f}x"
    print(line)
ours_ms={c: gm([r["_aat"] for r in rows if r["_c"]==c]) for c in CLS}
print(f"{'Ours(ms)':10s}"+"".join(f"{ours_ms[c]:>11.3f}" for c in CLS)+f"{aat_g:>9.3f}")

# ================= FIGURE (deck style, PROJECTED watermark) =================
TEAL,RED,GRAY,MUTED,INK="#1485A4","#C00000","#9AA6B2","#5C6773","#1F2933"
BG,GRID="#FFFFFF","#ECEEF1"
BCOL={"dense":"#eda100","cuBLAS":"#6c5ce7","cuSPARSE":"#2a78d6","opSparse":"#eb6834","HSMU":"#1baf7a","Ocean":"#8e44ad"}
fig=plt.figure(figsize=(12.8,6.2),facecolor=BG)
fig.text(0.5,0.965,"AAT (C = A·Aᵀ) — PROJECTED speedup over baselines",
         ha="center",va="top",fontsize=19,fontweight="bold",color=INK)
fig.text(0.5,0.918,"analytical model from AA phase breakdown + upper-triangle/tail-balance  ·  pending GPU validation",
         ha="center",va="top",fontsize=10,color=MUTED,style="italic")

# Panel A: overall geomean speedup per baseline
axA=fig.add_axes([0.07,0.13,0.40,0.72]); axA.set_facecolor(BG)
names=[d for d,_ in BASES]; vals=[proj_overall[d] for d in names]
aa_vals=[gm([num(r[c])/r["_T"] for r in rows if num(r[c]) and r["_T"]>0]) for _,c in BASES]
yy=np.arange(len(names))[::-1]
axA.barh(yy+0.18, vals, height=0.34, color=[BCOL[d] for d in names], edgecolor="white", linewidth=0.6, label="AAT (projected)")
axA.barh(yy-0.18, aa_vals, height=0.34, color=[BCOL[d] for d in names], alpha=0.30, edgecolor="white", linewidth=0.6, label="AA (measured)")
for y,v in zip(yy,vals):
    axA.text(v+0.12,y+0.18,f"{v:.2f}x",va="center",ha="left",fontsize=8.5,color=INK,fontweight="bold")
axA.set_yticks(yy); axA.set_yticklabels(names,fontsize=10,color=INK)
axA.set_xscale("log"); axA.set_xlim(1,40)
axA.set_xlabel("geomean speedup (baseline / Ours)",fontsize=10,color=INK)
axA.set_title("Overall (100 matrices)",fontsize=12,fontweight="bold",color=INK,loc="left")
axA.legend(loc="lower right",fontsize=8,framealpha=0.9,edgecolor=GRID)
axA.grid(axis="x",color=GRID,lw=0.5); axA.set_axisbelow(True)
for s in ["top","right"]: axA.spines[s].set_visible(False)
for s in ["left","bottom"]: axA.spines[s].set_color(MUTED)

# Panel B: by density class (grouped: AAT projected only, 6 baselines x 4 classes)
axB=fig.add_axes([0.555,0.13,0.42,0.72]); axB.set_facecolor(BG)
x=np.arange(len(CLS)); w=0.13
for i,(disp,col) in enumerate(BASES):
    hv=[cell[(disp,c)] for c in CLS]
    axB.bar(x+(i-2.5)*w, hv, w, color=BCOL[disp], edgecolor="white", linewidth=0.4, label=disp)
axB.set_yscale("log"); axB.set_xticks(x); axB.set_xticklabels([f"{c}\n({sum(1 for r in rows if r['_c']==c)})" for c in CLS],fontsize=9.5,color=INK)
axB.axhline(1,color=RED,ls="--",lw=1.0); axB.text(3.4,1.02,"break-even",color=RED,fontsize=7.5,ha="right")
axB.set_ylabel("geomean speedup (log)",fontsize=10,color=INK)
axB.set_title("By input density class",fontsize=12,fontweight="bold",color=INK,loc="left")
axB.legend(loc="upper right",fontsize=7.2,framealpha=0.9,edgecolor=GRID,ncol=2)
axB.grid(axis="y",color=GRID,lw=0.5); axB.set_axisbelow(True)
for s in ["top","right"]: axB.spines[s].set_visible(False)
for s in ["left","bottom"]: axB.spines[s].set_color(MUTED)
# Ours AAT ms annotation
axB2=axB.twinx(); axB2.set_ylim(axB.get_ylim())
# watermark
fig.text(0.5,0.5,"PROJECTED",fontsize=58,color=RED,alpha=0.05,ha="center",va="center",rotation=20,fontweight="bold",zorder=0)
fig.text(0.5,0.02,"H100 PCIe  ·  compute-only  ·  AAT_baseline ≈ AA_baseline (baselines compute full A·Aᵀ)  ·  NOT measured",
         ha="center",fontsize=8,color=MUTED)
os.makedirs("fig",exist_ok=True)
fig.savefig("fig/aat_projected.png",dpi=200,facecolor=BG,bbox_inches="tight")
fig.savefig("fig/aat_projected.pdf",facecolor=BG,bbox_inches="tight")
print("\nsaved fig/aat_projected.png (+.pdf)")
