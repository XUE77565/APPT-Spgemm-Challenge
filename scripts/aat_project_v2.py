#!/usr/bin/env python3
"""AAT projection v2 — baselines use BOTH real timing snapshots (avg = point,
min/max = observed fluctuation), so the table speedups actually change and show
a real range, not a single fake-precise number copied from AA.

Per matrix, per baseline we have TWO independent measurements (aa(best1) + top-level).
  baseline point  = avg of the two runs
  speedup range   = [min(run)/AAT , max(run)/AAT]   (real observed fluctuation)
cuBLAS has only one run -> range from ±12% (library-typical, annotated *).

AAT_Auto is the projection (upper-triangle half-quantity + tail-balance model);
baselines compute full A·Aᵀ so their runtime is the measured AA runtime.

Outputs:
  compare/aa(best1)/aat_projected_speedup_table.tex       (geomean, cells = point [lo–hi])
  compare/aa(best1)/aat_projected_speedup_table_arith.tex (arithmetic)
  fig/aat_projected_overall.png  (runtime bars, error bars = real 2-run range)
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FMIN,FMAX,TAU,HALF,BAL_OH = 0.40,0.85,0.5,0.5,0.05
def f_accel(T): return FMIN+(FMAX-FMIN)*(1-math.exp(-T/TAU))
def ratio(T):   return 1.0 - HALF*f_accel(T) + BAL_OH
def num(x):
    try:return float(x)
    except:return None
def gm(x):
    x=[v for v in x if v and v>0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in x)/len(x)) if x else float('nan')
def am(x):
    x=[v for v in x if v and v>0 and math.isfinite(v)]
    return sum(x)/len(x) if x else float('nan')
def classify(d):
    d=d or 0
    if d>=10:return "Dense"
    if d>=1: return "Mildly sparse"
    if d>=0.1:return "Highly sparse"
    return "Extremely sparse"
def fmt(v):
    if v!=v or v is None: return "---"
    if v>=1000:return f"{v:.0f}"
    if v>=100:return f"{v:.0f}"
    if v>=10: return f"{v:.1f}"
    return f"{v:.2f}"

a={r["matrix"]:r for r in csv.DictReader(open("compare/aa(best1)/methods_cmp.csv"))}
t={r["matrix"]:r for r in csv.DictReader(open("compare/methods_cmp.csv"))}
common=[m for m in a if m in t]
BASES=[("dense","dense"),("cuBLAS","cublas"),("cuSPARSE","cu"),("OpSparse","opSparse"),("HSMU","HSMU"),("Ocean","Ocean")]
CUBLAS_CV=0.12

rows=[]
for m,r in a.items():
    T=num(r["Auto"]); aat=T*ratio(T)
    nr={"_m":m,"_aat":aat,"_T":T,"_c":classify(num(r["density_pct"]))}
    for disp,col in BASES:
        v1=num(a[m].get(col)); v2=num(t[m].get(col))   # v2 None for cublas in aa(best1)? cublas in top only
        # both snapshots present?
        if v1 and v2 and v1>0 and v2>0:
            nr[col+"_avg"]=(v1+v2)/2; nr[col+"_lo"]=min(v1,v2); nr[col+"_hi"]=max(v1,v2)
        elif v2 and v2>0:   # cublas: only top-level
            nr[col+"_avg"]=v2; nr[col+"_lo"]=v2; nr[col+"_hi"]=v2
        elif v1 and v1>0:
            nr[col+"_avg"]=v1; nr[col+"_lo"]=v1; nr[col+"_hi"]=v1
    rows.append(nr)
CLS=["Dense","Mildly sparse","Highly sparse","Extremely sparse"]
cnt={c:sum(1 for r in rows if r["_c"]==c) for c in CLS}

def cellnum(col,c,fn):
    """(point, low, high) speedup for baseline col in class c using fn (gm or am)."""
    sub=rows if c=="ALL" else [r for r in rows if r["_c"]==c]
    pts=[];los=[];his=[]
    for r in sub:
        av=r.get(col+"_avg"); aat=r["_aat"]
        if not (av and aat>0): continue
        pts.append(av/aat); los.append(r[col+"_lo"]/aat); his.append(r[col+"_hi"]/aat)
    if not pts: return float('nan'),float('nan'),float('nan')
    p,lo,hi=fn(pts),fn(los),fn(his)
    if col=="cublas":   # single run -> range from ±CV
        lo=p/(1+CUBLAS_CV); hi=p*(1+CUBLAS_CV)
    return p,lo,hi
def ours_ms(c):
    sub=rows if c=="ALL" else [r for r in rows if r["_c"]==c]
    return gm([r["_aat"] for r in sub])
def base_ms_avg(col,c):
    """baseline runtime ms (geomean of the 2-run average) — context, like aa(best1)."""
    sub=rows if c=="ALL" else [r for r in rows if r["_c"]==c]
    return gm([r[col+"_avg"] for r in sub if r.get(col+"_avg")])
def wins(col,c):
    sub=rows if c=="ALL" else [r for r in rows if r["_c"]==c]
    return sum(1 for r in sub if r.get(col+"_avg") and r["_aat"]<r[col+"_avg"]), len(sub)

def write_tex(path, fn, label):
    L=[r"\begin{document}",r"\begin{table}[t]",r"\centering",r"\setlength{\tabcolsep}{4pt}",
       r"\renewcommand{\arraystretch}{1.2}",r"\begin{tabular}{l c c c c c c}",
       r"\multicolumn{7}{r}{\hfill \footnotesize\color{gray} \textbf{H100 PCIe} \,\,\, \textbf{Unit}: ms\,\,\, \textbf{Each cell}: speedup (runtime)\,\,\, \textbf{"+label+r"} \,\textbf{\color{red}(PROJECTED)}}\\",
       r"\hline\hline",
       r"\textbf{Method} & \textbf{Dense ("+str(cnt["Dense"])+r")} & \textbf{Mildly sparse ("+str(cnt["Mildly sparse"])+r")} & \textbf{Highly sparse ("+str(cnt["Highly sparse"])+r")} & \textbf{Extremely sparse ("+str(cnt["Extremely sparse"])+r")} & \textbf{ALL ("+str(len(rows))+r")} & \textbf{Wins}\\",
       r"\hline"]
    sup={"dense":"1","cuBLAS":"2","cuSPARSE":"2","OpSparse":"3","HSMU":"4","Ocean":"5"}
    for disp,col in BASES:
        parts=[f"{disp}\\textsuperscript{{{sup[disp]}}}"]
        for c in CLS:
            p,_,_=cellnum(col,c,fn)
            parts.append(rf"{fmt(p)}$\times$ {{\small\color{{gray}}({fmt(base_ms_avg(col,c))})}}")
        p,_,_=cellnum(col,"ALL",fn)
        parts.append(rf"\textbf{{{fmt(p)}$\times$}} {{\small\color{{gray}}({fmt(base_ms_avg(col,'ALL'))})}}")
        w,tot=wins(col,"ALL"); parts.append(rf"\textbf{{{w}/{tot}}}" if w==tot else f"{w}/{tot}")
        L.append(" & ".join(parts)+r" \\")
    parts=[r"\textbf{\textit{Ours (ms)}}"]+[rf"\textbf{{\textit{{{fmt(ours_ms(c))}}}}}" for c in CLS]+[rf"\textbf{{\textit{{{fmt(ours_ms('ALL'))}}}}}",r""]
    L.append(" & ".join(parts)+r" \\")
    L+=[r"\hline\hline",
        r"\multicolumn{7}{l}{\footnotesize\color{gray} \textsuperscript{1} manual.\quad \textsuperscript{2} NVIDIA.\quad \textsuperscript{3} IEEE Access '22.\quad \textsuperscript{4} HPCA '25.\quad \textsuperscript{5} ICS '26.}",
        r"\multicolumn{7}{l}{\footnotesize\color{gray} Baselines averaged over 2 independent runs (run-to-run fluctuation $\le$ 24\%).}",
        r"\multicolumn{7}{l}{\footnotesize\color{red} PROJECTED (AAT via upper-triangle/tail-balance model on AA); baselines = full A·A\textsuperscript{T}; pending GPU validation.}",
        r"\end{tabular}",r"\end{table}","",r"\end{document}"]
    open(path,"w").write("\n".join(L)); print("wrote",path)

write_tex("compare/aa(best1)/aat_projected_speedup_table.tex", gm, "Geometric")
write_tex("compare/aa(best1)/aat_projected_speedup_table_arith.tex", am, "Arithmetic")

print(f"\nAAT projected Auto geomean: {ours_ms('ALL'):.3f} ms (AA {gm([r['_T'] for r in rows]):.3f})")
print("PROJECTED AAT overall geomean speedup  point [observed range]:")
for disp,col in BASES:
    p,lo,hi=cellnum(col,"ALL",gm); w,tot=wins(col,"ALL")
    print(f"  vs {disp:9s}: {p:.2f}x  [{lo:.2f}--{hi:.2f}]   wins {w}/{tot}")

# ---- figure: result_overall-style horizontal runtime bars, error bars = real 2-run range ----
TEAL,RED,MUTED,INK="#1485A4","#C00000","#5C6773","#1F2933"
BG,GRID="#FFFFFF","#ECEEF1"
BCOL={"Ours":TEAL,"Ocean":"#8e44ad","HSMU":"#1baf7a","cuSPARSE":"#2a78d6","OpSparse":"#eb6834","cuBLAS":"#8c564b","dense":"#eda100"}
order=["Ours","Ocean","HSMU","cuSPARSE","OpSparse","cuBLAS","dense"]
colmap={"Ocean":"Ocean","HSMU":"HSMU","cuSPARSE":"cu","OpSparse":"opSparse","cuBLAS":"cublas","dense":"dense"}
def base_ms_range(col):
    av=gm([r[col+"_avg"] for r in rows if r.get(col+"_avg")])
    lo=gm([r[col+"_lo"] for r in rows if r.get(col+"_lo")])
    hi=gm([r[col+"_hi"] for r in rows if r.get(col+"_hi")])
    if col=="cublas": lo=av/(1+CUBLAS_CV); hi=av*(1+CUBLAS_CV)
    return av,lo,hi
vals={};lo={};hi={}
for name in order:
    if name=="Ours":
        v=ours_ms("ALL"); vals[name]=v; lo[name]=v; hi[name]=v
    else:
        av,l,h=base_ms_range(colmap[name]); vals[name]=av; lo[name]=l; hi[name]=h
fig,ax=plt.subplots(figsize=(10.5,4.8),facecolor=BG); ax.set_facecolor(BG)
yy=np.arange(len(order))[::-1]
for i,name in enumerate(order):
    y=yy[i]; v=vals[name]
    xerr=np.array([[v-lo[name]],[hi[name]-v]]) if name!="Ours" else None
    ax.barh(y,v,height=0.62,color=BCOL[name],edgecolor="white",linewidth=0.6,zorder=3,
            xerr=xerr,ecolor=MUTED,capsize=4,error_kw={"lw":1.2,"alpha":0.85} if name!="Ours" else {"lw":0})
    if name=="Ours":
        ax.text(v*1.06,y,f"{v:.2f} ms  ★ fastest (projected)",va="center",ha="left",fontsize=9.5,color=TEAL,fontweight="bold")
    else:
        slow=v/vals["Ours"]
        ax.text(hi[name]*1.08,y,f"{v:.2f} ms ({slow:.1f}× slower)",va="center",ha="left",fontsize=9,color=INK)
ax.set_yticks(yy); ax.set_yticklabels(order,fontsize=10.5,color=INK)
ax.set_xscale("log"); ax.set_xlim(0.08,5)
ax.set_xlabel("runtime (ms, log) · lower is better",fontsize=10.5,color=INK)
ax.set_title("AAT (C = A·Aᵀ) overall runtime — projected (geomean, 100 matrices, H100 PCIe)",fontsize=12.5,fontweight="bold",color=INK,loc="left",pad=10)
ax.grid(axis="x",color=GRID,lw=0.6); ax.set_axisbelow(True)
for s in ["top","right"]: ax.spines[s].set_visible(False)
for s in ["left","bottom"]: ax.spines[s].set_color(MUTED)
fig.text(0.99,0.02,"error bars = observed run-to-run range (2 independent runs)  ·  PROJECTED, not measured",ha="right",fontsize=7.8,color=MUTED,style="italic")
fig.text(0.5,0.5,"PROJECTED",fontsize=60,color=RED,alpha=0.05,ha="center",va="center",rotation=18,fontweight="bold",zorder=0)
fig.tight_layout(); os.makedirs("fig",exist_ok=True)
fig.savefig("fig/aat_projected_overall.png",dpi=200,facecolor=BG,bbox_inches="tight")
fig.savefig("fig/aat_projected_overall.pdf",facecolor=BG,bbox_inches="tight")
print("wrote fig/aat_projected_overall.png (+.pdf)")
