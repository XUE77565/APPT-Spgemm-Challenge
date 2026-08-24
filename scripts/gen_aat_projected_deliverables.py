#!/usr/bin/env python3
"""Build PROJECTED-AAT deliverables that mirror aa(best1)/ exactly:
  1. compare/aa(best1)/aat_projected_methods_cmp.csv  (Auto = projected AAT; baselines = AA)
  2. compare/aa(best1)/aat_projected_speedup_table.tex (exact speedup_table.tex format)
Run scripts/plot_methods_cmp.py on the CSV to get the matching bar figure.

Projection model = scripts/aat_project_model.py:
  AAT_Auto = AA_Auto * (1 - 0.5*f_accel(AA_Auto) + BAL_OH), baselines unchanged.
"""
import csv, math, os
FMIN,FMAX,TAU,HALF,BAL_OH = 0.40,0.85,0.5,0.5,0.05
def f_accel(T): return FMIN+(FMAX-FMIN)*(1-math.exp(-T/TAU))
def ratio(T):   return 1.0 - HALF*f_accel(T) + BAL_OH
def num(x):
    try:return float(x)
    except:return None
def gm(x):
    x=[v for v in x if v and v>0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in x)/len(x)) if x else float('nan')
def classify(d):
    d=d or 0
    if d>=10:return "Dense"
    if d>=1: return "Mildly sparse"
    if d>=0.1:return "Highly sparse"
    return "Extremely sparse"
def fmt_sp(v):
    if v!=v: return "---"
    if v>=100:return f"{v:.0f}"
    if v>=10: return f"{v:.1f}"
    return f"{v:.2f}"

b1={r["matrix"]:r for r in csv.DictReader(open("compare/aa(best1)/methods_cmp.csv"))}
top={r["matrix"]:r for r in csv.DictReader(open("compare/methods_cmp.csv"))}
cols=["matrix","n","sym","density_pct","cu","Auto","Auto_choice","Ocean","opSparse","HSMU","dense","cublas","cnnz"]
out_rows=[]
for m,r in b1.items():
    T=num(r["Auto"]); aat=T*ratio(T)
    nr=dict(r); nr["Auto"]=f"{aat:.4f}"; nr["cublas"]=top[m]["cublas"]
    out_rows.append(nr)
    nr["_c"]=classify(num(r["density_pct"])); nr["_aat"]=aat; nr["_T"]=T

# 1) projected CSV (same schema as methods_cmp.csv)
with open("compare/aa(best1)/aat_projected_methods_cmp.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in out_rows: w.writerow({k:r[k] for k in cols})
print("wrote compare/aa(best1)/aat_projected_methods_cmp.csv")

# 2) speedup_table.tex (mirror aa(best1)/speedup_table.tex exactly)
rows=out_rows
CLS=["Dense","Mildly sparse","Highly sparse","Extremely sparse"]
BASES=[("dense","dense","manual."),("cuBLAS","cublas","NVIDIA."),("cuSPARSE","cu","NVIDIA."),
       ("OpSparse","opSparse","IEEE Access '22."),("HSMU","HSMU","HPCA '25."),("Ocean","Ocean","ICS '26.")]
cnt={c:sum(1 for r in rows if r["_c"]==c) for c in CLS}
# per (baseline,class): geomean speedup + baseline geomean ms + wins
def cell(base_col,c):
    sub=rows if c=="ALL" else [r for r in rows if r["_c"]==c]
    sp=[num(r[base_col])/r["_aat"] for r in sub if num(r[base_col]) and r["_aat"]>0]
    bms=[num(r[base_col]) for r in sub if num(r[base_col])]
    win=sum(1 for r in sub if num(r[base_col]) and r["_aat"]<num(r[base_col]))
    return gm(sp), gm(bms), win, len(sub)
sp_all={}
for disp,col,_ in BASES:
    for c in CLS+["ALL"]:
        sp_all[(disp,c)]=cell(col,c)
# best per column (max geomean speedup) for bold
best={}
for c in CLS+["ALL"]:
    vals={disp:sp_all[(disp,c)][0] for disp,_,_ in BASES if sp_all[(disp,c)][0]==sp_all[(disp,c)][0]}
    best[c]=max(vals,key=vals.get) if vals else None
ours_ms={c:gm([r["_aat"] for r in (rows if c=="ALL" else [r for r in rows if r["_c"]==c])]) for c in CLS+["ALL"]}

L=[]
L.append(r"\begin{document}")
L.append(r"\begin{table}[t]")
L.append(r"\centering")
L.append(r"\setlength{\tabcolsep}{5pt}")
L.append(r"\renewcommand{\arraystretch}{1.2}")
L.append(r"\begin{tabular}{l c c c c c c}")
L.append(r"\multicolumn{7}{r}{\hfill \footnotesize\color{gray} \textbf{H100 PCIe} \,\,\, \textbf{Unit}: ms\,\,\, \textbf{Each cell}: speedup (runtime)\,\,\, \textbf{Geometric} mean \,\textbf{\color{red}(PROJECTED)}}\\")
L.append(r"\hline\hline")
L.append(r"\textbf{Method} & \textbf{Dense ("+str(cnt["Dense"])+r")} & \textbf{Mildly sparse ("+str(cnt["Mildly sparse"])+r")} & \textbf{Highly sparse ("+str(cnt["Highly sparse"])+r")} & \textbf{Extremely sparse ("+str(cnt["Extremely sparse"])+r")} & \textbf{ALL ("+str(len(rows))+r")} & \textbf{Wins}\\")
L.append(r"\hline")
sup={"dense":r"\textsuperscript{1}","cuBLAS":r"\textsuperscript{2}","cuSPARSE":r"\textsuperscript{2}","OpSparse":r"\textsuperscript{3}","HSMU":r"\textsuperscript{4}","Ocean":r"\textsuperscript{5}"}
for i,(disp,col,_) in enumerate(BASES):
    parts=[disp+sup[disp]]
    for c in CLS:
        sp,bms,win,tot=sp_all[(disp,c)]
        b=r"\textbf{" if best[c]==disp else ""
        be=r"}" if best[c]==disp else ""
        parts.append(rf"{b}{fmt_sp(sp)}$\times${be} {{\small\color{{gray}}({fmt_sp(bms)})}}")
    sp,bms,win,tot=sp_all[(disp,"ALL")]
    b=r"\textbf{" if best["ALL"]==disp else ""; be=r"}" if best["ALL"]==disp else ""
    parts.append(rf"{b}{fmt_sp(sp)}$\times${be} {{\small\color{{gray}}({fmt_sp(bms)})}}")
    parts.append(rf"\textbf{{{win}/{tot}}}" if win==tot else f"{win}/{tot}")
    L.append(" & ".join(parts)+r" \\")
parts=[r"\textbf{\textit{Ours (ms)}}"]
for c in CLS: parts.append(rf"\textbf{{\textit{{{fmt_sp(ours_ms[c])}}}}}")
parts.append(rf"\textbf{{\textit{{{fmt_sp(ours_ms['ALL'])}}}}}")
parts.append("")
L.append(" & ".join(parts)+r" \\")
L.append(r"\hline\hline")
L.append(r"\multicolumn{7}{l}{\footnotesize\color{gray} \textsuperscript{1} manual.\quad \textsuperscript{2} NVIDIA.\quad \textsuperscript{3} IEEE Access '22.\quad \textsuperscript{4} HPCA '25.\quad \textsuperscript{5} ICS '26.}")
L.append(r"\multicolumn{7}{l}{\footnotesize\color{red} PROJECTED from AA measurements + upper-triangle/tail-balance model; pending GPU validation. Not measured.}")
L.append(r"\end{tabular}")
L.append(r"\end{table}")
L.append("")
L.append(r"\end{document}")
open("compare/aa(best1)/aat_projected_speedup_table.tex","w").write("\n".join(L))
print("wrote compare/aa(best1)/aat_projected_speedup_table.tex")
print(f"\nAAT projected Auto geomean: {ours_ms['ALL']:.3f} ms (AA 0.181) -> {0.181/ours_ms['ALL']:.2f}x vs AA path")
print("overall projected speedup / wins:")
for disp,col,_ in BASES:
    sp,bms,win,tot=sp_all[(disp,"ALL")]
    print(f"  vs {disp:9s}: {sp:.2f}x   wins {win}/{tot}")
