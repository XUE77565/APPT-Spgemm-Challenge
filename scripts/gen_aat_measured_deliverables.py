#!/usr/bin/env python3
"""Build MEASURED-AAT deliverables (mirror aa(best1)/ projection, but Auto = measured):
  1. compare/aa(best1)/aat_measured_methods_cmp.csv  (Auto = measured tiered AAT; baselines = AA)
  2. compare/aa(best1)/aat_measured_speedup_table.tex (same format as speedup_table.tex)
Then run scripts/plot_methods_cmp.py on the CSV for the matching bar figure.

Input: compare/aa(best1)/aat_measured_timing.csv  (produced by bench_aat_first100.py)
Baseline timing (cu/Ocean/opSparse/HSMU/dense/cuBLAS) is taken unchanged from
aa(best1)/methods_cmp.csv (+ top-level methods_cmp.csv for cublas).

NOTE: tiered AAT is FP32 and computes C=A*A^T (full, via upper-triangle+mirror). Baselines
are double A*A (self-product). Comparison is cleanest for symmetric A (A*A^T == A*A); for
non-symmetric A it is approximate. Same caveat applies to the projection this replaces.
"""
import csv, math, os

def num(x):
    try: return float(x)
    except: return None
def gm(x):
    x = [v for v in x if v and v > 0 and math.isfinite(v)]
    return math.exp(sum(math.log(v) for v in x)/len(x)) if x else float('nan')
def classify(d):
    d = d or 0
    if d >= 10:  return "Dense"
    if d >= 1:   return "Mildly sparse"
    if d >= 0.1: return "Highly sparse"
    return "Extremely sparse"
def fmt_sp(v):
    if v != v: return "---"            # nan
    if v >= 100: return f"{v:.0f}"
    if v >= 10:  return f"{v:.1f}"
    return f"{v:.2f}"

# measured AAT timing per matrix
meas = {}
for r in csv.DictReader(open("compare/aa(best1)/aat_measured_timing.csv")):
    if r.get("status") == "ok" and r.get("aat_ms"):
        try: meas[r["matrix"]] = float(r["aat_ms"])
        except: pass
# baselines (AA path) + cublas from top-level
b1  = {r["matrix"]: r for r in csv.DictReader(open("compare/aa(best1)/methods_cmp.csv"))}
top = {r["matrix"]: r for r in csv.DictReader(open("compare/methods_cmp.csv"))}
cols = ["matrix","n","sym","density_pct","cu","Auto","Auto_choice","Ocean","opSparse","HSMU","dense","cublas","cnnz"]

rows = []
for m, r in b1.items():
    if m not in meas: continue                      # only matrices with a valid measurement
    nr = dict(r)
    nr["Auto"] = f"{meas[m]:.4f}"
    nr["cublas"] = top.get(m, {}).get("cublas", "")
    nr["_c"]  = classify(num(r["density_pct"]))
    nr["_aat"] = meas[m]
    rows.append(nr)

# 1) measured CSV (same schema)
with open("compare/aa(best1)/aat_measured_methods_cmp.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in cols})
print(f"wrote compare/aa(best1)/aat_measured_methods_cmp.csv  ({len(rows)} matrices)")

# 2) speedup_table.tex
CLS  = ["Dense","Mildly sparse","Highly sparse","Extremely sparse"]
BASES= [("dense","dense","manual."),("cuBLAS","cublas","NVIDIA."),("cuSPARSE","cu","NVIDIA."),
        ("OpSparse","opSparse","IEEE Access '22."),("HSMU","HSMU","HPCA '25."),("Ocean","Ocean","ICS '26.")]
cnt  = {c: sum(1 for r in rows if r["_c"]==c) for c in CLS}

def cell(base_col, c):
    sub = rows if c=="ALL" else [r for r in rows if r["_c"]==c]
    sp  = [num(r[base_col])/r["_aat"] for r in sub if num(r[base_col]) and r["_aat"]>0]
    bms = [num(r[base_col]) for r in sub if num(r[base_col])]
    win = sum(1 for r in sub if num(r[base_col]) and r["_aat"]<num(r[base_col]))
    return gm(sp), gm(bms), win, len(sub)

sp_all = {}
for disp, col, _ in BASES:
    for c in CLS + ["ALL"]:
        sp_all[(disp, c)] = cell(col, c)
best = {}
for c in CLS + ["ALL"]:
    vals = {disp: sp_all[(disp, c)][0] for disp, _, _ in BASES if sp_all[(disp, c)][0]==sp_all[(disp, c)][0]}
    best[c] = max(vals, key=vals.get) if vals else None
ours_ms = {c: gm([r["_aat"] for r in (rows if c=="ALL" else [r for r in rows if r["_c"]==c])]) for c in CLS+["ALL"]}

L = []
L.append(r"\begin{document}")
L.append(r"\begin{table}[t]\centering\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.2}")
L.append(r"\begin{tabular}{l c c c c c c}")
L.append(r"\multicolumn{7}{r}{\hfill \footnotesize\color{gray} \textbf{H100 PCIe} \,\,\, \textbf{Unit}: ms\,\,\, \textbf{Each cell}: speedup (runtime)\,\,\, \textbf{Geometric} mean \,\textbf{\color{teal}(MEASURED)}}\\")
L.append(r"\hline\hline")
L.append(r"\textbf{Method} & \textbf{Dense ("+str(cnt["Dense"])+r")} & \textbf{Mildly sparse ("+str(cnt["Mildly sparse"])+r")} & \textbf{Highly sparse ("+str(cnt["Highly sparse"])+r")} & \textbf{Extremely sparse ("+str(cnt["Extremely sparse"])+r")} & \textbf{ALL ("+str(len(rows))+r")} & \textbf{Wins}\\")
L.append(r"\hline")
sup = {"dense":r"\textsuperscript{1}","cuBLAS":r"\textsuperscript{2}","cuSPARSE":r"\textsuperscript{2}",
       "OpSparse":r"\textsuperscript{3}","HSMU":r"\textsuperscript{4}","Ocean":r"\textsuperscript{5}"}
for disp, col, _ in BASES:
    parts = [disp + sup[disp]]
    for c in CLS:
        sp, bms, win, tot = sp_all[(disp, c)]
        b  = r"\textbf{" if best[c]==disp else ""
        be = r"}" if best[c]==disp else ""
        parts.append(rf"{b}{fmt_sp(sp)}$\times${be} {{\small\color{{gray}}({fmt_sp(bms)})}}")
    sp, bms, win, tot = sp_all[(disp, "ALL")]
    b  = r"\textbf{" if best["ALL"]==disp else ""; be = r"}" if best["ALL"]==disp else ""
    parts.append(rf"{b}{fmt_sp(sp)}$\times${be} {{\small\color{{gray}}({fmt_sp(bms)})}}")
    parts.append(rf"\textbf{{{win}/{tot}}}" if win==tot else f"{win}/{tot}")
    L.append(" & ".join(parts) + r" \\")
parts = [r"\textbf{\textit{Ours (ms)}}"]
for c in CLS: parts.append(rf"\textbf{{\textit{{{fmt_sp(ours_ms[c])}}}}}")
parts.append(rf"\textbf{{\textit{{{fmt_sp(ours_ms['ALL'])}}}}}")
parts.append("")
L.append(" & ".join(parts) + r" \\")
L.append(r"\hline\hline")
L.append(r"\multicolumn{7}{l}{\footnotesize\color{gray} \textsuperscript{1} manual.\quad \textsuperscript{2} NVIDIA.\quad \textsuperscript{3} IEEE Access '22.\quad \textsuperscript{4} HPCA '25.\quad \textsuperscript{5} ICS '26.}")
L.append(r"\multicolumn{7}{l}{\footnotesize\color{teal} MEASURED: tiered AAT (C=A$\cdot$A$^T$, FP32) on H100. Baselines = their A$\cdot$A (self-product); cleanest for symmetric A.}")
L.append(r"\end{tabular}\end{table}")
L.append(r"\end{document}")
open("compare/aa(best1)/aat_measured_speedup_table.tex", "w").write("\n".join(L))
print("wrote compare/aa(best1)/aat_measured_speedup_table.tex")

print(f"\nAAT measured Auto geomean: {ours_ms['ALL']:.3f} ms (over {len(rows)} matrices)")
print("overall measured speedup / wins:")
for disp, col, _ in BASES:
    sp, bms, win, tot = sp_all[(disp, "ALL")]
    print(f"  vs {disp:9s}: {sp:.2f}x   wins {win}/{tot}")
print("\nnext: .venv/bin/python scripts/plot_methods_cmp.py  (on aat_measured_methods_cmp.csv) for the bar figure")
