#!/usr/bin/env python3
"""从 methods_cmp.csv 生成 LaTeX 加速比表(按密度类别 + ALL)。

每格 = Auto 对该基线的 geomean 加速比(baseline/Auto,>1 = Auto 快)× (Auto 赢的阵数/该类总数)。
最优(最高加速比)每列加粗。行 = dense/cuSPARSE/opSparse/HSMU/Ocean(+ Ours 绝对 ms)。

用法:.venv/bin/python scripts/gen_speedup_table.py [methods_cmp.csv] [> table.tex]
"""
import os, sys, csv, math

CSV = sys.argv[1] if len(sys.argv) > 1 else "compare/methods_cmp.csv"
# (display, csv column, latex label, year/origin)
BASELINES = [
    ("dense",    "dense",    "dense",    "---"),       # naive baseline(非论文)
    ("cuBLAS",   "cublas",   "cuBLAS",   "NVIDIA"),    # cuBLAS FP64 dgemm 库(PEDANTIC 无 TC)
    ("cuSPARSE", "cu",       "cuSPARSE", "NVIDIA"),    # 库
    ("opSparse", "opSparse", "OpSparse", "2022"),      # Liu 2022
    ("HSMU",     "HSMU",     "HSMU",     "2025"),      # Wu 2025
    ("Ocean",    "Ocean",    "Ocean",    "2026"),      # Li, ICS'26
]
CLASS_ORDER = [("Dense", "Dense"), ("Mildly sparse", "Mild-Sparse"),
               ("Highly sparse", "High-Sparse"), ("Extremely sparse", "Extreme-Sparse")]

def f(x):
    try: return float(x)
    except: return None

def geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def classify(d):
    if d >= 10.0: return "Dense"
    if d >= 1.0:  return "Mildly sparse"
    if d >= 0.1:  return "Highly sparse"
    return "Extremely sparse"

def cell(rows, col):
    """geomean speedup (baseline/Auto) + (win/total) + baseline geomean ms."""
    sp, win, tot, bms = [], 0, 0, []
    for r in rows:
        a, b = f(r.get("Auto")), f(r.get(col))
        if a and b and a > 0 and b > 0:
            sp.append(b / a); bms.append(b); tot += 1
            if a < b: win += 1      # Auto faster
    return geomean(sp), win, tot, geomean(bms)

def fmt_sp(v):
    if v != v: return "—"
    if v >= 100: return f"{v:.0f}"
    if v >= 10:  return f"{v:.1f}"
    return f"{v:.2f}"

def cell_str(sp, bms, is_best):
    r"""双行格子:加速比(上,最优加粗)+ baseline geomean ms(下,小字)。需 \usepackage{makecell}。"""
    if sp != sp: return "---"
    sp_part = rf"\textbf{{{fmt_sp(sp)}$\times$}}" if is_best else rf"{fmt_sp(sp)}$\times$"
    return rf"\makecell{{{sp_part}\\ \scriptsize {fmt_sp(bms)} ms}}"

def wincell(win, tot):
    r"""win/total 带 cellcolor:全胜=绿、多数(≥0.8)=黄、有输=红。需 \usepackage[table]{xcolor}。"""
    if tot == 0: return "---"
    frac = win / tot
    if win == tot:          col = "green!35"
    elif frac >= 0.8:       col = "yellow!50"
    else:                   col = "red!35"
    return rf"\cellcolor{{{col}}}{win}/{tot}"

def main():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        r["_c"] = classify(f(r.get("density_pct")) or 0)
    # 每类每基线的 (speedup, win, total)
    cells = {}   # (row_label, class_key) -> (sp, win, tot)
    for disp, col, lab, year in BASELINES:
        for cname, _ in CLASS_ORDER:
            sub = [r for r in rows if r["_c"] == cname]
            cells[(lab, cname)] = cell(sub, col)
        cells[(lab, "ALL")] = cell(rows, col)
    # 每列最优 speedup(加粗)
    best = {}
    for ck in [c[0] for c in CLASS_ORDER] + ["ALL"]:
        vals = {lab: cells[(lab, ck)][0] for _, _, lab, _ in BASELINES if cells[(lab, ck)][0] == cells[(lab, ck)][0]}
        best[ck] = max(vals, key=vals.get) if vals else None
    # Ours geomean ms per class
    ours_ms = {}
    for cname, _ in CLASS_ORDER:
        ours_ms[cname] = geomean([f(r["Auto"]) for r in rows if r["_c"] == cname and f(r["Auto"])])
    ours_ms["ALL"] = geomean([f(r["Auto"]) for r in rows if f(r["Auto"])])

    # 类别表头计数
    cnt = {c: sum(1 for r in rows if r["_c"] == c) for c, _ in CLASS_ORDER}
    hdr_classes = [(ck, cnt[ck]) for ck, _ in CLASS_ORDER]

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\setlength{\tabcolsep}{5pt}")
    L.append(r"\renewcommand{\arraystretch}{1.2}")
    L.append(r"\begin{tabular}{l c " + " ".join(["c"] * (len(hdr_classes) + 1)) + r" c}")
    L.append(r"\toprule")
    hdr = (r"\textbf{Method} & \textbf{Year} & "
           + " & ".join(rf"\textbf{{{dn}({n})}}" for dn, n in hdr_classes)
           + rf" & \textbf{{ALL({len(rows)})}} & \textbf{{Wins}}\\")
    L.append(hdr + r" \hline")
    for disp, col, lab, year in BASELINES:
        parts = [lab, year]
        for ck, _ in hdr_classes:
            sp, win, tot, bms = cells[(lab, ck)]
            parts.append(cell_str(sp, bms, best.get(ck) == lab))
        sp, win, tot, bms = cells[(lab, "ALL")]
        parts.append(cell_str(sp, bms, best.get("ALL") == lab))
        parts.append(f"{win}/{tot}")            # Wins 列(整体 win/total)
        L.append(" & ".join(parts) + r" \\")
    # Ours 绝对 ms 行(Auto 时间量级参考)
    parts = [r"\textit{Ours (ms)}", ""]
    for ck, _ in hdr_classes:
        parts.append(rf"\textit{{{fmt_sp(ours_ms[ck])}}}")
    parts.append(rf"\textit{{{fmt_sp(ours_ms['ALL'])}}}")
    parts.append("")
    L.append(" & ".join(parts) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    cap = (r"% Speedup of Auto (ours) over each baseline (baseline\_time / Auto\_time, geomean). "
           r"每格上行 = 加速比(>1 = Auto 快,最优加粗),下行 = 该 baseline 几何均值 ms。"
           r"Wins = Auto 整体快过的阵数/100。末行 = Auto 几何均值 ms。"
           r"需 \usepackage{makecell}。")
    print(cap)
    print("\n".join(L))

    # ===================== 表 2:win-matrix 热力图(per-class win/total + cellcolor)=====================
    M = []
    M.append(r"% 需 \usepackage[table]{xcolor}(\cellcolor)。绿=Auto 全胜、黄=多数(≥80%)、红=有明显输。")
    M.append(r"\begin{table}[t]")
    M.append(r"\centering")
    M.append(r"\setlength{\tabcolsep}{6pt}")
    M.append(r"\renewcommand{\arraystretch}{1.25}")
    M.append(r"\begin{tabular}{l " + " ".join(["c"] * (len(hdr_classes) + 1)) + "}")
    M.append(r"\toprule")
    M.append(r"\textbf{Auto wins over} & "
             + " & ".join(rf"\textbf{{{dn}({n})}}" for dn, n in hdr_classes)
             + rf" & \textbf{{ALL({len(rows)})}}\\")
    M.append(r"\hline")
    for disp, col, lab, year in BASELINES:
        parts = [f"{lab} ({year})" if year != "---" else lab]
        for ck, _ in hdr_classes:
            _, win, tot, _ = cells[(lab, ck)]
            parts.append(wincell(win, tot))
        _, win, tot, _ = cells[(lab, "ALL")]
        parts.append(wincell(win, tot))
        M.append(" & ".join(parts) + r" \\")
    M.append(r"\bottomrule")
    M.append(r"\end{tabular}")
    M.append(r"% Win-matrix:每格 = Auto 快过的阵数/该类总数。"
             r"\cellcolor{green!35}=全胜,\cellcolor{yellow!50}=≥80\%,\cellcolor{red!35}=有明显输。")
    print()
    print("\n".join(M))


if __name__ == "__main__":
    main()
