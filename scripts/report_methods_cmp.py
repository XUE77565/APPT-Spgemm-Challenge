#!/usr/bin/env python3
"""读 methods_cmp.csv → 仿 full_compare 格式的报告(基线 = cuSPARSE / opSparse / HSMU / dense):
  1) 每矩阵表(class/name/n + 各法 compute-only + Auto 对各法比值),按类别排序
  2) 按类别聚合(几何均值 + Auto 对各法比值均值)
  3) Auto vs cuSPARSE / opSparse / HSMU / dense 的赢/输(ratio>1 降序,赢/输计数 + 几何均值 + 输的清单)
所有 spgemm 法时间 = compute-only(排除 h2d/d2h);dense=naive scalar GEMM kernel time;HSMU=time2;opSparse=total。
用法:report_methods_cmp.py <methods_cmp.csv> [out.txt]
"""
import os, sys, csv, math

CLASS_ORDER = ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]
CLASS_TAG = {"Dense": "Dens", "Mildly sparse": "Mild", "Highly sparse": "High", "Extremely sparse": "Extr"}

def classify(d):
    try: d = float(d)
    except: return "?"
    if d >= 10.0: return "Dense"
    if d >= 1.0:  return "Mildly sparse"
    if d >= 0.1:  return "Highly sparse"
    return "Extremely sparse"

def fnum(r, key):
    v = r.get(key, "")
    try: return float(v)
    except: return None

def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def fmt(v, w=7, p=2):
    return f"{v:>{w}.{p}f}" if v is not None and v == v else f"{'-':>{w}}"

def ratio_gmean(rows, col, ref):
    xs = []
    for r in rows:
        a, b = fnum(r, col), fnum(r, ref)
        if a and b and b > 0: xs.append(a / b)
    return gmean(xs)

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "compare/methods_cmp.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(csv_path)[0] + "_report.txt"
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["_class"] = classify(r.get("density_pct", ""))
        r["_n"] = int(r["n"]) if r.get("n", "").isdigit() else 0
    rows.sort(key=lambda r: (CLASS_ORDER.index(r["_class"]) if r["_class"] in CLASS_ORDER else 99, r["_n"]))

    REFS = [("cu", "cu"), ("opSparse", "opSp"), ("HSMU", "HSMU"), ("dense", "dense")]
    out = []
    out.append("[compute-only:cudaEvent 纯 GPU(去边界 h2d/d2h);dense=naive scalar GEMM kernel;HSMU=time2;opSparse=total]  ms")
    out.append(f"{'cls':>4} {'name':<24}{'n':>7}{'cu':>9}{'opSp':>9}{'HSMU':>9}{'dense':>10}{'Auto':>9}"
               + "".join(f"{'A/'+short:>8}" for _, short in REFS))
    out.append("-" * 110)
    for r in rows:
        vals = {k: fnum(r, k) for k in ("cu", "opSparse", "HSMU", "dense", "Auto")}
        au = vals["Auto"]
        out.append(f"{CLASS_TAG.get(r['_class'],'?'):>4} {r['matrix']:<24}{r['_n']:>7}"
                   f"{fmt(vals['cu'],9,2)}{fmt(vals['opSparse'],9,2)}{fmt(vals['HSMU'],9,2)}"
                   f"{fmt(vals['dense'],10,2)}{fmt(au,9,2)}"
                   + "".join(fmt(au / vals[k] if (au and vals[k] and vals[k] > 0) else None, 8, 2) for k, _ in REFS))

    # ---- 按类别聚合 ----
    out.append(""); out.append("=" * 110)
    out.append("按类别聚合(几何均值 ms;A/X = Auto 对该法比值均值,<1 = Auto 快)")
    out.append("-" * 110)
    out.append(f"{'class':<20}{'#':>4}{'cu':>9}{'opSp':>9}{'HSMU':>9}{'dense':>10}{'Auto':>9}"
               + "".join(f"{'A/'+short:>8}" for _, short in REFS))
    for c in CLASS_ORDER:
        sub = [r for r in rows if r["_class"] == c]
        if not sub: continue
        gm = {k: gmean([fnum(r, k) for r in sub if fnum(r, k)]) for k in ("cu", "opSparse", "HSMU", "dense", "Auto")}
        out.append(f"{c:<20}{len(sub):>4}{fmt(gm['cu'],9,2)}{fmt(gm['opSparse'],9,2)}{fmt(gm['HSMU'],9,2)}"
                   f"{fmt(gm['dense'],10,2)}{fmt(gm['Auto'],9,2)}"
                   + "".join(fmt(gm['Auto'] / gm[k] if (gm['Auto'] == gm['Auto'] and gm[k] == gm[k] and gm[k] > 0) else None, 8, 2)
                             for k, _ in REFS))
    # 总体
    gm = {k: gmean([fnum(r, k) for r in rows if fnum(r, k)]) for k in ("cu", "opSparse", "HSMU", "dense", "Auto")}
    out.append(f"{'ALL('+str(len(rows))+')':<20}{len(rows):>4}{fmt(gm['cu'],9,2)}{fmt(gm['opSparse'],9,2)}{fmt(gm['HSMU'],9,2)}"
               f"{fmt(gm['dense'],10,2)}{fmt(gm['Auto'],9,2)}"
               + "".join(fmt(gm['Auto'] / gm[k] if (gm['Auto'] == gm['Auto'] and gm[k] == gm[k] and gm[k] > 0) else None, 8, 2)
                         for k, _ in REFS))

    # ---- Auto vs 各基线 赢/输 ----
    def vs_section(ref_col, ref_name):
        pairs = []
        for r in rows:
            t, ref = fnum(r, "Auto"), fnum(r, ref_col)
            if t and ref and ref > 0:
                pairs.append((r, t / ref))
        if not pairs:
            return
        win = [p for p in pairs if p[1] <= 1.0]
        lose = sorted([p for p in pairs if p[1] > 1.0], key=lambda p: -p[1])
        out.append(""); out.append("=" * 100)
        out.append(f"Auto vs {ref_name}:赢 {len(win)} / 输 {len(lose)},几何均值(Auto/{ref_name}) {gmean([p[1] for p in pairs]):.3f}×")
        out.append("-" * 100)
        out.append(f"{'name':<16}{'n':>8}{'C_nnz':>12}{ref_name:>12}{'Auto':>12}{'ratio':>9}")
        for r, ratio in lose[:20]:
            out.append(f"{r['matrix']:<16}{r['_n']:>8}{r.get('cnnz',''):>12}"
                       f"{fmt(fnum(r, ref_col),12,3)}{fmt(fnum(r, 'Auto'),12,3)}{ratio:>8.2f}×")
        if len(lose) > 20:
            out.append(f"  ... 另有 {len(lose)-20} 个")

    DISPLAY = {"cu": "cuSPARSE", "opSparse": "opSparse", "HSMU": "HSMU", "dense": "dense"}
    for col, _ in REFS:
        vs_section(col, DISPLAY[col])

    report = "\n".join(out) + "\n"
    open(out_path, "w").write(report)
    print(report)
    print(f"报告写出: {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
