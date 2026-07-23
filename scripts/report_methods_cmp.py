#!/usr/bin/env python3
"""读 methods_cmp.csv → 仿 full_compare_K=5.txt 格式的报告:
  1) 每矩阵表(class/name/n + 各法 compute-only + 比值),按类别排序
  2) 按类别聚合(均值 + 比值均值)
  3) Auto / m3 vs Ocean + vs HSMU 的赢/输(ratio>1 降序,赢/输计数 + 几何均值)
所有 spgemm 法时间 = compute-only(排除 h2d/d2h,由 compare_methods.py 解析);Ocean=GPU 阶段求和;HSMU=time2。
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

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

def fmt(v, w=7, p=2):
    return f"{v:>{w}.{p}f}" if v is not None and v == v else f"{'-':>{w}}"

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "compare/methods_cmp.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(csv_path)[0] + "_report.txt"
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["_class"] = classify(r.get("density_pct", ""))
        r["_n"] = int(r["n"]) if r.get("n", "").isdigit() else 0
    # 按 (class_order, n) 排序
    rows.sort(key=lambda r: (CLASS_ORDER.index(r["_class"]) if r["_class"] in CLASS_ORDER else 99, r["_n"]))

    out = []
    out.append("[compute-only:cudaEvent 纯 GPU(去边界 h2d/d2h,同 Ocean 口径);dense=dense baseline(-O0) cudaEvent kernel]  ms")
    out.append("class name                        n  dense   Ocean    HSMU     m3    Auto Auto/Oce Auto/base  m3/Oce")
    out.append("-" * 116)
    for r in rows:
        base, oc, hs, m3, au = (fnum(r, k) for k in ("baseline", "Ocean", "HSMU", "merge3", "Auto"))
        n_str = f"{r['_n']:,}"
        def ratio(a, b): return a / b if (a and b and b > 0) else None
        out.append(f"{CLASS_TAG.get(r['_class'],'?'):>5} {r['matrix']:<24}{n_str:>7}"
                   f"{fmt(base)}{fmt(oc)}{fmt(hs)}{fmt(m3)}{fmt(au)}"
                   f"{fmt(ratio(au, oc), 7, 2)}{fmt(ratio(au, base), 7, 2)}{fmt(ratio(m3, oc), 7, 2)}")

    # ---- 按类别聚合 ----
    out.append("")
    out.append("=" * 116)
    out.append("按类别聚合(compute-only 均值,毫秒;比值为该类各阵比值的均值)")
    out.append("-" * 116)
    out.append(f"{'class':<20}{'#':>4}{'Ocean':>9}{'HSMU':>9}{'dense':>10}{'m3':>8}{'Auto':>8}{'Auto/Oce':>10}{'Auto/base':>9}{'m3/Oce':>9}")
    for c in CLASS_ORDER:
        sub = [r for r in rows if r["_class"] == c]
        if not sub: continue
        oc = mean([fnum(r, "Ocean") for r in sub]); hs = mean([fnum(r, "HSMU") for r in sub])
        base = mean([fnum(r, "baseline") for r in sub])
        m3 = mean([fnum(r, "merge3") for r in sub]); au = mean([fnum(r, "Auto") for r in sub])
        ra = mean([fnum(r, "Auto") / fnum(r, "Ocean") for r in sub
                   if fnum(r, "Auto") and fnum(r, "Ocean")])
        rab = mean([fnum(r, "Auto") / fnum(r, "baseline") for r in sub
                    if fnum(r, "Auto") and fnum(r, "baseline")])
        rmo = mean([fnum(r, "merge3") / fnum(r, "Ocean") for r in sub
                    if fnum(r, "merge3") and fnum(r, "Ocean")])
        out.append(f"{c:<20}{len(sub):>4}{fmt(oc,9,2)}{fmt(hs,9,2)}{fmt(base,10,2)}"
                   f"{fmt(m3,8,2)}{fmt(au,8,2)}{fmt(ra,10,2)}{fmt(rab,9,2)}{fmt(rmo,9,2)}")

    # ---- vs 参照法(Ocean / HSMU)的赢/输 ----
    def vs_section(label, col, ref_col, ref_name):
        pairs = []
        for r in rows:
            t = fnum(r, col); ref = fnum(r, ref_col)
            if t and ref and ref > 0:
                pairs.append((r, t / ref))
        if not pairs:
            return
        win = [p for p in pairs if p[1] <= 1.0]      # 我方 ≤ 参照 = 赢
        lose = [p for p in pairs if p[1] > 1.0]       # 我方 > 参照 = 输
        lose.sort(key=lambda p: -p[1])
        out.append("")
        out.append("=" * 100)
        out.append(f"{label} vs {ref_name}:赢 {len(win)} / 输 {len(lose)},几何均值 {gmean([p[1] for p in pairs]):.3f}×")
        out.append("-" * 100)
        out.append(f"{'name':<16}{'n':>8}{'C_nnz':>10}{ref_name:>10}{label:>14}{'ratio':>9}")
        for r, ratio in lose[:20]:
            cnnz = r.get("cnnz", "")
            out.append(f"{r['matrix']:<16}{r['_n']:>8}{cnnz:>10}{fmt(fnum(r,ref_col),10,3)}{fmt(fnum(r,col),14,3)}{ratio:>8.2f}×")
        if len(lose) > 20:
            out.append(f"  ... 另有 {len(lose)-20} 个")

    vs_section("Auto", "Auto", "Ocean", "Ocean")
    vs_section("m3",   "merge3", "Ocean", "Ocean")
    vs_section("Auto", "Auto", "baseline", "baseline")

    report = "\n".join(out) + "\n"
    open(out_path, "w").write(report)
    print(report)
    print(f"报告写出: {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
