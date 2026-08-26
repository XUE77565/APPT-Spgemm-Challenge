#!/usr/bin/env python3
"""干净基准分析:Auto vs Ocean 全景 + 分档 + 输阵 Top + 类特征。用法:
  .venv/bin/python scripts/analyze_clean_bench.py [csv路径]
"""
import csv, math, sys, os

csv_path = sys.argv[1] if len(sys.argv) > 1 else "compare/ocean337/methods_cmp_v2.csv"
rows = [r for r in csv.DictReader(open(csv_path))]

def f(x):
    try: return float(x)
    except: return None

def gm(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

# 总览
pairs = []
for r in rows:
    a, o = f(r.get("Auto")), f(r.get("Ocean"))
    if a and o and a > 0 and o > 0:
        pairs.append((r["matrix"], int(r["n"] or 0), a, o, a / o, r.get("Auto_choice", "?")))
win = [p for p in pairs if p[4] <= 1.0]
lose = sorted([p for p in pairs if p[4] > 1.0], key=lambda p: -p[4])
print(f"== Auto vs Ocean(干净直测,{csv_path}) ==")
print(f"有效对阵 {len(pairs)} | 赢 {len(win)} | 输 {len(lose)} | geomean(Auto/Ocean) {gm([p[4] for p in pairs]):.3f}×")
buckets = [("<1万", lambda n: n < 10000), ("1-10万", lambda n: 10000 <= n < 100000),
           ("10-100万", lambda n: 100000 <= n < 1000000), (">100万", lambda n: n >= 1000000)]
print("\n== 按 n 分档 ==")
for name, pred in buckets:
    v = [p for p in pairs if pred(p[1])]
    if v:
        print(f"  {name:8} {len(v):3} 阵  gm {gm([x[4] for x in v]):6.2f}×  赢 {sum(1 for x in v if x[4] <= 1)}/{len(v)}")
print("\n== 输阵 Top 20(优化目标清单) ==")
print(f"  {'matrix':<32}{'n':>9}{'Auto':>9}{'Ocean':>9}{'ratio':>8}  choice")
for m, n, a, o, rt, ch in lose[:20]:
    print(f"  {m:<32}{n:>9}{a:>9.2f}{o:>9.2f}{rt:>7.2f}×  {ch}")
print(f"\n== 赢阵 Top 5 ==")
for m, n, a, o, rt, ch in sorted(pairs, key=lambda p: p[4])[:5]:
    print(f"  {m:<32}{n:>9}{a:>9.2f}{o:>9.2f}{rt:>7.2f}×  {ch}")
# merge3 被 Auto 选中的情况
m3 = [p for p in pairs if p[5].startswith("merge3") and "回退" not in p[5]]
print(f"\n== dispatcher 主动选 merge3:{len(m3)} 阵 ==")
for m, n, a, o, rt, ch in m3[:10]:
    print(f"  {m:<32} ratio {rt:.2f}×")
