#!/usr/bin/env python3
"""v24(DCFUSE)干净分析:vs v9 Ocean 列的 geomean/赢面/带结构/相对 v23 的变化。"""
import csv, math, sys

def load(p): return {r['matrix']: r for r in csv.DictReader(open(p))}
def f(x):
    try: return float(x)
    except: return None
def gm(xs): xs=[x for x in xs if x and x>0]; return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float('nan')

BASE = 'compare/ocean337/'
v24 = load(BASE + 'methods_cmp_v24_dcfuse.csv')
v23 = load(BASE + 'methods_cmp_v23_sortfree.csv')
v9  = load(BASE + 'methods_cmp_v9_oceansym.csv')

pairs = []
for m, r in v24.items():
    a, o = f(r['Auto']), f(v9.get(m, {}).get('Ocean'))
    a23 = f(v23.get(m, {}).get('Auto'))
    if a and o and o > 0:
        r23 = a23 / o if a23 else None
        pairs.append((m, a, o, a / o, r23, (a / a23 - 1) * 100 if a23 else None))

print(f"可比 {len(pairs)} 阵:geomean v24 = {gm([p[3] for p in pairs]):.4f}×  "
      f"赢 {sum(1 for p in pairs if p[3] < 1)} (v23 赢 {sum(1 for p in pairs if p[4] and p[4] < 1)})")
d = [(p[0], p[5]) for p in pairs if p[5] is not None and abs(p[5]) > 8]
if d:
    print(f"\n|Δ|>8% 的阵({len(d)};若非 DCFUSE 目标阵 = 疑污染,复核):")
    for m, dd in sorted(d, key=lambda x: x[1]):
        print(f"  {m[:24]:24s} {dd:+7.1f}%")
# 带结构 + 翻面清单
flips_win = [p for p in pairs if p[3] < 1 and p[4] and p[4] >= 1]
flips_lose = [p for p in pairs if p[3] >= 1 and p[4] and p[4] < 1]
print(f"\n新翻赢({len(flips_win)}):")
for m, a, o, r, a23, _ in sorted(flips_win, key=lambda p: p[3]):
    print(f"  {m[:24]:24s} {a23:8.2f}→{a:8.2f} vs Ocean {o:8.2f} = {r:.3f}×")
print(f"跌出赢({len(flips_lose)}):")
for m, a, o, r, a23, _ in sorted(flips_lose, key=lambda p: -p[3]):
    print(f"  {m[:24]:24s} {a23:8.2f}→{a:8.2f} vs Ocean {o:8.2f} = {r:.3f}×")
for lo, hi, tag in [(1.0, 1.5, '近差<1.5'), (1.5, 2.5, '中差'), (2.5, 99, '大差>2.5')]:
    sel = [p for p in pairs if lo <= p[3] < hi]
    if sel:
        print(f"  {tag:10s} {len(sel):3d} 阵 geomean {gm([p[3] for p in sel]):.3f}×")
