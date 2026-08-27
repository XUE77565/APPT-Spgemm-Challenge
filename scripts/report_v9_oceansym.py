#!/usr/bin/env python3
"""v9 Ocean 列(含 symbolic,stale 已修复)终版分析:双口径 geomean + symbolic 效应分布 + top losers。"""
import csv, math, sys

OLD = 'compare/ocean337/methods_cmp_v8_v4gate.csv'      # 不含 symbolic(08-26/27 各次直测混合)
NEW = 'compare/ocean337/methods_cmp_v9_oceansym.csv'    # 含 symbolic(08-27 夜,权威)

def load(p): return {r['matrix']: r for r in csv.DictReader(open(p))}
def f(x):
    try: return float(x)
    except: return None
def gm(xs): xs=[x for x in xs if x and x>0]; return math.exp(sum(math.log(x) for x in xs)/len(xs))

old, new = load(OLD), load(NEW)
pairs = []
for m, r in new.items():
    a, n, o = f(r['Auto']), f(r['Ocean']), f(old.get(m, {}).get('Ocean'))
    if a and n: pairs.append((m, a, n, o))

print(f"== v9(含 symbolic,stale 修复后)==")
print(f"可比 {len(pairs)} 阵:geomean Auto/Ocean = {gm([p[1]/p[2] for p in pairs]):.4f}×  "
      f"(赢 {sum(1 for p in pairs if p[1]<p[2])} / 输 {sum(1 for p in pairs if p[1]>=p[2])})")
# 不含 symbolic 口径:旧列数字(它本身不含 symbolic)。⚠ 旧列自身的 stale:wb-edu=1.469
# (echo water_tank,所有历史 CSV 同病)→ 从该口径剔除。
OLD_STALE = {'wb-edu'}
print(f"对照旧列(不含 symbolic,同一 Auto,剔 {OLD_STALE}):geomean = "
      f"{gm([p[1]/p[3] for p in pairs if p[3] and p[0] not in OLD_STALE]):.4f}×  "
      f"({sum(1 for p in pairs if p[3] and p[0] not in OLD_STALE)} 阵)")
# symbolic 效应(干净子集:两列都非 stale —— 新列 != 旧列的符号性变化大多来自 symbolic)
clean = [p for p in pairs if p[3] and abs(p[2]-p[3]) > 0.01]
sym_up = [p for p in clean if p[2] > p[3]]
print(f"\nsymbolic 计入后 Ocean 变慢的阵:{len(sym_up)} / 变化>1% 共 {len(clean)};中位 +{sorted((p[2]-p[3])/p[3] for p in sym_up)[len(sym_up)//2]*100 if sym_up else 0:.0f}%")
# top losers(新口径)
pairs.sort(key=lambda p: -p[1]/p[2])
print("\n== top-20 losers(Auto/Ocean,含 symbolic 新口径)==")
for m, a, n, o in pairs[:20]:
    print(f"  {m[:28]:28s} Auto={a:9.2f}  Ocean={n:8.2f}  ratio={a/n:6.2f}×  (旧Ocean={o if o else 'NA'})")
# 头部空间
r2050 = [p for p in pairs if 2 <= p[1]/p[2] < 5]; r50 = [p for p in pairs if p[1]/p[2] >= 5]
tot = gm([p[1]/p[2] for p in pairs])
print(f"\n结构:≥5× {len(r50)} 阵(占 {math.log(tot) and 0 or 0:.0f}%),2-5× {len(r2050)} 阵,<2× {len(pairs)-len(r2050)-len(r50)} 阵")
if r50:
    gm_all = gm([p[1]/p[2] for p in pairs]); gm_no50 = gm([p[1]/p[2] for p in pairs if p[1]/p[2] < 5])
    print(f"全部 1× 的 geomean = {gm_no50:.3f}×(≥5× 的 {len(r50)} 阵贡献头部空间 {gm_all/gm_no50:.3f}×)")

# docs/20 同款分桶表(新口径重算)
def bucket(rows_by, edges, get):
    print(f"\n== 按 {rows_by} 分桶(新口径)==")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [p for p in pairs if lo <= get(p) < hi]
        if sel:
            print(f"| {lo}-{hi} | {len(sel)} | {gm([x[1]/x[2] for x in sel]):.2f}× |")
nn = {r['matrix']: f(r['n']) for r in new.values()}
cnnz = {r['matrix']: f(r.get('cnnz')) for r in new.values()}
bucket('n', [0, 10000, 50000, 200000, 2000000, 10**12], lambda p: nn.get(p[0]) or 0)
print("| (按 C 行长 = cnnz/n) |")
bucket('C行长', [0, 100, 500, 2000, 10000, 10**12],
       lambda p: (cnnz.get(p[0]) or 0) / max(nn.get(p[0]) or 1, 1))
