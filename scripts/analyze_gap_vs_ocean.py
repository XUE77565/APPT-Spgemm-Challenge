#!/usr/bin/env python3
"""全量 337 阵 Auto vs Ocean 差距结构分析。

用法: python3 scripts/analyze_gap_vs_ocean.py [csv路径]
输出: 总体统计 / 尺寸分桶 / 簇分析 / 头部空间 / 赢阵清单
"""
import csv, math, sys, re
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "compare/ocean337/methods_cmp_clean.csv.csv"

def f(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None

rows = []
for r in csv.DictReader(open(CSV)):
    auto, ocean = f(r.get("Auto")), f(r.get("Ocean"))
    if auto and ocean:
        r["_auto"], r["_ocean"], r["_ratio"] = auto, ocean, auto / ocean
        r["_n"] = f(r.get("n")) or 0
        r["_cnnz"] = f(r.get("cnnz")) or 0
        r["_crow"] = r["_cnnz"] / r["_n"] if r["_n"] else 0  # 输出平均行长
        rows.append(r)

n_all = len(rows)
ratios = [r["_ratio"] for r in rows]
gm = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
wins = sum(1 for x in ratios if x < 1)
print(f"=== 总体(有双方数据 {n_all} 阵)===")
print(f"geomean(Auto/Ocean) = {gm:.3f}×   赢 {wins} / 输 {n_all - wins}")
print(f"中位数 = {sorted(ratios)[len(ratios)//2]:.2f}×   最差 = {max(ratios):.1f}×   最好 = {min(ratios):.3f}×")

# ---- 尺寸分桶(按 n) ----
buckets = [(0, 10e3, "<10k"), (10e3, 50e3, "10k-50k"), (50e3, 200e3, "50k-200k"),
           (200e3, 2e6, "200k-2M"), (2e6, 1e18, ">2M")]
print("\n=== 按 n 分桶 ===")
print(f"{'桶':>10} {'阵数':>4} {'geomean':>8} {'赢':>3} {'输':>4}  最差3阵")
for lo, hi, label in buckets:
    grp = [r for r in rows if lo <= r["_n"] < hi]
    if not grp:
        continue
    g = math.exp(sum(math.log(r["_ratio"]) for r in grp) / len(grp))
    w = sum(1 for r in grp if r["_ratio"] < 1)
    worst = sorted(grp, key=lambda r: -r["_ratio"])[:3]
    wnames = ", ".join(f"{r['matrix']}({r['_ratio']:.1f})" for r in worst)
    print(f"{label:>10} {len(grp):>4} {g:>7.2f}× {w:>3} {len(grp)-w:>4}  {wnames}")

# ---- 按 C 平均行长(输出局部性) ----
print("\n=== 按输出平均行长 C_nnz/n ===")
b2 = [(0, 100, "<100"), (100, 500, "100-500"), (500, 2000, "500-2k"),
      (2000, 10000, "2k-10k"), (10000, 1e18, ">10k")]
for lo, hi, label in b2:
    grp = [r for r in rows if lo <= r["_crow"] < hi]
    if not grp:
        continue
    g = math.exp(sum(math.log(r["_ratio"]) for r in grp) / len(grp))
    w = sum(1 for r in grp if r["_ratio"] < 1)
    print(f"{label:>10} {len(grp):>4}  geomean {g:>6.2f}×  赢{w}输{len(grp)-w}")

# ---- 名称簇(SAT/电路/网格/图) ----
def cluster(name):
    if name.startswith("c-") or name.startswith("c_"): return "SAT(c-*)"
    if "blowey" in name: return "blowey*"
    if "mult_dcop" in name: return "mult_dcop"
    if name.startswith("TSOPF"): return "TSOPF"
    if re.match(r"^(Ga|Ge|Si|aSi)", name): return "DFT(Ga/Ge/Si)"
    if re.search(r"(FEM|Flan|Emilia|Fault|Bump|Stoke|Sync|transient|therm|oil|cage|Pres_Poisson|bdwcg| Inline_1|robot|HPCG|pwtk|af_.*|kkt|nx |ASIC)", name): return None  # 杂项留给通用
    if re.search(r"(graph|email|web|cit|p2p|soc|wiki|road|europe|osm|belgium|germany|lux|netherlands|rgg| delaunay|ca-|loc|huck|USA)", name, re.I): return "图/网络"
    return None

# ---- top losers 表 ----
print("\n=== Top 40 输阵(按 ratio)===")
print(f"{'matrix':<24} {'n':>9} {'C_nnz':>11} {'C行均长':>8} {'Auto(ms)':>9} {'Ocean(ms)':>9} {'ratio':>6}")
for r in sorted(rows, key=lambda r: -r["_ratio"])[:40]:
    print(f"{r['matrix']:<24} {int(r['_n']):>9} {int(r['_cnnz']):>11} {r['_crow']:>8.0f} "
          f"{r['_auto']:>9.2f} {r['_ocean']:>9.2f} {r['_ratio']:>5.1f}×")

# ---- 头部空间:把某簇修到 1× 后的全局 geomean ----
print("\n=== 头部空间(把某组修到 1× 后的全局 geomean)===")
def gm_if(pred, label):
    grp = [r for r in rows if pred(r)]
    if not grp:
        return
    logs = [math.log(1.0 if pred(r) else r["_ratio"]) for r in rows]
    g = math.exp(sum(logs) / len(logs))
    print(f"  {label:<28} {len(grp):>3} 阵  → geomean {g:.3f}×  (省 {gm-g:.3f})")

top40 = set(r["matrix"] for r in sorted(rows, key=lambda r: -r["_ratio"])[:40])
gm_if(lambda r: r["matrix"] in top40, "top40 全修")
gm_if(lambda r: r["_ratio"] >= 5, "ratio≥5 全修")
gm_if(lambda r: 2 <= r["_ratio"] < 5, "2≤ratio<5 修到 1×")
gm_if(lambda r: r["_ratio"] < 2, "ratio<2 修到 1×")
gm_if(lambda r: cluster(r["matrix"]) == "SAT(c-*)", "SAT c-* 簇")
gm_if(lambda r: cluster(r["matrix"]) == "blowey*", "blowey 簇")
gm_if(lambda r: r["_n"] <= 80000 and r["_crow"] >= 500, "中尺寸(n≤80k,C行长≥500)")

# ---- 赢阵 ----
print("\n=== 赢阵清单(全部)===")
for r in sorted(rows, key=lambda r: r["_ratio"]):
    if r["_ratio"] < 1:
        print(f"  {r['matrix']:<24} n={int(r['_n']):>8}  Auto {r['_auto']:>8.2f} vs Ocean {r['_ocean']:>8.2f}  = {r['_ratio']:.3f}×")

# ---- 中尺寸 c-*/blowey 结构画像 ----
print("\n=== 新 top 簇画像(n 30k-80k 不规则)===")
mid = [r for r in rows if 20000 <= r["_n"] <= 120000 and r["_ratio"] >= 5]
if mid:
    print(f"{'matrix':<24} {'n':>8} {'C_nnz':>11} {'C行均长':>8} {'ratio':>6} {'Auto':>8} {'Ocean':>8}")
    for r in sorted(mid, key=lambda r: -r["_ratio"]):
        print(f"{r['matrix']:<24} {int(r['_n']):>8} {int(r['_cnnz']):>11} {r['_crow']:>8.0f} {r['_ratio']:>5.1f}× {r['_auto']:>8.1f} {r['_ocean']:>8.1f}")
