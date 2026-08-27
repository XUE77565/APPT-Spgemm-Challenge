#!/usr/bin/env python3
"""重跑 refresh_ocean_sym 批首被 stale-stats 污染的阵(Ocean 列值恰好 4.027 的 43 阵)。
用修好的 run_ocean(删旧 stats + 查 rc)逐阵覆写 CSV 的 Ocean 列。"""
import csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import compare_methods as cm

CSV = "compare/ocean337/methods_cmp_v9_oceansym.csv"
rows = list(csv.DictReader(open(CSV)))
fields = rows[0].keys()
bad = [i for i, r in enumerate(rows)
       if r["Ocean"] and abs(float(r["Ocean"]) - 4.027) < 0.001]
print(f"待重跑 {len(bad)} 阵 → {CSV}", flush=True)
for k, i in enumerate(bad):
    r = rows[i]
    p = cm.find_mtx(r["matrix"])
    if not p:
        print(f"[{k+1}/{len(bad)}] {r['matrix']}: 未找到", flush=True); continue
    t0 = time.time()
    v = cm.run_ocean(p)
    r["Ocean"] = round(v, 3) if v is not None else "DNF"
    print(f"[{k+1}/{len(bad)}] {r['matrix']:24s} Ocean={r['Ocean']}  ({time.time()-t0:.1f}s)", flush=True)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print("done", flush=True)
