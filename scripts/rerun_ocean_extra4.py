#!/usr/bin/env python3
"""补跑 4 个疑似 stale(与前一矩阵完全同值,含 wb-edu 1.462=water_tank echo)。
在 rerun_ocean_bogus.py(43 阵)完成后运行。"""
import csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
import compare_methods as cm

CSV = "compare/ocean337/methods_cmp_v9_oceansym.csv"
NAMES = ["af_3_k101", "atmosmodj", "conf5_4-8x8-15", "wb-edu"]
rows = list(csv.DictReader(open(CSV)))
fields = rows[0].keys()
by_name = {r["matrix"]: r for r in rows}
for k, name in enumerate(NAMES):
    r = by_name.get(name)
    if not r:
        print(f"[{k+1}/4] {name}: CSV 无此阵", flush=True); continue
    p = cm.find_mtx(name)
    if not p:
        print(f"[{k+1}/4] {name}: 未找到 mtx", flush=True); continue
    t0 = time.time()
    v = cm.run_ocean(p)
    r["Ocean"] = round(v, 3) if v is not None else "DNF"
    print(f"[{k+1}/4] {name:20s} Ocean={r['Ocean']}  ({time.time()-t0:.1f}s)", flush=True)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print("done", flush=True)
