#!/usr/bin/env python3
"""Hybrid Value 消融:noHybrid(v3 基线) vs refresh6(Hybrid+自动路由)。

用法: python3 scripts/ablation_hybrid.py [new_csv]
输出: 总体 geomean 变化、逐阵 delta、受益/退步清单、受污染窗口阵复核。
"""
import csv, math, sys

OLD = "compare/ocean337/methods_cmp_v3_nohybrid.csv"
NEW = sys.argv[1] if len(sys.argv) > 1 else "compare/ocean337/methods_cmp_clean.csv.csv"

def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        try:
            a = float(r["Auto"])
            if a > 0:
                d[r["matrix"]] = (a, float(r["Ocean"]) if r.get("Ocean") else None, r.get("Auto_choice", ""))
        except (ValueError, TypeError):
            pass
    return d

old, new = load(OLD), load(NEW)
both = [m for m in old if m in new]
gm_old = math.exp(sum(math.log(old[m][0]) for m in both) / len(both))
gm_new = math.exp(sum(math.log(new[m][0]) for m in both) / len(both))
print(f"=== Hybrid 消融(双方有数据 {len(both)} 阵)===")
print(f"geomean(Auto): noHybrid {gm_old:.3f}ms → Hybrid {gm_new:.3f}ms  ({(gm_new/gm_old-1)*100:+.1f}%)")

vs_ocean_old = [old[m][0]/old[m][1] for m in both if old[m][1]]
vs_ocean_new = [new[m][0]/new[m][1] for m in both if new[m][1]]
if vs_ocean_old:
    g1 = math.exp(sum(math.log(x) for x in vs_ocean_old)/len(vs_ocean_old))
    g2 = math.exp(sum(math.log(x) for x in vs_ocean_new)/len(vs_ocean_new))
    w1 = sum(1 for x in vs_ocean_old if x < 1); w2 = sum(1 for x in vs_ocean_new if x < 1)
    print(f"vs Ocean: {g1:.3f}× (赢{w1}/{len(vs_ocean_old)}) → {g2:.3f}× (赢{w2}/{len(vs_ocean_new)})")

delta = {m: new[m][0]/old[m][0] - 1 for m in both}
better = sorted((m for m in delta if delta[m] <= -0.03), key=lambda m: delta[m])
worse  = sorted((m for m in delta if delta[m] >= 0.03),  key=lambda m: -delta[m])
print(f"\n受益 ≥3%: {len(better)} 阵")
for m in better[:15]:
    print(f"  {m:<26} {old[m][0]:>9.2f} → {new[m][0]:>9.2f} ms  ({delta[m]*100:+.1f}%)")
print(f"\n退步 ≥3%: {len(worse)} 阵")
for m in worse[:15]:
    print(f"  {m:<26} {old[m][0]:>9.2f} → {new[m][0]:>9.2f} ms  ({delta[m]*100:+.1f}%)")

only_new = [m for m in new if m not in old]
only_old = [m for m in old if m not in new]
if only_new: print(f"\n新增出数({len(only_new)}): {', '.join(sorted(only_new)[:20])}")
if only_old: print(f"新 DN F/丢失({len(only_old)}): {', '.join(sorted(only_old)[:20])}")

print("\n=== 受污染窗口复核(refresh6 #41-45,make 事故)===")
for m in ["Si5H12", "Si87H76", "SiN", "SiO", "Si10H16", "Si2H6", "Si41Ge41H72", "Si34H36", "SiNa", "SiDL"]:
    if m in old and m in new:
        print(f"  {m:<14} {old[m][0]:>8.2f} → {new[m][0]:>8.2f}  ({delta.get(m,0)*100:+.1f}%)")
