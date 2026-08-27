#!/usr/bin/env python3
"""代表性输阵的相位 profiling:跑 spgemm_test 解析 hash-prof 各相位 + est/dup/overflow 诊断行。

用法: python3 scripts/profile_top_losers.py bloweya c-58 ...
输出: 每阵相位表(ms + 占比) + 关键诊断量,并对照 CSV 里 Ocean 时间。
"""
import csv, os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCEAN_CSV = os.path.join(REPO, "compare/ocean337/methods_cmp_clean.csv.csv")
DIR = os.path.join(REPO, "data/ocean/square")

ocean = {}
for r in csv.DictReader(open(OCEAN_CSV)):
    try:
        v = float(r["Ocean"])
        if v > 0:
            ocean[r["matrix"]] = v
    except (ValueError, TypeError):
        pass

names = sys.argv[1:]
for name in names:
    mtx = os.path.join(DIR, name + ".mtx")
    if not os.path.exists(mtx):
        cand = [f for f in os.listdir(DIR) if f.startswith(name)]
        print(f"!! {name}: 未找到 ({cand[:3]})")
        continue
    env = dict(os.environ, METHOD="adaptive", USE_MEMPOOL="1", MP_HOST_MB="8192")
    p = subprocess.run([os.path.join(REPO, "spgemm_test"), mtx], env=env,
                       capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr

    phases = []  # (tag, ms) 全轮次;round 以 h2d 重复出现为界
    for m in re.finditer(r"\[(?:hash|mrg3|merge)-prof\] (\S+)\s+([0-9.]+)", out):
        tag = m.group(1)
        if tag.startswith("TOTAL"):
            continue
        phases.append((tag, float(m.group(2))))
    # 取最后一轮完整序列(测量态);h2d/d2h 不计入 compute(compare_methods 同口径)
    rounds, cur = [], []
    for tag, ms in phases:
        if tag == "h2d" and cur:
            rounds.append(cur); cur = []
        cur.append((tag, ms))
    if cur:
        rounds.append(cur)
    last = rounds[-1] if rounds else []
    first = [(t, m) for t, m in last if t not in ("h2d", "d2h")]
    total = sum(ms for _, ms in first)

    diag = {}
    for pat, key in [
        (r"total_flop=(\d+)", "flop"),
        (r"total_est=(\d+)", "est"),
        (r"C_nnz=(\d+)", "C_nnz"),
        (r"overflow=(\d+)", "overflow_cols"),
        (r"ovf_cnt=(\d+)", "ovf_rows"),
        (r"avg_product=([0-9.]+)", "avg_prod"),
        (r"(\d+) x (\d+), nnz=(\d+)", "dims"),
    ]:
        m = re.search(pat, out)
        if m:
            diag[key] = m.groups() if len(m.groups()) > 1 else m.group(1)

    oc = ocean.get(name)
    print(f"\n===== {name} =====(测量轮,{len(rounds)} 轮)")
    if "dims" in diag:
        n, _, nnzA = diag["dims"]
        print(f"A: {n}×{n} nnzA={nnzA}   flop={diag.get('flop','-')}  est={diag.get('est','-')}"
              f"  C_nnz={diag.get('C_nnz','-')}")
        if diag.get("flop") and diag.get("est") and int(diag["est"]) > 0:
            print(f"dup因子(flop/est) = {int(diag['flop'])/int(diag['est']):.1f}×")
    if "overflow_cols" in diag or "ovf_rows" in diag:
        print(f"overflow: {diag.get('overflow_cols','-')} 列, retry 行 = {diag.get('ovf_rows','-')}")
    print(f"{'相位':<16} {'ms':>9} {'占比':>6}")
    for tag, ms in first:
        print(f"{tag:<16} {ms:>9.3f} {ms/total*100:>5.1f}%")
    print(f"{'TOTAL(compute)':<16} {total:>9.3f}")
    if oc:
        print(f"{'Ocean(CSV)':<16} {oc:>9.3f}   → Auto/Ocean = {total/oc:.2f}×")
