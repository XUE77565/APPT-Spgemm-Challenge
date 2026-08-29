#!/usr/bin/env python3
"""DCFUSE 聚焦复核:疑点阵 ×3 重复 + 相位级分解,分离噪声与真实回归。"""
import os, subprocess, re, statistics, sys

BIN, DIR = "./spgemm_test", "data/ocean/square"

def run(mtx, dcfuse, timeout=900, mb="8192"):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash", MP_HOST_MB=mb, DCFUSE=str(dcfuse))
    try:
        r = subprocess.run([BIN, os.path.join(DIR, mtx + ".mtx")],
                           capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    phases = {}
    for line in r.stderr.splitlines():
        m = re.search(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s+ms", line)
        if m:
            phases[m.group(1)] = float(m.group(2))
    total = phases.get("TOTAL(GPU)")
    if total is None:
        return None
    return total - phases.get("h2d", 0.0) - phases.get("d2h", 0.0), phases

if __name__ == "__main__":
    mats = sys.argv[1:] or ["bloweya", "SiO2", "mult_dcop_03", "c-58"]
    for m in mats:
        mb = "32768" if m in ("TSOPF_FS_b300_c2", "vsp_south31", "rajat16") else "8192"
        offs, ons = [], []
        ph_off, ph_on = None, None
        for _ in range(3):
            r = run(m, 0, mb=mb)
            if r: offs.append(r[0]); ph_off = r[1]
            r = run(m, 1, mb=mb)
            if r: ons.append(r[0]); ph_on = r[1]
        if not offs or not ons:
            print(f"{m:16s} FAIL offs={offs} ons={ons}")
            continue
        mo, mn = statistics.median(offs), statistics.median(ons)
        d = (mn - mo) / mo * 100
        print(f"{m:16s} off={mo:8.2f}(n={len(offs)}) on={mn:8.2f}(n={len(ons)}) {d:+6.1f}%")
        # 相位差(最后一次)
        keys = sorted(set(ph_off) | set(ph_on))
        for k in keys:
            a, b = ph_off.get(k, 0.0), ph_on.get(k, 0.0)
            if abs(b - a) > 0.15 and k not in ("h2d", "d2h", "TOTAL(GPU)"):
                print(f"    {k:16s} {a:8.3f} → {b:8.3f}  ({b-a:+.3f})", flush=True)
