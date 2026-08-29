#!/usr/bin/env python3
"""PIPEFUSE A/B(交替 ×N 取中位,docs/56 方统计学):
MinHash 集(toll 目标)+ gate 集(construct/merge 空转回归)+ cnnz 一致性。
用法:CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/ab_pipefuse.py [N=4]"""
import os, subprocess, re, statistics, sys

BIN = os.environ.get("PF_BIN", "./spgemm_test")
DIR = "data/ocean/square"
MH_SET = ["TSOPF_FS_b39_c7", "brainpc2", "c-64", "Ga3As3H12", "mult_dcop_03", "bloweya"]
GATE_SET = ["pwtk", "333SP"]          # avg_product≤64:PIPEFUSE 下 construct/merge 空转
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

def run(m, pf):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash", MP_HOST_MB="8192", PIPEFUSE=str(pf))
    try:
        r = subprocess.run([BIN, os.path.join(DIR, m + ".mtx")],
                           capture_output=True, text=True, env=env, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    ph = {}
    for line in r.stderr.splitlines():
        mm = re.search(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s+ms$", line)
        if mm: ph[mm.group(1)] = float(mm.group(2))
    nz = re.search(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    t = ph.get("TOTAL(GPU)")
    if t is None: return None
    return {"comp": t - ph.get("h2d", 0) - ph.get("d2h", 0),
            "mh_c": ph.get("mh_construct", 0), "mh_m": ph.get("mh_merge", 0),
            "nnz": int(nz.group(1)) if nz else -1}

if __name__ == "__main__":
    print(f"{'matrix':16s} {'off':>9s} {'on':>9s} {'delta':>7s} | mh_c {str():>0}| mh_merge off→on | nnz")
    for m in MH_SET + GATE_SET:
        a, b = [], []
        for _ in range(REPS):
            ra = run(m, 0); rb = run(m, 1)
            if ra and rb: a.append(ra); b.append(rb)
        if not a: print(f"{m:16s} FAIL"); continue
        ma, mb = statistics.median([x["comp"] for x in a]), statistics.median([x["comp"] for x in b])
        mc_a = statistics.median([x["mh_m"] for x in a]); mc_b = statistics.median([x["mh_m"] for x in b])
        ok = "OK" if a[0]["nnz"] == b[0]["nnz"] else f"NNZ {a[0]['nnz']} vs {b[0]['nnz']}"
        print(f"{m:16s} {ma:9.2f} {mb:9.2f} {(mb/ma-1)*100:+6.1f}% | {mc_a:6.3f}→{mc_b:6.3f} | {ok}",
              flush=True)
