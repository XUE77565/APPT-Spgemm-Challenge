#!/usr/bin/env python3
"""DCFUSE A/B:env DCFUSE=0(legacy)vs 1(瘦身)对目标+回归阵。
输出 compute-only(TOTAL−h2d−d2h,取最后 timed run)+ dense_count/dense_direct 相位 + cnnz 一致性。"""
import os, re, subprocess, sys

BIN = "./spgemm_test"
DIR = "data/ocean/square"
TARGETS = ["brainpc2", "c-64", "TSOPF_FS_b39_c7"]
REGRESS = ["pwtk", "333SP", "3Dspectralwave2", "bcsstk30", "Ga3As3H12"]

def run(mtx, dcfuse, timeout=600):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash", MP_HOST_MB="8192", DCFUSE=str(dcfuse))
    try:
        r = subprocess.run([BIN, os.path.join(DIR, mtx + ".mtx")],
                           capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    phases = {}
    for line in r.stderr.splitlines():
        m = re.search(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s+ms", line)
        if m:
            phases[m.group(1)] = float(m.group(2))   # 后值覆盖前值 = 取最后 timed run
    total = phases.get("TOTAL(GPU)")
    if total is None:
        return None
    comp = total - phases.get("h2d", 0.0) - phases.get("d2h", 0.0)
    nz = re.search(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    return {"comp": comp, "cnt": phases.get("dense_count"), "direct": phases.get("dense_direct"),
            "nnz": int(nz.group(1)) if nz else -1}

print(f"{'matrix':18s} {'off':>9s} {'on':>9s} {'delta':>8s} | {'cnt off':>8s} {'cnt on':>8s} | {'dir off':>8s} {'dir on':>8s} | nnz match")
for m in TARGETS + REGRESS:
    a = run(m, 0)
    b = run(m, 1)
    if a is None or b is None:
        print(f"{m:18s} FAIL(a={a is not None},b={b is not None})")
        continue
    d = (b["comp"] - a["comp"]) / a["comp"] * 100
    ok = "✓" if (a["nnz"] == b["nnz"] and a["nnz"] > 0) else f"⚠ {a['nnz']}vs{b['nnz']}"
    print(f"{m:18s} {a['comp']:9.2f} {b['comp']:9.2f} {d:+7.1f}% | "
          f"{a['cnt'] if a['cnt'] else 0:8.3f} {b['cnt'] if b['cnt'] else 0:8.3f} | "
          f"{a['direct'] if a['direct'] else 0:8.2f} {b['direct'] if b['direct'] else 0:8.2f} | {ok}",
          flush=True)
