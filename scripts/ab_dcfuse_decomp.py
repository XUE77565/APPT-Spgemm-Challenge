#!/usr/bin/env python3
"""DCFUSE 三分分解:off / on(默认预算) / on+budget0(强制坍缩)→ 隔离 no_collapse 贡献。"""
import os, subprocess, re

BIN, DIR = "./spgemm_test", "data/ocean/square"

def run(mtx, dcfuse, budget=None, timeout=600):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash", MP_HOST_MB="8192", DCFUSE=str(dcfuse))
    if budget is not None:
        env["DCF_BUDGET_GB"] = str(budget)
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
    nz = re.search(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    return {"comp": total - phases.get("h2d", 0.0) - phases.get("d2h", 0.0),
            "cnt": phases.get("dense_count") or 0, "nnz": int(nz.group(1)) if nz else -1}

if __name__ == "__main__":
    print(f"{'matrix':18s} {'off':>9s} {'on':>9s} {'on+bud0':>9s} | cnt: {'off':>7s} {'on':>7s} {'bud0':>7s}")
    for m in ["brainpc2", "c-64", "TSOPF_FS_b39_c7", "Ga3As3H12"]:
        a = run(m, 0)
        c = run(m, 1, budget=0)
        b = run(m, 1)
        print(f"{m:18s} {a['comp']:9.2f} {b['comp']:9.2f} {c['comp']:9.2f} | "
              f"{a['cnt']:7.3f} {b['cnt']:7.3f} {c['cnt']:7.3f} | nnz {a['nnz']}/{b['nnz']}/{c['nnz']}",
              flush=True)
