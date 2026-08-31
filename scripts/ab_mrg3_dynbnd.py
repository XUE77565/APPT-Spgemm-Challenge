#!/usr/bin/env python3
"""merge3 动态负载均衡 A/B(docs/66):MRG3_DYN_BND off/on 交替 × N 取中位。
纪律:两版 nnz 必须逐位一致,否则红旗立即停(边界错 = 桶划分错 = 假 cnnz)。
用法:.venv/bin/python scripts/ab_mrg3_dynbnd.py [--reps 4] [--only band_n32000_x1024 ...]
"""
import argparse, os, re, statistics, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

BIN = CM.BIN
DEFAULT = [
    "data/synth_disp/band_n32000_x1024.mtx",   # 宽带生态位(等宽桶全落桶0,预期最大收益)
    "data/synth_disp/band_n8000_x1024.mtx",
    "data/synth_disp/band_n2000_x1024.mtx",
    "data/synth_disp/band_n32000_x128.mtx",    # 窄带(预期小/负,守卫)
    "data/synth_disp/band_n32000_x16.mtx",
    "data/synth_disp/er_n32000_x16.mtx",       # ER 均匀(等宽≈均衡,边界 kernel 纯开销)
    "data/synth_disp/skew_n32000_x16.mtx",     # 偏斜(若存在)
]

def run(mtx, dyn):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="merge3", MP_HOST_MB="8192",
               MRG3_DYN_BND=("1" if dyn else "0"))
    try:
        r = subprocess.run([BIN, mtx], capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return None, None
    t = CM.compute_only_from_prof(r.stderr, "mrg3-prof")
    m = re.search(r"Result C[^:]*:\s*\d+ x \d+, nnz = (\d+)", r.stdout)
    return t, (int(m.group(1)) if m else -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    mats = a.only or [m for m in DEFAULT if os.path.exists(m)]
    for mtx in mats:
        off, on, nz = [], [], None
        for _ in range(a.reps):   # 交替 off/on:同环漂移免疫(docs/59 纪律)
            t0, n0 = run(mtx, False)
            t1, n1 = run(mtx, True)
            if t0: off.append(t0)
            if t1: on.append(t1)
            if n0 is not None and n1 is not None and n0 != n1:
                print(f"  {os.path.basename(mtx):24s} ⚠NNZ RED FLAG off={n0} on={n1} —— 停,查边界")
                return 1
            if (n0 is None or n0 < 0) and t0 is not None:
                print(f"  {os.path.basename(mtx):24s} ⚠nnz 未解析(off),红旗纪律不满足", flush=True)
            nz = n1
        if not off or not on:
            print(f"  {os.path.basename(mtx):24s} FAIL(no timing)"); continue
        m0, m1 = statistics.median(off), statistics.median(on)
        d = (m1 / m0 - 1) * 100
        sp = f"[{min(off):.1f},{max(off):.1f}]|[{min(on):.1f},{max(on):.1f}]"
        print(f"  {os.path.basename(mtx):24s} off={m0:9.2f} on={m1:9.2f} Δ={d:+6.1f}%  {sp} nnz={nz}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
