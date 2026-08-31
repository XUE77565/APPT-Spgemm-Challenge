#!/usr/bin/env python3
"""Ocean-A 双梯 OR-退化路由验证(docs/64 机制 A):DIM2=0/1 对比。
确定性部分(负载免疫):diter 路由行数、cnnz 逐位一致(红旗纪律)。
计时部分在 load<6 净窗才可信,否则仅作趋势参考。
用法:.venv/bin/python scripts/ab_dim2_or.py [--reps 3]
"""
import argparse, os, re, statistics, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

BIN = CM.BIN
DIR = "data/ocean/square"
# c-big=目标(straggler 19491/19578 挂 16× 门);F2/333SP=docs/64 回归线;其余=diter 常客守卫
MATS = ["c-big", "F2", "333SP", "c-58", "bcsstk30", "brainpc2", "mult_dcop_03"]

def run(m, dim2):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash", MP_HOST_MB="8192",
               DIM2=("1" if dim2 else "0"))
    try:
        r = subprocess.run([BIN, os.path.join(DIR, m + ".mtx")], capture_output=True, text=True,
                           env=env, timeout=1800)
    except subprocess.TimeoutExpired:
        return None, None, None
    t = CM.compute_only_from_prof(r.stderr, "hash-prof")
    nz = re.search(r"Result C[^:]*:\s*\d+ x \d+, nnz = (\d+)", r.stdout)
    diter = re.search(r"diter bin=(\d+) 行", r.stderr)
    return t, (int(nz.group(1)) if nz else -1), (int(diter.group(1)) if diter else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    for m in MATS:
        p = os.path.join(DIR, m + ".mtx")
        if not os.path.exists(p): print(f"  {m}: 缺文件,跳过"); continue
        t0, n0, d0 = run(m, False)
        if t0 is None: print(f"  {m}: off 跑失败"); continue
        best_t1, n1, d1 = [], None, None
        for _ in range(a.reps):
            t1, n1_, d1_ = run(m, True)
            if t1: best_t1.append(t1); n1, d1 = n1_, d1_
        med0, med1 = t0, (statistics.median(best_t1) if best_t1 else None)
        if n0 != n1:
            print(f"  {m:14s} ⚠NNZ RED FLAG off={n0} on={n1} —— 路由改输出,停!"); return 1
        dd = f"{d0}→{d1}(+{(d1-d0) if d0 is not None and d1 is not None else '?'})" if (d0 is not None and d1 is not None) else "n/a"
        pct = f"{(med1/med0-1)*100:+6.1f}%" if med1 else "  FAIL"
        print(f"  {m:14s} off={med0:9.2f} on={str(med1):>9} {pct}  diter行:{dd}  nnz={n0}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
