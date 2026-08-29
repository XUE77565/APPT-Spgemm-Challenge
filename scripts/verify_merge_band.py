#!/usr/bin/env python3
"""带状合成阵三方验证:merge3 vs hash(我们)vs Ocean(compute-only 同口径)。

背景(memory):disp_synth.csv 记录 band_n32000_x1024 merge3 23.9ms vs hash 841ms(35×),
但从未对过 Ocean —— 本脚本补上第三方,检验"带状阵 merge 大幅领先 Ocean"是否成立。

口径:
  hash   : [hash-prof] TOTAL(GPU) − h2d − d2h(compute-only)
  merge3 : [mrg3-prof] 同上
  Ocean  : run_ocean(analysis+estimation+symbolic+numeric+epilogue+prologue,compute-only)

用法:.venv/bin/python scripts/verify_merge_band.py [--dir data/synth_disp] [--timeout 300]
     [--only band_n32000_x1024 ...] [--big-last]
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

BIN = CM.BIN


def run_ours(mtx, method, timeout):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD=method, MP_HOST_MB="8192")
    try:
        r = subprocess.run([BIN, mtx], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    out = r.stdout + r.stderr
    tag = "hash-prof" if method == "hash" else "mrg3-prof"
    t = CM.compute_only_from_prof(r.stderr, tag)
    nz = re.search(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    if t is None:
        return None, out[-500:]
    return t, (int(nz.group(1)) if nz else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/synth_disp")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定阵(不带 .mtx)")
    ap.add_argument("--big-last", action="store_true", help="大阵(fp>15亿)放最后并逐个确认")
    args = ap.parse_args()

    names = sorted(p[:-4] for p in os.listdir(args.dir)
                   if p.endswith(".mtx") and p.startswith("band"))
    if args.only:
        names = [n for n in names if n in args.only]
    # 按 .mtx 大小升序 = 小阵先行;--big-last 时保持该序(天然大阵靠后)
    names.sort(key=lambda n: os.path.getsize(os.path.join(args.dir, n + ".mtx")))

    print(f"{'matrix':24s} {'hash':>9s} {'merge3':>9s} {'Ocean':>9s} | "
          f"{'m3/Oce':>7s} {'hash/Oce':>8s} {'m3/hash':>8s}  cnnz")
    rows = []
    for name in names:
        p = os.path.join(args.dir, name + ".mtx")
        th, zh = run_ours(p, "hash", args.timeout)
        tm, zm = run_ours(p, "merge3", args.timeout)
        to = CM.run_ocean(p, timeout=args.timeout)
        cnnz = zh if isinstance(zh, int) and zh > 0 else (zm if isinstance(zm, int) else -1)
        # cnnz 一致性(三方都成功时)
        clash = ""
        if (isinstance(zh, int) and zh > 0 and isinstance(zm, int) and zm > 0
                and zh != zm):
            clash = f" ⚠nnz {zh}vs{zm}"
        rows.append((name, th, tm, to, cnnz))
        fmt = lambda v: f"{v:9.2f}" if isinstance(v, float) else f"{'FAIL' if v is None else v:>9s}"
        r_m3o = f"{tm/to:7.2f}" if isinstance(tm, float) and isinstance(to, float) else "      -"
        r_ho = f"{th/to:8.2f}" if isinstance(th, float) and isinstance(to, float) else "       -"
        r_mh = f"{tm/th:8.2f}" if isinstance(tm, float) and isinstance(th, float) else "       -"
        print(f"{name:24s} {fmt(th)} {fmt(tm)} {fmt(to)} | {r_m3o} {r_ho} {r_mh}  {cnnz}{clash}",
              flush=True)

    # 汇总
    ok = [r for r in rows if isinstance(r[2], float) and isinstance(r[3], float)]
    print(f"\nmerge3/Ocean 可比 {len(ok)}/{len(rows)}:")
    for name, th, tm, to, cz in sorted(ok, key=lambda r: r[2] / r[3]):
        win = "★ merge3 胜 Ocean" if tm < to else ""
        print(f"  {name:24s} {tm:9.2f} vs {to:9.2f} = {tm/to:6.2f}×  {win}")


if __name__ == "__main__":
    main()
