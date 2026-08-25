#!/usr/bin/env python3
"""我们的 hash vs Ocean 的逐相位 profiling(同矩阵、同卡、同条件连跑)。

我方 : METHOD=hash 跑 spgemm_test,解析 [hash-prof] cudaEvent 相位
       (h2d/d2h 剔除 = compute-only;accumulate 含 ultra;mh_* 为 MinHash sizing)
Ocean: ocean/convert + ocean/spgemm(bench_detail track_stage_time)→ stats.json
       (analysis/estimation/numeric/epilogue/prologue,口径与 compare 相同)

输出:相位对齐表 + 差距归因。用法:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/profile_us_vs_ocean.py 333SP AS365 ...
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

OCEAN_CWD = os.path.join(CM.REPO, "ocean")
OCEAN_STATS = CM.OCEAN_STATS

# 我方相位 → 语义分组(与 Ocean 对齐)
OURS_GROUPS = {
    "sizing (MinHash)": ["mh_construct", "mh_merge", "est_scan"],
    "binning":           ["binning"],
    "accumulate":        ["accumulate"],
    "compact+sort":      ["compact+sort"],
    "cnnz_scan":         ["cnnz_scan"],
    "memset":            ["memset"],
}


def run_ours(mtx):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash", MP_HOST_MB="8192", CU_REF="0")
    try:
        r = subprocess.run([CM.BIN, mtx], capture_output=True, text=True, env=env, timeout=900)
    except subprocess.TimeoutExpired:
        return None, None
    rx = re.compile(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s+ms")
    phases = {}
    for line in (r.stderr or "").splitlines():
        m = rx.search(line)
        if m:
            phases[m.group(1)] = float(m.group(2))   # 最后一轮 timed
    total = phases.get("TOTAL(GPU)")
    comp = total - phases.get("h2d", 0) - phases.get("d2h", 0) if total else None
    adapt = re.search(r"\[adapt\].*", r.stdout)
    return phases, comp


def run_ocean(mtx):
    csr = "/tmp/prof_ocean.csr"
    try:
        subprocess.run([CM.OCEAN_CONV, mtx, csr], capture_output=True, timeout=600)
        subprocess.run([CM.OCEAN_RUN, csr, CM.OCEAN_CFG], cwd=OCEAN_CWD,
                       capture_output=True, timeout=900)
        t = json.load(open(OCEAN_STATS))["timing"]
    except Exception:
        return None, None
    def s(*keys):
        tot = 0.0
        for grp in keys:
            v = t.get(grp)
            if isinstance(v, dict):
                tot += sum(v.values())
        return tot
    comp = (s("analysis", "estimation", "numeric", "epilogue") + t.get("prologue", 0))
    return t, comp


def main():
    names = sys.argv[1:]
    for name in names:
        p = CM.find_mtx(name)
        if not p:
            print(f"## {name}: 未找到"); continue
        print(f"\n{'='*72}\n## {name}")
        ours, ocomp = run_ours(p)
        oce, ocomp2 = run_ocean(p)
        if ours is None or oce is None:
            print("  某侧跑失败"); continue
        # 我方分组
        print(f"{'相位':<26}{'ours(ms)':>10}{'Ocean(ms)':>11}{'ours/Ocn':>9}")
        ours_g = {g: sum(ours.get(k, 0) for k in ks) for g, ks in OURS_GROUPS.items()}
        ocn_map = {
            "sizing (MinHash)": sum(oce["estimation"][k] for k in
                                    ("hll_construct", "hll_merge", "sampling", "malloc")),
            "binning":           sum(oce["analysis"].values()),
            "accumulate":        sum(oce["numeric"].values()),
            "compact+sort":      sum(oce["epilogue"].values()),
        }
        for g in ["sizing (MinHash)", "binning", "accumulate", "compact+sort", "cnnz_scan", "memset"]:
            ov, nv = ours_g.get(g, 0), ocn_map.get(g, 0)
            if ov == 0 and nv == 0:
                continue
            r = f"{ov/nv:8.2f}x" if nv > 0 else "      —"
            print(f"{g:<26}{ov:>10.3f}{nv:>11.3f}{r:>9}")
        print(f"{'— 其余 numeric 明细 —':<26}")
        for k, v in sorted(oce["numeric"].items(), key=lambda kv: -kv[1]):
            if v > 0.01: print(f"  numeric.{k:<20}{v:>8.3f} ms")
        if ocomp is None or ocomp2 is None or not ocomp2:
            print(f"{'COMPUTE-ONLY 合计':<26}  我方={'复跑失败/无TOTAL' if ocomp is None else f'{ocomp:.3f}'}  Ocean={ocomp2}")
        else:
            print(f"{'COMPUTE-ONLY 合计':<26}{ocomp:>10.3f}{ocomp2:>11.3f}"
                  f"{(ocomp/ocomp2):>8.2f}x")


if __name__ == "__main__":
    main()
