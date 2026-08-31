#!/usr/bin/env python3
"""LADDER v1 标定采数(docs/67 §6,净窗用):每阵两跑(默认混跑 / 纯 hash),取相位分解
→ 标定 CSV:搬走行数、Σest、accumulate/compact/retry 的墙钟差 → 供拟合 hash 侧
【有效】成本(含 compact 摊账 + SMEM 串行化),即 t_hash_eff 的修正系数。

用法:.venv/bin/python scripts/collect_ladder_calib.py [--outdir compare/ladder_calib] [mat ...]
"""
import argparse, csv, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

DIR = "data/ocean/square"
DEFAULT = ["c-64", "c-58", "brainpc2", "mult_dcop_03",          # 混跑偏好类
           "Ga41As41H72", "crankseg_2", "3Dspectralwave2",       # hash 偏好类
           "Ge99H100", "crankseg_1", "pkustk14", "nd24k"]        # 中间带
PHASES = ["accumulate", "compact+sort", "retry", "dense_direct", "dense_count", "TOTAL(GPU)"]

def run(m, force_hash):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash", MP_HOST_MB="8192")
    if force_hash: env["DITER_MIN_FLOP"] = "999999999"
    r = subprocess.run([CM.BIN, os.path.join(DIR, m + ".mtx")], capture_output=True, text=True,
                       env=env, timeout=1800)
    ph = {}
    for name in PHASES:
        mm = re.findall(r"\[hash-prof\] " + re.escape(name) + r"\s+([0-9.]+)", r.stderr)
        ph[name] = float(mm[-1]) if mm else None
    return ph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*", default=DEFAULT)
    ap.add_argument("--outdir", default="compare/ladder_calib")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "calib.csv")
    new = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if new: w.writerow(["matrix"] + [f"{p}_mixed" for p in PHASES] + [f"{p}_purehash" for p in PHASES])
    for m in a.mat:
        mixed, pure = run(m, False), run(m, True)
        if mixed["TOTAL(GPU)"] is None or pure["TOTAL(GPU)"] is None:
            print(f"{m}: FAIL"); continue
        w.writerow([m] + [mixed[p] for p in PHASES] + [pure[p] for p in PHASES])
        f.flush()
        dm = {p: (pure[p] - mixed[p]) for p in PHASES if mixed[p] is not None and pure[p] is not None}
        print(f"{m}: ΔTOTAL={dm['TOTAL(GPU)']:+.1f} Δacc={dm['accumulate']:+.1f} "
              f"Δcompact={dm['compact+sort']:+.1f} Δretry={dm['retry']:+.1f} Δdd={dm['dense_direct']:+.1f}", flush=True)
    print(f"\n→ {out}(拟合:Δacc+Δcompact+Δretry 对 Σest 回归 = 有效每槽成本)")

if __name__ == "__main__":
    main()
