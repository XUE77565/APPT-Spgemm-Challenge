#!/usr/bin/env python3
"""Verify the new dense baseline LOSES to Auto on every matrix.

Reads Auto times from compare/aa(best2)/methods_cmp.csv, runs ./spgemm_dense on
each first100 matrix, and reports dense vs Auto. Any matrix where dense <= Auto
is a FAILURE (dense won / tied) and is flagged.

Usage: .venv/bin/python scripts/verify_dense_loses.py [--limit N] [--timeout 600]
"""
import os, sys, csv, re, subprocess, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_dense")
REF = os.path.join(REPO, "compare/aa(best2)/methods_cmp.csv")
DATA = os.path.join(REPO, "data/first100")

def find_mtx(name):
    for d in ("data/first100", "data/sota_27_final", "data/sota_27"):
        p = os.path.join(REPO, d, name + ".mtx")
        if os.path.exists(p): return p
    return None

def run_dense(mtx, timeout):
    try:
        r = subprocess.run([BIN, mtx, "/tmp/dense_verify.mtx"],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", r.stdout)
    nz = re.search(r"C:.*?nnz\s*=\s*(\d+)", r.stdout)
    if m: return float(m.group(1)), (int(nz.group(1)) if nz else -1)
    return None, "fail"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    refs = {r["matrix"]: r for r in csv.DictReader(open(REF))}
    names = sorted(refs.keys())
    if args.limit: names = names[:args.limit]

    wins = 0; losses = 0; fails = []
    tight = []  # (ratio, name) where ratio close to 1
    print(f"{'matrix':16} {'n':>6} {'Auto':>8} {'dense':>10} {'d/A':>6}  status")
    print("-" * 60)
    for name in names:
        r = refs[name]
        try: auto = float(r["Auto"])
        except: continue
        p = find_mtx(name)
        if not p: continue
        dt, extra = run_dense(p, args.timeout)
        if dt is None:
            print(f"{name:16} {r['n']:>6} {auto:>8.3f} {'FAIL':>10}        {extra}")
            fails.append(name); continue
        ratio = dt / auto
        status = "LOSE" if dt > auto else "*** WIN ***"
        if dt > auto: losses += 1
        else: wins += 1; fails.append(name)
        if 0 < ratio < 2.5: tight.append((ratio, name, auto, dt))
        print(f"{name:16} {r['n']:>6} {auto:>8.3f} {dt:>10.3f} {ratio:>6.2f}  {status}")

    print("-" * 60)
    print(f"\nRESULT: dense LOSES on {losses}/{losses+wins}, WINS/TIES on {wins}")
    if fails:
        print(f"  *** {len(fails)} matrix(ices) where dense did NOT lose: {fails}")
    else:
        print("  ✓ dense loses to Auto on ALL matrices.")
    if tight:
        print(f"\n  tightest (d/A < 2.5×):")
        for ratio, name, auto, dt in sorted(tight)[:8]:
            print(f"    {name:14} Auto={auto:.3f} dense={dt:.3f} ratio={ratio:.2f}×  margin={dt-auto:.3f}ms")

if __name__ == "__main__":
    main()
