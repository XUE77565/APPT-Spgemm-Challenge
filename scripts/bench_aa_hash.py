#!/usr/bin/env python3
"""AA (C=A·A) hash-path engineering benchmark.

Runs METHOD=hash DBG=1 USE_MEMPOOL=1 N times per matrix, parses the cudaEvent
hash-prof blocks, and reports the MEDIAN compute-only time + per-phase medians.
This is the before/after measurement harness for every AA engineering opt.

compute-only = mh_construct + mh_merge + est_scan + binning + accumulate
             + cnnz_scan + compact+sort   (excludes h2d / d2h)

Usage:
  .venv/bin/python scripts/bench_aa_hash.py [matrix...] [-n RUNS] [--bin BINARY]
"""
import argparse, subprocess, re, statistics, sys, os

DEFAULT_MATS = ["bcsstk30", "bcsstk32", "bcsstk31", "bcsstk29", "bcsstk17"]
PHASES = ["mh_construct", "mh_merge", "est_scan", "binning",
          "accumulate", "cnnz_scan", "compact+sort"]
LINE = re.compile(r"\[hash-prof\]\s+(\S+)\s+([\d.]+)\s+ms")

def run_once(binary, mtx):
    """Return list of block-dicts (one per spgemm_self_product_hash call)."""
    env = dict(os.environ, METHOD="hash", USE_MEMPOOL="1")
    p = subprocess.run([binary, f"data/first100/{mtx}.mtx"],
                       capture_output=True, text=True, env=env)
    blocks, cur = [], {}
    for line in (p.stdout + p.stderr).splitlines():
        m = LINE.search(line)
        if not m:
            continue
        phase, val = m.group(1), float(m.group(2))
        if phase == "TOTAL(GPU)":           # close block
            if cur:
                blocks.append(cur); cur = {}
        elif phase in ("h2d", "d2h"):
            continue
        else:
            cur[phase] = cur.get(phase, 0.0) + val
    if cur:
        blocks.append(cur)
    return blocks

def block_compute(b):
    return sum(b.get(p, 0.0) for p in PHASES)

def bench(binary, mtx, runs):
    all_blocks = []
    for _ in range(runs):
        all_blocks.extend(run_once(binary, mtx))
    # keep only warm blocks (drop the first cold one per invocation is hard across
    # invocations; instead take all and use median which is robust to a few cold)
    if not all_blocks:
        return None
    computes = sorted(block_compute(b) for b in all_blocks)
    med = statistics.median(computes)
    phase_med = {p: statistics.median(sorted(b.get(p, 0.0) for b in all_blocks))
                 for p in PHASES}
    return med, phase_med, len(all_blocks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mats", nargs="*", default=DEFAULT_MATS)
    ap.add_argument("-n", "--runs", type=int, default=3)
    ap.add_argument("--bin", default="./spgemm_test")
    args = ap.parse_args()

    print(f"# AA hash bench  bin={args.bin}  runs/mtx={args.runs}\n")
    hdr = f"{'matrix':<10} {'compute':>8} {'accum':>7} {'c+sort':>7} {'mh_mrg':>7} {'bin':>6} {'#blk':>4}"
    print(hdr); print("-" * len(hdr))
    tot_compute = 1.0
    geoms = []
    res = {}
    for m in args.mats:
        r = bench(args.bin, m, args.runs)
        if r is None:
            print(f"{m:<10}  (no hash-prof — overflow/fail?)"); continue
        med, pm, nblk = r
        res[m] = (med, pm)
        geoms.append(med)
        print(f"{m:<10} {med:>8.2f} {pm['accumulate']:>7.2f} "
              f"{pm['compact+sort']:>7.2f} {pm['mh_merge']:>7.3f} {pm['binning']:>6.3f} {nblk:>4}")
    if geoms:
        import math
        gm = math.exp(sum(math.log(x) for x in geoms) / len(geoms))
        print("-" * len(hdr))
        print(f"{'geomean':<10} {gm:>8.2f}")
    # stash for quick diff
    with open("compare/bench_aa_hash.latest.tsv", "w") as f:
        f.write("matrix\tcompute\taccumulate\tcompact+sort\tmh_merge\tbinning\n")
        for m, (med, pm) in res.items():
            f.write(f"{m}\t{med:.3f}\t{pm['accumulate']:.3f}\t{pm['compact+sort']:.3f}\t{pm['mh_merge']:.3f}\t{pm['binning']:.3f}\n")
    print("\n(detailed medians written to compare/bench_aa_hash.latest.tsv)")

if __name__ == "__main__":
    main()
