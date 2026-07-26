#!/usr/bin/env python3
"""HLL vs MinHash estimator comparison on the hash method (first100).

Runs METHOD=hash twice per matrix — once with EST_METHOD=hll, once unset (MinHash,
the default) — and compares:
  · compute-only time (cudaEvent prof, excludes h2d/d2h — same basis as Ocean),
  · over-alloc factor (est / C_nnz),
  · C_nnz correctness (must match between the two).

Reuses compare_methods.{BIN, REPO, CALL_TIMEOUT, compute_only_from_prof,
find_mtx, mtx_header} so the timing basis is identical to the main compare.

Output: compare/hll_vs_minhash_<ts>.csv + a printed summary.
"""
import os, sys, re, csv, math, subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

REPO = CM.REPO
BIN = CM.BIN
TIMEOUT = CM.CALL_TIMEOUT
OVER_RE = re.compile(r"C_nnz=(\d+)\s*\(est=(\d+),\s*([\d.]+)x over-alloc\)")


def run_hash(mtx, est):
    """est: 'hll' | None (unset → MinHash default). Returns dict(comp, cnnz, over) or None."""
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash")
    if est is None:
        env.pop("EST_METHOD", None)
    else:
        env["EST_METHOD"] = est
    try:
        r = subprocess.run([BIN, mtx], capture_output=True, text=True, env=env, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    comp = CM.compute_only_from_prof(r.stderr, "hash-prof")
    m = OVER_RE.search(r.stderr)
    return dict(comp=comp,
                cnnz=int(m.group(1)) if m else -1,
                over=float(m.group(3)) if m else float("nan"))


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    d = os.path.join(REPO, "data/first100")
    names = sorted(p[:-4] for p in os.listdir(d) if p.endswith(".mtx"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(REPO, "compare", f"hll_vs_minhash_{ts}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fout = open(out, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["matrix", "n", "density_pct", "HLL_ms", "MinHash_ms", "ratio_MH_HLL",
                "HLL_overalloc", "MH_overalloc", "HLL_cnnz", "MH_cnnz"])

    N = len(names)
    print(f"HLL vs MinHash on {N} matrices -> {out}\n", flush=True)
    hll_t, mh_t, hll_o, mh_o = [], [], [], []
    hll_wins = mh_wins = ties = cnnz_bad = 0
    for i, name in enumerate(names):
        p = CM.find_mtx(name)
        if not p:
            print(f"[{i+1}/{N}] {name}: not found, skip", flush=True)
            continue
        n, _sym, _annz, dens = CM.mtx_header(p)
        rh = run_hash(p, "hll")
        rm = run_hash(p, None)
        if not rh or not rm or rh["comp"] is None or rm["comp"] is None:
            print(f"[{i+1}/{N}] {name:14}: skip (run/parse failed)", flush=True)
            continue
        ratio = rm["comp"] / rh["comp"]            # >1 → MinHash slower → HLL wins
        if abs(ratio - 1) < 0.03:
            ties += 1
        elif ratio > 1:
            hll_wins += 1
        else:
            mh_wins += 1
        if rh["cnnz"] != rm["cnnz"]:
            cnnz_bad += 1
        hll_t.append(rh["comp"]); mh_t.append(rm["comp"])
        hll_o.append(rh["over"]); mh_o.append(rm["over"])
        w.writerow([name, n, round(dens, 4), round(rh["comp"], 4), round(rm["comp"], 4),
                    round(ratio, 3), round(rh["over"], 2), round(rm["over"], 2),
                    rh["cnnz"], rm["cnnz"]])
        flag = "HLL<" if ratio > 1.03 else ("MH<" if ratio < 0.97 else "=")
        print(f"[{i+1}/{N}] {name:14} n={n:<6} HLL={rh['comp']:8.3f} MH={rm['comp']:8.3f} "
              f"MH/HLL={ratio:5.2f} {flag}  over {rh['over']:.2f}/{rm['over']:.2f}", flush=True)
    fout.close()

    print("\n================ SUMMARY ================", flush=True)
    print(f"matrices run       : {len(hll_t)} / {N}")
    print(f"geomean compute    : HLL {geomean(hll_t):.4f} ms | MinHash {geomean(mh_t):.4f} ms")
    print(f"geomean MH/HLL     : {geomean(mh_t) / geomean(hll_t):.3f}  (>1 = MinHash slower)")
    print(f"geomean over-alloc : HLL {geomean(hll_o):.2f}x | MinHash {geomean(mh_o):.2f}x")
    print(f"win/loss (compute) : HLL faster {hll_wins} | MinHash faster {mh_wins} | tie {ties}")
    print(f"C_nnz mismatch     : {cnnz_bad}")
    print(f"csv                : {out}", flush=True)


if __name__ == "__main__":
    main()
