#!/usr/bin/env python3
"""Re-fit the 4-variable hash/merge3 dispatcher on the CURRENT kernels
(MinHash-default hash + flop_ub merge3).  The in-tree formula
(spgemm_adaptive.cu:72) was fit on older kernels (HLL hash, pre-flop_ub merge3),
so the crossover may have shifted.

Per first100 matrix we collect, matching spgemm_adaptive.cu exactly:
  features : lfp = log10(A_nnz^2 / n), ln = log10(n),
             lmr = log10(max_row_nnz), lsk = log10(skew),
             skew = max_row_nnz / (A_nnz / n)   (symmetric matrices expanded)
  timings  : hash compute-only (METHOD=hash, MinHash default) and merge3 compute-only
  label y  : 1 if hash faster (hash_t < merge3_t); hash overflow -> merge3 (0)

Fits margin-cost-weighted logistic regression (sample_weight = |hash_t - merge3_t|,
so decisive / large matrices drive the fit, not noisy micro-matrices) and reports
the formula in the code's form  score = a*lfp + b*ln + c*lmr + d*lsk + e,  hash iff
score < 0.  Compares accuracy + total compute time vs the current formula.

Output: compare/dispatcher_fit_<ts>.csv + printed report.
"""
import os, sys, math, subprocess
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

REPO = CM.REPO
TIMEOUT = CM.CALL_TIMEOUT

# current in-tree coefficients (spgemm_adaptive.cu:72), hash iff score<0
CUR_W = np.array([-1.3085, 1.2131, 1.9815, -2.4331])
CUR_B = 1.2943


def load_features(path):
    """n, A_nnz(expanded), max_row_nnz(expanded), skew, flop_proxy — matching the loader."""
    n = 0; sym = False; sized = False; rowcnt = {}
    with open(path) as f:
        first = f.readline(); sym = "symmetric" in first.lower()
        for l in f:
            if l.startswith("%"):
                continue
            p = l.split()
            if not p:
                continue
            if not sized:
                n = int(p[0]); sized = True; continue
            r = int(p[0]); c = int(p[1])
            rowcnt[r] = rowcnt.get(r, 0) + 1
            if sym and r != c:                      # mirror off-diagonal (matrix_utils.cu:108-109)
                rowcnt[c] = rowcnt.get(c, 0) + 1
    nnz = sum(rowcnt.values())
    maxrow = max(rowcnt.values()) if rowcnt else 0
    avg = nnz / n if n else 0.0
    skew = maxrow / avg if avg > 0 else 0.0
    flop = nnz * nnz / n if n else 0.0
    return n, nnz, maxrow, skew, flop


def feat_vec(n, maxrow, skew, flop):
    return np.array([math.log10(max(flop, 1.0)), math.log10(n),
                     math.log10(max(maxrow, 1)), math.log10(max(skew, 1.0))])


def run_method(mtx, key):
    return CM.run_spgemm_method(mtx, key)           # (comp, wall, cnnz, choice) or None


def main():
    os.environ.pop("EST_METHOD", None)              # hash = MinHash default (current)
    d = os.path.join(REPO, "data/first100")
    names = sorted(p[:-4] for p in os.listdir(d) if p.endswith(".mtx"))
    N = len(names)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(REPO, "compare", f"dispatcher_fit_{ts}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"collecting hash(MinHash) vs merge3 + features on {N} matrices ...", flush=True)
    rows = []
    import csv
    fout = open(out, "w", newline="")
    cw = csv.writer(fout)
    cw.writerow(["matrix", "n", "maxrow", "skew", "flop", "hash_ms", "merge3_ms", "winner"])
    for i, name in enumerate(names):
        p = CM.find_mtx(name)
        if not p:
            continue
        n, nnz, maxrow, skew, flop = load_features(p)
        x = feat_vec(n, maxrow, skew, flop)
        rh = run_method(p, "hash"); rm = run_method(p, "merge3")
        if not rh or not rm or rh[0] is None or rm[0] is None:
            print(f"[{i+1}/{N}] {name}: skip (run failed)", flush=True); continue
        ht, _, hc, _ = rh
        mt, _, mc, _ = rm
        if hc < 0:                                  # hash overflow -> merge3 wins
            hw = 0; ht = math.inf
        else:
            hw = 1 if ht < mt else 0
        rows.append(dict(name=name, n=n, maxrow=maxrow, skew=skew, flop=flop,
                         x=x, ht=ht, mt=mt, hw=hw))
        cw.writerow([name, n, maxrow, round(skew, 3), round(flop), round(ht, 4),
                     round(mt, 4), "hash" if hw else "merge3"])
        print(f"[{i+1}/{N}] {name:14} n={n:<6} hash={ht:8.3f} mrg3={mt:8.3f} "
              f"→ {'hash' if hw else 'merge3'}  (lfp={x[0]:.2f} ln={x[1]:.2f} lmr={x[2]:.2f} lsk={x[3]:.2f})",
              flush=True)
    fout.close()

    X = np.array([r["x"] for r in rows])
    y = np.array([r["hw"] for r in rows])
    sw = np.array([max(abs(r["ht"] - r["mt"]), 1e-6) for r in rows])   # cost weight

    # margin-cost-weighted fit
    clf = LogisticRegression(C=1e6, max_iter=20000, solver="lbfgs")
    clf.fit(X, y, sample_weight=sw)
    w, b = clf.coef_[0], clf.intercept_[0]
    # sklearn: hash(y=1) iff w·X+b>0  =>  code form score = -(w·X+b), hash iff score<0
    W, B = -w, -b

    def pick(scores_lt0):
        return scores_lt0

    new_hash = clf.predict(X) == 1                       # bool, True=hash
    cur_hash = np.array([(CUR_W.dot(r["x"]) + CUR_B) < 0 for r in rows])

    def cost(pick_hash):
        s = 0.0
        for r, ph in zip(rows, pick_hash):
            if ph:
                s += r["mt"] if not math.isfinite(r["ht"]) else r["ht"]   # hash w/ fallback
            else:
                s += r["mt"]
        return s

    oracle = sum(min(r["ht"] if math.isfinite(r["ht"]) else math.inf, r["mt"]) for r in rows)
    # decisive subset: relative margin > 10% (label is meaningful)
    relm = np.array([abs(r["ht"] - r["mt"]) / min(r["ht"] if math.isfinite(r["ht"]) else r["mt"], r["mt"])
                     for r in rows])
    decisive = relm > 0.10

    print("\n================ FIT RESULT ================", flush=True)
    print(f"N = {len(rows)}   (hash-wins {int(y.sum())} / merge3-wins {int((1-y).sum())}; "
          f"decisive(rel>10%) {int(decisive.sum())})")
    print(f"\nNEW formula  (margin-cost-weighted logistic):")
    print(f"  score = {W[0]:+.4f}*lfp {W[1]:+.4f}*ln {W[2]:+.4f}*lmr {W[3]:+.4f}*lsk {B:+.4f}")
    print(f"  hash iff score < 0   (= a*A_nnz^2/n < ... see design doc)")
    print(f"\naccuracy (all)           : new {np.mean(new_hash==y)*100:5.1f}%   | current {np.mean(cur_hash==y)*100:5.1f}%")
    if decisive.sum():
        print(f"accuracy (decisive>10%)  : new {np.mean(new_hash[decisive]==y[decisive])*100:5.1f}%   | "
              f"current {np.mean(cur_hash[decisive]==y[decisive])*100:5.1f}%")
    print(f"\ntotal compute time (ms)  : always-hash {cost([True]*len(rows)):.2f} | "
          f"always-merge3 {cost([False]*len(rows)):.2f}")
    print(f"                           oracle {oracle:.2f} | current {cost(cur_hash):.2f} | new {cost(new_hash):.2f}")
    print(f"  (new vs current saves {cost(cur_hash)-cost(new_hash):+.2f} ms; gap-to-oracle "
          f"new {cost(new_hash)-oracle:+.2f} / current {cost(cur_hash)-oracle:+.2f})")

    print("\n--- NEW-formula misclassifications ---")
    nm = 0
    for r, ph in zip(rows, new_hash):
        if ph != r["hw"]:
            nm += 1
            chosen = "hash" if ph else "merge3"
            ct = (r["ht"] if math.isfinite(r["ht"]) else r["mt"]) if ph else r["mt"]
            bt = r["mt"] if r["hw"] == 0 else r["ht"]
            print(f"  {r['name']:14} chose {chosen:6} lost {ct-bt:+.3f} ms  (|Δ|={abs(r['ht']-r['mt']):.3f})")
    if nm == 0:
        print("  (none)")
    cur_misc = [r["name"] for r, ph in zip(rows, cur_hash) if ph != r["hw"]]
    print(f"\ncurrent-formula misclassifications ({len(cur_misc)}): {cur_misc}")
    print(f"\ncoeffs to paste (a*lfp b*ln c*lmr d*lsk e) = "
          f"({W[0]:.4f}, {W[1]:.4f}, {W[2]:.4f}, {W[3]:.4f}, {B:.4f})")
    print(f"csv: {out}", flush=True)


if __name__ == "__main__":
    main()
