#!/usr/bin/env python3
"""Element-wise value comparison of two Matrix-Market sparse matrices.

Used to verify hash-SPA output values == cuSPARSE reference (nnz match alone
can't catch a value-corruption bug in accumulate). Both inputs must be the SAME
matrix product; reads MM coordinate format.

Usage:
  .venv/bin/python scripts/compare_mtx_values.py <ref.mtx> <test.mtx> [rel_tol]

Exit 0 if nnz match AND max rel err <= rel_tol (default 1e-9); else exit 1.
"""
import sys, math

def read_mm(path):
    with open(path) as f:
        first = f.readline()
        if not first.startswith("%%MatrixMarket"):
            raise SystemExit(f"{path}: not a Matrix Market file")
        # header words: %%MatrixMarket matrix coordinate real general/symmetric
        parts = first.split()
        fmt = parts[1:4] if len(parts) >= 4 else ["matrix", "coordinate"]
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        M, N, nnz = (int(x) for x in line.split())
        d = {}
        for _ in range(nnz):
            r, c, v = f.readline().split()
            r, c, v = int(r), int(c), float(v)
            d[(r, c)] = d.get((r, c), 0.0) + v          # sum any duplicates
        return d, (M, N, nnz), "symmetric" in first

def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    ref_path, test_path = sys.argv[1], sys.argv[2]
    rel_tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9

    ref, ref_shape, ref_sym = read_mm(ref_path)
    test, test_shape, test_sym = read_mm(test_path)
    if ref_sym and not test_sym:
        # ref symmetric stores only upper triangle; expand for comparison
        add = {(c, r): v for (r, c), v in ref.items() if r != c}
        for k, v in add.items():
            ref[k] = ref.get(k, 0.0) + v

    rk, tk = set(ref), set(test)
    only_ref = rk - tk
    only_test = tk - rk
    common = rk & tk

    # global magnitude scale: relative error vs the matrix's max |value|, NOT vs
    # each (near-zero) entry — accumulation-order rounding on cancelled-to-near-zero
    # entries otherwise looks huge in per-entry rel terms but is ~1e-12 globally.
    gmax = 1.0
    for d in (ref, test):
        for v in d.values():
            if abs(v) > gmax:
                gmax = abs(v)

    max_abs = max_grel = 0.0
    worst = None
    for k in common:
        a, b = ref[k], test[k]
        d = abs(a - b)
        grel = d / gmax
        if d > max_abs:
            max_abs = d
            worst = (k, a, b, grel)
        max_grel = max(max_grel, grel)

    print(f"ref   : nnz={len(ref)}  shape={ref_shape}")
    print(f"test  : nnz={len(test)}  shape={test_shape}")
    print(f"common={len(common)}  only_ref={len(only_ref)}  only_test={len(only_test)}")
    print(f"global |max|={gmax:.3g}   max abs err = {max_abs:.3e}   max rel-vs-global = {max_grel:.3e}")
    if worst:
        print(f"worst: (row,col)={worst[0]}  ref={worst[1]:.6g}  test={worst[2]:.6g}  rel-g={worst[3]:.3e}")

    ok = (len(only_ref) == 0 and len(only_test) == 0 and max_grel <= rel_tol)
    print(f"\n=> {'PASS' if ok else 'FAIL'} (rel_tol={rel_tol}, vs global |max|)")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
