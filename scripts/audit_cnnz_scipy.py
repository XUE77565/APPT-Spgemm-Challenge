#!/usr/bin/env python3
"""CSV cnnz 列 vs scipy 结构口径真值全量审计(docs/59 §3)。
CPU 密集(每阵一次 scipy SpGEMM)——勿在 refresh 跑批期间运行。
用法:.venv/bin/python scripts/audit_cnnz_scipy.py [--csv compare/ocean337/methods_cmp_v25_cntfix.csv] [--limit N] [--min-nnz 0]"""
import argparse, csv, warnings
warnings.filterwarnings("ignore")
import scipy.io as sio, scipy.sparse as sp, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="compare/ocean337/methods_cmp_v25_cntfix.csv")
ap.add_argument("--limit", type=int, default=10**9)
ap.add_argument("--min-n", type=int, default=0)
args = ap.parse_args()

rows = [r for r in csv.DictReader(open(args.csv)) if r.get("cnnz")]
rows.sort(key=lambda r: int(r["n"]))
bad = ok = skip = 0
for i, r in enumerate(rows):
    if i >= args.limit: break
    n = int(r["n"])
    if n < args.min_n: skip += 1; continue
    try:
        A = sp.csr_matrix(sio.mmread(f"data/ocean/square/{r['matrix']}.mtx"))
        S = A.copy(); S.data = np.ones_like(S.data)
        truth = (S @ S).nnz
    except Exception as e:
        print(f"  {r['matrix'][:24]:24s} scipy FAIL({e})", flush=True); skip += 1; continue
    ours = int(float(r["cnnz"]))
    if ours == truth:
        ok += 1
    else:
        bad += 1
        print(f"  ✗ {r['matrix'][:24]:24s} ours={ours:>12} truth={truth:>12} diff={ours-truth:+,} ({(ours/truth-1)*100:+.3f}%)", flush=True)
print(f"\n审计完成:对 {ok} / 错 {bad} / 跳过 {skip}")
