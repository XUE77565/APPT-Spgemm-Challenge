#!/usr/bin/env python3
"""DIRECT5 / LLB A/B(docs/29 §3):3 配置 × 靶阵+回归阵,compute-only 口径(同 CSV)。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compare_methods as cm

# (靶阵 DIRECT5)(靶阵 LLB)(回归)
MATS = ["pre2", "mult_dcop_03", "c-58", "bloweya", "soc-Slashdot0902", "email-Enron",
        "TSOPF_RS_b39_c7", "exdata_1", "Ge99H100",
        "pwtk", "333SP", "bcsstk30", "Ga3As3H12"]
CONFIGS = [("base", {}), ("D5", {"DIRECT5": "1"}), ("LLB", {"LLB": "1"})]

# CSV cnnz 对照
import csv
cnnz_ref = {}
for p in ["compare/ocean337/methods_cmp_v9_oceansym.csv"]:
    for r in csv.DictReader(open(p)):
        if r.get("cnnz"): cnnz_ref[r["matrix"]] = r["cnnz"]

print(f"{'matrix':22s} {'base':>9s} {'D5':>9s} {'LLB':>9s}   cnnz(参考 {len(cnnz_ref)})", flush=True)
for m in MATS:
    p = cm.find_mtx(m)
    if not p:
        print(f"{m}: 未找到", flush=True); continue
    row = []
    cnnz = None
    for name, extra in CONFIGS:
        for k, v in extra.items(): os.environ[k] = v
        for k in ("DIRECT5", "LLB"):
            if k not in extra: os.environ.pop(k, None)
        t0 = time.time()
        r = cm.run_spgemm_method(p, "adaptive")
        el = time.time() - t0
        if r is None:
            row.append("DNF/TO")
        else:
            row.append(f"{r[0]:.2f}")
            cnnz = r[2]
        #print(f"  [{name}] {r[0] if r else 'NA'} ({el:.0f}s)", flush=True)
    ref = cnnz_ref.get(m, "?")
    ok = "=" if (cnnz and str(cnnz) == str(ref)) else ("≠" if cnnz else "")
    print(f"{m:22s} {row[0]:>9s} {row[1]:>9s} {row[2]:>9s}   cnnz={cnnz} ref={ref} {ok}", flush=True)
print("done", flush=True)
