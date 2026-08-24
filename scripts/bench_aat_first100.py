#!/usr/bin/env python3
"""bench_aat_first100.py — run the hardened att_aat_tiered (FP32, tiered C=A*A^T) on each
first100 matrix via `./test_att_tiered benchmtx <file>`, collect measured AAT timing ->
compare/aa(best1)/aat_measured_timing.csv  (columns: matrix,aat_ms,nnzA,nnzC,LMH,status).

GPU required (run on a healthy GPU; CUDA_VISIBLE_DEVICES honored from the environment).
Each matrix is bounded by a per-matrix timeout; failures/overflows are recorded, not fatal.
"""
import csv, glob, os, re, subprocess, time

BIN      = "./test_att_tiered"
DATA     = "data/first100"
OUT      = "compare/aa(best1)/aat_measured_timing.csv"
TIMEOUT  = 150          # seconds per matrix (safety; tiny matrices finish in <1s)

# AAT_MS <name> total=0.1234 nnzA=.. nnzC=.. L/M/H=a/b/c status=0 ovf=0
RE_TOTAL  = re.compile(r"total=([\d.]+)")
RE_NNZC   = re.compile(r"nnzC=(\d+)")
RE_NNZA   = re.compile(r"nnzA=(\d+)")
RE_LMH    = re.compile(r"L/M/H=(\d+)/(\d+)/(\d+)")
RE_STATUS = re.compile(r"status=(\d+)")
RE_OVF    = re.compile(r"ovf=(\d+)")

def main():
    mtxs = sorted(glob.glob(os.path.join(DATA, "*.mtx")))
    print(f"[bench_aat] {len(mtxs)} matrices; binary={BIN}; out={OUT}")
    rows = []
    for i, p in enumerate(mtxs):
        name = os.path.splitext(os.path.basename(p))[0]
        t0 = time.time()
        rec = {"matrix": name, "aat_ms": "", "nnzA": "", "nnzC": "", "LMH": "", "status": ""}
        try:
            r = subprocess.run([BIN, "benchmtx", p], capture_output=True, text=True,
                               timeout=TIMEOUT)
            out = (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            rec["status"] = "timeout"; rows.append(rec)
            print(f"  [{i+1:3d}/{len(mtxs)}] {name:24s} TIMEOUT({TIMEOUT}s) skip")
            continue
        status = RE_STATUS.search(out); ovf = RE_OVF.search(out); total = RE_TOTAL.search(out)
        if not total or (status and status.group(1) != "0") or (ovf and ovf.group(1) != "0"):
            rec["status"] = "fail"; rows.append(rec)
            tail = out.strip().replace("\n", " ")[-100:]
            print(f"  [{i+1:3d}/{len(mtxs)}] {name:24s} FAIL/overflow skip :: {tail}")
            continue
        aat = float(total.group(1))
        rec.update({"aat_ms": f"{aat:.4f}",
                    "nnzA": (RE_NNZA.search(out) or ["",""])[1] if RE_NNZA.search(out) else "",
                    "nnzC": (RE_NNZC.search(out) or ["",""])[1] if RE_NNZC.search(out) else ""})
        lmh = RE_LMH.search(out)
        rec["LMH"] = f"{lmh.group(1)}/{lmh.group(2)}/{lmh.group(3)}" if lmh else ""
        rec["status"] = "ok"; rows.append(rec)
        print(f"  [{i+1:3d}/{len(mtxs)}] {name:24s} {aat:8.4f} ms  nnzC={rec['nnzC']:>9}  {time.time()-t0:4.1f}s")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["matrix", "aat_ms", "nnzA", "nnzC", "LMH", "status"])
        w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n[bench_aat] wrote {OUT}: {len(ok)}/{len(rows)} ok, "
          f"{sum(1 for r in rows if r['status']=='fail')} fail, "
          f"{sum(1 for r in rows if r['status']=='timeout')} timeout")

if __name__ == "__main__":
    main()
