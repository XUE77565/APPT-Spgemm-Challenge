#!/usr/bin/env python3
"""跑 spgemm_dense(-O0,朴素 dense matmul)在 first100 全阵,结果存 compare/dense_baseline.csv(matrix,dense_ms)。
   dense 的 cudaEvent 只计 kernel(不含 h2d/d2h)= 与 hash/Ocean compute-only 同口径。
   每阵 DENSE_TIMEOUT 秒(默认 200);超时记 'timeout',失败记 'fail'。断点续跑(跳过已 cached)。
   用法:.venv/bin/python scripts/run_dense_baseline.py   [DENSE_TIMEOUT=200]
"""
import os, sys, re, csv, subprocess, glob, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_dense")
DATA = os.path.join(REPO, "data/first100")
CACHE = os.path.join(REPO, "compare/dense_baseline.csv")
TIMEOUT = int(os.environ.get("DENSE_TIMEOUT", "200"))

def main():
    if not os.path.exists(BIN):
        print(f"[dense] binary 不存在 {BIN},先 make dense", file=sys.stderr); sys.exit(1)
    mtxs = sorted(glob.glob(os.path.join(DATA, "*.mtx")))
    done = {}
    if os.path.exists(CACHE):
        for row in csv.DictReader(open(CACHE)):
            done[row["matrix"]] = row["dense_ms"]
    rows = []
    for mtx in mtxs:
        name = os.path.splitext(os.path.basename(mtx))[0]
        if name in done:
            rows.append({"matrix": name, "dense_ms": done[name]})
            continue
        out = "/tmp/dense_out.mtx"
        t0 = time.time()
        try:
            r = subprocess.run([BIN, mtx, out], capture_output=True, text=True, timeout=TIMEOUT)
            m = re.search(r"Kernel time:\s*([0-9.]+)", r.stdout)
            ms = f"{float(m.group(1)):.3f}" if m else "fail"
        except subprocess.TimeoutExpired:
            ms = "timeout"
        except Exception:
            ms = "fail"
        wall = time.time() - t0
        rows.append({"matrix": name, "dense_ms": ms})
        print(f"[{name}] dense={ms} ms (wall {wall:.1f}s)", flush=True)
        with open(CACHE, "w") as f:
            w = csv.DictWriter(f, fieldnames=["matrix", "dense_ms"])
            w.writeheader(); w.writerows(rows)
        if os.path.exists(out):
            try: os.remove(out)
            except OSError: pass
    print(f"[dense] done → {CACHE} ({len(rows)} matrices)", flush=True)

if __name__ == "__main__":
    main()
