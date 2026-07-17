#!/usr/bin/env python3
# Ocean SpGEMM baseline(compute-only = 各 GPU 阶段求和)。
# Ocean 的 timed 阶段(analysis/estimation/numeric/epilogue/prologue)全是 GPU 计算,
# 文件 IO + 二进制启动在阶段之外(类比稀疏法不计 mtx 读入)。单位:ms(stats.json 原生 ms,
# 注意 run_compare_ocean.sh 曾误 ×1000;本脚本不乘)。与稀疏法 compute-only(去 h2d/d2h)同口径。
#
# 用法:.venv/bin/python suitesparse_crawl/baseline_ocean.py [data_dir] [out.csv]

import sys, glob, json, subprocess
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO / "data" / "first100")
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else (HERE / "baseline_ocean.csv")
OCEAN = REPO / "ocean"
CSR_DIR = Path("/tmp/ocean_csr")
CSR_DIR.mkdir(exist_ok=True)
BENCH = OCEAN / "config" / "bench_detail.json"
STATS = OCEAN / "stats.json"
TIMEOUT = 300


def ocean_compute_ms(mtx):
    """跑一次 Ocean bench_detail,读 stats.json 求 GPU 阶段和(ms)。"""
    name = Path(mtx).stem
    csr = CSR_DIR / f"{name}.csr"
    if not csr.exists():
        subprocess.run([str(OCEAN / "convert"), str(mtx), str(csr)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    try:
        # cwd=OCEAN:spgemm 把 stats.json 写到 CWD,故在其目录下跑 → 写到 ocean/stats.json
        r = subprocess.run([str(OCEAN / "spgemm"), str(csr), str(BENCH)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=TIMEOUT, cwd=str(OCEAN))
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    if r.returncode != 0:
        return None, f"rc={r.returncode}"   # 不读 stats(避免上一个矩阵的脏数据)
    try:
        s = json.load(open(STATS))
        t = s["timing"]
        an  = t["analysis"]["product_calc"] + t["analysis"]["reduce"] + t["analysis"]["mem_cpy"]
        est = (t["estimation"]["hll_construct"] + t["estimation"]["hll_merge"]
               + t["estimation"]["malloc"] + t["estimation"]["sampling"])
        num = sum(t["numeric"].values())
        epi = t["epilogue"]["sort"] + t["epilogue"]["copy"] + t["epilogue"]["scan"]
        return (an + est + num + epi + t["prologue"]), "ok"   # 原生 ms,不 ×1000
    except Exception as e:
        return None, f"ERR:{str(e)[:24]}"


def main():
    print(f"Ocean baseline(compute-only = GPU 阶段求和,ms)")
    files = sorted(glob.glob(str(DATA_DIR / "*.mtx")))
    rows = []
    for i, f in enumerate(files):
        name = Path(f).stem
        t, status = ocean_compute_ms(f)
        if t is None:
            print(f"[{i+1}/{len(files)}] {name} → SKIP ({status})")
        else:
            print(f"[{i+1}/{len(files)}] {name} → {t:.3f} ms")
        rows.append({"name": name, "ocean_ms": (t if t is not None else float("nan")), "status": status})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)   # 增量写
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n写出 {OUT_CSV}  (ok {ok}/{len(rows)})")


if __name__ == "__main__":
    main()
