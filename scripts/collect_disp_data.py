#!/usr/bin/env python3
"""采集 dispatcher 重拟合数据:(features, t_hash, t_merge3) × 矩阵集。

每矩阵 3 次运行(与 spgemm_adaptive.cu 完全同口径):
  METHOD=adaptive → 解析 [adapt] 行的 n/maxrow/skew/fp(=nnz²/n),特征与 C 端零漂移
  METHOD=hash     → hash-prof compute-only(cudaEvent TOTAL−h2d−d2h);OVERFLOW 记 flag
  METHOD=merge3   → mrg3-prof compute-only

输出增量 CSV(可断点续跑):matrix,n,nnz未知省略,maxrow,skew,fp,t_hash,t_merge3,hash_overflow
用法:.venv/bin/python scripts/collect_disp_data.py --dir data/synth_disp --out compare/disp_synth.csv
GPU 选择:外层 CUDA_VISIBLE_DEVICES=1(默认不动 GPU0 的跑批)。
"""
import argparse
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

BIN = CM.BIN
FIELDS = ["matrix", "n", "maxrow", "skew", "fp", "t_hash", "t_merge3", "hash_overflow"]


def run_one(mtx, method, timeout=CM.CALL_TIMEOUT):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD=method,
               MP_HOST_MB=os.environ.get("MP_HOST_MB", "8192"))
    try:
        r = subprocess.run([BIN, mtx], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, ""
    return r, r.stdout + r.stderr


def parse_adapt(out):
    m = re.search(r"\[adapt\] n=(\d+) maxrow=(\d+) skew=([\d.]+) → (\w+)"
                  r" \(score=([-\d.]+) fp=(\d+) mr=(\d+) sk=([\d.]+)\)", out)
    if not m:
        return None
    return {"n": int(m.group(1)), "maxrow": int(m.group(2)), "skew": float(m.group(3)),
            "fp": int(m.group(6))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    names = sorted(p[:-4] for p in os.listdir(args.dir) if p.endswith(".mtx"))
    done = set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            done.add(r["matrix"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=FIELDS)
    if not os.path.exists(args.out) or os.path.getsize(args.out) == 0:
        w.writeheader()

    print(f"采集 {len(names)} 阵(已有 {len(done)})→ {args.out}")
    # ⚠ 安全护栏(2026-08-25,GPU1 wedge 教训):病态输入不进 hash/merge3 路径
    #   fp>15亿 → merge3 gapped buffer int 溢出区(Ga/band_n32000x1024 同款炸弹)
    #   maxrow>10万 → merge3 行内长循环 + 超时 SIGKILL 在 CUDA teardown = wedge 机制
    FP_GUARD = int(os.environ.get("DISP_FP_GUARD", "1500000000"))
    MR_GUARD = int(os.environ.get("DISP_MR_GUARD", "100000"))
    for i, name in enumerate(names, 1):
        if name in done:
            continue
        p = os.path.join(args.dir, name + ".mtx")
        row = {"matrix": name}
        r, out = run_one(p, "adaptive", args.timeout)
        feat = parse_adapt(out) if out else None
        if feat is None:
            # adaptive 都没打出特征 = 该阵本身有问题,绝不盲跑 hash/merge3
            print(f"[{i}/{len(names)}] {name}: adapt 行缺失/失败(跳过,不跑两路径)", flush=True)
            row.update({"t_hash": "", "t_merge3": "", "hash_overflow": ""})
            w.writerow(row); fout.flush()
            continue
        row.update(feat)
        if feat["fp"] > FP_GUARD or feat["maxrow"] > MR_GUARD:
            why = "fp" if feat["fp"] > FP_GUARD else "maxrow"
            print(f"[{i}/{len(names)}] {name}: SKIP({why}={feat['fp'] if why=='fp' else feat['maxrow']}"
                  f" 超护栏,防 wedge)", flush=True)
            row.update({"t_hash": "", "t_merge3": "", "hash_overflow": ""})
            w.writerow(row); fout.flush()
            continue
        # t_hash
        rh, outh = run_one(p, "hash", args.timeout)
        th = CM.compute_only_from_prof(rh.stderr if rh else "", "hash-prof") if rh else None
        ovf = "OVERFLOW" in outh
        row["t_hash"] = round(th, 3) if th is not None else ""
        row["hash_overflow"] = int(ovf)
        # t_merge3
        rm, outm = run_one(p, "merge3", args.timeout)
        tm = CM.compute_only_from_prof(rm.stderr if rm else "", "mrg3-prof") if rm else None
        row["t_merge3"] = round(tm, 3) if tm is not None else ""
        w.writerow(row); fout.flush()
        print(f"[{i}/{len(names)}] {name:<22} n={feat['n']:<8} fp={feat['fp']:<10} "
              f"hash={row['t_hash'] or 'OVF':>9} merge3={row['t_merge3'] or 'FAIL':>9}", flush=True)
    fout.close()
    print("完成 →", args.out)


if __name__ == "__main__":
    main()
