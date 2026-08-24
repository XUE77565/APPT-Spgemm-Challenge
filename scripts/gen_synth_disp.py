#!/usr/bin/env python3
"""生成 dispatcher【解析标定】用的合成矩阵族(H100 硬件成本模型,零测试集泄漏)。

设计:H100 上两条路径的 compute-only 成本假设
  hash  : t ≈ α·flops(atomic-bound accumulate)+ c·nnz(sizing/binning)+ o_h
  merge3: t ≈ β·flops(warp-merge 比较吞吐)+ c'·nnz + o_m
标定需要 (features, t_hash, t_merge3) 三元组覆盖特征空间。三个合成族:

  A. 均匀族(ER 方阵)   : n ∈ {2k,8k,32k,128k,512k,1.5M}, 平均度 d ∈ {4,16,48}
                         → 控制 (n, nnz, flop) 的均匀网格
  B. 偏斜族(重行)       : 同 n 网格, 行长 ~ 对数正态(σ 控制 skew), maxrow 逼近 HASH_CAP
  C. 带状族(局部性)     : 半带宽 b ∈ {16,128,1024}, 结构阵的列域集中特性

输出 data/synth_disp/<family>_n<k>_d<b>.mtx(coordinate real general)。
生成确定性(seed 固定);nnz 上限 2500 万。
用法:.venv/bin/python scripts/gen_synth_disp.py
"""
import os
import numpy as np
import scipy.sparse as sp
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "synth_disp"
SEED = 20260825
MAX_NNZ = 25_000_000


def save_mtx(A, path):
    A = sp.csr_matrix(A, dtype=np.float64)
    A.sort_indices()
    n, nnz = A.shape[0], A.nnz
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"% synthetic dispatcher-calibration matrix\n")
        f.write(f"{n} {n} {nnz}\n")
        coo = A.tocoo()
        # 大阵分行写,避免一次性 join 爆内存
        rows = coo.row
        cols = coo.col
        vals = coo.data
        order = np.argsort(rows, kind="stable")
        rows, cols, vals = rows[order], cols[order], vals[order]
        buf = []
        for i in range(nnz):
            buf.append(f"{rows[i]+1} {cols[i]+1} {vals[i]:.6g}")
            if len(buf) >= 200000:
                f.write("\n".join(buf) + "\n")
                buf = []
        if buf:
            f.write("\n".join(buf) + "\n")
    return n, nnz


def er_matrix(n, deg, rng):
    p = min(deg / n, 0.5)
    A = sp.random(n, n, density=p, format="csr", random_state=rng,
                  data_rvs=lambda k: rng.uniform(0.5, 1.5, k))
    return A


def skewed_matrix(n, deg, sigma, rng):
    """行长度 ~ 对数正态:控制 skew/maxrow(重行族,逼近 hash HASH_CAP)。"""
    mean_row = max(deg, 2)
    logits = rng.normal(np.log(mean_row), sigma, n)
    lens = np.minimum(np.round(np.exp(logits)).astype(int), max(2 * n, 64))
    total = int(lens.sum())
    total = min(total, MAX_NNZ)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    scale = total / max(indptr[-1], 1)
    if scale < 1.0:  # 截回总量
        lens = np.maximum((lens * scale).astype(int), 0)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
    idx = np.concatenate([rng.choice(n, size=max(l, 0), replace=False) if l < n
                          else rng.permutation(np.arange(n))
                          for l in lens]).astype(np.int32)
    ptr = indptr.astype(np.int32)
    data = rng.uniform(0.5, 1.5, len(idx))
    return sp.csr_matrix((data, idx, ptr), shape=(n, n))


def banded_matrix(n, bw, rng):
    """半带宽 b 的带状阵(列域局部,结构阵特性)。"""
    offs = rng.integers(-bw, bw + 1, size=int(n * (2 * bw + 1) * 0.25))
    rows = (rng.integers(0, n, size=len(offs)) + 0).astype(np.int64)
    rows = np.clip(rows, 0, n - 1)
    cols = np.clip(rows + offs, 0, n - 1)
    mask = rows != cols
    r1, c1 = rows[mask], cols[mask]
    r2, c2 = cols[mask], rows[mask]
    rows = np.concatenate([r1, r2]); cols = np.concatenate([c1, c2])
    vals = rng.uniform(0.5, 1.5, len(rows))
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ns = [2000, 8000, 32000, 128000, 512000, 1500000]
    plan = []
    for n in ns:
        for deg in (4, 16, 48):
            est = n * deg
            if est > MAX_NNZ:
                continue
            plan.append(("er", er_matrix, (n, deg), {"n": n, "deg": deg}))
    for n in ns:
        for sigma in (0.8, 1.6, 2.4):
            est = n * 24
            if est > MAX_NNZ:
                continue
            plan.append(("skew", skewed_matrix, (n, 24, sigma), {"n": n, "sigma": sigma}))
    for n in ns:
        for bw in (16, 128, 1024):
            if n * bw * 0.5 > MAX_NNZ:
                continue
            plan.append(("band", banded_matrix, (n, bw), {"n": n, "bw": bw}))

    print(f"共 {len(plan)} 个合成阵 → {OUT}")
    for i, (fam, fn, args, desc) in enumerate(plan, 1):
        tag = f"{fam}_n{args[0]}_x{args[1]}"
        path = OUT / f"{tag}.mtx"
        if path.exists() and path.stat().st_size > 0:
            print(f"[{i:>2}/{len(plan)}] skip {tag}")
            continue
        r2 = np.random.default_rng(SEED + i)
        A = fn(*args, r2)
        n, nnz = save_mtx(A, path)
        print(f"[{i:>2}/{len(plan)}] {tag:<22} n={n:<8} nnz={nnz:<10} -> {path.name}")


if __name__ == "__main__":
    main()
