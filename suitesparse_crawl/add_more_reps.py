#!/usr/bin/env python3
"""
在现有 representatives.csv 基础上【按密度分层】补充代表矩阵,把每类补到 TARGET 个。
保留已选/已下载的,只新增,不产生孤儿。新增项按密度分位选取,保证稀疏度覆盖。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CLF = pd.read_csv(HERE / "classification_square.csv")
REPS = pd.read_csv(HERE / "representatives.csv")
COLS = list(REPS.columns)            # class,id,name,group,n,nnz,density_pct,sparsity_pct,in_local_data,url_mm

TARGET = 8
CAP_N, CAP_NNZ = 500_000, 10_000_000     # 可 profiling 尺寸档(与 classify 一致)
CLASS_ORDER = ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]

have = set(REPS["name"])
added = []
for c in CLASS_ORDER:
    cur = len(REPS[REPS["class"] == c])
    need = TARGET - cur
    if need <= 0:
        print(f"[{c}] 已有 {cur},跳过")
        continue
    cand = CLF[(CLF["class"] == c) & (CLF["n"] <= CAP_N) & (CLF["nnz"] <= CAP_NNZ)
               & (~CLF["name"].isin(have))]
    pool_label = f"tier(n≤{CAP_N:,}&nnz≤{CAP_NNZ:,})"
    if len(cand) < need:                       # 该类 tier 不够,放宽到全类
        cand = CLF[(CLF["class"] == c) & (~CLF["name"].isin(have))]
        pool_label = "放宽到全类"
    cand = cand.sort_values("density_pct")
    m = len(cand)
    idx = [min(m - 1, max(0, int(round(q * (m - 1))))) for q in np.linspace(0.05, 0.95, need)]
    picked = cand.iloc[idx].drop_duplicates("name")
    picked = picked.reindex(columns=COLS)
    picked["in_local_data"] = False            # 新增的尚未下载
    added.append(picked)
    have.update(picked["name"])
    print(f"[{c}] 已有 {cur} + 新增 {len(picked)} → {cur+len(picked)}  ({pool_label})")
    for _, r in picked.iterrows():
        print(f"    {r['name']:<30} n={int(r['n']):>9,}  nnz={int(r['nnz']):>10,}  "
              f"密度={r['density_pct']:.3e}%")

if added:
    new = pd.concat(added, ignore_index=True)
    out = pd.concat([REPS, new], ignore_index=True)
    out.to_csv(HERE / "representatives.csv", index=False)
    print(f"\nrepresentatives.csv 更新:{len(REPS)} → {len(out)} 个代表")
else:
    print("\n无需新增")
