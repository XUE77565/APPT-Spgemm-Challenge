#!/usr/bin/env python3
# cuBLAS 稠密 GEMM baseline:把稀疏 A 稠密化,A@A 走 cuBLAS sgemm(torch.mm)。
# 作为新的低开销 baseline 取代 cuSPARSE(尤其小矩阵上 cuSPARSE 的 workEstimation 固定开销偏大)。
#
# A 的语义与 C 端 read_matrix_market 完全一致(scipy.mmread 自动:symmetric 展开全矩阵 /
# pattern 填 1.0;complex 取实部),经 bcsstk30 验证 nnz=2043492 与 C 端一致。
# 精度:关闭 TF32,走真 FP32 cuBLAS sgemm,与稀疏法(float32)对齐。
# 计时:仅 GEMM kernel(densify 不计入 baseline,因稀疏法无此步);warmup 后取 min。
#
# 用法:.venv/bin/python suitesparse_crawl/baseline_cublas.py [data_dir] [out.csv]

import sys, glob, time
from pathlib import Path
import numpy as np
import scipy.io as sio
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO / "data" / "first100")
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else (HERE / "baseline_cublas.csv")

# 关闭 TF32:走真 FP32 sgemm,精度与稀疏法一致(否则 TF32 快但精度低、不公平)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def read_A(path):
    """复刻 read_matrix_market 语义(scipy 已处理 symmetric 展开 / pattern=1.0)→ CSR float32。"""
    A = sio.mmread(str(path))
    if np.iscomplexobj(A):
        A = A.real
    return A.tocsr().astype(np.float32)


def time_gemm(A_csr, device):
    """densify → cuBLAS sgemm。返回 (ms, status)。OOM/ERR 返回 (None, 原因)。"""
    n = A_csr.shape[0]
    torch.cuda.empty_cache()
    try:
        Ad = torch.from_numpy(A_csr.toarray()).to(device)
        C = torch.empty((n, n), device=device, dtype=torch.float32)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); return None, "OOM_alloc"
    except RuntimeError as e:
        torch.cuda.empty_cache(); return None, f"ERR:{str(e)[:32]}"
    # 自适应 reps:小矩阵多轮取 min;大矩阵极少轮(单次已几十~百秒)
    reps = 50 if n < 2000 else (5 if n < 15000 else 2)
    try:
        torch.mm(Ad, Ad, out=C)        # warmup(初始化 cuBLAS / kernel 选优)
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.mm(Ad, Ad, out=C)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    except torch.cuda.OutOfMemoryError:
        del Ad, C; torch.cuda.empty_cache(); return None, "OOM_gemm"
    except RuntimeError as e:
        del Ad, C; torch.cuda.empty_cache(); return None, f"ERR:{str(e)[:32]}"
    ms = min(ts) * 1000.0
    del Ad, C; torch.cuda.empty_cache()
    return ms, "ok"


def main():
    device = torch.device("cuda:0")
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(device)} | TF32={torch.backends.cuda.matmul.allow_tf32}")
    files = sorted(glob.glob(str(DATA_DIR / "*.mtx")))
    rows = []
    for i, f in enumerate(files):
        name = Path(f).stem
        try:
            A = read_A(f)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] {name}: READ ERR {e}"); continue
        n, nnz = A.shape[0], A.nnz
        t, status = time_gemm(A, device)
        if t is None:
            print(f"[{i+1}/{len(files)}] {name}: n={n} nnz={nnz} → SKIP ({status})")
        else:
            print(f"[{i+1}/{len(files)}] {name}: n={n} nnz={nnz} → {t:.3f} ms")
        rows.append({"name": name, "n": n, "A_nnz": nnz, "cublas_ms": (t if t is not None else np.nan), "status": status})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)   # 增量写,大矩阵跑挂也不丢前面
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n写出 {OUT_CSV}  (ok {ok}/{len(rows)})")


if __name__ == "__main__":
    main()
