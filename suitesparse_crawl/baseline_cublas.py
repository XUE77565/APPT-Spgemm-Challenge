#!/usr/bin/env python3
# cuBLAS 稠密 GEMM baseline(Python ctypes 直调系统 libcublas,零下载/零额外依赖)。
# 把稀疏 A 稠密化,A@A 走 cuBLAS sgemm,作为新的低开销 baseline 取代 cuSPARSE。
#
# A 的语义与 C 端 read_matrix_market 一致(scipy.mmread 自动:symmetric 展开全矩阵 /
# pattern 填 1.0;complex 取实部),经 bcsstk30 验证 nnz=2043492 与 C 端一致。
# 精度:FP32 sgemm(cuBLAS 原生 FP32,无 TF32 降精度,与稀疏法 float32 对齐)。
# 计时:仅 sgemm kernel(densify/transfer 不计入,稀疏法无此步);warmup 后取 min。
# A@A 自乘:opN/opN 即可(行列序差异对计时无影响,只算时间不算结果)。
#
# 用法:.venv/bin/python suitesparse_crawl/baseline_cublas.py [data_dir] [out.csv]

import sys, glob, time, ctypes
from ctypes import c_int, c_float, c_void_p, POINTER, byref, c_size_t
from pathlib import Path
import numpy as np
import scipy.io as sio
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (REPO / "data" / "first100")
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else (HERE / "baseline_cublas.csv")

CUDA_LIBDIR = "/usr/lib/x86_64-linux-gnu"

def _load(name):
    for cand in (name, f"{CUDA_LIBDIR}/{name}", f"{CUDA_LIBDIR}/{name}.12"):
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    raise OSError(f"找不到 {name}(确认 CUDA 已装)")

cudart = _load("libcudart.so")
cublas = _load("libcublas.so")

# ---- cudart ----
cudart.cudaMalloc.argtypes = [POINTER(c_void_p), c_size_t];        cudart.cudaMalloc.restype = c_int
cudart.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]; cudart.cudaMemcpy.restype = c_int
cudart.cudaDeviceSynchronize.argtypes = [];                        cudart.cudaDeviceSynchronize.restype = c_int
cudart.cudaFree.argtypes = [c_void_p];                             cudart.cudaFree.restype = c_int
cudaMemcpyHostToDevice = 1

# ---- cublas(handle 一次创建,全程序复用)----
cublas.cublasCreate_v2.argtypes = [POINTER(c_void_p)];             cublas.cublasCreate_v2.restype = c_int
cublas.cublasDestroy_v2.argtypes = [c_void_p];                     cublas.cublasDestroy_v2.restype = c_int
cublas.cublasSgemm_v2.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_int,
                                  POINTER(c_float), c_void_p, c_int, c_void_p, c_int,
                                  POINTER(c_float), c_void_p, c_int]
cublas.cublasSgemm_v2.restype = c_int
CUBLAS_OP_N = 0

HANDLE = c_void_p()
if cublas.cublasCreate_v2(byref(HANDLE)) != 0:
    raise RuntimeError("cublasCreate_v2 失败")


def read_A(path):
    """复刻 read_matrix_market 语义(scipy 已处理 symmetric 展开 / pattern=1.0)→ CSR float32。"""
    A = sio.mmread(str(path))
    if np.iscomplexobj(A):
        A = A.real
    return A.tocsr().astype(np.float32)


def time_gemm(A_csr):
    """densify → cuBLAS sgemm。返回 (ms, status)。"""
    n = A_csr.shape[0]
    bytes2 = n * n * 4
    Ad = np.ascontiguousarray(A_csr.toarray(), dtype=np.float32)
    dA = c_void_p(); dC = c_void_p()
    # 分配设备显存(大矩阵可能 OOM → 跳过)
    if cudart.cudaMalloc(byref(dA), bytes2) != 0:
        return None, "OOM_dA"
    if cudart.cudaMalloc(byref(dC), bytes2) != 0:
        cudart.cudaFree(dA); return None, "OOM_dC"
    if cudart.cudaMemcpy(dA, Ad.ctypes.data, bytes2, cudaMemcpyHostToDevice) != 0:
        cudart.cudaFree(dA); cudart.cudaFree(dC); return None, "ERR_memcpy"
    alpha = c_float(1.0); beta = c_float(0.0)
    # warmup(初始化 cuBLAS / kernel 选优)
    r = cublas.cublasSgemm_v2(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                              byref(alpha), dA, n, dA, n, byref(beta), dC, n)
    if r != 0:
        cudart.cudaFree(dA); cudart.cudaFree(dC); return None, f"ERR_sgemm({r})"
    cudart.cudaDeviceSynchronize()
    reps = 50 if n < 2000 else (5 if n < 15000 else 2)
    ts = []
    for _ in range(reps):
        cudart.cudaDeviceSynchronize()
        t0 = time.perf_counter()
        cublas.cublasSgemm_v2(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                              byref(alpha), dA, n, dA, n, byref(beta), dC, n)
        cudart.cudaDeviceSynchronize()
        ts.append(time.perf_counter() - t0)
    cudart.cudaFree(dA); cudart.cudaFree(dC)
    return min(ts) * 1000.0, "ok"


def main():
    print(f"cuBLAS baseline(ctypes 直调系统 libcublas)")
    files = sorted(glob.glob(str(DATA_DIR / "*.mtx")))
    rows = []
    for i, f in enumerate(files):
        name = Path(f).stem
        try:
            A = read_A(f)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] {name}: READ ERR {e}"); continue
        n, nnz = A.shape[0], A.nnz
        t, status = time_gemm(A)
        if t is None:
            print(f"[{i+1}/{len(files)}] {name} → SKIP ({status})")
        else:
            print(f"[{i+1}/{len(files)}] {name} → {t:.3f} ms")
        rows.append({"name": name, "n": n, "A_nnz": nnz, "cublas_ms": (t if t is not None else np.nan), "status": status})
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)   # 增量写,大矩阵跑挂也不丢
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n写出 {OUT_CSV}  (ok {ok}/{len(rows)})")
    cublas.cublasDestroy_v2(HANDLE)


if __name__ == "__main__":
    main()
