# APPT SpGEMM Challenge — Hash or Merge? Adaptive GPU Acceleration for SpGEMM

100 个 SuiteSparse 矩阵上的 SpGEMM(C = A·A 与 C = A·Aᵀ),NVIDIA H100 / CUDA C++。
核心:**自适应 merge–hash dispatcher(Auto)**——按 h2d 阶段可读的特征(n / 列偏斜 σ / nnz²/n)
逐阵选 merge 或 hash 累加路径,配 MinHash O(nnz) 输出尺寸估计。

论文初稿见 `paper/DATE0822-1.pdf`,答辩讲稿 `paper/pre.md`(配 `paper/pre0729.pptx`)。

## 目录结构

| 目录 | 内容 |
|---|---|
| `src/` | CUDA 源码:`main.cu`(入口)、`spgemm_adaptive.cu`(dispatcher)、`spgemm_kernel_hash.cu`(hash SPA + MinHash)、`spgemm_merge.cu`(列域分桶 merge)、`spgemm_kernel_manual.cu`(ESC 基线)、`att_tiered.cu`(A·Aᵀ 三层实现)、`matrix_utils.cu`(mtx 读取)、`mempool.cu`(锁页/设备内存池) |
| `include/` | 头文件 |
| `test/` | `test_att_tiered.cu` 自生成阵正确性 + bench |
| `scripts/` | 全部运行/对比/画图脚本(均可在任意 CWD 调用) |
| `compare/` | 对比交付物。**稳定 CSV = `methods_cmp.csv`**(7 法 × 100 阵:cu/Auto/Ocean/opSparse/HSMU/dense/cublas);`aa(best1)/` = 论文引用的数据快照 + speedup 表 |
| `fig/` | 论文/答辩图(`scripts/plot_*.py` 生成) |
| `paper/` | 文章 PDF + 讲稿 + deck |
| `data/` | 96+ 基准矩阵 `.mtx`(不入库) |
| `ocean/` `spECK/` `nsparse/` `external_sota/` | 外部 SOTA 基线源码(不入库) |

详细用法(单阵运行、环境变量、profiling)见 **USAGE.md**。

## 编译

```bash
make                    # → spgemm_test(主二进制,DBG=1 开 cudaEvent profiling)
make dense              # → spgemm_dense(tiled64 dense 基线)
make cublas             # → spgemm_dense_cublas(cuBLAS FP64 基线)
# att_tiered(A·Aᵀ 三层实现):
nvcc -O3 -arch=sm_90 -std=c++17 -Iinclude src/att_tiered.cu test/test_att_tiered.cu -o test_att_tiered
```

## 复现对比(论文口径)

```bash
bash scripts/run_method_cmp.sh          # 默认 resume:已有结果跳过,秒级
REPORT_ONLY=1 bash scripts/run_method_cmp.sh   # 只读 CSV 出报告+图(~1s)
REFRESH=auto bash scripts/run_method_cmp.sh    # 重跑一列(改 binary 后)
FRESH=1 bash scripts/run_method_cmp.sh         # 备份+全量重跑
```

计时口径:Auto = cudaEvent `TOTAL − h2d − d2h`(compute-only),各基线同口径详见
`scripts/compare_methods.py`。

## 主要结果(first100 全量,geomean)

Auto 相对基线胜场:cuSPARSE **91/100**、Ocean **84/100**、opSparse/HSMU/dense/cuBLAS **100/100**。
分档与加速比表:`compare/aa(best1)/speedup_table.tex`、`compare/methods_cmp_report.txt`。

## 已知待办

- A·Aᵀ tiered 实现的实测 bench 未跑(现有 `compare/aa(best1)/aat_projected_*` 为投影值,
  实测脚本 `scripts/bench_aat_first100.py` + `gen_aat_measured_deliverables.py` 已就位)。
- `compare/paper_cmp.csv` 为 stale(device-pool 旧值),以 `methods_cmp.csv` 为准。
