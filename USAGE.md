# 脚本用法手册

> 日期:2026-07-16
> 当前代码有三种方法:cuSPARSE(T1)+ Gustavson ESC(T4)+ **Gustavson Merge**(T4b,串行 k-way merge,对照 ESC 的 sort)。outer/colwise/inner 已移除。

---

## 一、编译与运行

### `make`
```bash
make                    # 编译 spgemm_test
make clean && make      # 清理重编
```

### `./spgemm_test <matrix.mtx> [att]`
```bash
# A·A(C = A × A)
USE_MEMPOOL=1 ./spgemm_test data/first100/1138_bus.mtx

# A·Aᵀ 上三角(att 模式)
USE_MEMPOOL=1 ./spgemm_test data/first100/1138_bus.mtx att

# 环境变量:
#   USE_MEMPOOL=0/1  pinned 内存池(0=legacy cudaMallocHost,1=arena,默认 1)
#   MP_HOST_MB=512   arena 大小(默认 256MB)
```

---

## 二、批量跑 benchmark

### `scripts/run_aa.sh`
```bash
bash scripts/run_aa.sh                         # A·A,默认 legacy
USE_MEMPOOL=1 bash scripts/run_aa.sh           # A·A,pinned 池
USE_MEMPOOL=0 AA_RESULTS_DIR=results/aa/legacy bash scripts/run_aa.sh
TIMEOUT=120 bash scripts/run_aa.sh             # 缩短每矩阵超时
```
- 跑 `data/first100/` 全部 100 矩阵,日志→`results/aa/first100_aa/log/`。
- `AA_RESULTS_DIR=...` 覆盖输出目录。

### `scripts/run_att.sh`
```bash
bash scripts/run_att.sh                        # A·Aᵀ att 模式
```

### `scripts/test_read.sh`
```bash
bash scripts/test_read.sh                     # 只测 read_matrix_market
```

### `scripts/run_all.sh`
```bash
bash scripts/run_all.sh                      # 全集(data/random)
```

---

## 三、Profiling

### `suitesparse_crawl/profile_aa.py`
```bash
.venv/bin/python suitesparse_crawl/profile_aa.py [LOG_DIR]
# 默认读 results/aa/first100_aa/log/
```
- 解析 dbg 阶段时间戳,输出:
  - `profile_aa.csv`(每矩阵×方法 分阶段明细)
  - `profile_aa_summary.csv`(每矩阵汇总)
  - `charts/profile_aa.png`(阶段构成堆叠图)
  - `charts/profile_totals.png`(cu vs gust 总耗时对照)
- 表格含:h2d / d2h / 符号(count+scan) / 计算(expand) / 排序(sort) / 去重(reduce) / 收尾(final) / 打包(pack)
  - 注:`profile_aa.py` 现已解析第三法 **Merge**(T4b)的 `Time` 与 `[merge]` 阶段;merge 的 merge 阶段映到「排序」列,与 gust.sort 直接对照

### `suitesparse_crawl/analyze_merge.py`(Merge vs ESC 专项分析)
```bash
.venv/bin/python suitesparse_crawl/analyze_merge.py [summary.csv] [out_dir]
# 默认读 suitesparse_crawl/profile_aa_summary.csv
# 图表默认写到 compare/merge_vs_esc_<时间戳>/(每次运行新建文件夹,自包含:图 + 源 CSV)
#   可用 argv[2] 或 MERGE_COMPARE_DIR=compare/xxx 指定文件夹名
```
- 打印:并行/串行 merge vs ESC 的胜负计数 + 几何均值、并行 vs 串行加速、各类别分布
- 出图到 `compare/<folder>/`(与 ocean_compute_only 等同目录约定):
  - `merge2_vs_esc_scatter.png` — merge(par) vs ESC 总耗时(log-log),虚线下方=并行 merge 赢
  - `merge2_speedup_over_serial.png` — 并行相对串行的加速 vs C 输出 nnz(>1=并行更快)
  - `merge2_winloss_by_class.png` — 并行 merge vs ESC 各类别 赢/平/输 堆叠条

### `suitesparse_crawl/profile_att.py`
```bash
.venv/bin/python suitesparse_crawl/profile_att.py [LOG_DIR]
# 默认读 results/first100_att/log/
```

---

## 四、A/B 对比(pinned 池 vs legacy)

### `scripts/ab_profile.sh`
```bash
bash scripts/ab_profile.sh                    # 交错跑两路(每矩阵 legacy→pool 背靠背)
```
- 产物:`profile_aa_legacy.csv` / `profile_aa_pool.csv`

### `suitesparse_crawl/compare_ab.py`
```bash
.venv/bin/python suitesparse_crawl/compare_ab.py \
    suitesparse_crawl/profile_aa_legacy.csv \
    suitesparse_crawl/profile_aa_pool.csv
```
- 输出:对比表(均值+中位数)+ `charts/ab_compare.png`

### `suitesparse_crawl/compare_pinned.py`
```bash
.venv/bin/python suitesparse_crawl/compare_pinned.py [no.log] [yes.log] [out.png]
# 默认读 compare/pinned-and-arena/*.log
```
- 输出:每方法两根柱(有/无 arena)的堆叠对比图

### `scripts/regen_pinned_compare.sh`
```bash
bash scripts/regen_pinned_compare.sh          # 一键:build → 两路 → profile → 对比图
```

---

## 五、Ocean 对比

### 部署(已完成)
```bash
cd ocean/
# 配置:NUM_SM=114(include/Common.h),sm_90(Makefile),已修 CUDA 12.0 兼容
make -j8                                     # 生成 spgemm + convert
```

### Ocean 用法
```bash
cd ocean/
./convert ../data/first100/1138_bus.mtx /tmp/1138_bus.csr
./spgemm /tmp/1138_bus.csr config/bench.json
./spgemm /tmp/1138_bus.csr config/bench_detail.json    # 带阶段计时(stats.json)
```

### `scripts/run_compare_ocean.sh` ★ 一键对比(纯计算)
```bash
bash scripts/run_compare_ocean.sh
```
- 流程:build 两边 → 跑 spgemm_test(USE_MEMPOOL=1)→ 跑 Ocean → 合并 → 出图
- 产物:`compare/ocean_compute_only/`
  - `compute_scatter_bucket.png`(scatter + 分桶柱状)
  - `gust_phase_vs_ocean.png`(Gustavson 阶段拆解 vs Ocean)
  - `compute_comparison.csv`(合并数据)
- 只统计计算部分(不含 H2D/D2H)

---

## 六、Ocean 部署信息

| 项 | 值 |
|---|---|
| 路径 | `ocean/` |
| 兼容性修复 | `cuda::std::numeric_limits` → `0x7FFFFFFF`;`c++14` → `c++17` |
| 配置 | `NUM_SM=114`(H100 PCIe),`SHARED_MEMORY_KB=128`,`sm_90` |
| 阶段计时 | `config/bench_detail.json`(`track_stage_time=true`)→ `stats.json` |
| 阶段 | analysis / estimation(HLL) / numeric(hash+dense+ESC) / epilogue(indirect sort) |
| 代码参考 | `kernels/AccumulatorHash.cuh`(hash 累加器)、`kernels/Epilogue.cuh`(indirect sort) |

---

## 七、关键开关汇总

| 开关 | 值 | 作用 |
|---|---|---|
| `USE_MEMPOOL` | 0/1 | pinned 池(1=arena,0=legacy cudaMallocHost) |
| `MP_HOST_MB` | 256/512/.. | arena 大小(MB) |
| `AA_RESULTS_DIR` | 路径 | run_aa.sh 的输出目录 |
| `TIMEOUT` | 秒 | run_aa.sh 每矩阵超时 |
| `CU_REF` | 0/1 | spgemm.h 里 cuSPARSE 计时块(1=启用) |
| `DBG` | 0/1 | spgemm.h 里 dbg 打桩(1=打阶段戳) |
| `WRITE_MTX` | 0/1 | spgemm.h 里是否写 .mtx(1=写) |
| `TEST_READ` | 0/1 | spgemm.h 里只测读入(1=读后退出) |

---

## 八、目录结构概览

```
spgemm-challenge/
├── src/
│   ├── main.cu                        # 入口:cuSPARSE(T1) + Gustavson(T4)
│   ├── spgemm_kernel_cusparse.cu      # cuSPARSE A·A / A·Aᵀ
│   ├── spgemm_kernel_manual.cu        # Gustavson ESC(expand→sort→reduce→finalize)
│   ├── spgemm_kernel_formulations.cu  # att 方法(outer/colw/inner 的 att 版本)+ 共用 esc_merge
│   ├── matrix_utils.cu                # read/write Matrix Market
│   └── mempool.cu                     # pinned host arena + pinned_d2h_alloc/host_free
├── include/
│   ├── spgemm.h                       # 宏(TEST_READ/DBG/WRITE_MTX/CU_REF/USE_MEMPOOL)+ 函数声明
│   └── memool.h                       # 池 API
├── scripts/
│   ├── run_aa.sh                      # A·A benchmark
│   ├── run_att.sh                     # A·Aᵀ att benchmark
│   ├── run_all.sh                     # 全集
│   ├── test_read.sh                   # 只测读入
│   ├── ab_profile.sh                  # 交错 A/B(legacy vs pool)
│   ├── regen_pinned_compare.sh        # 一键 pinned 对比
│   └── run_compare_ocean.sh           # ★ 一键 vs Ocean(纯计算)
├── suitesparse_crawl/
│   ├── profile_aa.py                  # A·A profiling
│   ├── profile_att.py                 # A·Aᵀ profiling
│   ├── compare_ab.py                  # A/B 对比(legacy vs pool)
│   └── compare_pinned.py             # 有/无 arena 堆叠柱对比
├── ocean/                             # Ocean SpGEMM(已部署)
│   ├── spgemm                         # Ocean 可执行
│   ├── convert                        # .mtx → .csr
│   ├── config/bench.json              # 标准 benchmark
│   └── config/bench_detail.json       # 带阶段计时
├── compare/                           # 对比图/数据
├── ref/                               # 研究文档(创新方向/文献/融合方案/Ocean 解读)
├── worklog/                           # 工作日志(profiling/h2d 副作用/changelog)
└── data/first100/                     # 100 个测试矩阵(.mtx)
```
