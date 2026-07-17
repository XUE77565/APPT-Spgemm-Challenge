# Session 总览:ESC 全局排序 → k-way merge 加速

> 日期:2026-07-16,分支 `feature/optiESC`。
> 目标(用户):把 Gustavson ESC 里的 `thrust::sort_by_key` 全局排序改成 k-way merge,利用「A 是 CSR → 每行列有序」的结构省掉 sort+reduce。
> **ESC baseline 完全保留**;merge 为并列新函数。全程未提交,改动留工作树。
> 分阶段详记见 [[merge_serial_v1_2026-07-16]]、[[merge_parallel_v2_2026-07-16]]。

---

## 起点(已存在)

- `spgemm_self_product_manual`(Gustavson ESC):`count → scan → expand → sort_by_key → reduce_by_key → finalize`。
- 瓶颈(profiling):`sort_by_key` 均值 0.75ms,bcsstk30 **21.6ms**(占该阵 34%);真正计算 expand 才 0.13ms。CUB radix 已到顶,「换 sort 实现」无效。

## 阶段 1:串行 merge v1(摸交叉点 + 定瓶颈)

- 新增 `src/spgemm_merge.cu`:`expand_serial_kernel`(每行 thread0 串行写连续有序段)+ `merge_serial_kernel`(thread0 k-way merge+去重求和)+ host `spgemm_self_product_merge`。
- 正确性:全 100 矩阵 C_nnz 与 cu/ESC 三者完全一致。
- 结果:赢 49 / 输 39,几何均值 1.377×;排除 bp_* 重行族后 1.097×。
- **两个瓶颈**:① bp_* 重行(822×822 但一行 num_k≈300)单 thread0 straggler → 11–17× 慢;② 大矩阵每输出重读 num_k 段头(散列全局读,latency-bound,bcsstk30 merge 79ms)。

## 阶段 2:并行 merge v2(治本,翻盘 ESC)

- 同文件加 `merge_warp_kernel` + host `spgemm_self_product_merge2`。两处相对 v1:
  - **① 无 expand 阶段(Tier 0 融合)**:merge 直接读 A,砍掉「写 COO + 读回」来回 traffic。
  - **② 行内并行(Tier 2)**:每行一个 block=1 warp(32 线程),`__shfl_xor_sync` min/sum 归约协作归并。
- 正确性:全 100 矩阵四法(cu/ESC/serial/par)C_nnz 完全一致。
- 结果:**赢 85 / 输 12,几何均值 0.910×(整体比 ESC 快)**;并行 vs 串行 1.50× 加速。
  - merge kernel 本身 0.45ms < ESC sort 0.75ms;4 类全不输 ESC,且与 cuSPARSE 基本打平。
  - 仍输的 12 个:小矩阵 + 重行(bp_*/bcsstk08/bcspwr01)—— sort 固定开销低、N 小压不过。
  - ⚠️ 上面 0.910×/85-12 是 **legacy(未开 arena)** 数据;阶段 3 的 arena 重测更准,以此为准。

## 阶段 3:分析深化 + arena 重测 + cuSPARSE 对比

### arena 发现与重测(关键)
- 发现 `scripts/run_aa.sh` 未设 `USE_MEMPOOL` → 默认宏 `0` → 全量 benchmark 走 **legacy `cudaMallocHost`**(每次 D2H 现锁页),d2h 膨胀且抖动(小矩阵 d2h ~0.7-1.4ms)。merge 的 D2H 经 `pinned_d2h_alloc`,arena-gated(`g_use_mempool` 真→arena,假→cudaMallocHost)。
- `USE_MEMPOOL=1 bash scripts/run_aa.sh` 重测,d2h 降到 ~0.01-0.02ms。
- **arena 下结果(作准)**:par vs ESC **几何均值 0.747×、赢 89/输 10、4 类全赢**;并行 vs 串行 2.01× 加速。上轮 B 类「d2h 噪声输」全部翻盘为赢。

### merPar vs cuSPARSE(新加 par/cu 列)
- `profile_aa.py`(逐矩阵 `mrgP/cu` + 类别 `par/cu`)+ `analyze_merge.py`(汇总 `par vs cuSPARSE` 行 + 类别 `par/cu`)加列;类别表比值改**几何均值**(原算术均值被 bp_* 带偏:Highly par/gust 错显 1.13,实际 0.89)。
- **结果:par vs cuSPARSE 几何均值 0.861×、赢 83/输 15** → merPar **同时赢 ESC 和 cuSPARSE**。
- 输 cu 的 15 个分三类:① **小矩阵+重行**(bp_*×9 + bcsstk08,10 个)—— cu 的 hash SPA 处理重行 >> k-way merge;② **超大 bcsstk30**(1 个,1.11×,cu 精调 hash kernel 天花板);③ **边界小矩阵**(bcsstm13/can_256/bcsstk13/arc130,4 个,1.08-1.16×)—— merge kernel 本身 ≥ cu,输在 6 段 pipeline 的 launch/sync 固定开销。

### profiling bug 修复
- `profile_aa.py` 的 `method_durations` 原按 PHASE_ORDER 标签顺序算时长,但 merge 执行顺序是 `merge→final(行scan)→compact`,与 PHASE_ORDER(compact 在 final 前)不一致 → `final` 出现负值、compact/pack 错分。改成**按真实时间戳排序**算时长。merge kernel 时间本就正确,只影响 compact/final/pack 的小开销拆分。

### merPar 慢矩阵分析(对 ESC,前后对照)
- legacy 下 15 个慢分两类:重行算法性(bp_*+bcsstk08:merge kernel 0.96-1.61ms vs ESC sort 0.12-0.26ms)+ d2h/overhead 噪声(bcsstk28/bcspwr01/…)。**开 arena 后后者全翻盘为赢,只剩重行算法性输** —— 印证噪声判断。

## 阶段 4:baseline 从 cuSPARSE 换成 cuBLAS(Python ctypes 直调)

> 用户判断 cuSPARSE 不是好 baseline(同类稀疏库,小矩阵上 workEstimation 固定开销大、不干净)。改用 cuBLAS **稠密 GEMM** 作 baseline,更能判别「该稀疏还是该稠密」。

- **实现选择**:torch/cupy 下载成本过高(torch+cu12 依赖 ~2-3GB,pytorch.org 不通、阿里云 1.7MB/s 慢、wget 挂过)。最终 **ctypes 直调系统已装的 `/usr/lib/.../libcublas.so`(CUDA 12.0),零下载**,仍是「Python 调 cuBLAS」。
- **`suitesparse_crawl/baseline_cublas.py`**:scipy 读 A(与 C 端同语义:symmetric 展开/pattern=1.0,已验 bcsstk30 nnz 一致)→ 稠密化 float32 → `cublasSgemm_v2`(FP32,关 TF32 公平)→ 计时(仅 kernel,warmup 取 min)。A@A 自乘 opN/opN。全 100 矩阵 0 OOM/0 skip。
- **集成进 `analyze_merge.py`**:cuBLAS 成**主 baseline**(汇总置顶 `par vs cuBLAS`、by-class 加 `cuBLAS`+`par/cuBLAS`、新图 `merge2_vs_cublas_scatter.png`);cuSPARSE 标「旧 baseline」保留。
- **结果(crossover 戏剧性)**:
  - par vs cuBLAS:赢 22 / 输 78,几何均值 **2.245×**(整体输 —— cuBLAS 在大量小矩阵上极快)。
  - 分类 par/cuBLAS:Dense 8.75×、Mildly 5.08×(小矩阵稠密 GEMM 近最优,merge 输);Highly 1.49×;**Extremely sparse 0.12×(merge 快 ~8× —— 大 n 稀疏,稠密 O(n³) 灾难)**。
  - 大矩阵 bcsstk30:cuBLAS 1172ms vs merge(par) 16ms(merge 快 ~73×);bcsstk32 cuBLAS 4388ms。
- **判据**:vs cuSPARSE/ESC(同类稀疏)merge 整体赢;vs cuBLAS(稠密)小矩阵输、大稀疏矩阵碾压 —— **cuBLAS 是更有判别力的 baseline**(揭示「何时该稀疏」)。

### 计时口径:端到端 vs compute-only(可切)
- 默认 `*_time`(merge2_time/man_time/cu_time)= **端到端**(chrono 包整个函数,含 h2d+d2h);cuBLAS `cublas_ms` = **纯 sgemm kernel**(连 densify 都没算)→ 不对称,cuBLAS 被低估。
- **compute-only**(用户选 B):稀疏法时间 = 各 dbg 阶段求和**去 h2d/d2h**(`profile_aa.py` 新增 `*_compute_t` 列、`analyze_merge.py` 用之);cuBLAS 保持 kernel。两边只比 GPU 计算,公平。
  - 结果(compute-only):par vs cuBLAS 几何均值 **1.293×**(端到端 2.245×)、par vs gust **0.634×**、并行 vs 串行 **2.45×**。Highly sparse par/cuBLAS 从 1.49×→**0.91×(基本打平)**,Extremely 0.12×→**0.06×**。
- 切换:analyze_merge 默认 compute-only(读 `*_compute_t`);要看端到端改回读 `*_time`。

### 加 Ocean baseline(compute-only = GPU 阶段求和)
- `suitesparse_crawl/baseline_ocean.py`:每矩阵跑 `ocean/spgemm csr bench_detail.json`(cwd=ocean),读 `stats.json` 求 analysis+estimation+numeric+epilogue+prologue 阶段和。
- **修了旧 `run_compare_ocean.sh` 两个 bug**:① spgemm 把 stats.json 写 **CWD** 不是 ocean/(旧脚本读 ocean/stats.json → 脏数据,这就是 three_method_profiling/ocean_phases.csv 全=bcsstk30 的原因);② stats.json 单位是 **ms**,旧脚本误 `×1000`(当秒)。
- 验证:bcsstk30 wall-clock 0.9s,阶段和 2.293(若是秒就超 wall,不可能)→ 确是 ms。Ocean compute bcsstk30≈2.4ms、bcsstk32≈1.8ms(hash SpGEMM 很快)。全 100 ok。
- **结果(compute-only)par vs Ocean:赢 69 / 输 29,几何均值 1.021×(整体打平)**。类别 par/Ocean:Dense 0.51×、Mildly 0.79×、**Highly 1.59×(Ocean 赢)**、Extremely 0.87×。
- **关键判据**:Ocean 是 merge 在**大矩阵上的真正对手**(Highly sparse 上 hash SpGEMM 1.59× 快于 merge)—— 与 cuBLAS 相反(merge 大矩阵碾压 cuBLAS,却输 Ocean)。即大稀疏矩阵上 hash(Ocean)> merge > dense(cuBLAS)。
- 集成:profile_aa.py 合并 ocean_ms(逐矩阵/类别表加 Ocean 列);analyze_merge.py 加 `par vs Ocean` 行 + 柱状图第 4 方法。

## 阶段 5:终极 6 法对比 + par vs Ocean 实情 + 严谨性/口径审计

### 终极脚本 `scripts/run_full_compare.sh`
- 一键跑全 6 法(cuSPARSE / gust(ESC) / merge(ser) / merge(par) / Ocean / cuBLAS)first100,出比较日志 + **6 方法 compute-only 柱状图**(`profile_methods_bar.png`),打包到 `compare/full_compare_<ts>/`。
- stage 2 内联循环跑 spgemm_test(USE_MEMPOOL=1);stage 3/4 tee Python baseline 进度;每矩阵打印 `[i/100]`。`profile_aa.py` 的 `chart_methods_bar` 扩到 6 方法(加 merge-ser)。

### compute-only 口径审计(结论:一致、无 bug)
- 4 个 C 法 = 各 dbg 阶段求和**去 h2d/d2h**;`h2dmalloc`(cuSPARSE 的诊断桩)因**不在 `CU_PHASES`** 被 `method_durations` 自动排除(正是 changelog 想要)→ **未误计入**(此前一度怀疑的"漏算"是读了旧 CSV 的误判,实测 `[cu] compute=7.68ms` 正确纳入)。
- cuBLAS = 仅 `cublasSgemm_v2`(densify/malloc/H2D/D2H 在计时区外);Ocean = GPU 阶段求和(文件IO/启动在阶段外)。
- 判据:每法 compute-only = **它自己在 GPU 上的全部工作,减主机 h2d/d2h**,口径一致。小 caveat:C 法的 `pack`(D2D 打包)+ 设备 malloc 计入,而 cuBLAS 排除了它的 malloc → cuBLAS 略占便宜,但都很小。

### par vs Ocean 实情(诚实定位:**par 没比 SOTA 快**)
- par vs Ocean(compute-only,全 100):**赢 48 / 输 48**(个数打平),但**几何均值 par/Ocean = 1.225×(par 慢 22%)**,总耗时 par 69ms vs Ocean 39ms(**慢 80%**)。
- 大矩阵(C_nnz>100K,18 个):par **慢 2×**(39 vs 16ms);小/中(82 个):1.09× 接近。
- 本质:**Ocean 的 hash 累加器 > par 的 k-way merge**(hash O(中间项) O(1) 插入;par 每输出一次归约+散列读)。规模(bcsstk30 4.8×)和重行(bp_* 5×)放大差距。
- 定位:par **赢 ESC(0.65×)、cuSPARSE(0.85×)**,大稀疏矩阵**碾压 cuBLAS**(0.01× 总时间),但**对 Ocean 整体慢 22%、大矩阵慢 2×** → Ocean 最快、par 第二。

### 测量严谨性审计(结论:方向可靠,非最严谨)
- **Ocean**:`bench_detail.json` warmup 10 + bench 10,报 **`累加/iters` = 均值**(`Utils.h:256`,**非 min**)→ 冷启动避开,长尾**未完全避开**(均值被慢轮拉高)。
- **C 法**:3 warmup + **单次计时**(main.cu)。冷启动避开,但单次 → 实测 par 抖动 **~5.7%**(bcsstk30 连跑 5 次 15.22–16.10ms),其余法 ~2-3%。
- **不对称**:Ocean 10 次均值(~±2%)比 par 单次(~±6%)稳。大结论(gap ≫ 5% 噪声)稳健;精确比值带 ±5% 不确定。要最严谨:C 法也多轮取 **min**(比 Ocean 的 mean 更压长尾)。

### 比值聚合口径(日志 = 几何均值,非总和比)
- 日志类别表的 par/cuBL、par/Oce、par/gust = **逐矩阵比值 A_i/B_i 的几何均值**(`exp(mean(log(r)))`,等权、抗离群),**不是 ΣA/ΣB,不是算术均值**。
- 三种口径数字差大(par/Ocean:①几何 1.225× / ②算术 1.53× / ③总和比 1.80×;par/cuBLAS ①1.55× vs ③0.01× 极端)。
- 含义:① = 「典型矩阵等权」(对 par 偏乐观,小矩阵稀释大矩阵劣势);③ = 「总吞吐/总时间」(大矩阵主导,对 par 更严苛)。两者都说 par 慢于 Ocean,幅度不同(22% vs 80%)。

## 工具链(本 session 新增/扩展)

- **`profile_aa.py`**:从 cu+gust 两法扩到 **4 法**(加 `merge`/`mrg2` tag、Time/nnz 解析、列、阶段映射);merge 的 merge 阶段映到「排序」列与 gust.sort 直接对照。
- **`analyze_merge.py`**(新):merge 专项分析,出 3 图(scatter / speedup / winloss),**输出到 `compare/merge_vs_esc_<时间戳>/`**(自包含:图 + 源 CSV),每次运行新建文件夹。
- **`main.cu`**:warmup + T4b(merge serial)+ T4c(merge2 par)计时块。
- `compare/` 约定:不同 compare 对象各占一个子文件夹(v1 归档 `compare/merge_serial_v1_2026-07-16/`、v2 当前 `compare/merge_vs_esc_<ts>/`,与既有 ocean_compute_only 等一致)。

## 产物清单

- 代码:`src/spgemm_merge.cu`(v1+v2)、`src/main.cu`、`include/spgemm.h`、`Makefile`、`suitesparse_crawl/profile_aa.py`(含 6 法 `chart_methods_bar`)、`suitesparse_crawl/analyze_merge.py`、`suitesparse_crawl/baseline_cublas.py`(ctypes cuBLAS)、`suitesparse_crawl/baseline_ocean.py`(Ocean 阶段求和)、`scripts/run_full_compare.sh`(终极 6 法一键)、`USAGE.md`。
- worklog:`merge_serial_v1_2026-07-16.md`、`merge_parallel_v2_2026-07-16.md`、本文件。
- 数据:`results/aa/first100_aa/log/*.log`(100,含 `[merge]`/`[mrg2]` 桩)、`suitesparse_crawl/profile_aa{,_summary}.csv`、`baseline_cublas.csv`、`baseline_ocean.csv`。
- 图:`compare/full_compare_<ts>/`(终极 6 法:日志 + `profile_methods_bar.png` 6 法柱状图 + 各 baseline CSV)、`compare/merge_vs_esc_<ts>/`(merge 专项)、`suitesparse_crawl/charts/profile_methods_bar.png`、`profile_aa.png`。

## 复现

```bash
# 终极一键(6 法 + 日志 + 柱状图 → compare/full_compare_<ts>/):
bash scripts/run_full_compare.sh
# 分步:
make                                                  # 编译(4 法)
USE_MEMPOOL=1 bash scripts/run_aa.sh                  # 全 100 矩阵(开 arena,d2h 干净;作准)
.venv/bin/python suitesparse_crawl/profile_aa.py      # 分阶段 + 汇总(4 法,含 par/cu 列)
.venv/bin/python suitesparse_crawl/analyze_merge.py   # → compare/merge_vs_esc_<ts>/(含 par vs cu)
```

## 下一步候选

- **追平 Ocean(最难)**:大矩阵上 k-way merge 算法层面难赢 hash(Ocean geo 1.225×、总时间 1.80× 领先 par)。路径:① Tier 1(shared 缓存段头)+ 更激进并行压 merge kernel;② 大矩阵改用 hash 累加器思路(本质换路线);③ 重行 hash 化/拆行治 bp_*+bcsstk08。
- **严谨性升级**:C 法计时从单次改成「3 warmup + N(如10)timed 取 **min**」(main.cu),与 Ocean 同等迭代且用 min 比 Ocean 的 mean 更压长尾 → 消除 par ~6% 单次抖动。
- **双口径报表**:日志/图同时报 ①几何均值 + ③总和比(ΣA/ΣB),两个视角都给(par/Ocean 1.225× vs 1.80× 都显示)。
- **自适应派发**(方向 B):小矩阵走 sort、中大走 merge/hash,理论全矩阵不输 ESC/cuSPARSE。
- **小矩阵固定开销**:par 的 6 段 pipeline(count/scan/merge/compact/final/pack)在小/对角矩阵上输 Ocean 的 launch/sync → 融合阶段减开销。
