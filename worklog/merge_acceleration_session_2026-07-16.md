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

## 工具链(本 session 新增/扩展)

- **`profile_aa.py`**:从 cu+gust 两法扩到 **4 法**(加 `merge`/`mrg2` tag、Time/nnz 解析、列、阶段映射);merge 的 merge 阶段映到「排序」列与 gust.sort 直接对照。
- **`analyze_merge.py`**(新):merge 专项分析,出 3 图(scatter / speedup / winloss),**输出到 `compare/merge_vs_esc_<时间戳>/`**(自包含:图 + 源 CSV),每次运行新建文件夹。
- **`main.cu`**:warmup + T4b(merge serial)+ T4c(merge2 par)计时块。
- `compare/` 约定:不同 compare 对象各占一个子文件夹(v1 归档 `compare/merge_serial_v1_2026-07-16/`、v2 当前 `compare/merge_vs_esc_<ts>/`,与既有 ocean_compute_only 等一致)。

## 产物清单

- 代码:`src/spgemm_merge.cu`(v1+v2)、`src/main.cu`、`include/spgemm.h`、`Makefile`、`suitesparse_crawl/profile_aa.py`、`suitesparse_crawl/analyze_merge.py`、`USAGE.md`。
- worklog:`merge_serial_v1_2026-07-16.md`、`merge_parallel_v2_2026-07-16.md`、本文件。
- 数据:`results/aa/first100_aa/log/*.log`(100,含 `[merge]`/`[mrg2]` 桩)、`suitesparse_crawl/profile_aa{,_summary}.csv`。
- 图:`compare/merge_vs_esc_<ts>/`(v2 三图 + CSV)、`compare/merge_serial_v1_2026-07-16/`(v1 三图)、`suitesparse_crawl/charts/profile_aa.png`(4 法阶段构成)。

## 复现

```bash
make                                                  # 编译(4 法)
USE_MEMPOOL=1 bash scripts/run_aa.sh                  # 全 100 矩阵(开 arena,d2h 干净;作准)
.venv/bin/python suitesparse_crawl/profile_aa.py      # 分阶段 + 汇总(4 法,含 par/cu 列)
.venv/bin/python suitesparse_crawl/analyze_merge.py   # → compare/merge_vs_esc_<ts>/(含 par vs cu)
```

## 下一步候选

- **Tier 1**(shared 缓存段头 `A_col_idx[seg_ptr[p]]`):再压 merge kernel,拉近 bcsstk30 与 cu 的差距(现 1.11×)。
- **重行更强并行 / 拆行**:治 bp_*+bcsstk08 这 10 个对 cu、ESC 双输的重行 straggler(hash SPA 在此天克 k-way merge)。
- **自适应派发**(方向 B):小矩阵走 sort、中大走 merge,理论全矩阵不输 ESC/cu。
