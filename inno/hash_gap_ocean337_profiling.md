# ocean337 上我方 hash vs Ocean 的差距归因(profiling 报告)

> 日期:2026-08-25。方法:同矩阵同卡(GPU0)连跑,我方 `[hash-prof]` cudaEvent 相位 vs
> Ocean stats.json 相位(`scripts/profile_us_vs_ocean.py`);kernel 级 nsys 佐证。
> 背景:ocean337 全量 Auto/Ocean = 4.04×(hash 路径 2.26× + 67 个 merge3 回退阵 28.5×)。
> 本报告只分析 hash 路径本身的差距。

## 1. 相位对齐表(ms,compute-only)

| 相位 | 333SP(371万行,均度6) | | | AS365(380万行) | | | pwtk(22万行,均度~50) | | |
|---|---|---|---|---|---|---|---|---|---|
| | ours | Ocean | 比 | ours | Ocean | 比 | ours | Ocean | 比 |
| sizing(MinHash) | 5.70 | **0.00** | — | 6.22 | **0.00** | — | 1.00 | 0.56 | 1.78× |
| binning | 5.74 | 0.69 | 8.4× | 5.61 | 1.26 | 4.5× | 0.37 | 0.17 | 2.2× |
| **accumulate** | **21.97** | **4.11** | **5.4×** | **24.01** | **4.65** | **5.2×** | 6.28 | 5.33 | **1.18×** |
| compact+sort | 5.03 | 1.37 | 3.7× | 5.18 | 1.41 | 3.7× | 0.71 | 0.62 | 1.15× |
| **合计** | **38.48** | **6.19** | **6.2×** | **41.08** | **7.34** | **5.6×** | **8.40** | **6.72** | **1.25×** |

关键对照:**pwtk 上 accumulate 几乎持平(1.18×)——差距不是"hash 本身慢",
而是"百万行 × 每行极短"这个形态触发的每行固定开销**。Ocean 对这种阵走的是
`usparse_main` kernel(与 pwtk 上的 `numeric.hash` 不同)——它有专门的**小行批量路径**。

## 2. kernel 级证据(nsys,333SP 我方)

```
hash_spa_kernel   grid 3,423,159 × 256线程   20.5ms   ← 主 accumulate
hash_compact_copy grid 3,423,159 × 256线程    4.8ms
mh_merge_kernel   grid 3,712,815 ×  32线程    4.5ms
compute_bucket    grid    14,504 × 256线程    2.9ms
scatter_rows      grid    14,504 × 256线程    2.8ms
```

333SP 平均每行只有 **~6 个中间积**(maxrow=28),却为一个行launch一个 **256 线程 CTA**:
8 个 warp 伺候 6 个乘法,组机制(G=16)、SMEM 表管理、CTA 建立/退出全是纯开销。
3.42M 个 CTA × 每行 6 项 = 有效利用率个位数百分比。Ocean 的 usparse_main(4.0ms)
按 CTA 批量处理多行,把每行固定开销摊掉。

## 3. 差距分解(333SP,总 Δ=32.3ms)

| 来源 | Δ(ms) | 占比 | 机制 |
|---|---|---|---|
| accumulate | 17.9 | **55%** | 1 CTA/行 × 256 线程处理 ~6 项;Ocean 小行批量 |
| sizing | 5.7 | 18% | 我方 MinHash 两遍(mh_construct 1.1 + mh_merge 4.5)对 2221万 nnz 全额征收;**Ocean 该阵 estimation=0**(小均度走更便宜的 sizing/上界) |
| binning | 5.1 | 16% | compute_bucket+scatter_rows 逐行分桶 3.7M 行;Ocean analysis 仅 0.69 |
| compact+sort | 3.7 | 11% | compact_copy 同样 1 CTA/行 |

## 3.5 Ocean "没有 sizing" 的真相:Ana1 内部分流器(源码 SpGEMM.cuh:429-515)

Ocean 在 analysis 阶段就算出**每行精确乘积数**(Σ_k len(B[k,:]),即每行 flop 上界)与
avg_product = 总乘积/行数,然后**按 avg_product 分流工作流**:

```
avg_product ≤ usparse_avg_product_threshold(=64, Utils.h:29)
   → ana1_type=0 "ultrasparse/spark" 工作流:
     ✗ 不跑 HLL(整个 estimation 相位为 0 —— 333SP/AS365 即此)
     ✓ sizing 直接用每行乘积数(均度低的行 dup 因子≈1,上界≈准界,免费)
     ✓ numeric = sparseKernelLauncher:每 CTA NUMERIC_USPARSE_K_ROWS_PER_BLOCK 行
       × 每行 spark_kernel_size 线程(16/32 按 avg 档;另有 4 线程/行 sub-warp hash 变体)
     ✓ 超过 avg 的行由 classifyOutlierRows 挑出走单独 padded kernel(spark_outlier)
avg_product > 64 → ana1_type=1:HLL 估计 + numeric.hash(pwtk 走此路,estimation 0.56ms)
```

**⇒ Ocean 本身就是个矩阵内 dispatcher**(avg_product 阈值 64 选工作流);我方差距的
sizing/accumulate/binning 三项全部源于没有这条 ultrasparse 分支:
- 我方 STREAMLINE_NNZ 门按【总 nnz】(100k),333SP 22M nnz → MinHash 全额征收;
  应学它按【avg_product ≤ 64】免 sizing(用 flop_ub);
- 我方 hash_spa 1 CTA/行×256 线程 vs 它的【多行/CTA × 按均度定每行线程数 + outlier 分流】;
  sparseKernelLauncher 里甚至有 hashNumericSubWarpKernel<4>(4 线程/行)。

## 4. 修复路线图(按 ROI)

1. **小行批量路径(主攻,55%)**:对 ht_size ≤ 阈值(如 64)的行,每 CTA 处理
   R=32-128 行(按行填充 warp,组内 j 并行),一个 kernel 吃掉 bin0/1 的大军;
   预计 accumulate 21.9→~5ms。**这是 Ocean usparse_main 的对应物**。
2. **小行免 MinHash sizing(18%)**:均度低(如 avg_row<16)时跳过 mh 两遍,
   用 flop_ub/简单上界定桶(我们本有 STREAMLINE_NNZ 门槛,应改成均度门槛)。
3. **binning 融合/轻量化(16%)**:compute_bucket 与 scatter 融合;或小行全走
   统一 config 免分桶。
4. **compact 小行批量(11%)**:与 1 同构。
5. 合计预估:333SP 型 38.5→**~9-10ms**(vs Ocean 6.2,余 ~1.5×:其残差是
   usparse_main 本身的吞吐优势,可再攻)。

pwtk 型(均度高)无需改动,已 1.18-1.25× 持平。

## 5. 与 merge3 回退问题的关系

67 个回退阵(28.5×)是另一个独立问题(HASH_CAP=16384 单行上限 + int 溢出);
两线修完,Auto/Ocean 预计 4.04× → **~1.6-2×**,再靠 dispatcher 重拟合收尾。
