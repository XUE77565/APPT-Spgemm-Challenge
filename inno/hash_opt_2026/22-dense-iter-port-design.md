# 22 · Ocean 机制移植设计:按行 dense-iter 分流(2026-08-26)

目标:治 docs/20 战线 1(中尺寸 5-25× 簇)。基于 Ocean 源码解剖 + 重行量化验证。

## 0. 核心洞察(实measured)

Ocean 对 c-58 的 numeric = **dense 4.03ms + hash 0.44ms** —— 不是"dense 替代 hash",而是
**按行分流:重行 → dense(iter 窗口内核),轻行 → hash**。CPU 精算重行占比:

| 阵 | n | Σflop | flop_row>4096 | 占 Σflop |
|---|---|---|---|---|
| c-58 | 37.6k | 79M | 7343 行(19.5%) | **89.3%** |
| bloweya | 30k | 101M | 10002 行(33.3%) | **99.6%** |
| email-Enron | 36.7k | 52M | 2523 行(6.9%) | 55.0% |
| mult_dcop_03 | 25k | 520M | 22774 行(90.4%) | **100%** |

我们的 69.5ms accumulate = 全部行挤 hash(重行探测链长+大表低 occupancy)。
Ocean:重行直接寻址免探测,轻行小表 hash → 4ms 出头。

**dense 为什么对低密度也赢**:clear+扫描是线性 SMEM 带宽(~29TB/s 聚合),hash 探测+CAS 是
依赖链+随机访问;c-58 行均 2096 积/37.6k span,clear 成本 ≈ 3× 积成本,仍大胜。

## 1. Phase A:按行分流(最小改动,先拿大头)

**数据流零改动**:dense-iter 行的输出仍写 tmp(est-gapped)+ 现有 compact 路由(窗口序天然有序)。

1. `hash_binning_kernel` 加一路:行满足 `flop_row ≥ DENSE_ITER_MIN_FLOP`(默认 4096,env 可调)
   → 新 bin(dense-iter);优先级高于 heavy(est>HASH_CAP)
2. accumulate:该 bin 走**现有** `hash_dense_window_kernel`(已在,修 3 个参数问题即可跑 per-row 集合):
   - launch 用行列表(bucket_rows)而非 A_rows 全体 —— 现 kernel 用 `blockIdx.x` 当行号,加 indirection
   - SMEM 窗口 W 与 512→1024 线程按 bin 大小配
3. 现有 retry/compact/est 语义全保留(ovf 行照旧路由)

预期(用现 kernel 未升级的速度保守估):bloweya 类从 hash 34ms → dense_win ~15-20ms;
c-58 → 重行 dense_win ~20ms + 轻行 hash ~5ms ≈ 25ms(仍 4-5× Ocean,但 3× 于现在)。

## 2. Phase B:内核升级(对标 Ocean iter 的 9.4×,docs/20 §3)

现 `hash_dense_window_kernel` 与 Ocean `denseNumericIterKernel` 的差距逐项修:

| # | Ocean 有 | 我们缺 | 改法 |
|---|---|---|---|
| B1 | 1024 线程 + ~128KB SMEM → 2 CTA/SM | 512 线程 + 195KB → 1 CTA/SM(25% occ) | W 降到 ~12k cols,blockDim 1024 |
| B2 | localLoadBalance:2^k 线程/组 × 每组一个 A 条目 | 固定 1 warp/条目(mult_dcop 半数线程闲) | 组大小 = clamp(avg_b_col_len, 32..256) |
| B3 | start_map 游标续算:窗口边界=各条目 atomicMin 下一未消费列,零空窗零重扫 | 每(窗口×条目) 2 次 lower_bound,固定窗口网格 | SMEM 存每条目 B 游标;条目多时复用 prefix 区(Ocean getSMBufferLoc 同款) |
| B4 | bitmap 1bit/col + word 级 ballot 提取 | dflag u8 + dpref int(13B/col) | dval(8B)+bitmap(1/8B)=8.125B/col;denseCompact 式 word 前缀 |
| B5 | symbolic 先行直接写终态 | tmp + 独立 compact | Phase C(方案 5),est-gapped 先保正确性 |

B1+B2 是大头(occupancy 翻倍 × 空闲消除);B3 治大 n 多窗重搜;B4 把提取扫描成本 /8。

## 3. Phase C:直接写终态(= 方案 5,治 compact 税 + DNF 内存)

dense 行的 symbolic 免费获得:bitmap popcount = 精确 per-row distinct → 两遍法
(count→offset→value 直写 CSR),hash 轻行走 tmp+compact(轻行 compact 便宜)。
配合 docs/21 Fix2(compact 原地化,含 C_rp[i] ≥ E_off[i] 守卫的推导)作为过渡。

## 4. 覆盖判据与 dispatcher 接线(→ 任务⑦)

按行规则(Phase A):`flop_row ≥ 4096 → dense-iter`。更精细(Phase B 后标定):
- `flop_row ≥ k × span_row`(k≈0.2)→ dense(成本模型:flop×c_hash vs span×c_clear+flop×c_dense)
- 轻行 → 现有 hash bins(它们在 Ocean 侧也是 hash 0.44ms 量级,我们的差距主要在重行)
- 与矩阵级 dense_win/dense 门合并:矩阵级门(15% 密度)可降为 1% 或删除(按行规则自然覆盖)

## 5. 验收
- 单阵 A/B:c-58/bloweya/Enron/mult_dcop/soc + 6 个 hash 主场阵(回归)× {Phase A, A+B1B2, 全家}
- 全量 refresh + 消融(docs/23);目标:中尺寸簇 7-15× → ≤2×,geomean 2.28 → ≤1.7
- 正确性:C_nnz 精确 + 0 乱序(DBG check)+ 与 cuSPARSE 对拍抽 3 阵

## 附:受污染窗口记录
refresh6 运行中我误 make 了一次(#41-45 约 5 阵短暂用了含 Fix0+1 的 binary,~19:44-19:48),
消融分析时对这 5 阵(Si5H12/Si87H76/SiN/SiO 等)单独复核;compute 相位理论上中性(frees 在 prof 外,
cap 对 flop_row<n 的阵不触发)。
