# 20 · 全量瓶颈分析:我们 vs Ocean 差在哪里(2026-08-26)

数据基础:`compare/ocean337/methods_cmp_clean.csv.csv`(337 阵,v3 refresh,**不含 Hybrid 的最新 kernel**)+
8 阵 GPU 相位 profiling + 9 阵 Ocean 直跑决策/分相位解剖。工具:`scripts/analyze_gap_vs_ocean.py`、
`scripts/profile_top_losers.py`(可复用)。

> ⚠ 数据勘误:CSV 里 bloweya Auto=114.4ms 是脏数据(复测 50.6ms,兄弟阵 bloweybl 47.8ms 佐证,
> 两次独立运行一致)。真 ratio ≈ 10.9×。全量 geomean 2.279× 受此影响 <0.01,结论不变。

## 1. 总体结构

**geomean(Auto/Ocean) = 2.279×,赢 7 / 输 314**(有双方数据的 321 阵;另有 16 阵我们 DNF,见 §5)。

| 按 n 分桶 | 阵数 | geomean | 备注 |
|---|---|---|---|
| <10k | 16 | 1.36× | 固定开销主导 |
| 10k-50k | 120 | **2.65×** | 最差桶,新 top losers 所在 |
| 50k-200k | 78 | 2.43× | |
| 200k-2M | 80 | 1.98× | |
| >2M | 27 | 2.01× | 巨型图已修,剩 ~2× 结构差 |

**按输出行长(C_nnz/n)——最强单变量信号:**

| C 行长 | 阵数 | geomean |
|---|---|---|
| <100 | 64 | 2.08× |
| 100-500 | 137 | 1.65×(最好) |
| 500-2k | 67 | **3.54×**(最差) |
| 2k-10k | 44 | 3.27× |
| >10k | 9 | 3.91× |

**头部空间分解(修到 1× 后的 geomean)**——差距不是少数离群点,是双层结构:

- 修完 ratio≥5 的 24 阵 → 只到 1.918×(省 0.36)
- 修完 2≤ratio<5 的 **140 阵** → 到 1.443×(**省 0.84,大头在中段**)
- 修完 ratio<2 的 157 阵 → 到 1.877×(省 0.40)

⇒ top losers 和"到处慢 1.5-2.5×"是**两个独立问题**,都要修才能到 1.0×。

## 2. 相位 profiling:时间花在哪(8 代表阵,复现 CSV ±2%)

| 阵 | n | C密度 | dup | accumulate | retry | compact+sort | 合计 | Ocean | ratio |
|---|---|---|---|---|---|---|---|---|---|
| bloweya | 30k | 11% | 1.0 | 34.0 (67%) | 3.7 | 10.2 | 50.6 | 4.66 | 10.9× |
| c-58 | 37.6k | 2.9% | 1.5 | 69.5 (83%) | 4.7 | 7.2 | 83.8 | 5.67 | 14.8× |
| email-Enron | 36.7k | 2.3% | 1.2 | 34.8 (78%) | 2.5 | 5.5 | 44.5 | 3.28 | 13.6× |
| soc-Slashdot0902 | 82k | 1.2% | 1.1 | 63.7 (74%) | 6.9 | 13.1 | 85.9 | 8.49 | 10.1× |
| mult_dcop_03 | 25k | 82% | 1.0 | 86.9 (80%) | 0.6 | 17.7 | 109.2 | 11.89 | 9.2× |
| a5esindl | 60k | 4.9% | 1.0 | 40.5 (58%) | 3.6 | 22.8 | 69.9 | 10.29 | 6.8× |
| Ge99H100 | 113k | 0.8% | 9.2 | 36.7 (53%) | 3.2 | 24.2 | 69.8 | 16.75 | 4.2× |
| pre2 | 659k | 0.05% | 1.1 | 11.4 (14%) | 3.4 | **61.9 (75%)** | 82.2 | 12.84 | 6.4× |

要点:
- **accumulate 占 53-83%**(除 pre2);这些阵 **dup 只有 1.0-1.5**——不是 Ga3 那种高 dup SMEM 串行化
  (Hybrid Value 治的是那种,对这些阵无效),是 hash 探测+CAS 本身太贵
- **compact+sort 是第二杀手**:普遍 8-35%,pre2 极端到 75%
- 我们的 accumulate 吞吐只有 **1.1-3 G product/s**;Ocean 同阵全管线 13-56 G/s

## 3. Ocean 对照解剖(直跑 stats.json + 源码阅读)

| 阵 | Ocean numeric dense | Ocean hash 系 | Ocean 走法 |
|---|---|---|---|
| bloweya | 3.14 | 0.07 | dense(iter) |
| c-58 | 4.03 | 0.44 | **dense(iter)** ← 2.9% 密度、dup 1.5 仍走 dense! |
| email-Enron | 1.08 | 0.91 | 混合 |
| soc-Slashdot0902 | 2.07 | 3.03 | 混合 |
| a5esindl | 9.12 | 0.10 | dense |
| mult_dcop_03 | 9.25 | 0.05 | dense(iter) |
| Ge99H100 | 0.01 | **13.54** | hash(dup 9.2,正是 hybrid value 场景) |
| pre2 | 4.21 | 5.72 | 混合 |
| web-Google | 0.12 | 2.37 | hash |

**Ocean 敢对 2.9% 密度、dup=1.5 的 c-58 用 dense,且快 17×。** 机制(源码拆解,`ocean/kernels/`):

1. **按行分 bin,不是按矩阵选路**(`numericBinning`,Analysis.cuh):每行按
   `range = b_col_idx_right - left + 1`(**列跨度**)选 dense 档,跨度 ≤ SMEM 档位的行进
   dense 静态 bin(6 档,最小档 sub-warp 32 线程/行),任意跨度进 bin6 = **iter 窗口内核**;
   hash 档并行存在,**7 个 stream 同时跑**
2. **iter 内核(大跨度/整 n 跨度)四个关键设计**(AccumulatorDense.cuh:611):
   - **数据驱动窗口**:`while (start_b_idx <= end_b_idx)`,下一窗起点 = 各 A 条目下一个未消费列的
     `atomicMin`(**start_map 游标续算**,零重复扫描、零空窗)
   - **localLoadBalance 动态子组**:按(行长, 乘积数, B 最长列)把 CTA 切成 2^k 线程/组的子组,
     每个 A 条目一个子组——不固定 warp/条目
   - **1024 线程大 CTA + 128KB SMEM/CTA** → 2 CTA/SM;我们 512 线程 + 195KB → **1 CTA/SM、
     25% occupancy,且 warp/条目固定导致 mult_dcop 一半线程闲**
   - **symbolic 先行 → numeric 直接写终态 CSR**(symbolic_dense 只花 0.4-3.3ms);
     **没有 compact 阶段**,epilogue 只有按需 indirect sort(0-2.9ms)
3. **bitmap 1bit/列**(atomicOr + word 级 ballot 扫描,提取近零成本)vs 我们 u8 flag+int pref
4. 我们的 dense_win(mult_dcop 实测走了这条路)**同算法家族内差 9.4×**(86.9 vs 9.25)——
   差距=occupancy+固定 warp 切分+lower_bound 重搜+后接独立 compact

## 4. 四条战线(按 geomean 收益排序)

### 战线 1:dense 家族大扩建(治 5-25× 簇 + 中段)——最大单杠杆
- **dispatcher**:从"矩阵级 15% 密度门"改为 **Ocean 式按行 binning**(列跨度 + est),
  dense 静态 6 档 + iter 兜底,hash bin 并存
- **iter 内核重写四要素**:数据驱动窗口游标 / 动态子组 / 1024 线程 2CTA/SM / 直接写终态
- 覆盖判据从"密度"改为"**flop 对 clear+scan 成本的比值**":2.3% 密度(Enron)依然大胜 hash,
  0.05%(pre2)才该留给 hash
- 预期:bloweya/c-58/Enron/soc/mult_dcop/a5esindl 类从 7-15× → ~1.5×

### 战线 2:hash 内核自身 2.5-5× 结构差(治 Ge99/web-Google/road 类)
- Ocean hash 13.5ms vs 我们 accumulate 36.7ms(Ge99, dup 9.2)——这正好是 **Hybrid Value 的
  场景(dbe19cd 已上线,refresh6 待验证全量净效果)**
- 剩余:他们的 load-balance 子组、20-stream、行级混合(hybrid_hashmap 限流)

### 战线 3:compact+sort 税(8-75%,一个病根三笔账)
- pre2 的 61.9ms(75%)、全体 8-35% 税、**rajat 族的 DNF 根因就是它**(见 §5)
- 治法 = 方案 5(compact-accumulate 融合)+ 学习 Ocean "symbolic 先行直接写终态";
  est-gapped 中间态是双重内存+双重带宽的来源

### 战线 4:16 阵 DNF(rajat16-28 ×7、c-73/c-73b/c-big、cage15、wb-edu、audikw_1、
dielFilterV3real、Cube_Coup_dt0、TSOPF_FS_b300_c2、mouse_gene)
- rajat16 复现:`C_nnz=891M (est 1.01×) → compact 阶段 cudaMalloc 10.7GB OOM`——
  est 超配 + tmp/compact 双缓冲 ≈ 2.5× C_nnz 的内存峰值;Ocean 同阵 46.9ms 轻松跑
- wb-edu Ocean 只要 **1.47ms**(9.8M 阶!)、mouse_gene 1726ms(它自己也慢)
- 战线 3 的直接写终态顺带治好大部分;修完前这 16 阵在 geomean 外白送

## 5. 结论

剩余 2.279× 的构成:**~1.0× 在 dense 家族缺失(战线 1)+ ~0.3-0.5× 在 hash 结构差(战线 2,
Hybrid Value 已瞄准)+ compact 税(战线 3)+ DNF 白送(战线 4)**。中段 140 阵(2-5×)与
top 24 阵(5-25×)是同一批根因在不同尺寸下的表现。

下一步顺序建议:**①refresh6(Hybrid 全量,已在盘)→ ②方案 5 直接写终态(治 compact+DNF,
工作量最小收益最广)→ ③dense 家族扩建(按行 binning + iter 内核重写,主战役)→ ④hash
load-balance 子组化**。

---
*复现:`python3 scripts/analyze_gap_vs_ocean.py`、`python3 scripts/profile_top_losers.py bloweya c-58 ...`;
Ocean 解剖:`cd ocean && ./convert ../data/ocean/square/c-58.mtx /tmp/c.csr && ./spgemm /tmp/c.csr config/analysis.json`。*
