# DATE 2027 追平 Ocean 总计划(2026-08-25 制定)

> 截稿:**2026-09-07**(13 天)。会议 2027-03-22~24 Dresden。
> 现状:ocean337(337 阵)Auto geomean 19.9ms vs Ocean 6.5ms = **4.04×**。
> 目标:**T1(保底)≤1.2× 且多类别反超;T2(冲刺)geomean 追平/反超**。
> 依据:`hash_gap_ocean337_profiling.md`(差距分解)+ `mmspgemm_comparison.md`/
> `bhsparse_baseline.md`(novelty 定位)+ 8-25 全量数据。

---

## 0. 安全协议(wedge 三连教训:7-27、7-30、8-25,最高优先级)

**目标:任何情况下不再出现 GPU requires reset。**

已落地:
- ✅ `scripts/gpu_check.sh`——任何跑批/计时前必跑(nvidia-smi 状态 + 1-thread kernel+D2H 冒烟;双检 `[N/A]/requires reset`)
- ✅ `collect_disp_data.py` 熔断:fp>15 亿或 maxrow>10 万 → SKIP 不进 hash/merge3;
  adaptive 行缺失 → 不盲跑两路径(GPU1 wedge 的直接死因就是盲跑病态合成阵)
- ✅ 长跑一律 tmux + 增量 CSV(断点安全)+ 监控告警

待落地(Phase 1 一并做,代码级根治):
- `spgemm_merge.cu`:① gapped buffer 分配前 sanity(`total_flop` 非法/超界 → 干净报错退出,
  绝不进 cudaMalloc);② `bucket_merge*_kernel` 的 `while(true)` 加 `MRG3_LOOP_CAP`
  (照抄 att_tiered 的 ATT_LOOP_CAP 手法:不变量破坏 → 错误输出而非挂死)
- 合成阵生成器(gen_synth_disp.py)补 fp 上限约束(重生成时生效)
- 纪律:GPU1 reset 前单卡作业;每批前 gpu_check;监控见单阵超 2×timeout 立即人工介入
  (不要等超时杀——SIGKILL 落在 CUDA teardown 就是 wedge 机制)

---

## 1. Phase 1:杀回退灾难(8/25-27)→ 4.04× ⇒ **~2.3×**

| # | 任务 | 说明 | 验收 |
|---|---|---|---|
| 1.1 | merge3 int 溢出修复 | `total_flop`/`d_row_off` 64 位化 + kernel 内索引 64 位(§0 的 sanity+cap 同步落地) | Ga3As3H12/Ga41As41H72/TSOPF 跑通且值对 |
| 1.2 | 重行不再回退 | 抬 HASH_CAP/改走 ultra 路径,让 67 个 hash 溢出阵留在 hash | 67 阵 Auto_choice 不再是 merge3(回退) |
| 1.3 | 333SP 4 行乱序修复 | compact_sort bin8/9 边界(超重行) | sorted check 0 违规 |
| 1.4 | REFRESH=Auto | 只刷受影响列(gpu_check→tmux→监控) | 主表更新,无 wedge |

## 2. Phase 2:ultrasparse 路径(8/27-30)→ ~2.3× ⇒ **~1.3×**

照 Ocean 源码蓝图(它自己的矩阵内 dispatcher,`SpGEMM.cuh:429-515`):
| # | 任务 | 机制 | 预期 |
|---|---|---|---|
| 2.1 | avg_product≤64 免 MinHash | sizing 用 flop_ub(分析阶段白拿);门槛从总nnz 改均积 | 333SP 型省 5.7ms(18%) |
| 2.2 | 小行批量 accumulate kernel | 多行/CTA × 每行按均度定线程数(16/32 档,参考其 sub-warp<4>)+ outlier 行 classifyOutlier 同款分流 | accumulate 21.9→~5ms(55%) |
| 2.3 | compact 小行批量化 | 与 2.2 同构 | compact 5.0→~1.5ms(11%) |
| 2.4 | binning 融合/轻量化 | compute_bucket+scatter 融合;小行统一 config 免分桶 | binning 5.7→~1ms(16%) |

新 kernel 安全要求:先在安全阵集(bcsstk30/pwtk/exdata_1/333SP)对拍旧 hash 输出
(C_nnz+值+序),全对才进 REFRESH;loop cap 与 sanity 随代码落地。

## 3. Phase 3:无泄漏 dispatcher + 带状阵杀手锏(8/30-9/1)→ **~1.3× ⇒ 1.0-1.2×**

- **数据**:disptrain(GPU0 在跑,213 真阵带熔断)+ 合成 35 阵(病态 24 个待 fp 约束重生成重采)
- **模型双路**:① H100 解析(成本模型 α·flop+c·nnz+o,合成数据标定)② 不相交 ML(加权 logistic)
- **⚠ 新特征——局部性(带宽)**:`band_n32000_x1024` 上 merge3 23.9ms vs hash 841ms(**35×**),
  现有 4 特征(n/nnz/maxrow/skew)抓不到带状性 → 加"平均行列跨度/行长度"特征
  (Ocean 的 analysis 里有 b_col_idx_left/right 同款信息)。带状/局部性阵 = merge 主场,
  ocean337 里电路/网格阵不少 → 这是反超 Ocean 的类别级武器(Ocean 无 merge 族)
- **部署**:公式进 `spgemm_adaptive.cu`(score<0 逻辑不动);双测试集(first100+ocean337)
  vs oracle 验收:准确率 ≥ 现版、总 compute ≤ oracle+5%

## 4. Phase 4:终测(9/1-4)

- 最终 binary 全量重跑 ocean337 + first100(gpu_check→tmux→监控→REFRESH)
- 报告/图表管线刷新(含 bhSparse/MMSpGEMM 两代 merge 基线列)

## 5. Phase 5:论文(8/25 起并行,9/7 截)

- **framing**:GPU SpGEMM for EDA 稀疏工作负载(ocean337 本就是电路仿真阵套件;AiSpGEMM DATE'25 先例)
- 主张链:①hash/merge 双家族自适应(Ocean 只有 hash 族;带状阵 35× 优势类别)②无泄漏
  dispatcher 方法学(合成标定+不相交训练,对抗"profiling 拟合"质疑)③merge 线复活
  (值域分桶 vs bhSparse'15 rank/MMSpGEMM'25 rank+sort,查重文档就绪)
- 诚实呈现:first100(平价偏赢)与 ocean337(追平)双基准
- 砍掉:AAT tiered(未实测,降为 future work)

## 6. 风险表

| 风险 | 概率 | 缓解 |
|---|---|---|
| Phase 2 新 kernel 出 bug/延期 | 中 | 正确性对拍门槛;若 8/30 未过,只保 2.1(免sizing)+2.2,砍 2.3/2.4 |
| 再 wedge | 低(护栏后) | §0 协议;GPU1 找管理员 reset 增加冗余卡 |
| 单卡排队(采集×开发×终测) | 中 | 时段错开;终测前暂停采集 |
| 追不平(卡在 1.3×) | 中 | T1 故事兜底:类别反超+方法学+双基准,DATE 的 EDA framing 降低对 geomean 的依赖 |
| disptrain 病态真阵 SKIP 太多 | 低 | 样本 213 足够;SKIP 阵记录在案 |

## 7. 当前在跑

- disptrain(GPU0,带熔断,56/213)→ Phase 3 数据
- GPU1 待管理员 `sudo nvidia-smi --gpu-reset -i 1`
