# 29 · 方案5 实现:dense 行精确计数 + 直写终态(2026-08-27 夜,DIRECT5)

> docs/27 §4.1 的落地。**交通量经济模型**(从 Ocean type1/type2 路由反推):
> 两遍(count+direct)vs 单遍 gapped 的临界 ≈ dup 3×(1+est/C) ≈ **6-8** —— 恰是 Ocean
> `decideMatrixType` 的 compaction≥8 门与我们 Phase A v4 门的 `dup<8`。**所以本方案只对
> dense/Phase A 行启用**(v4 门已保证 dup<8),hash 行不动(结构已同 Ocean type2 est 工作流)。
> 外部支点:MH-SpGEMM(ICCD'25)的 bitmap-symbolic(docs/28 A1)同思想。

## 1. 机制

```
legacy(dense 行):accumulate 窗口扫 → 写 gapped tmp(est 槽)→ cnnz_scan → compact copy → dC
DIRECT5:        count pass(只标记 flag,1B/列)→ 精确 row_nnz
                accumulate(hash 行照旧,gapped)→ cnnz_scan(全行精确)
                → dense numeric pass 直写 dC_ci/d_val @ dC_rp(窗口升序天然有序)
                → compact 只处理 hash 行
```

- **count pass**(`hash_dense_count_kernel`):与数值 pass 同构的窗口扫描,但只
  `csmem[j]=1` 标记(免 dval 8B/列、免 atomicAdd、免 A/B 值加载)。窗口常数
  `DENSE_CNT_W=65536`(SMEM 1B/列 ≈ 64KB,是数值窗口 14980 的 4.4 倍宽 → pre2 类
  44 窗/行降到 ~1/4)。计数 = 每窗口 flag 求和的块归约。**精确、无 cap、永不 ovf。**
- **direct pass**(`hash_dense_direct_kernel`):`hash_dense_window_kernel` 的直写变体,
  base = `dC_rp[i]`(精确 int 偏移)替代 `d_off[i]`(est-gapped ll),cap = count pass 的
  row_nnz。legacy 提取段的 `upper_tri` 过滤是死代码(标记时已滤),direct 版删掉防洞。
- **tmp 缩容**:dense 行 est 置 0(`zero_est_kernel`/memset)+ **d_off 重扫**(inclusive scan
  ToLL)→ gapped 偏移坍缩进 hash 侧空间 → `tmp_slots = total_est − Σest_dense`。
  ⚠ d_off 是全行 est 前缀,不重扫则 heavy/hash 行偏移仍指旧空间 → 越界(实现时抓到)。
  heavy scratch(`d_scr_key/val`)同口径改 tmp_slots。鲸鱼阵 Phase A 行恰是 est 大头。
- **est 下溢检查改 hash 侧**:`C_nnz − Σnnz_dense ≤ tmp_slots`(dense 行计数精确可合法
  超 其 est,legacy 口径会假阳性回退)。
- **互斥**:Fix2 in_place 强制关(dense_nr>0 时)——direct 写终态与 tmp 前缀紧缩重叠,
  且 est 置 0 后 margin 数学失义。

## 2. 开关与边界

- `DIRECT5=1` 开(env,默认 0 = 行为与 refresh8 binary 逐位一致,全部新分支 inert)。
- 启用面:矩阵级 dense(n≤14980 且 ≥15% 稠密)/ dense_win(n≤200k 同密度门)/ Phase A
  bin(BIN_DITER,v4 门:est≥2048 ∧ flop≥4096 ∧ 16flop≥span ∧ dup<8)。
- 计时相位:`dense_count`(含三 memset + est 置零 + d_off 重扫)/ `dense_direct`
  (compact 前)。profile_top_losers.py 直接可见。
- 流同步:count/rescan 在 legacy stream 0;bin_s 多流由 legacy-default-stream 语义隐式
  排序(cudaStreamCreate 非 non-blocking),与既有 binning→bin-stream 次序同理。

## 3. 预期与验收(待跑,refresh 完成后)

| 类别 | 阵 | 预期 |
|---|---|---|
| 窗口多/pre2 类 | pre2(44 窗/行,compact 75% 税)| 大赢:count 窗口 4.4× 宽 + compact 消失 |
| Phase A mixed | c-58 / bloweya / soc / Enron | 中赢:gapped 写+copy 消失,多付 0.4× count |
| 矩阵级 dense | exdata_1 类 | 小-中赢(accumulate 本就占主导)|
| 鲸鱼内存 | cage15 / c-73 / rajat | tmp 峰值 −Σest_dense(Phase A 行 est 大头),DNF 余量 |
| 回归护栏 | pwtk / 333SP / bcsstk30 / Ga3(hash 行为主)| 应 ±0(count/direct 不触 hash 路径)|

A/B:`DIRECT5=1` vs 默认,profile_top_losers.py(三坑:MP_HOST_MB≥8192 / 取末轮 / 排除
h2d-d2h-TOTAL);C_nnz 与乱序校验(DBG 的 hash_check_sorted + 与 CSV cnnz 对表)。

## 4. 同场实现:localLoadBalance 移植(LLB=1,默认关)

- `row_span_kernel` 顺产 `max_b_len`(它已在扫 B 行,零额外成本);`local_load_balance()`
  device 函数 = Ocean `AccumulatorCommon.cuh:67` 同款语义:起步均值**排除最长 B 行**
  (`(flop−maxb)/(a_len−1)`,straggler 由夹逼吸收)→ 取最近 2^k → 按 max_sub_iter vs
  num_iters 的 2× 失衡双向调 → clamp。
- `hash_spa_kernel` 的 G 选择替换:静态梯(ht_size→G∈{4..32})→ 按行动态 log2(G)∈[2,8]
  (**上限 HASH_BLOCK=256:整 block 伺候一个 k**,我们的内层无 warp 内在,合法)。
  治 Ge99 类 4.4×(重行 B 长方差大,统一 G 必然失衡)。
- 移植修正:原版在 avg_ops>2^ubound 时 `1<<(ubound−log_nthr)` 负移位 UB(其 bin 行长
  有界故未触发),我们加了预钳位。
- A/B:`LLB=1` vs 0,阵 = Ge99H100 / c-58 / bloweya / soc / Enron + 回归 pwtk/333SP/bcsstk30。

## 5. 同场修复:成功路径逐行数组泄漏

审计发现:d_bkid/d_sort/d_est/**d_flop**/d_span_lo/d_span_len(/d_maxbl)此前**只在早退
路径释放,成功路径全漏**(USE_DEV_POOL 默认关 = cudaMalloc 模式)≈ 28B×A_rows/调用,
warmup+bench 多轮累加 —— 鲸鱼阵(c-73 峰值 ~62GB、rajat 类 OOM 史)的隐性推手。
已补尾部释放(d2h 后区域,与既有 12 buffer 尾释放同址,不进 hash-prof 相位;早退路径补
d_flop)。docs/21 Fix0 修了 retry 缓冲,这批是当年漏网的另一半。

## 6. A/B 结果(2026-08-27 夜,compute-only 同 CSV 口径,`compare/ocean337/ab_direct5_llb.log`)

| 阵 | base | D5 | LLB | 判 |
|---|---|---|---|---|
| mult_dcop_03 | 113.40 | **77.76(-31%)** | 113.40 | D5 大胜(top1 loser 7.9×→~5.4×)|
| c-58 | 12.89 | **11.64(-10%)** | 12.09(-6%) | 双赢 |
| bloweya | 19.61 | **18.26(-7%)** | **18.20(-7%)** | 双赢 |
| email-Enron | 8.66 | 8.32(-4%) | **8.07(-7%)** | 双赢 |
| pre2 | 81.70 | 81.00(-1%) | 79.47(-3%) | **D5 无感 = 其行未过 v4 门(span),瓶颈是路由非 compact** |
| soc | 24.32 | 24.11 | 23.89 | 微 |
| exdata_1 | 14.82 | ~~38.30(+159%)~~ → 修复后 21.2≈base | 14.80 | **回归已修**(见下)|
| Ge99H100 | 74.34 | 75.12 | 74.10 | **LLB 无感:Ge99 重行疑走 heavy(hash_global)路径,LLB 只补了 hash_spa** |
| pwtk | 8.73 | 8.74 | 8.93(+2%) | LLB 微回归 |
| 333SP/bcsstk30/Ga3 | ±0.1-1% | 同 | 同 | 回归干净 |

**exdata_1 回归根因**:dense_mode 的 legacy 内核按 **n 尺寸**配 SMEM(exdata n=6001→78KB→2CTA/SM),
direct 内核是固定 DENSE_MAX_N 窗口(195KB→1CTA/SM)—— occupancy 减半(相位实证:direct 33.4ms
vs legacy accumulate 13.5ms)。**修复 = dense_mode 排除出 DIRECT5**(其 compact 仅 1.4% 无税可省);
保留 dense_win_mode + BIN_DITER(两路同为固定窗口,D5 严格减活)。修复后 exdata_1 D5 = 21.2 ≈ base ✓。
cnnz 14/14 阵全部与 CSV 精确一致 ✓。

**默认策略**:DIRECT5/LLB 仍默认关;**refresh9(DIRECT5=1,Auto 列,tmux)全量验证 geomean 后再定**。
LLB 净效应温和(-7%~+2%),且 Ge99(原 4.4× 主诉求)无感 —— 待查 heavy 路径(下一步:hash_global_kernel
同款 LLB)。TSOPF_RS_b2383 非矩阵级 dense_win(D5 -0.8% 噪声),矩阵级 dense_win 的直接案例本轮未逮到
(结构上两路同固定窗口,D5 仍应纯赢)。

## 7. 实现状态

- 代码:`src/spgemm_kernel_hash.cu`(hash_dense_count_kernel / hash_dense_direct_kernel /
  dense_sum_kernel[unsigned ll 原子 —— sm_90 无 signed ll atomicAdd 重载]/ zero_est_kernel
  + caller 七处)。nvcc -c 编译干净(跑批期间未 make,binary 未动)。
- **未测**:跑批(refresh_ocean_sym)完成后 make + 冒烟 + A/B。
- 潜在风险清单:①count/direct 同构性(过滤位置、窗口划分无关性——disjoint cover 已验)
  ②d_off 重扫后所有 hash 侧消费者一致性(accumulate/compact/retry/heavy 均改用新空间)
  ③matrix-level 全 zero est 的 tmp_slots=0 → dev_alloc(0) 已确认安全(bump 1B)。
