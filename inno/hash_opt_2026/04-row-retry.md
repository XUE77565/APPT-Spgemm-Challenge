# 04 · 行级重试(Row-Level Retry)

> commit `de4abe7`。对标 Ocean `AccumulatorHash.cuh` 的 `out_overflow_row_ids` 机制:
> 概率 sizing 必有行级欠估尾部,**与其整阵回退,不如把溢出行收集起来用精确上界重跑**。

## 1. 动机:欠估尾部的算术

无偏 estimator 的行级噪声 ~σ ≈ D/√m ≈ 9%D(m=128 partitions)。任何 EXPAND < 1+4σ/D
都会让大矩阵(行数 > 1/P)几乎必然出现至少一行欠估。旧行为:该行表满 → overflow 标志 →
**整阵回退 merge3**(ocean337 上 = 28.5× 灾难)。这就是 est 一直不敢收紧的结构性原因。

## 2. 机制(五步)

```
① 收集:所有 kernel 的溢出点(probe 满 / extract 超槽)记录行号
   → d_ovf_rows[d_ovf_cnt++](atomicExch 去重门,见 §3)
② 重跑:对收集的行,用每行精确 flop(D ≤ flop 恒真)定表
   → ht = pow2(2×flop),走 hash_global_kernel(表必然够大:2×flop ≥ 2×D)
③ 重试区:独立 tmp 区(槽位 = flop,scan 得行偏移),输出写这里
④ 排序:heavy_seg + cub DeviceSegmentedRadixSort 对重试区分段排序
⑤ 路由:compact 阶段主循环按 d_row_ovf 标记跳过这些行,
   末段用重试区指针单独 compact_copy
```

关键接线细节:
- **重试前必须清 accumulate 留下的旧 overflow 标志**,再让 retry_prep 对"不可重试行"
  (flop 超全局表上限)重置标志 → 仍有则整阵回退(正确性兜底)。
- row_nnz 在重试 kernel 里被写成真实计数,下游 cnnz_scan/C_row_ptr 自然正确。

## 3. 陷阱两则(都真实踩过)

1. **probe 风暴行重复记录**:一个表满的行,每个失败的元素都会记一次 → 几千次 append
   刷爆 `d_ovf_rows`(容量 = A_rows)→ 后来的行被挤掉没重试 → **静默数据损坏**
   (表现:2 行乱序)。修:`if (!atomicExch(&row_ovf[i], 1)) { append }` 去重门。
2. **重试前不清旧标志** → `if (!ovf2)` 永假 → 重试跑了但结论永远是失败。修:进重试块
   先 `cudaMemset(d_overflow, 0)`。

## 4. 收益

- **EST_EXPAND 1.5 → 1.15**(欠估尾部由重试兜底,不再需要保险垫)→ est 普降 ~2×;
- bcsstk30/3Dspectralwave2 从"守卫触发→整阵回退"变"5 行重试 0.74ms 后继续";
- 重试成本 ~0.3-0.7ms(欠估行通常 < 10 行)。

## 5. 与 Ocean 的对照

Ocean 在 hashSymbolic/numeric kernel 里维护 `out_num_overflow_rows/out_overflow_row_ids`
+ 全局 buffer bitmap 池(getUniqueGlobalBuffer),溢出行交 `sparseOutlierLauncher` 重跑
(其 spark_outlier_kernel_size 按 max_product 对齐)。我们的差异:**用 flop 精确上界定表**
(它们按行 max_product 分档)——更省内存,代价是 dup 高的行表偏松(2×flop)。
