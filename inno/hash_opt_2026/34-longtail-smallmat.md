# 34 · 长尾小阵攻坚(2026-08-28 上午班)

> 目标:近赢带(1.0-1.2× 的 12+ 小阵)翻赢。解剖:管理相位 39%(nemeth18)/compact 23%(tsyl201)/
> retry 10%(cant)。结合 docs/27(Ocean 数据流)+ docs/33(文献四轮)。

## 1. 本轮三试(1 正 2 否决)

| 试 | 结果 | 教训 |
|---|---|---|
| **D2H 合并**(est_scan 的 total_est 延到 binning 单 sync)| ✅ 保留:nemeth18 -2.6%/crplat2 -1%,余中性 | 小阵每个同步点 ~10μs |
| est=flop 小阵门(nnz<1M ∧ flop<250M 免 MinHash)| ❌ 否决:nemeth18 **+166%**/crplat2 **+840%**/cant +28% | flop 当 est 把表撑大 dup 倍(dup 4-9),1CTA/行大表利用率崩;**MinHash 正是廉价 dup 探测器,不可省**(Ocean 用 HLL 同理)|
| D5H(docs/31 §3.5)| ❌ 否决(count 全价)| count pass 只该打 csort 大行 |

## 2. 剩余小阵税的结构(v11,nemeth18 compute 0.72ms)

count_flop 9% + mh 11% + est_scan 7% + binning 12% + cnnz_scan 7% = **46% 非累积相位**
(多数是 launch/同步延迟,~11 个相边界 × ~2.1μs + 3 次 sync 往返)。
accumulate 62% 内还有 1CTA/行 的利用率问题(轻行)。

## 3. 下一步(按 docs/33 文献 + 本轮教训排序)

1. **PDL**(sm_90 programmatic dependent launch):consumer 内核加 `cudaGridDependencySynchronize()`
   + host `cudaLaunchKernelEx` PSS 属性,逐相边界省 ~2.1μs;估计链(count_flop→row_span→mh×2→
   compute_bucket→scatter)先装。预期小阵 -3~6%。注意:无 device 侧 sync 的纯属性不安全。
2. **轻行 (R 行 × T 线程) 2D 模板**(spECK 32 行/块):bin dispatcher 加 R 维映射,治 accumulate
   利用率(nemeth18 的 0.44ms 大头)。
3. **fingerprint 探测**(8-16bit/槽早退):中阵 1.5-3× 的探测长尾(docs/33 A1)。
4. tsyl201 的 compact 23%:轻量 count 只打 csort bins(6-10)行 —— D5H 的正确子集。

## 4. 关键数字备忘

- 赢面 6/336(v11);近赢带:nemeth01 1.023/nemeth18 1.029(D5H 修后 0.70 vs Ocean 0.70 刀锋)/
  crplat2 1.063/pkustk09 1.067/fp 1.077/nemeth19 1.091/laminar 1.118/nemeth20 1.130/oilpan 1.139/cant 1.171
- STREAMLINE_NNZ(100k)是死宏,活门仅 avg_product≤64 —— 小阵全走 MinHash(本轮证明确实该走)。
