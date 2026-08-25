# 05 · 批量 Kernel 族(warp-per-row)

> commit `a72f51a` → `96e633e`。治"每行固定开销":370 万行的矩阵上,1 CTA × 256 线程
> 伺候平均 ~6 个中间积,启动/表初始化/同步全是纯开销(profiling 定位为 333SP 型 55% 差距源)。

## 1. kernel 家族

### hash_spa_batched_kernel(累加,bin0:est≤64 且行长≤32)
- **warp-per-row**(8 行/CTA):每 warp 一张 SMEM 私有表(BATCH_HT=128 槽 = est≤64 的 2× 余量);
- k 串行 / j 32 路并行(lane stride)——所以门有"行长≤32"(k 多的行 warp 串行反而慢,
  3Dspectralwave2 曾回归 62→81ms 的教训);
- **融合有序 extract**:pack 非空槽 → count-rank(每项数比它小的)→ `tmp[base+rank]` 直写有序,
  下游 compact 免排序(同时吃掉 compact 差距的 11%)。

### hash_compact_copy_warp_kernel(compact,bin0)
- 同 warp-per-row,3M 行的 copy 从"每行一 CTA"变 8 行/CTA:6.0→2.1ms。

### warp 聚合原子(binning:compute_bucket + scatter)
- 3.7M 行对 12 个全局计数器 atomicAdd = 串行化点(单 kernel 2.9ms);
- `__match_any_sync` 找同 bin 的 lane 组 → leader 一次 `atomicAdd(popc)`;
- scatter 同理,成员按 warp 内 rank 领槽:binning 总耗时 5.9→0.24ms。

### count_intermediates_par_kernel(warp-per-row 重映射)
- 旧:每行一 block(3.7M blocks!)5.4ms → warp-per-row 0.75ms(对齐 Ocean analysis 0.69)。

## 2. 分桶路由(改后全景)

```
est ≤ 16                → ultra(线程级线性去重,CAP=32)
16 < est ≤ 64 且 rk ≤ 32 → bin0 批量 kernel(私有 128 槽表)
16 < est ≤ 64 且 rk > 32 → 原 hash_spa(G 组 k 并行,治长链)
64 < est ≤ HASH_CAP      → 原 hash_spa 梯(bin1-9,表 32<<bi,≤4096 带 2× 松弛)
est > HASH_CAP           → heavy 全局表(02 文档)
任何溢出/欠估            → 行级重试(04 文档)
```

⚠ 门的经验:**批量门必须独立于表梯计算**——曾把门写成 `bi <= 1`,而 2× 表目标把
est~36 的行推到 bi=2,333SP 的主力行全漏出批量路径(86→79ms 才发现)。

## 3. 实测(333SP 相位演变)

| 相位 | 优化前 | 优化后 |
|---|---|---|
| count_flop | —(MinHash 路径无) | 0.75ms |
| MinHash 两遍 | 6.3ms | **0(avg_product≤64 门免除)** |
| binning | 5.9ms | 0.24ms |
| accumulate | 21.97ms | **6.47ms** |
| compact+sort | 5.0ms | 2.06ms |
| **compute-only 合计** | **38.5ms** | **9.7ms**(vs Ocean 6.19 = 1.55×) |

## 4. 与 Ocean 的机制对照

Ocean 小行走 ESC 语义(`ESCKernelDispatcher<16/32>`:每行 16/32 线程,展开+块内排序,
重行 `hashNumericSubWarpKernel<4>`),并有 `classifyOutlierRows` 分流。我们保留 hash
语义(表去重)批量——**对高 dup 行更优**(exdata_1 型 76× dup,ESC 要物化 76× 中间积)。
这是论文里"我们的 ultrasparse 路径 vs Ocean 的 ESC 路径"差异化点。
