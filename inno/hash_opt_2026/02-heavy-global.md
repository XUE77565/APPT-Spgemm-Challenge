# 02 · Heavy 全局表路径

> commit `7dad171`。est > HASH_CAP(16384)的重行不再整阵回退 merge3(ocean337 上 67 阵
> 28.5× 灾难的主源),改走 global-memory 开放寻址表(对标 Ocean numeric.overflow)。

## 机制

```
bin 11(est > HASH_CAP)→ 每行一张全局表(ht = pow2(2×est) ≥ 2×D,容量必够)
  → 表区 arena:exclusive scan 行偏移(注意:exclusive scan 输出 [k]=Σ_{i<k},恰为行 k 偏移;
     inclusive scan 才需要 +1 偏移模式——这两个别搞混,见 §连环 bug①)
  → hash_global_kernel:CTA=行,G=32 组结构同 hash_spa,atomicCAS/atomicAdd 打全局表
  → extract 无序写 tmp → heavy_seg + cub DeviceSegmentedRadixSort 分段排序
  → compact 阶段该 bin 走 compact_copy(读排序后的 scratch)
```

护栏:单表 ≤ GLOBAL_HT_MAX_SLOTS(4M 槽);表区 arena ≤ GHT_MAX_BYTES(24GB,超则按
overflow 回退——TSOPF 曾因 estimator 高估把 arena 顶到 82GB 触发此熔断,estimator
修复后 ~2GB 正常通行)。

## 连环 bug 三则(验尸记录,防再犯)

1. **exclusive scan 套了 inclusive 的 "+1 偏移"模式**:`scan(..., d_off + 1)` + 读 `[n]`
   ——行偏移全体错位一格,行 0 和行 1 共用表 0 → 数据竞争 → 非确定崩溃。
   验尸方法:dump `tab_off[0,1,n-1,n]` 与 `ht[0,...]` 对账,`tab_off[1]=0` vs `ht[0]=131072`
   一眼定罪。修复:exclusive scan 直写 `[0..n-1]`,总量 = `[n-1] + ht[n-1]`。
2. **混型 thrust scan 的类型纪律**:int 输入 → long long 输出的包装必须 `device_ptr<long long>`
   包 ll 指针(编译错误还算幸运;更隐蔽的是把 `device_ptr<int>` 套在 ll 指针上)。
   最稳:输入也升 ll(heavy_ht 本身 ll)。
3. **重试路径的 scratch 指针**:cub SortPairs 需要 out ≠ in;scratch 按 total_est 分配,
   compact 该 bin 传 scratch 指针而非 tmp。

## 实测

- 3Dspectralwave2:599 重行、arena 1.26GB、accumulate 内全局表 ~10ms 级——整阵从
  merge3 回退(29ms)变 hash 13.6ms 反超 Ocean(18.7)。
- TSOPF_b39(estimator 修复后):14030 重行 arena ~2GB,hash 全程 ~76ms,**胜 Ocean(400)5.3×**
  ——Ocean 对重行阵(电网类)是弱项,这是我们反超的第一块阵地。
