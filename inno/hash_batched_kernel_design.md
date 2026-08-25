# 小行批量 accumulate kernel 设计规格(Phase 2 主杠杆,2026-08-25 深夜写)

> 目标:攻 hash 差距的 55%(profiling:`hash_gap_ocean337_profiling.md`)。
> 现状:333SP 型(370 万行×均度 6)accumulate 30ms vs Ocean usparse_main ~4ms —— 我们 1 CTA/行×256 线程伺候 ~6 个积。
> 本文档供下个会话直接实现;前置状态见 commit d0b5d8c。

## 设计(照 Ocean ESCKernelDispatcher 形态 + 我们 hash 语义)

**目标行**:est ≤ 64 的行(现 bin0-2 + ultra;333SP 99% 行在此)。其余行仍走现有 per-row 路径。

**kernel `hash_spa_batched_kernel<TPB, RPB>`**:
- grid = ceil(n_target_rows / RPB),block = TPB(256)线程;
- **每 warp 一行**(RPB = TPB/32 = 8 行/CTA);warp 内 32 线程 = j 并行(k 串行,同 hash_spa 的组结构但 G=32):
  ```
  row i = target_rows[blockIdx*RPB + warp]
  for (p = rs; p < re; p++) {            // k 串行(小行 k 数少)
      k = A_col[p]; a = A_val[p]
      for (q = ks+lane; q < ke; q += 32) { // j 32 路并行
          j = B_col[q]; v = a*B_val[q]
          插入 warp 私有表(SMEM per-warp 段,ht=64:col[64]+val[64] per warp)
      }
  }
  ```
- **表放 SMEM,per-warp 段**:8 warps × 64 槽 × 12B = 6KB/CTA ✓ occupancy 极好;
  插入:lane 独立 probe 自己 warp 的表(atomicCAS/atomicAdd SMEM),无跨 warp 同步;
  probe 上界 64(必终止);溢出(distinct>64)→ overflow 标志 + 该行记入 retry。
- **extract 融合**(免 compact_copy 二遍):warp 扫自己 64 槽,计 rank(warp 内 shfl 前缀)→
  有序写 tmp(base+rank);count-sort 免了(64 槽 warp 内排序:warp bitonic 或直接
  count-rank,32 线程×2 轮)。**这一步同时吃掉 compact 差距的 11%**。
- est ≤ 16 的行(现 ultra):并入同 kernel 或保留 hash_ultra(先并入,RPB 内无区别)。

**接线**:
- compute_bucket:est ≤ 64 → 新 batched bin(替代 bin0-2+ultra 的这些行;ultra 阈值并入);
- host:一个 launch 吃掉全部小行,删 per-bin 0-2 循环;
- compact 阶段:这些行已有序 → hash_compact_copy 路径(现状 bin0-5 已如此,零改动)。

**正确性门禁(安全协议)**:
1. gpu_check → 2. bcsstk30/3Dspectralwave2/pwtk/333SP 四阵对拍旧路径(C_nnz+值+0 乱序)
→ 3. est_synth 抽 3 阵对拍 → 4. 才进 REFRESH。

**预期**:333SP accumulate 30→~6ms,compact 5→~2ms,总 ~84→~50ms compute-only(vs Ocean 6.2
仍差 8×?→ 不对,Ocean 6.2 是全程;我们 50 里 d2h/h2d 已扣)——重新核算:compute-only 目标
~25-30ms,然后 binning(5.2)与 est(flop 路径 0.75+0.04)已接近 Ocean 对应项,残差在
accumulate 内的原子吞吐,再攻。

## 附:Ocean 参照(读它源码的产出)
- `ESCKernelDispatcher<16/32>`:NUMERIC_USPARSE_K_ROWS_PER_BLOCK 行/CTA × 每行 16/32 线程
  (它小行走 ESC 语义=展开+排序;我们保留 hash 语义=表去重,对 dup 高的行更优——论文差异化点);
- `hashNumericSubWarpKernel<4, UB>`:4 线程/行的 hash 变体(它也有!)—— 我们 warp(32)/行
  是更粗粒度,可加 <8,32> sub-warp 档;
- outlier 行(超均度)单独 kernel —— 我们已有 heavy bin,等价。
