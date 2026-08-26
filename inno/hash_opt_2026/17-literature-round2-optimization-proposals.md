# 17 · 文献深读第二轮:5 个新优化方案(2026-08-26)

> 广度搜索 2020-2025 文献 + 精读 HSMU/MAGNUS/NVIDIA cuCollections/Hive 代码与论文,
> 结合我们对 Ocean 的差距分析(accumulate 原子吞吐 12.4 vs 22.4 G/s),提出 5 个方案。

## 差距回顾(优化必须攻克的数字)

```
我们的 accumulate 吞吐: 12.4 G-products/s(germany_osm)
Ocean 的 accumulate 吞吐: 22.4 G-products/s(germany_osm)
差距: 1.8×,全部在 hash 表的 CAS + atomicAdd 上

bcsstk30 全流程(compute-only 3.8ms):
  sizing 0.4 | binning 0.05 | accumulate 1.93 | retry 0.74 | compact 0.68
  → 固定开销(sizing+retry+compact) = 1.82ms = 48%
```

---

## 方案 1:预排序二分寻址(学 HSMU)⭐⭐⭐⭐⭐

**来源**:[HSMU-SpGEMM (HPCA 2025)](https://github.com/wuminqaq/HSMU-SpGEMM) 源码实读

**核心发现**:HSMU 的 numeric 阶段根本不做 hash 探测——它用**二分查找**已排序的列号数组:

```cpp
// HSMU src/small/numeric.cuh:341
hash = Binary_search_for_hash_loction(shared_col, start, end, bcol);
atomicAdd(shared_val + hash, aval * bval);
```

**机制**:
1. 轻量 symbolic pass 先得到每行的**精确列集合**(已排序);
2. numeric 时对每个积 `bcol`,在 sorted 列号数组上**二分查找**其位置(O(log d));
3. 直接 `atomicAdd(shared_val[pos], val)` ——**零 CAS、零冲突、零探测链**;
4. 输出天然有序(col 数组有序,按位置写)。

**vs 我们的 hash SPA**:
| | hash SPA(我们) | 二分寻址(HSMU) |
|---|---|---|
| 每积操作 | CAS + Add | **仅 Add** |
| 查找 | O(1) 均摊(有冲突) | O(log d)(确定性) |
| CAS 冲突 | 热列串行化(dup 76×) | **不存在** |
| 输出有序 | 需 compact+sort | **天然有序** |
| 前置 | est(概率) | **精确 symbolic**(二分可用) |

**实现路径**:用我们的 flop 精确上界 + 一遍轻量 symbolic(只写列号,不写值)→ sorted col 数组
→ numeric 用二分替换 hash。

**预期**:accumulate 12.4→20+ G/s(CAS 消除);compact 减半(免排序);
但 symbolic 增加一遍 O(nnz) 开销(~1ms)。

**⚠ 注意**:这实际上把我们的 hash 家族向 merge 家族靠近了一步(用"已知列集合"替代
"探测发现列集合")——论文叙事需调整。

---

## 方案 2:Warp 协作探测(学 NVIDIA cuCollections)⭐⭐⭐⭐

**来源**:[NVIDIA cuCollections 博客](https://developer.nvidia.com/blog/maximizing-performance-with-massively-parallel-hash-maps-on-gpus/)

**机制**:每 4 线程协作探测一个 key 的插入(而非每线程独立探测):
```
1. 4 线程各检查 hash 表的 4 个不同槽位(coalesced load)
2. __ballot_sync + __shfl_sync 快速判断哪个槽位空闲
3. 一致决定插入位置 → 一次 CAS
```
NVIDIA 实测:**插入吞吐 +13%,查找吞吐 +40%**(高负载因子时)。

**实现**:改 `hash_spa_batched_kernel` 的 j-loop:
```cpp
// 现在:每 lane 独立 CAS 探测
for (q = ks + lane; q < ke; q += 32) { ... }

// 改为:4-lane 协作(每组 4 线程处理 1 个积,同时探测 4 个候选槽)
for (q = ks + lane/4; q < ke; q += 8) {
    // 4 线程同时探测 slot+0/1/2/3
    // ballot 选出空闲 → leader CAS
}
```

**预期**:插入吞吐 +13%(保守估计 accumulate 提升 ~10%)。

---

## 方案 3:Dense Bitmap 累加器(学 HSMU mask.cuh)⭐⭐⭐

**来源**:HSMU `src/small/mask.cuh`

**机制**:对 dense-enough 行,用 bitmap + atomicOr 标记列存在性,prefix sum 得有序输出:
```cpp
// HSMU mask.cuh: Form_mask_array_for_CSR_kernel
atomicOr(&mask_array[i], 1 << k);  // 标记列 col 存在
// 之后 prefix sum over bitmap → sorted positions
```

**我们已有 dense 路径的补充**:我们的 dense 路径(exdata_1 类)用 `double vals[n]`——
每列 8 字节。改用 bitmap(1 bit/列)+ prefix sum 可将 SMEM 需求降 64×,
使 dense 路径能覆盖 n 更大的矩阵(TSOPF n=28k 需要 28k/8=3.5KB bitmap vs 226KB vals)。

**实现**:dense 路径分两阶段——(1) bitmap 累加 + atomicOr 标记,(2) prefix sum + 值累加。
或者直接在现有 dense kernel 里把 flags 从 byte 改为 bit。

**预期**:dense 路径覆盖范围扩大(从 n≤14980 到 n≤200k+);TSOPF 类直接受益。

---

## 方案 4:行内列域多片(MAGNUS hierarchical multisplit)⭐⭐⭐

**来源**:[MAGNUS (ICS 2025)](https://arxiv.org/html/2607.22866v2) — "hierarchical multisplit"

**机制**:把重行按列域切成独立 chunk(类似我们的 merge3 列桶思想!),每 chunk
用自己的局部 hash 表:
```
row i 的积 → multisplit 按列范围分到 K 个 chunk
chunk 0: [0, n/K) 的积 → 独立 hash 表(chunk-local)
chunk 1: [n/K, 2n/K) 的积 → 独立 hash 表
...
每个 chunk 的表更小(≤n/K slots)→ SMEM 占用降 K 倍 → occupancy 升 K 倍
```

**vs 我们的现状**:我们的 heavy 路径(hash_global_kernel)用全局内存大表——
慢(全局原子)。MAGNUS 的做法是**不等表大到全局,而是切小到 SMEM**。

**实现**:把 est > HASH_CAP 的行,切成 K = ceil(est/HASH_CAP) 个列域片,
每片走现有 hash_spa(SMEM 表)。

**预期**:heavy 行速度提升 3-5×(SMEM vs 全局原子);TSOPF/Ga 类受益。

---

## 方案 5:固定开销瘦身(工程)⭐⭐

**来源**:自身 profiling(bcsstk30 固定开销占 48%)

三个子项:
1. **retry 无宿主化**:Ocean 在 kernel 内 switch_to_gmem,我们在 host 端倒手;
   改为 kernel 内溢出时直接切全局表,免 host 往返(省 0.74ms);
2. **sizing 合并**:count_flop + MinHash + scan 三阶段 → 单 kernel 融合
   (读 B 行一次,同时产出 flop 和 sketch;省 launch 开销);
3. **compact 与 accumulate 融合**:batched kernel 的 extract 已有序,
   直接写 CSR 而非先写 tmp 再 copy(省 0.68ms)。

---

## 优先级与实施路径

| 方案 | 预期收益 | 难度 | 时间 | 优先级 |
|---|---|---|---|---|
| **1. 二分寻址** | accumulate 1.6× + compact 减半 | 中 | 1 天 | ⭐⭐⭐⭐⭐ |
| **2. 协作探测** | accumulate +10% | 低 | 2 小时 | ⭐⭐⭐⭐ |
| **4. 行内多片** | heavy 行 3-5× | 中 | 半天 | ⭐⭐⭐⭐ |
| **3. Bitmap dense** | dense 覆盖 n 扩 10× | 低 | 2 小时 | ⭐⭐⭐ |
| **5. 开销瘦身** | 小阵 30-50% | 中 | 1 天 | ⭐⭐⭐ |

**推荐实施顺序**:2(快速验证)→ 1(核心突破)→ 4(补齐 heavy)→ 3 → 5

**预期总体效果**:方案 1+2 攻 accumulate(12.4→25+ G/s,超 Ocean);
方案 4 攻 heavy 行;方案 3+5 攻固定开销 → 综合 geomean 从 2.27× → 1.0-1.2×。

---

## 参考文献

1. [HSMU-SpGEMM (HPCA 2025)](https://github.com/wuminqaq/HSMU-SpGEMM) - 二分寻址 + bitmap
2. [MAGNUS (ICS 2025)](https://arxiv.org/html/2607.22866v2) - hierarchical multisplit
3. [NVIDIA cuCollections](https://developer.nvidia.com/blog/maximizing-performance-with-massively-parallel-hash-maps-on-gpus/) - warp 协作探测
4. [Hive hash (2025)](https://arxiv.org/html/2510.15095v1) - warp-cooperative dynamic hash
5. [Hash-based Multi-phase SpGEMM](https://arxiv.org/html/2512.12036v1) - AIA 间接访存加速
6. [Ocean (arXiv 2604.19004)](https://arxiv.org/html/2604.19004v1) - HLL 估计 + 混合累加器
