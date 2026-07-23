# 工程优化细节

> 这些是在实现创新点的过程中做的工程优化,不是独立的创新主张,但对性能有实际贡献。

---

## 1. accumulate G 从 ht_size 自适应(commit 494cd0c)

### 问题

hash SPA accumulate 阶段,一个 block(256 线程)处理一行。线程分 G 个组,每组处理一个 k,组内 G 线程并行处理 j:

```
G = group size(每组线程数)
num_groups = 256 / G(并行处理的 k 数)
```

固定 G=16 对小阵浪费严重:bp_0 每行只有 4 个 k、4 个 j → 16 组中 12 组空闲(75% 线程浪费)。

### 做法

G 从 ht_size(已有的 kernel 参数 = HLL 估计的 hash 表大小)直接推:

```cpp
int G = (ht_size <= 64) ? 4 : (ht_size <= 256) ? 8 : (ht_size <= 1024) ? 16 : 32;
```

ht_size 编码了行的"重量":小 → 少 distinct 列 → 少 j → 小 G(更多 k 并行);大 → 多 j → 大 G(更多 j 并行)。零 scan、零额外开销。

### 效果

| 矩阵 | ht_size | G(旧) | G(新) | accumulate |
|---|---|---|---|---|
| bp_0 | 32 | 16 | **4** | 0.28→0.13ms(**2× 快**) |
| bcsstk08 | 32-128 | 16 | **4-8** | 0.47→0.23ms(**2× 快**) |
| bcsstk30 | 1024 | 16 | 16 | 持平 |

### 为什么不是创新点

"根据工作量调线程分组"是 GPU 编程常识。Ocean 的 localLoadBalance(per-row 动态 G,已发表)更精细。我们的 ht_size → G 映射是一个简单 threshold table,是工程优化而非算法创新。

### 尝试过的更复杂版本(localLoadBalance,已回滚)

Ocean 式 per-row 动态 G:accumulate 内 scan B-row 统计 → 计算最优 G。实测净负(scan 开销 +22% cycles > G 优化收益)。Ocean 能用是因为 stats 在 estimation 阶段已算好(d_num_products),我们在 accumulate 内部做太贵。

---

## 2. hll_merge blockDim 32→64(commit c8a7ae5)

### 问题

hll_merge(kernel Phase2)用 32 线程(blockDim=HLL_M/4=32)做 packed __vmaxu4 merge。Ocean 用 64 线程(b_rows_per_iter=2)→ 2× 吞吐。

### 做法

blockDim 32→64(HLL_M/2),smem 128→256(HLL_M×2)。kernel 代码已支持 b_rows_per_iter>1(j-loop cross-batch max),只改 host launch 参数 2 行。

### 效果

bcsstk30: hll_merge 0.205→0.139ms(1.5× 加速)。

### 为什么不是创新点

纯对齐 Ocean 的已有设计,我们的 kernel 从一开始就写了 j-loop 支持多 batch,只是 host 没用。对齐不等于创新。

---

## 3. compact+sort 融合(commit 0bba357)

### 做法

把 compact + 全局 thrust::sort + split_key 三个阶段融成一个 per-row BlockRadixSort kernel(hash_compact_sort_kernel)。每个 block 从 tmp(gapped)读 → BlockRadixSort 行内按 col 排 → 直接写 CSR(packed,有序)。

### 效果

compact+sort 从 2.48ms 降到 1.01ms。省掉 d_key buffer(−200MB)。

### 为什么不是创新点

Ocean 的 compactAndSort + sortOutputDyn 是同样的设计。我们的实现细节不同(count-sort for 小行 + BlockRadixSort for 大行),但概念是已有的。

---

## 4. 小阵 pipeline 精简 streamlin(commit e08f451)

### 做法

小阵(A_nnz<100k)用并行 count(1 kernel)替 HLL 两阶段(2 kernel)。省 1 kernel + merge。

### 效果

bp_0 有效(0.36→0.31ms),但 net wash(部分阵因 flop 估松 → 进 bin8 → sort 稍慢)。保留备用。

### 为什么不是创新点

只是跳过 HLL 用更简单的估计,工程上合理但非算法创新。
