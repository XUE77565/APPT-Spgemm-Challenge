# 19 · 方案实验进度 + 剩余差距结构(2026-08-26)

## 方案实验汇总

| 方案 | 来源 | 结果 | 状态 |
|---|---|---|---|
| 2. warp 协作探测 v1 | NVIDIA cuCollections | bcsstk30 +4%/pwtk +9% 退步 | ❌ 否决 |
| 2. warp 协作探测 v2(混合) | 改进版 | 高 dup 持平,中密度退步 4-10% | ❌ 否决 |
| 1. 预排序二分寻址 | HSMU (HPCA'25) | 8 阵 A/B 持平(exdata -4%,germany +7%) | ❌ 无净收益 |
| **4. 行内列域多片** | **MAGNUS (ICS'25)** | **待实验** | ⏳ |
| 3. Bitmap dense | HSMU mask | 待实验 | ⏳ |
| 5. 固定开销瘦身 | 自身 profiling | 待实验 | ⏳ |

## 方案 2/1 的教训

**为什么文献中的优化在我们的场景下不生效:**

1. **方案 2(协作探测)**: cuCollections 的 +13% 需要 load factor >80%;
   我们的表 ~50%,平均探测链 1.2 步,协作的 ballot 开销 > 省的探测

2. **方案 1(二分寻址)**: HSMU 设计给"不知道列集合"的通用 SpGEMM;
   我们已有精确 est(MinHash + flop 上界),不需要 symbolic pass——
   **我们的 est 创新已替代了 HSMU 的 symbolic 需求**

## 剩余差距结构(为什么还差 ~1.5×)

从 Ga3(最大差距阵之一)的相位分解:

```
Ga3As3H12 (n=61349, avg_product=60k, dup~76×, C=62M nnz):

  我们的 accumulate = 74.2ms → 49.6 G/s
  Ocean 的 numeric   ≈ 30ms  → ~120 G/s

  差距 = 2.4×, 全部在 accumulate 阶段的原子操作吞吐
```

**根因:热列的 atomicAdd 串行化**

- Ga3 的 dup 因子 76× → 每个输出列平均被写 76 次
- 这些写入都打到同一个 SMEM 地址 → atomicAdd 串行化(独木桥)
- H100 SMEM atomicAdd(double) 吞吐 ~1/clk/bank
- 理论上限 ≈ 32 banks × 1.755 GHz = 56 G/s (如果我们 49.6 则已接近)

**Ocean 为什么能达到 120 G/s?**
- Ocean 用 `atomicAdd_block`(shared memory) 的 `double` 版本
- 关键:它的 **负载均衡让多个 SM 同时工作不同的行**,不存在单行瓶颈
- 更重要:它的 **hybrid accumulator 把 value 放全局内存**(atomicAdd 全局是 L2 吞吐,
  比 SMEM 更高带宽——多个 SM 的 L2 写可以并行)

**这解释了为什么方案 1 不工作**:它只去掉了 CAS(首次插入),没解决
dup 后续的 atomicAdd 串行化(每次写入都要原子)。

## 下一步(按剩余差距的本质)

### 方案 4 修正版: Hybrid Value(学 Ocean 的真正杀手锏)

Ocean 120 G/s 的秘密:**keys 在 SMEM(快速 CAS)+ values 在全局内存(L2 原子)**

- SMEM atomicCAS: 确定位置(1 步,快)
- 全局 atomicAdd(double): 累加值(L2 原子吞吐 > SMEM, 多 SM 并行)

我们目前: keys + values 都在 SMEM → SMEM 原子吞吐成为瓶颈

**改法**: batched kernel 的 `t_val` 从 SMEM 指针改为全局内存指针:
```cpp
// 现在: double *t_val = (double*)(bsmem[...] + BATCH_HT);  // SMEM
// 改为: double *t_val = global_val_pool + row_idx * BATCH_HT;  // 全局
```

这样 CAS 在 SMEM(快),Add 在全局(高吞吐),各自最优。

### 方案 5: compact-accumulate 融合

Ga3 的 compact = 10.6ms(8% of total)。batched kernel 的 extract 已有序,
直接写 CSR 而非先写 tmp 再 copy → 省 10ms。
