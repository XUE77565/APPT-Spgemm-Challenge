# docs/70 dense 两侧算法对照(Ocean AccumulatorDense.cuh 精读;09-02)

## 1. Ocean 的 dense 怎么算(逐 kernel 实读)

**symbolic(denseSymbolicKernel)**:
- SMEM = **位图**(uint32 字,1 bit/列),尺寸 = 行 span 起始列 start 到末列(end_element_b),
  min(档位上限, span+1) —— 不含值数组!
- 乘积循环(localLoadBalance 分组,同我们 LLB):每乘积 `bitmap |= bit(col−start)`(atomicOr)
- 计数 = 位图 popcount → **精确 nnz**(一次扫描 span/32 字)

**numeric(denseNumericKernel)**:
- SMEM = 值数组(span doubles)+ 位图 + 前缀数组;乘积 `val[col−start] += a*b` + 位图置位
- **denseCompact 发射**:位图字内 popcount 前缀(并行,span/32 项)→ 按置位位升序直写
  【symbolic 给的精确偏移】—— 位图升序 = 天然有序,免排序免串行 prefix
- 溢出行 → 全局逃逸 buffer(机制 B 原型)

**大 span**:分档(BLOCK_SIZES 梯 × 每档 DENSE 尺寸),超档走 denseNumericIterKernel(迭代窗)
与 static 变体 —— **它们也有窗,但窗宽按档位分级,非单一固定值**。

## 2. 我们的 dense(hash_dense_window/count/direct)

- 固定窗 PB2_W=5400;每窗:clear **dflag(1B/列)+dval(8B/列)** = 9B/列
- 乘积累加:同(dflag/dval 原子)✓
- 发射:**warp0 串行 prefix 5400 项**/窗(docs/63 §5:块并行版回归)→ 有序写
- 方案5:count(dense_count)→ 精确偏移 → dense_direct 直写(= Ocean 的 symbolic→numeric 同构!)

## 3. 差异清单(逐项,带数量级)

| 维度 | Ocean | 我们 | 差距量级 |
|---|---|---|---|
| symbolic 占位 | **位图 1 bit/列** | flag 1B/列 | clear 流量 8×(numeric 才有值数组) |
| 发射前缀 | **位图字 popcount 前缀(span/32 项,并行)** | warp0 串行 5400 项/窗 | **32× 少且并行** |
| 窗宽 | 档位分级(多尺寸) | 单一 5400 | 尾部适配性 |
| numeric 写 | 精确偏移直写(symbolic 先给) | 方案5 同构 ✓ | 平 |
| 值累加 | SMEM atomicAdd(span 数组) | 同 ✓ | 平 |
| LLB 分组 | localLoadBalance | 同(LLB)✓ | 平 |

**Ga 行(span 112k/est 4096)算账**:Ocean symbolic = 位图 14KB clear + popcount 3.5k 字;
我们 = 21 窗 × (clear 48KB + 串行 prefix 5400)——**位图版把 docs/63 的 O(span) 税砍 ~32×**。

## 4. "全面学 Ocean"dense 改造策略(排 ROI)

1. **dense_count 位图化**(symbolic pass):dense_count_kernel 改 span 窗位图置位+popcount
   —— 直接砍 Ga 族/3Dspec2 类 counting 主税,改动面小(单 kernel)
2. **dense_direct 发射位图化**:dflag(1B)→位图 + 字级并行 popcount 前缀替 warp0 串行
   —— 砍每窗 5400 串行(docs/63 §5 块并行版回归的教训 = 字级并行才便宜)
3. 窗宽分档(远)
4. 与 D5H 解封协同:bitmap symbolic = counting 更便宜 → dup 门可放宽 → precise 覆盖面扩大
