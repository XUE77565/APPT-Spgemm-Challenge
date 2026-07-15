# 四种 SpGEMM 公式的融合加速方案

> 日期:2026-07-15
> 背景:gust(行向)、outer(外积)、colwise(列向)、inner(内积)四种公式产生**同一批中间项**,只是顺序不同。能否从各自抽取优势结构,组合成更快的算法?
> 关联:`ref/sort_innovation_directions.md`(创新方向)、`ref/spgemm_papers_2025.md`(文献)、`worklog/profiling_analysis.md`(瓶颈)

---

## 〇、核心前提:四种积产生同一批中间项

四种公式计算的 C = A·A 是同一个矩阵,中间项 `{(i, j, A[i,k]×A[k,j])}` 也是**完全相同的集合**,区别只在**外层循环轴 = 产生顺序**:

```
Gustavson (外层=i):  对每行 i,遍历 k∈A[i,:],再遍历 j∈A[k,:]
                      → 中间项按 i 分段(row-grouped)✓ 实测验证
Outer     (外层=k):  对每个 k,遍历 i∈A[:,k],再遍历 j∈A[k,:]
                      → 中间项按 k 分段(k-grouped)✗ 非 row-grouped
Colwise   (外层=j):  对每列 j,遍历 k∈A[:,j],再遍历 i∈A[k,:]
                      → 中间项按 j 分段
Inner     (外层=ij): 对每个 (i,j),算 dot(A[i,:], A[:,j])
                      → 不存中间项(实时算点积,merge of sorted lists)
```

**融合的本质**:不是把四种算法"混在一起跑",而是从每种里抽取它的**结构优势**,组合成一种新方法。

---

## 各自的结构优势

| 方法 | 结构优势 | 结构劣势 |
|---|---|---|
| **Gustavson** | 中间项**行分段**(block i 只写 row i);行间天然隔离 | A[k,:] 被多个 i 反复读(冗余) |
| **Outer** | A[k,:] **只读一次**(k 处理完就不碰了) | 中间项不按行分段(需全局 sort) |
| **Colwise** | 可以**按列累加**(适合 CSC 输出) | 同 outer,中间项不按行 |
| **Inner** | **无中间项**(直接出值,无 sort/reduce);每个点积 = merge of sorted lists | O(n²) 点积,稀疏时大部分为 0(浪费) |

---

## 融合方案

### 融合 1:Gustavson 行分段 + Inner 的"每个 k 贡献有序链" → k-way merge 免排序 ★★★ 最创新

**关键洞察**:Inner product 算 `C[i,j] = dot(A[i,:], A[:,j])` 时,A[i,:] 的每个 k 对应 A[k,j],而 **A[k,:] 的 col 是有序的(CSR)**。所以每个 k 对行 i 贡献一条**有序的 j 链**:

```
行 i 的中间项来源(A 是 CSR → 每条链有序):
  k=0 → j ∈ cols(A[0,:]) = [0, 1]     ← 有序!
  k=1 → j ∈ cols(A[1,:]) = [0, 2]     ← 有序!
  k=2 → j ∈ cols(A[2,:]) = [1, 3]     ← 有序!

现在 ESC:expand 打散 → 全局 sort → reduce
k-way merge:直接归并 3 条有序链 → 归并时就地求和(遇到同 j 累加)
```

**复杂度**:k-way merge 是 **O(中间项数)**,sort 是 **O(N log N)**。大矩阵上差距巨大(bcsstk30: 2M 中间项,sort 23ms vs merge 预估 ~2ms)。

**融合了**:
- Gustavson 的**行分段结构**(block i 处理行 i,不跨行)。
- Inner product 的洞察:**每个 k 贡献一条有序列链**(因为 A 是 CSR)。
- **不需要全局排序**——归并替代排序。

**GPU 实现**:每个 block 负责一行 i。block 内线程协作做 k-way merge。可用:
- **两两归并**(merge pairs of sorted lists, log k 层)——每层用 CUB DeviceMerge 或 block-level merge。
- **基于 shared-mem 的多路归并**(heap/priority queue)——串行性较高,需 block 协作。
- **CUB::DeviceSegmentedRadixSort + reduce**(如果 merge 太难实现,先排序再 reduce,但失去 O(N) 优势)。

**创新性**:★★★ GPU 上 merge-based SpGEMM 极少见(文献里要么 hash 要么 sort)。这是 `ref/sort_innovation_directions.md` 方向 A 的具体化。

**风险**:GPU 上 k-way merge 的并行化是难点(归并天然串行)。

---

### 融合 2:Gustavson 行累加 + Outer 的 A 缓存 → Blocked Gustavson ★★ 最实用

**问题**:Gustavson 里,行 i₁ 和 i₂ 如果都包含 A[i,k]≠0,则 A[k,:] 被**读了两次**(block i₁ 一次,block i₂ 一次)。

**Outer 的优势**:A[k,:] 只读一次(k 处理完就结束)。但代价是中间项不按行分段。

**融合**:让一个 block 负责**多个输出行** G={i₁, i₂, ...}(选共享 k 的行):

```
Block 负责 G = {i₁, i₂, i₃}

for k in ∪ A[i,:] for i in G:      ← outer 的 k 轴迭代
    load A[k,:] → shared memory      ← 只读一次!
    for i in G:
        if A[i,k] ≠ 0:
            acc_i += A[i,k] × A[k,:]  ← gust 的 per-row 累加(hash/dense)
```

**效果**:
- A[k,:] 从 global memory **只读一次**(outer 的优势),缓存在 shared memory(所有行共享)。
- 每行仍有自己的累加器(gust 的优势)→ 中间项**不展开到全局 COO**——直接在 shared-mem 累加器里去重。
- 如果用 hash 累加器 → **完全免 sort**。

**这融合了**:
- Gustavson 的 **per-row 累加器**(每行独立 hash/dense)。
- Outer product 的 **A[k,:] 只读一次**(k-axis 迭代 + shared-mem 缓存)。

**选行策略**:哪些行分到同一个 block?→ **共享 k 的行**。
- 如果 A 是带状结构(很多矩阵是),邻近行天然共享 k。
- 可用矩阵重ordering(Islam et al. 2025)增强行间共享。
- 或简单贪心:按行 k 集合的 Jaccard 相似度聚类。

**创新性**:★★ TileSpGEMM(PPoPP'22)做了 2D tile;这里是"1D 行组"(更灵活的行选择),且结合 hash 累加器(免 sort)。

---

### 融合 3:Gustavson 结构发现 + Inner 点积填值 → 分段 Inner Product ★★

**思路**:不对所有 (i,j) 算点积(太贵),只对 expand 发现已知非零的 (i,j) 算:

```
Phase 1 (symbolic,Gustavson 式轻量 expand):
  对行 i,遍历 k∈A[i,:], j∈A[k,:],用 hash/bitmap 记录哪些 j 非零 → C 的结构

Phase 2 (numeric,Inner 式 merge 点积):
  对每个已知的 (i,j):
      C[i,j] = merge_dot(A[i,:], A[:,j])
      ← 两条有序链(CSR)的归并点积,O(nnz_i + nnz_j)
```

**这融合了**:
- Gustavson 的 expand(快速发现哪些 (i,j) 非零 → 跳过 O(n²) 的全扫)。
- Inner product 的 **merge-based 点积**(无 sort,直接算值;每个点积是两条有序链的归并)。

**效果**:完全免 sort。Phase 1 ≈ symbolic pass(hash/bitmap);Phase 2 ≈ numeric pass(merge 点积)。两遍但都 O(nnz),无 O(N log N)。

**与 cuSPARSE 的区别**:cuSPARSE 也做 symbolic-numeric 两遍,但 numeric 也用 hash 累加器。这里 numeric 用 **merge-based 点积**(利用 A 的 CSR 有序性,每次点积是 O(merge) 的)。

---

### 融合 4:Gustavson + Colwise 列过滤 → 减少无效中间项 ★

**思路**:expand 时预过滤——只展开那些 j 在"有效列范围"内的中间项:

```
expand 时:
  for k in A[i,:]:
      for j in A[k,:]:
          if j 在 i 行的"候选列"集合内(quick reject):
              write (i, j, val)
```

需要额外的列索引结构(哪些 j 对行 i 可能有贡献)。效果:减少无效中间项 → sort 输入更小。

**创新性**:★ 辅助优化,可和局部去重(路线 D)叠加。

---

## 评估

| 融合 | 融合了什么 | 免 sort | 创新性 | 难度 | 预期收益 |
|---|---|:--:|:--:|:--:|---|
| **1. k-way merge** | gust 行分段 + inner 有序链 | ✅ | ★★★ | 高 | sort O(NlogN)→O(N),大矩阵 10×+ |
| **2. Blocked gust** | gust 行累加 + outer 的 A 缓存 | ✅(hash) | ★★ | 中 | A 读取减半 + 免 sort |
| 3. 分段 inner | gust 结构 + inner 点积 | ✅ | ★★ | 中 | 两遍 O(nnz),无 sort |
| 4. 列过滤 | gust + colwise | 部分 | ★ | 低 | 减少 expand 输出 |

---

## 推荐

- **冲论文 / 创新性优先**:**融合 1(k-way merge)**。利用 gust 行分段 + A 的 CSR 有序性,用 O(N) 归并替代 O(N log N) 排序。GPU merge-based SpGEMM 是文献空白。
- **工程 / 快速提速**:**融合 2(blocked Gustavson + hash)**。一个 block 处理多行(共享 A[k,:] 缓存),每行用 shared-mem hash 去重。结合你的 pinned/arena IO → 端到端最优。
- **两者可叠加**:blocked Gustavson 框架里,每个 block 对每行做 k-way merge(而非 hash 或 sort)→ 既共享 A 读取,又用 merge 免排序 → 最大化融合。

---

## 与已有文献的关系

| 融合方案 | 相关文献 | 差异(你的创新点) |
|---|---|---|
| k-way merge | Register-Aware(merge accumulator,NPC'18) | GPU 全局 k-way merge + 行分段结构,Register-Aware 是寄存器级 |
| Blocked gust | TileSpGEMM(PPoPP'22)、MOSparse(2025) | 1D 行组(非 2D tile)+ 可选 hash/merge 累加器 |
| 分段 inner | cuSPARSE symbolic-numeric | numeric 用 merge-based 点积(非 hash),利用 CSR 有序性 |
| Blocked + merge | 无直接对应 | **空白**:共享 A 缓存 + merge 累加,文献里没有 |

---

## 附:本项目可复用的结构事实

1. **gust/inner 中间项按行分段**(实测验证)——融合 1/2 的前提。
2. **A 是 CSR,每行 col 有序**——融合 1(有序链)的前提。
3. **outer/colw 中间项按外层轴(k/j)分段**,不按行——融合 2 的 k 轴迭代天然适配 outer 的结构。
4. **你的 pinned+arena IO 优化**(端到端比 Ocean 快 1.76×)——任何融合方案都应套上这个 IO。
5. **sort 占 gust 计算的 49-77%**(随矩阵增大)——融合 1/2/3 都旨在消除它。

相关文档:`ref/sort_innovation_directions.md`、`ref/spgemm_papers_2025.md`、`worklog/profiling_analysis.md`、`ref/papers/ocean_estimation_spgemm.md`。
