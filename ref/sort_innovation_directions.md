# SpGEMM ESC 排序瓶颈:创新方向 brainstorm

> 日期:2026-07-14
> 背景:本项目 ESC(Expand–Sort–Compress)合并里,`thrust::sort_by_key`(对 64 位 key=row<<32|col 的全局基数排序)占合并阶段 ~72%(均值 ~0.76ms,大矩阵 bcsstk30 ~21ms)。
> 实测:CUB radix 复用缓冲、分段排序都基本无效——因为 CUB radix 已做空位跳过等优化,sort 本身到顶。
> 文献基础见 `worklog/spgemm_acceleration_paper_survey.md`。

---

## 核心判断

**sort 本身已被 CUB 优化到顶,创新空间在"绕过 / 替代排序"。** 文献里 fast SpGEMM(spECK / Balanced Hashing / cuSPARSE)普遍**不用全局排序**,而是 hash 累加器或 merge。所以真正的创新不在"让 sort 更快",而在**利用数据结构把 sort 省掉或替代**。

这套数据有两个被全局 sort 浪费掉的**结构**:
1. **gust / inner 中间项已按行分段**(实测验证);
2. **中间项的 col 来自 A 的各行,A 是 CSR → 每行的 col 本身有序**。

---

## 方向 A:用 k-way merge 替代排序 —— 利用输入的列有序性 ★最贴结构

**文献基础**:SpArch(ISCA'19)层次化 matrix merging;Gustavson 原始算法本质是行归并。但 GPU 上多数人用 hash,**merge-based 在 GPU 上是冷门**。

**点子**:对行 i,它的中间项 col 是「A 的若干行(k∈row_i)的列」的并集——**每个 k 的列有序**。现在 expand 把它们打散写进 COO 再全局排;其实只要对每个 i 做一次 **k 路归并**(归并 row_i 涉及的那些有序列链),就能**直接得到行内 col 有序**,完全不用 sort。归并 O(中间项数),sort O(N log N)。

**创新点**:GPU 上的 k-way merge kernel 不常见(多数用 hash),而且它**严格保持 col 有序**(hash 出来无序,还得再排)。可在 gust/inner(行分段)直接用,outer/colw 要先按行重排。难点:GPU 上的 k-way merge(优先队列/堆是串行的,需 block 级协作归并)——这本身就是一个可发的工程贡献。

---

## 方向 B:per-row hash 累加器 + 基于代价模型的自适应派发 ★spECK 的理论升级

**文献基础**:spECK(可适配 hash)、Register-Aware(sort/merge/hash 三种 accumulator 对比,NPC'18)、Spada(ASPLOS'23,按稀疏模式自适应)。

**点子**:不用 sort,每行一个 hash accumulator(col→val)。**创新在"自适应"**:spECK 的适配是启发式的;这里做**per-row 代价模型**,根据该行的中间项数、col 取值范围、重复率,在 {register-hash / shared-mem-hash / 归并 / 小批量 sort} 里**理论最优地选一个**,逐行派发。短行用 register hash(几十个 col),长行用 merge 或 sort。

**创新点**:把 spECK 的"启发式适配"升级成"有代价模型保证的逐行派发",并给出**切换阈值的理论分析**(什么行长度/重复率下 hash 比 sort 赢)。first100 行长分布差异大,很适合自适应。

---

## 方向 C:expand 时局部去重,缩小 sort 的输入

**文献基础**:NSparse(merge-preprocessing)、Segmented Merge 原语。

**点子**:现在 expand 把**所有**中间项(含重复)写进 COO 再 sort 再 reduce。若 expand 时**每个 block 先在自己行内做一次局部 hash 去重**(把同 (i,j) 的贡献先加掉),再写出,则**全局 sort 的输入 N 变小**(只写 distinct (i,j))。sort 成本 ∝ N,N 缩小直接提速。

**创新点**:把 reduce 的一部分工作前移到 expand(block-local 去重),减少全局 sort 数据量。对**重复率高**的矩阵(同一 (i,j) 由多个 k 贡献)收益特别大。难点:block-local hash 在 shared mem 的碰撞处理 + 与全局 reduce 的衔接。

---

## 方向 D:单 kernel 融合 sort+reduce+final —— 砍内存往返 ★工程性强、确定性收益

**文献基础**:Segmented Merge(GPU 原语)、MatRaptor 的 ESC 数据流。

**点子**:现在合并是 **3 趟**(sort 读写一遍 + reduce_by_key 读写一遍 + finalize+scan 又一遍),每趟还有 cudaMalloc/同步。对行分段的 gust/inner,写**一个 block-per-row 的融合 kernel**:每个 block 把自己行的中间项 load 进 shared mem,在 shared mem 里**排+去重+写 C 的该行**,一趟搞定。省掉 3 趟全局内存往返 + 多次 cudaMalloc + 多次同步。

**创新点**:不是算法层面(sort 还在做),而是**内存带宽 / launch 开销**层面的融合——对中小矩阵(first100 为主)这部分固定开销占比大。贡献在"把 ESC 的三阶段在 GPU 上融成一个 row-parallel kernel",并可分析内存带宽节省。

---

## 方向 E:symbolic–numeric 两阶段(彻底不全局排序)★cuSPARSE 路线

**文献基础**:cuSPARSE 自身、经典 Gustavson 两阶段。

**点子**:完全换路线:阶段一(symbolic)用 per-row accumulator 算出 C 的结构(row_ptr + 每行哪些 col),**不展开所有中间项**;阶段二(numeric)按结构填值。这样**根本不存在"展开 N 个中间项再全局排"这一步**,sort 自然消失。

**创新点(在本项目语境下)**:本项目是 ESC(sort-based)的对照实现,把"ESC vs symbolic-numeric(hash-based)"在**同一套数据/同一套 GPU 上**做严谨对比、并给出"什么矩阵特征下哪种赢"的判据,本身就是一个有价值的实证贡献(补 Spada 的思路)。创新不在算法(两阶段经典),在**针对这批矩阵的判据 + 混合策略**。

---

## 评估:创新性 × 可落地 × 文献空白度

| 方向 | 创新性 | 可落地 | 文献空白度 | 备注 |
|---|:--:|:--:|:--:|---|
| **A: k-way merge 替代排序** | 高 | 中 | 高 | GPU 上冷门;利用"行分段+输入列有序"独有结构 |
| **B: 代价模型驱动的逐行自适应派发** | 高 | 中 | 中 | spECK 基础上的理论升级,有公式 |
| D: 融合单 kernel | 中 | 高 | 低 | 工程融合,中小矩阵收益确定 |
| C: expand 局部去重缩 sort | 中 | 中 | 中 | 对高重复率矩阵收益大 |
| E: symbolic-numeric 对比/混合 | 中 | 高 | 低 | 算法经典,判据/混合是贡献 |

---

## 推荐

- **冲论文 / 创新性优先**:A 或 B。A 利用了独有的"行分段 + 输入列有序"结构,GPU merge 又是冷门,故事好讲;B 给 spECK 加代价模型,有理论。
- **稳妥提速**:D(融合单 kernel)最确定能落地、对中小矩阵收益明确。
- **混合可行**:C(局部去重缩 sort)可与 A/D 叠加。

---

## 附:本项目可复用的结构事实(支撑 A/B/C/D)

1. **gust / inner 中间项按行分段**(实测,gust count 精确 + 一块一行 expand 保证);outer/colw 按外层轴 k/j 分段(非行)。
2. **A 是 CSR,每行 col 有序** → 行 i 的中间项 col 是若干有序列链的并集 → 可 k-way merge(方向 A 的前提)。
3. **CUB radix sort 已优化到位**(空位跳过)→ "换 sort 实现"类优化(复用缓冲、分段)基本无效,已实测。
4. first100 行长分布差异大 → 自适应派发(方向 B)有发挥空间。

相关文档:`worklog/spgemm_acceleration_paper_survey.md`(文献)、`worklog/profiling_analysis.md`(瓶颈)。
