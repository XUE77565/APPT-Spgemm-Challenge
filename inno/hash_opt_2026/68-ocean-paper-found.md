# docs/68 Ocean 本尊论文定位 + ESC 源码精读(文献补给，09-01)

**发现**:[arXiv 2604.19004](https://arxiv.org/html/2604.19004v1) "Fast Estimation-Based
Sparse General Matrix-Matrix Multiplication" **就是 Ocean 的论文**(337 方阵套件 = 我们的
ocean337;HLL construct-merge = 我们的 MinHash 对位;ESC + 混合 hash + dense = 其三路累加器;
A100 63.7 GFLOPS vs spECK 46.2/opSparse 24.2/cuSPARSE 3.39;H100 108.0;自报 1.4-2.8× over SOTA)。

**论文口径对位**(写论文/查重直接用):
- HLL 估计 ×1.5 系数 → 上取整到累加器尺寸(我们的 est_expand 1.15-1.5 同物)
- 三路累加器:hash / dense / **ESC**;"给该行选资源需求最少的配置" = 其双梯本质
- 337 方阵 @ A100/H100 = 我们的评测集同源 → 引用其数字可直接对照

**ESC 源码精读(AccumulatorESC.cuh)**:
- count-rank-compact 融合:每线程持 1 乘积,SMEM scratch 全键扫描 → 同键求和 +
  计 rank + ballot 定 leading(首现),leading 线程直写有序位;n_elements = 副产物
- **无表、无 init、无探测**;工作量 O(P²/NTHREADS) 比较
- **门 = product_num ≤ NTHREADS(≤32)**:只覆盖超小行(比我们 est-65-256/384 乘积的
  目标类更小)→ **Ocean 里 in-2004 类也是 hash 行**,其 22k 地板税差额必来自 bin 尺寸
  (prime 梯 331/673/1327…)/块配置/全链,非单 kernel 魔法(与 docs/67 §10 结论一致)
- ESC 模式对我们 ultra 类(est≤16)可借鉴(现存 ultra 线性 kernel 同位),增量有限

**另**:ACM 3774654(北京理工 2024-25)= ML 选策略 + hash load factor/multiplier 启发式
调参 —— hash 乘数(我们固定 2654435761)与松弛系数是未调维度,记为低优先实验项。

**对 v2 的修正**:BATCH2 v2(块内多行子分块)方向不变;ESC 不适配我们的目标类(O(P²)
在 384 乘积 = 150k 比较/行)。
