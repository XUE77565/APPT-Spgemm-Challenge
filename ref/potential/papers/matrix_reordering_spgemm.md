# Improving SpGEMM Through Matrix Reordering and Cluster-wise Computation
- SC 2025 (arXiv:2507.21253); Islam, Xu, Dai, Buluç
- 全文: https://arxiv.org/abs/2507.21253

## 核心

对 SpGEMM 的输入矩阵 A 做**行重排序 + 层次聚类**,使相似行(共享 B 行的行)聚在一起 → 提升 B 矩阵的缓存复用。

## 要点

- **层次聚类(hierarchical clustering)**: 对 A 的行按 Jaccard 相似度聚类 → 相似行放一起 → 处理时 B 的行命中率更高。
- **行聚类格式(row-clustered format)**: 新的 CSR 变体,记录每个行簇的 B 行集合。
- **cluster-wise computation**: 同一簇内的行批量处理,B 行只读一次(类似 blocked Gustavson)。
- 性能: 平均加速 **1.39×**(110 个矩阵);预处理开销 < 20× 单次 SpGEMM。
- **10 种重排序算法对比**: graph partitioning 给出最好的 SpGEMM 加速,但预处理时间高。

## 对你的启示

- **blocked Gustavson 的数据支持**: 这篇证明了"把共享 A[k,:] 的行聚到一起处理"有效。你的 `ref/formulation_fusion_ideas.md` 融合方案 2(blocked Gustavson)就是类似思路——这篇提供了理论 + 实测支撑。
- **预处理成本 vs 收益**: 重排序需要预处理(聚类/图分区),但只需做一次 → 如果矩阵重复使用(如迭代求解),收益摊薄。
- **与你的结合**: 如果你的矩阵 A 固定(如 A·A·A 链式计算),预先对 A 做行重排序 → 你的 Gustavson expand 天然受益(相似行的 A[k,:] 更可能命中 L2 cache)。
