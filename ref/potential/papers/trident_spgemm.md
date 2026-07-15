# Trident: Communication-Avoiding SpGEMM via Trident Partitioning
- ICS 2026 (arXiv:2603.21444); Bellavita, Pichetti, Pasquali, Vella, Guidi
- 全文: https://arxiv.org/abs/2603.21444

## 核心

层次感知的 2D 分布式 SpGEMM 算法,利用**节点内 GPU 互联带宽 >> 节点间**的特点。

## 要点

- **Trident 分区**: 将矩阵分为三层(intra-GPU / intra-node / inter-node),按带宽层次分配数据 → 减少节点间通信量 2×。
- **异步通信**: 用 RDMA overlap 通信与计算。
- 性能: 比 2D SpGEMM 快 2.38×(最高),几何均值 1.54×。Markov Clustering 加速 2×。

## 对你的启示

- 你的 pinned/arena 优化是**单 GPU 内**的 communication-avoiding(H2D/D2H 传输优化)。
- Trident 是**多 GPU/多节点**层面的 CA。两者层次不同但思想一致:减少不必要的数据搬运。
- 如果未来扩展到多 GPU,Trident 的层次分区思想可借鉴。
