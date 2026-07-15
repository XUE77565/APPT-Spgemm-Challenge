# Accelerating SpGEMM on GPUs with Processing Near HBMs
- arXiv:2512.12036, Dec 2025; Li, Min, Yie, Kim, Ahn, Sim, Lee, Kim
- 全文:https://arxiv.org/abs/2512.12036

## 核心

Hash-based **Multi-phase SpGEMM** on GPU + **AIA**(加速间接访存,near-HBM processing)。

## 要点(从摘要)

- **hash 多阶段**:用 hash 累加器替代全局排序,分多阶段执行(symbolic-numeric 思路)。
- **AIA** = Acceleration of Indirect Memory Access:near-memory processing 硬件技术,加速 SpGEMM 中的不规则间接访存(hash 表查找等)。
- 软硬协同(hardware-software co-designed)框架。
- 性能:vs cuSPARSE,Graph Contraction 时间减 76.5%、Markov Clustering 减 58.4%;GNN workload 1.95× speedup,大矩阵上最高 4.18×。

## 对你的启示

- **hash 替代 sort**:这篇直接验证"hash 多阶段替代 ESC 全局排序"是当前(2025)的主流加速路径。
- **多阶段 = symbolic-numeric**:分阶段先算结构(symbolic,可用 hash/estimation 轻量化)、再算数值(numeric),避免一次全展开 + 全局排序。
- **AIA 是硬件角度**:你做不了(需要 near-HBM 定制硬件),但它的 hash 多阶段算法部分可以借鉴。
- **对应你的创新方向 B/E**(hash 累加器 + symbolic-numeric 两阶段)。

## 局限

- 摘要级信息(13 pages, 11 figures)。需要看全文了解 hash 多阶段的具体 kernel 设计、阶段划分、fallback 机制。
- AIA 依赖定制硬件(HBM processing),你无法复现;但软件部分的 hash 多阶段可借鉴。
