# 12 · 近年文献调研:可借鉴的点(2026-08-26)

> 目标:追平乃至超越 Ocean。本文档记录从近年并行计算/稀疏矩阵文献中找到的可借鉴方向,
> 按"我们能用/需硬件/理论启发"分类,标注优先级和预期收益。

## 一、Ocean 自身的深读(最重要:了解对手的每一寸)

论文:[arXiv 2604.19004](https://arxiv.org/html/2604.19004v1) | 代码:已完整读过

### 核心设计(我们已知)+ 新发现

| 机制 | Ocean 做法 | 我们的现状 | 可借鉴? |
|---|---|---|---|
| 工作流选择 | avg_product < 64 → 免估计;ER/CR 采样 → HLL vs symbolic | avg_product ≤ 64 → flop ✓ | **已对齐** |
| ESC 短行 | 2/4 行并发 × 64 线程 | ultra 单线程一行 ✓(更轻) | **已超越**(插入排序免排序) |
| Hybrid Hash | keys SMEM + values 全局 | 全 SMEM | **已在 heavy 路径实现** |
| **流水线** | **单流**(论文明确无 multi-stream!) | 单流 | **超越机会!** |
| **Compaction** | **平均 8% 运行时** | 插入排序后 ~5% | **我们更优**(Step4 后) |
| **HLL 估计** | construct-and-merge 两步 | **MinHash 算术均值** | 各有优劣;我们已修无偏 |
| 溢出处理 | 单一 fallback kernel | **行级重试** | **我们更优** |

### 新识别的超越机会

1. **Multi-stream 重叠**:Ocean 论文明说不用 multi-stream——h2d || sizing || accumulate
   三段重叠是白送的加速(论文 §Multi-Stream: "does not describe");
2. **Compaction 已胜**:我们的插入排序 ultra 路径 epilogue ~5% vs Ocean 8%;
3. **行级重试**:Ocean 的 overflow fallback 是"退到 symbolic 重跑"——我们的 flop 上界重试更精准。

## 二、近期论文扫描(2024-2025)

### [Optimizing General SpGEMM on GPU(2025)](https://dl.acm.org/doi/10.1145/3774654)
- **ML 模型选 sizing 方法**(轻量学习模型预测最优估计方式)
- 可借鉴:**dispatcher 的 ML 化**(已在 roadmap,不新但确认方向正确)

### [Hash-based Multi-phase SpGEMM + AIA(arXiv 2512.12036)](https://arxiv.org/html/2512.12036v1)
- **Acceleration of Indirect Memory Access(AIA)**:硬件级技术,减少间接访存延迟
- 需要自定义内存控制器 → **论文引用即可,不可直接用**

### [Multi-GPU SpGEMM(2025 Concurrency&Comp)](https://www.research.unipd.it/retrieve/baab0f3d-ea71-4fa9-8076-49618c2629c2/Concurrency%2520and%2520Computation%2520-%25202025%2520-%2520Mavliutov%2520-%2520Multi%2520GPU%2520Sparse%2520Matrix%2520by%2520Sparse%2520Matrix%2520Multiplication.pdf)
- nsparse 改造的多 GPU 分布式 SpGEMM
- 不适用(单卡竞赛)

### [CB-SpMV: Cache-Friendly SpMV(arXiv 2605.18515)](https://arxiv.org/abs/2605.18515)
- **warp-level 归约 + 数据聚合平衡算法**
- 可借鉴:**聚合思想已用**(warp 聚合原子);**缓存友好布局**值得注意

### [SGAP: Sparse Tensor Algebra GPU 编译](https://dl.acm.org/doi/10.1145/3674179.3674203)
- **atomic parallelism 的系统化优化空间分析**
- 理论启发:原子操作的"并行度-争用"权衡有系统化框架

## 三、核心可落地方向(按优先级)

### A. Multi-Stream 流水线(最直接, Ocean 没做)
```
Stream 1: h2d 传输 A 矩阵          (~10-20ms 大阵)
Stream 2: count_flop + MinHash sizing  (~1-5ms)
Stream 3: accumulate + compact
```
h2d 与 sizing 天然无依赖(count 读 host 侧或已到 device 的数据即可启动),
accumulate 依赖前两者。**预期省 5-15%**(大阵 h2d 占比高)。
→ **我们做**:`cudaStreamCreate×2`,count 用 stream2,h2d 在 stream1。

### B. Sub-warp 档位(巨型图第二程, Ocean 的 spark 思路)
```
est ≤ 8   → 8 lanes/行(4 行/warp,寄存器表)
est ≤ 16  → 16 lanes/行(2 行/warp)
est ≤ 64  → 32 lanes/行(1 行/warp)← 现批量 kernel
```
Ocean 用 16/32 线程档 + 4 线程 sub-warp 变体——我们只有 32 档。
巨型图 avg_product≈5 → 8-lane 档浪费 4× 线程。**预期 accumulate 2× 提升**。
→ **我们做**:batched kernel 加 `BATCH_LANES` 模板参数,路由加档。

### C. CUDA Graph(小阵 launch 开销)
整 pipeline(8-12 个 kernel)固化成 Graph,一次 launch。
Ocean 也没用(论文无提及)。**小阵省 30-50µs launch 延迟**。
→ **我们做**:对小阵(A_rows < 100k)构建 Graph 缓存。

### D. L2 Persist Window(H100 硬件特性)
`cudaAccessPolicyWindow` 把 A 矩阵钉在 L2(50MB on H100)。
巨型图(germany_osm A=276MB)不能全放,但 B 的热点行(频繁被引用)可以。
→ **实验性**,预期 5-10%。

### E. Warp-consolidation(理论启发,暂缓)
把遍历同一条 B 行的多个 warp 合并读取——需要改 kernel 结构,收益不确定。

## 四、论文定位(调研后的策略更新)

| 层面 | 我们的现状 vs Ocean | 论文叙事 |
|---|---|---|
| 算法 | 双家族自适应(独有) | 主贡献 |
| Sizing | MinHash 无偏修正(独有) | 数学贡献 |
| Ultra 路径 | 插入排序免 compact(优) | 工程贡献 |
| 行级重试 | flop 上界精准重试(优) | 工程贡献 |
| **流水线** | **multi-stream + Graph(Ocean 无)** | **工程超越点** |
| 巨型图 | sub-warp 档(补齐后对齐) | 工程 |
