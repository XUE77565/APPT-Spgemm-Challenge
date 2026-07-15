# KAMI: Communication-Avoiding GEMM within a Single GPU
- SC 2025; Wang, Du, Li, Tian, Sun, Liu(SSSLab, China University of Petroleum)
- 全文: https://www.ssslab.cn/assets/papers/2025-wang-KAMI.pdf

## 核心

首次将分布式 **Communication-Avoiding(CA)理论**迁移到**单 GPU 内部**:把 tensor core 当计算单元、寄存器当本地存储、shared memory 当通信介质,设计 1D/2D/3D CA 矩阵乘法算法。支持 SpMM/SpGEMM(基于 Z-Morton 稀疏块存储)。

## 关键映射(分布式 → 单 GPU)

| 概念 | 经典 CA(分布式) | KAMI(单 GPU) |
|---|---|---|
| 计算单元 | 进程(CPU/GPU) | warp(tensor core) |
| 本地存储 | DRAM | **线程寄存器** |
| 通信介质 | 网络(Send/Recv) | **shared memory**(LD/ST) |
| 性能度量 | 执行时间 | **GPU 时钟周期** |

## 要点

- **寄存器 > shared memory**: 寄存器延迟 1 cycle,shared memory 20+ cycle(20×)。KAMI 尽量把数据放寄存器,shared memory 只做 warp 间通信。
- **1D/2D/3D 三种算法**: 矩阵分块后,各 warp 持有子矩阵;通过 shared memory 交换子矩阵(= 通信);tensor core 做乘法(= 计算)。阶段交替通信/计算。
- **时钟周期理论分析**: 用 cycle 而非秒建模 → 更精确的性能预测。
- **稀疏扩展(SpMM/SpGEMM)**: 用 Z-Morton 块序存储稀疏矩阵(16×16 块),对齐 tensor core 形状。SpGEMM 需要 symbolic phase(经典 SPA)。
- **性能**: GEMM 在 GH200 上比 cuBLASDx 快 5.20×(FP64)。SpGEMM 也有评测但性能较低(索引开销 + 不规则访存)。

## 对你的启示

- **寄存器利用**: 你的 Gustavson expand kernel 几乎不用 shared memory(只一个 `__shared__ int pos`)。KAMI 的思路是:把更多数据放寄存器/shared memory,减少 global memory 访问。这对你的 expand + hash 累加器路线有参考价值。
- **CA 思想适配 SpGEMM**: KAMI 证明了 CA 在单 GPU 内也有效。你的 SpGEMM 可以借鉴"寄存器做累加器、shared memory 做 warp 间交换"的思路——这正是 spECK / Register-Aware 的做法。
- **Z-Morton 块序**: 如果你的矩阵有块结构(block-sparse),Z-Morton 序可提升缓存命中。对你的矩阵集(first100)可能不太适用(多为非结构化稀疏)。
- **SpGEMM 部分**: KAMI 的 SpGEMM 性能不如其 GEMM(因为索引/不规则)。说明"CA GEMM 方法直接扩展到 SpGEMM 收益有限"——SpGEMM 的瓶颈在算法层(hash/sort/merge),不是 GEMM 层的通信模式。
