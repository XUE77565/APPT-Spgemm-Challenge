# 16 · 为什么 Multi-Stream 收益只有 0-2%?(2026-08-26)

> 回答"按道理不应该很大吗?为什么 Ocean 用 20 个?"——用 Amdahl 定律 + GPU 硬件解释。

## 简短回答

**Multi-stream 不能创造 SM——它只能让 kernel 竞争现有的 114 个 SM。**
当一个大 bin 的 kernel 已经填满所有 SM 时,其他 bin 的 kernel 并行启动只是"排队换了个窗口",
总吞吐不变。

## 详细解释(给小白的比喻)

把 GPU 想象成一个有 **114 个窗口**的银行:
- 每个 kernel = 一批客户排一个队
- **串行**(单流):第一个队全办完,第二个队才开始 → 总时间 = Σ(各队时间)
- **并行**(多流):所有队同时叫号 → 总时间 = max(最长的队) + 调度开销

听起来并行应该快很多?但有一个关键约束:
**窗口数是固定的(114 个)。** 如果第一个队有 12,000 个客户,第二个队有 5,000 个,
并行时 114 个窗口被两个队轮流占用——总服务速率不变!第一个队的 12,000 人
不会因为第二个队在旁边排队就变快。

**只有当每个队都很短(加起来也填不满 114 个窗口)时,并行才真正有效。**

## bcsstk30 的实际数字

| 相位 | 时间 | 能否并行 |
|---|---|---|
| sizing(MinHash) | 0.4ms | ❌ 依赖链:count → est → bin |
| binning | 0.05ms | ❌ 依赖 est |
| **accumulate(各 bin)** | **1.93ms** | ✅ 可并行 |
| retry | 0.74ms | ❌ 必须等所有 bin 完成 |
| compact | 0.68ms | ❌ 依赖 accumulate |
| **总计(compute-only)** | **3.8ms** | |

Amdahl 定律:可并行部分占 1.93/3.8 = 51%。
- **完美并行**(accumulate = max(bin) = 1.0ms):新总 = 3.8 − 0.93 = 2.87ms → **1.32× 加速**
- **实际并行**(考虑 SM 竞争 + SMEM 占用限制):只拿到 **1.02×**

## 为什么实际 << 理论

1. **SMEM 占用不均**:bin8(192KB/CTA)独占 SM,bin4(24KB/CTA)8 个/SM。
   并行时 bin8 的 CTA "霸占" SM,bin4 的并行度反而下降;
2. **CTA 调度开销**:跨 stream 的 CTA 调度比同 stream 内更贵(需要硬件仲裁);
3. **最大的 bin 主导**:bcsstk30 的 bin4 独自就占 accumulate 时间的 50%+,
   并行化其他 bin 只能省 50% × (1 − max/sum) 的部分。

## 那 Ocean 的 20 stream 是摆设吗?

不完全是,但作用有限:
1. Ocean 的 bin 分布经过专门的 analysis 阶段做负载均衡,各 bin 更均匀 → 理论收益更大;
2. Ocean 用 stream 做 **symbolic 和 numeric 的部分重叠**(symbolic 的尾部 bin 与 numeric
   的头部 bin 交叠)——这是阶段间流水,不是阶段内并行;
3. **Ocean 论文没有报 stream 消融数据**——20 stream 更可能是"做了不亏"的工程完备性,
   而非核心性能来源。

## 教训(补入小白讲解)

**"并行"在 GPU 上不等于"更快"——GPU 本身已经是大规模并行的。**
Multi-stream 的真正用途是:
- 当各 kernel 都很小时(各自填不满 GPU),并行可以合并利用空闲 SM;
- 当不同 kernel 用不同资源时(如一个 SMEM 密集 + 一个寄存器密集),可以互补;
- 当有流水线依赖时(如 h2d || compute),stream 可以重叠不相关的操作。

对于 SpGEMM 的 per-bin accumulate,大部分矩阵是"一个大 bin + 几个小 bin"的分布,
大 bin 已经填满 GPU → multi-stream 的收益天然有限。
