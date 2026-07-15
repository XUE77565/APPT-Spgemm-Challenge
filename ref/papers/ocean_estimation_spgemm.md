# Ocean: Fast Estimation-Based SpGEMM on GPU
- ICS 2026 (arXiv:2604.19004); Sun et al., Cornell HPC
- 全文:https://arxiv.org/html/2604.19004v1
- 代码:https://github.com/CornellHPC/Ocean-SpGEMM

## 核心(四项贡献)

1. **HyperLogLog 替代 symbolic pass** —— 用概率基数估计预测每行输出 nnz,替代精确的符号计算。symbolic 从 28% runtime 降到 4%。**首次将 HLL 用于稀疏线性代数**。
2. **代价预测 + 工作流选择** —— 用 Input Expansion Ratio (ER) 和 Output Compression Ratio (CR) 动态选择:estimation-based / symbolic-based / upper-bound。
3. **混合累加器(hash + dense + ESC)** —— hash 用于中等行、dense 用于短/密集行、**ESC 用于极短行**(中间项 < 64)。
4. Ocean 整体比 spECK 快 1.4×、HSMU 2×、opSparse 2.6×、TileSpGEMM 3.5×。

## ★ 直接适用于你的 ESC sort 瓶颈的技术

### ★★★ 间接排序(Indirect Sorting,§4.2)—— 最直接可用

Ocean 的 hash 累加器输出行后需排序(为了 CSR 格式)。为加速:

- **不直接排 key-value 对**,而是排 **key-pointer 对**。
- ptr 指向对应的 value(在 on-chip 内存中,地址空间小 → ptr ≤ 14 bit)。
- **key + ptr 打包成一个 32 位整数**(key 占高位、ptr 占低位)。
- radix 排序时,**忽略 ptr 位(begin_bit/end_bit 只排 key 位)**。
- 排序后按 ptr 重排 value → 写到目标位置。

**效果**:排序数据量从 64-bit key+32-bit value(**12 字节/元素**)降到 **4 字节/元素**(32-bit key+ptr)。内存流量、寄存器压力、排序 pass 数全面降低。

**对你**:你现在排 64-bit key (row<<32|col) + 32-bit float = 12 字节。改用 indirect sorting:32-bit key(col 部分)+ 14-bit ptr(指向 value)打包在 32-bit 里 → 4 字节/元素。**排序数据量减 3×**。

### ★★ ESC 累加器用于短行(§3.3)—— 你的 ESC 有位置

Ocean 明确指出:**ESC 累加器适合极短行**(中间项 < 64),因为:
- ESC 不需要知道输出大小(不像 hash 需要分配表)。
- ESC 的配置只取决于中间项数。
- 多个短行可并行处理(一个 block 处理多行)。

**对你**:你的 ESC 全行用(不管长短)。Ocean 的洞察:ESC 对短行最优,长行该用 hash。你可以做**混合**:短行保留 ESC、长行切 hash → 各取所长。

### ★★ 增强 hash 累加器(§3.3)—— shared+global 协作

Ocean 发现:**hash 表的 value 可以放 global memory**,性能影响很小:
- index 操作(read/compare/swap)复杂,必须留 shared mem。
- value 只做 atomicAdd,**fire-and-forget** 模式(SASS 级单指令),global 延迟可接受。
- FP64 shared-mem atomic 不是原生支持的(编译成 CAS 循环),反而比 global 慢。
→ hash 表可以处理 **3× 长的行**,不需要全部挤在 shared mem。

### ★ HyperLogLog 替代 symbolic pass(§3.1)

- 对 B 的每行建 HLL sketch(constant memory, ~32-64 byte/row)。
- 按 A 的每行合并对应的 sketch(element-wise max)→ 估计 C 每行 nnz。
- 误差可控(32 registers → 平均 13% 相对误差,overflow < 1.2%)。
- symbolic 从 28% runtime → 4%。

**对你**:你的 count_intermediates kernel 是精确 symbolic(算上界)。HLL 可以用更轻量的估计替代。不过你的 symbolic 不算特别贵(~0.05ms),收益可能不如 indirect sorting 大。

## Ocean 的间接排序 vs 你现在的 sort

| | 你现在 | Ocean indirect sort |
|---|---|---|
| 排序对象 | 64-bit key + 32-bit val = 12B/elem | 32-bit (key+ptr packed) = 4B/elem |
| 排序 pass | ~4 (64-bit, CUB 跳空位) | ~4 (32-bit, 一样) 但内存流量 **3×** 少 |
| 排序后 | 直接得到 sorted (key,val) | sorted (key,ptr), 需额外 gather val |
| 适用 | 全局 ESC | hash 累加器 per-row output |

**关键**:indirect sort 减少的是**每元素的字节数**(12B → 4B),radix pass 数可能不变(都 ~4),但**每 pass 的内存读写量减 3×**。radix 是内存带宽 bound 的 → 流量减 3× → sort 快 ~3×。

## 数据(Ocean vs 竞品,A100 square 337 matrices)

| 方法 | #best | avg GFLOPS | vs spECK |
|---|--:|--:|--:|
| **Ocean** | **294/337** | **63.7** | **1.4×** |
| spECK | 22 | 46.2 | 1.0× |
| HSMU | 0 | 32.1 | 0.69× |
| opSparse | 16 | 24.2 | 0.52× |

## 代码

开源,MIT-style:https://github.com/CornellHPC/Ocean-SpGEMM
CUDA C++,支持 A100/H100。直接可比。
