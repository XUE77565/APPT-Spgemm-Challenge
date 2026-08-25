# 13 · Sub-warp 8 线程档实验:否决(2026-08-26 方向 B)

> 巨型图第二程的尝试。结论:**warp-per-row 的 idle 线程不是巨型图瓶颈;真瓶颈是 atomic 吞吐**。

## 假说

Ocean 对超稀疏行用 16/32 线程档(甚至 4 线程 sub-warp),我们 warp-per-row 固定 32 线程。
germany_osm avg_product=4.9 → 32 线程中 ~27 idle → 减到 8 线程/行(4 行/warp)应提速 4×?

## 实现

`hash_subwarp8_kernel`:8 线程/行 × 4 行/warp,SMEM 小表 32 slot,插入排序提取。
路由:est ≤ 8 → bin12(subwarp8)。

## 实测(否决)

| 矩阵 | warp-per-row(32 线程) | subwarp8(8 线程) |
|---|---|---|
| germany_osm | 49ms | **55ms(更差!)** |
| 333SP | 58ms | 61ms |
| bcsstk30 | 9.0 | 9.0(不变) |

## 为什么更差

1. **idle 线程免费**:SM 到来的 warp 无论 3 线程还是 32 线程活跃,执行成本差异微小
   (SIMT 模型下 inactive lane 只是 predicated off,不占额外 slot);
2. **8 线程的 j-loop 串行化**:B 行长度 2-3 时,8 线程做 j 循环 = 1 步,但 k 循环
   仍串行且线程更少,导致每行总迭代数相同但并行度更低;
3. **4 行/warp 的 __syncwarp 子组同步**(0xFF << row_slot*8)开销 > warp 整体同步。

## 正确的下一步

巨型图 accumulate 4.5ms vs Ocean ~2.5ms 的差距(1.8×)不在线程数,在:
- **atomicAdd 吞吐**:56M products ÷ 4.5ms = 12.4 G/s vs Ocean 22.4 G/s;
- 可能的改进方向:去掉 CAS(直接寻址表,类似 dense 路径但按行局部性)、
  或 j-loop 展开(每 k 预取 B 行头到寄存器减少重复访存)。

→ 转向方向 A(multi-stream,Ocean 未做的超越点)。
