# 06 · 剩余差距归因 + 超越路线

> 2026-08-26 晨状态:六阵两胜四追(1.4×-3.3×)。本文档 = 下一步的全部候选,按预期收益排序。

## A. 中尺寸行批量(exdata_1 类 3.3× 的主攻,预期收益最大)

**现状**:est ∈ (64, 2048] 的行仍走 1 CTA/行的 hash_spa(表 128-2048 槽)。
exdata_1:6001 行 × 平均 143k 积/行,accumulate 36.9ms = 23 G-prod/s(bcsstk30 同 kernel
达 91 G-prod/s)。差距源:行少(6001)→ CTA 少 → SM 不满;表大 → SMEM/CTA 占用高 →
驻留 CTA 少;76× dup → 热点列原子串行。

**方案 A1(推荐):表分片并行** —— 一行一张大表改成 K 张子表(列值域分片,merge3 的
值域思想!),每子表 ≤ 128 槽独立 probe:并行度 × K,热点列分散到不同子表。
`j & (K-1)` 或列域切分选片;extract 各片独立 count-rank,片间按值域拼接天然有序。
**这也是超越点:hash 的热点原子串行是 Ocean 同款结构瓶颈,值域分片是我们 merge 线的
独有武器移植**。

**方案 A2**:多行共享 CTA(8 行 × 中表):SMEM 96KB,8 warps 各伺候一行;行少时填满 SM。
**方案 A3**:hot-column 预分桶(先一遍 histogram 找热点列单独通道)——复杂,A1 失败再上。

## B. 批量 kernel 自身 1.6×(333SP 型最后一段)

batched 6.47ms vs Ocean usparse_main 4.0。候选:
- count-rank extract O(n²):n≤64 → 每行 ~400 次 SMEM 读;换 warp bitonic(log²·n/32 并行)
  或 32-lane 插入排序网络。预计省 ~1ms。
- k 串行循环 unroll/预取 B 行头(合并 gm 读)。

## C. 重试开销回收(pwtk/bcsstk30 的 1.4-1.6×)

- 重试 0.3-0.74ms 里大半是 fixed 启动(scan×2 + cub query);把 flop-scan 与 count_flop
  融合、cub temp 预分配复用。
- est 更准 → 欠估行更少 → 重试更少(estimator 修复后 bcsstk30 仅 5 行)。

## D. 内存/带宽面

- d2h(TOTAL 内,compute-only 外):333SP 40.7ms = 21GB/s——pinned 池带宽没吃满,
  可分 stream 重叠(论文 wall-clock 口径有用)。
- est 已 1.5-2×,tmp buffer 接近下界;再降需精确 symbolic(放弃,MinHash 是论文点)。

## E. 系统级(超越的加分项)

- **CUDA Graph** 固化整 pipeline launch 序列(小阵 launch 开销,对标 spECK 的瘦管道);
- 多 stream:h2d || sizing || accumulate 重叠(Ocean 用 streams[1] 做 numeric);
- L2 persist window 给 A 矩阵(H100 cudaAccessPolicyWindow)。

## 优先级建议

```
1. A1 表分片(今天)→ exdata_1 3.3× → ~1.5×,四追变四平
2. B extract 优化 + C 重试开销(半天)
3. REFRESH=Auto 全量重跑 ocean337 → 真实 Auto/Ocean geomean
4. disptrain 采集(119/213)→ dispatcher 重拟合(hash 变了,分布也变了)
5. E 系统级(论文前最后一轮)
```
