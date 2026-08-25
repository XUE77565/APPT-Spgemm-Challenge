# 07 · 自适应 EXPAND 与 retry 快路径:一次数据驱动的否决(2026-08-26)

> Step 1 尝试记录。结论:**两个方向都被实测否决**,保留 EXPAND=1.15 均一值 + thrust 重试路径。
> 这是有价值的负结果:它把 bcsstk30 的 1.76× 精确归因,并排除了两条歧路。

## 问题:bcsstk30 为什么从 7 月的 ~1.2× 掉到 1.76×?

相位对比(compute-only 3.99ms vs Ocean 2.27):
- accumulate 1.95ms —— **与 Ocean(1.76)持平,算法没输**;
- **retry 0.74ms(30%)** —— EXPAND 1.15 下 143 行欠估触发,纯 launch/同步延迟;
- compact 0.68ms + sizing 链 ~0.6ms —— 结构性略贵。

## 尝试 1:小矩阵自适应 EXPAND(1.4,免重试)

| EXPAND | retry | compact | TOTAL |
|---|---|---|---|
| 1.15 | 0.75 | 0.68 | **9.03** |
| 1.20 | 0.64 | 0.87 | 9.46 |
| 1.25 | 0.63 | 1.06 | 9.50 |
| 1.30 | 0.63 | 1.06 | 10.44 |
| 1.40 | 0(无) | 1.45 | 9.64 |

**否决**:retry 省的钱被 est 变松 → 行上移进 csort 档(compact 涨)完全吃掉——零和,
且 1.15 全线最优。教训:**紧 est 永远是对的,代价要从 retry 的实现里省,不是从 est 的松紧里买**。

## 尝试 2:retry 快路径(单 block scan + packed 偏移,省 thrust 多 launch)

**否决 + 一次险情**:
1. 首版 `retry_pack_off_kernel` 误留 `extern __shared__` 声明而 launch 未传尺寸 → 0 字节
   共享写 = 未定义行为 → GPU 0 空转挂死(30% util 僵尸)。**kill -9 后 GPU 自愈,冒烟通过,
   未 wedge(幸运)**。教训:`extern __shared__` 声明必须与 launch 的 smem 参数成对出现。
2. 修掉后 3Dspectralwave2 仍挂(h_ovf=513 vs bcsstk30 的 143 通过——未定位根因),
   关闭快路径即恢复。收益本来就 ~0.2ms,整体移除。

## 结论与去向

- EXPAND 保持 1.15(HASH_EXPAND env 可调留作实验);
- bcsstk30 类要追平,正解在:**retry 无宿主化**(Ocean 在 kernel 内 switch_to_gmem,
  零宿主往返)或 sizing/compact 链再瘦身高 —— 排在 heavy-hybrid 之后的优先级;
- 本文档的价值:排除法收敛了设计空间。
