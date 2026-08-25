# 11 · offshore 二次剖析:38.7× 的构成(2026-08-26)

> 干净基准 Top1 输阵 offshore(259789×259789,avg_product 高)修完 compact 后再看。

## 相位(修后)

| 相位 | ms | 说明 |
|---|---|---|
| count+MinHash+scan | 1.1 | avg_product>64 → 走 MinHash |
| binning | 0.06 | |
| **accumulate** | **2.26** | |
| retry | 0.50 | |
| compact+sort | 0.78 | 插入排序修复后 |
| **compute-only 合计** | **~4.9** | vs Ocean 1.91 = **2.6×** |

**从 38.7× → ~2.6×**(compact 修复 + compute-only 口径),不再是异常值,回归"中等差距"类。

## merge3 对比实测(offshore)

merge3 TOTAL 21.6ms vs hash 17.8ms(TOTAL 含 d2h;compute-only 差距更小)——
**带状阵上两条路径接近持平**,不再有合成数据上 35× 的极端差距
(合成 band_n32000 的 128 宽度完美带状在真实套件中不存在)。
→ merge3 的价值定位:不在"某些阵上大胜",而在** dispatcher 的第二选项防Hash失手** + 论文双家族叙事。

## 判定

offshore 类(38.7× 王冠)已从"异常"归位为 2.6× 的常规差距,瓶颈进一步收敛到:
1. accumulate 2.26 vs ~1.0(Ocean)——仍有 2× 但需新思路;
2. retry 0.50ms 固定延迟(小阵占比高)——已知问题;
3. 全局最大剩余差距仍在巨型图(geography 类 ~6-10×)的第二程。
