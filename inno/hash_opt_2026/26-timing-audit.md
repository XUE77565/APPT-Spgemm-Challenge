# 26 · 计时口径全面审计(2026-08-27 夜,用户质询触发)

> ## ⚠ 补遗(2026-08-27 夜二班,全量刷新完成后)——本表数字以此为准
>
> **终版:geomean Auto/Ocean = 1.9919×(含 symbolic 权威口径,336 阵;wb-edu 剔除,见下)。
> 不含 symbolic 口径 = 2.1585×。审计预测 "~2.0×" 命中。** 基线 `compare/ocean337/methods_cmp_v9_oceansym.csv`
> (run_ocean 已含 rc 守卫)。本文档其余部分为审计过程记录。
>
> **刷新过程揭出 stale-stats 污染事故**(修正了历次数字的三重来源):
> 1. **批首 43 阵冻结值 4.027**:20:40-20:47 窗口 spgemm 静默崩(根因未定,疑似外部 GPU 占用),
>    旧 run_ocean 吞 rc → 解析上一阵残留 stats.json。333SP 真值 6.15/Ga3 37.05/Cube_Coup 87.8 实锤。
>    另 af_3_k101/atmosmodj/conf5_4-8x8-15 同病(与前阵完全同值签名扫出)。
> 2. **wb-edu "1.47ms/~1000×" 是假的**:Ocean 的 spgemm 在 wb-edu 上**跨批确定性崩溃**
>    (`CUDA kernel error in Wrappers.cuh:902 illegal memory access` = 其 epilogue 排序内核越界),
>    所有历史 CSV 的 wb-edu 值都是 echo 前序 water_tank(1.469)。真值 = **DNF**;我方 910ms 正常
>    完成且正确。历史 geomean 被 wb-edu 一项虚罚 ~2%(1000×^(1/337))。**任务④关闭:异常不存在,
>    是 Ocean 的健壮性缺陷。**
> 3. **修复**:run_ocean 跑前删 stats.json + 查 convert/spgemm rc + 缺文件 → DNF;
>    rerun_ocean_bogus.py(43)+ rerun_ocean_extra4.py(4)定向重跑。今后该类失败显示 DNF 而非假数据。
>
> **新口径结构变化(vs docs/20 时代 2.279×)**:
> - **≥5× 只剩 5 阵**(mult_dcop×3 7.7-7.9× / vsp_south31_slptsk 5.6× / pre2 5.3×)—— top losers
>   大换血(Ge99 4.4× 已跌出 top20),头部空间仅 1.019×。**前 24 阵头部战线基本打赢,进入中段**。
> - 2-5× = 145 阵(大头),<2× = 186 阵;全 1× 的 geomean = 1.956×。
> - 按 n:最差 = 10k-200k(2.12×)中尺寸;按 C 行长:2k+(2.6-3.1×)重行最差(docs/24
>   "hash 速度∝表大小"结论在新口径下不变)→ LLB/DIRECT5/PhaseB v2 正对此。
> - symbolic 效应:226 阵变慢(中位 +14%),小阵主导(FEM_3D_thermal2 +195%)。
>
> docs 20/23/24/25 的 vs Ocean 数字均为不含 symbolic 的历史口径(其结论的结构性判断不受影响,
> 涉及 Ocean 绝对值的对比按本补遗 1.99×/2.16× 换算)。

## 结论先行

**口径形态对齐,但 Ocean 列漏计 symbolic —— 当前 CSV 对我们不利(保守)5-20%/阵。**
真实 geomean 估计 2.195× → **~2.0×**。修正脚本已就位(run_ocean 补 symbolic),
Ocean 列全量刷新待跑(~2h GPU,下一步动作)。

## 逐项审计

| # | 项 | 我方(Auto) | Ocean(CSV) | 判定 |
|---|---|---|---|---|
| 1 | 口径形态 | 逐相位 cudaEvent elapsed **求和**(hash_prof.h `total += ms`,非墙钟) | 逐阶段 timer 求和 | ✅ 对齐 |
| 2 | h2d/d2h | 排除(脚本 TOTAL−h2d−d2h) | 排除(stats.json 另算) | ✅ 对齐 |
| 3 | 相位间隙 | 排除(prof 块外的 alloc/attr/小 D2H) | 排除 | ✅ 对齐 |
| 4 | 估计器 vs symbolic | 计入(count_flop+MinHash+est_scan+binning) | **CSV 漏计 symbolic**(type-0 阵真实跑) | ❌ **对我们不利 5-20%** |
| 5 | 取轮 | 末轮(verify+5 warmup 后,全热) | warmup=1+bench=1(第 2 轮) | ⚠️ 利我们 1-3%(微小) |
| 6 | 分配 | dev_alloc 在 prof 外 | estimation.malloc 计入 | ⚠️ 利我们 μs 级(可忽略) |
| 7 | retry/overflow | 计入(retry 相位) | 计入(numeric.overflow) | ✅ 对齐 |
| 8 | compact vs epilogue | 计入(cnnz_scan+compact+sort) | 计入(sort+copy+scan) | ✅ 对齐 |

## #4 的证据(为什么必须补 symbolic)

c-58 直跑 Ocean:`Iteration 1 done. Time: 6.139 ms`;分项和不含 symbolic = 5.673(CSV 值),
含 symbolic = 6.225 —— **论文 iteration 时间(6.14)明确含 symbolic**,CSV 用的 5.67 是
弱化版基准。实测样本漏计量:c-58 0.55ms(9.7%)/Ge99 3.37ms(20%)/pre2 2.13/mult_dcop 2.49/
bloweya 0.55/soc 0.65/a5esindl 1.06/Enron 0.27/web-Google 0.99。

**修正**(scripts/compare_methods.py 已改,待全量刷):
```python
sym = sum(t.get("symbolic", {}).values())   # type-0 阵的 symbolic pass 计入
return an + est + sym + num + epi + prologue
```

## 赢阵(当前口径下,修正后只会更多)

mouse_gene **0.684×** / 3Dspectralwave 0.836× / 3Dspectralwave2 0.882×(dense 窗口)/
pkustk02 0.919 / raefsky3 0.929 / TSC_OPF_1047 0.955(小阵固定开销)。

## 待办(下一窗口第一动作)

`--refresh-col Ocean` 全量(修正版 run_ocean,tmux + Monitor,~2h)→ 更新全部消融结论的
Ocean 侧数字(docs 20/23/24/25 的 ratio 表按新列重算)→ geomean 报告含两个口径
(含/不含 symbolic)以便论文引用时自选。
