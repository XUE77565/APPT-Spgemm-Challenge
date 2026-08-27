# 26 · 计时口径全面审计(2026-08-27 夜,用户质询触发)

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
