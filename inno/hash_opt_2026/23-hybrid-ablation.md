# 23 · Hybrid Value 全量消融(refresh6,2026-08-26 夜)

口径:v3 noHybrid 基线(`methods_cmp_v3_nohybrid.csv`)vs refresh6(`methods_cmp_clean.csv.csv`,
= 2246a36 Hybrid kernel + dbe19cd 自动路由,两者合计)。321 阵双方有数据。

## 总体

| | noHybrid | Hybrid | Δ |
|---|---|---|---|
| geomean Auto | 13.013ms | 12.907ms | **-0.8%** |
| vs Ocean | 2.279×(赢7) | **2.261×**(赢6) | -0.018 |

**结论:Hybrid Value 全量净效果 ≈ 0.8%,不改变战局。** docs/20 的诊断被全量数据证实:
差距主体在中尺寸低 dup 簇(dense 家族缺失)+ compact 税,不在高 dup 原子路径。

## 细节

- 受益 ≥3%:18 阵。真收益 = **af_shell2/6/10(-34~-48%)、TSOPF_RS_b678_c2(-31%)、
  c-64b(-18%)、c-57(-13%)、bcsstk30(-10%)**;bloweya 的 -56% 大部分是 v3 脏基线
  (114→真实 ~50)的假象
- 退步 ≥3%:25 阵。delaunay/rgg/atmos 系 +4.7~5.9%(小型系统性,疑 hybrid 条件边缘);
  **Zd_Jac2 +38.5% 异常**(n<50k 不该触发 hybrid,待复测——可能 v3 该阵被 contention 恩惠)
- **8 阵 A/B 的收益没兑现**:pwtk +0.4%(非 -29%)、TSOPF_FS_b39_c7 -2.7%(非 -10%)、
  Ga3 -0.4%(非 -5%)。根因:**Hybrid Value 只实现 bin0(batched,est≤64 行)的 value 池**,
  pwtk 行均 est~255 → 全在 bin2-5 → 摸不到 hybrid。af_shell 类(bin0 占比高)才吃到
- 受污染窗口(refresh6 #41-45 Si 系):±1.2% 内 = make 事故无痕 ✓

## 下一步(已在盘)

Phase A 按行 dense-iter 分流(源码已编译过)= 主攻低 dup 重行(c-58 类 89% flop);
Hybrid Value 若要兑现 pwtk 类,需把全局 value 池扩到 per-row hash tiers(bin1-10)——
排队在 Phase B 之后(收益上限按本消融看很小)。

*工具:`scripts/ablation_hybrid.py`;refresh6 日志 `compare/ocean337/refresh_auto_v4.log`。*
