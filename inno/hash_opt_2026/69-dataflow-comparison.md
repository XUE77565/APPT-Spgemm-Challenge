# docs/69 我们 vs Ocean 数据流逐相位对照(用户钦点:读码先行,策略后置;09-02 起)

**方法**:两码并读,逐相位核实行数/遍数/拷贝;差异行标注 [已核实源码]/[待 profiling]。

## 1. 管线对照(第一轮,已核实)

| 相位 | 我们(hash_product) | Ocean(SpGEMM.cuh) | 差异判定 |
|---|---|---|---|
| 行统计 | count_flop(flop+span) | analysisKernel(products/maxB/range) | 同构 |
| 估计 | MinHash construct+merge(2×snap 梯)×expand | HLL construct+merge ×1.5 | 同构,系数不同 |
| 分桶 | compute_bucket(绝对门+diter/方案5) | binning(双梯相对+ESC≤32乘积+outlier) | 判据不同(我们=自拟合绝对门) |
| symbolic/count | 方案5 dense_count(dense 候选)+ D5H MODE=1(**默认关,与 dense 互斥,bins1-10**) | hashSymbolic/denseSymbolic/ESC 全域(配 est 或 precise 工作流) | **核心差异:我们 precise 覆盖面≈0(封印),Ocean 按工作流全域** |
| numeric | hash_spa gapped 写 / dense_direct 直写 / D5H MODE=2 | hashNumeric/denseNumeric 直写【精确偏移】 | 我们 est 行写 gapped |
| 善后 | **compact(gapped→精确拷贝)+sort** | epilogue_sort(原地行排序,无拷贝;indirect_sort 可选) | **我们多付拷贝税(已知:Ga41 14.5ms/in-2004 14.6-22.7/c-64纯hash 10.8)** |
| 溢出 | retry(hash_global 重跑) | 全局 hashmap 逃逸(binning 时分流) | 结构不同;我们 retry 风暴已被 harness 双 expand 吃掉 |

## 2. 长尾含义

est 工作流的结构代价 = compact 拷贝 + retry;precise 工作流代价 = 数一遍。
Ocean 的 Ana2 门 = compaction<1.5(dup 低,counting 便宜)才付两遍。
**我们的 D5H = 该机制的现成实现,但被封印(默认关 ∧ 与 dense 行互斥 ∧ bins1-10)。**

## 3. 策略队列(数据流证据排序)

1. **D5H 解封**(C 做对版):解除互斥(含 dense 行的阵也开),按 dup 门控
   (行级 dup<~3 才 precise),bins 扩到 1-12。预期 = 长尾低 dup 阵砍 compact 拷贝税。
   验证法:相位(compact 消失量)+ nnz + 同窗双 binary。
2. per-bin 模板(Ocean-D):hashNumeric 的 BLOCK_SIZES 梯实例化 vs 我们 runtime ht。
3. ESC 化 ultra 行(≤32 乘积):O(P²) 但 P≤32 → 1024 比较,免表免 init。
4. (远)indirect_sort / hybrid 全局 value 池(Ocean config 项)。

## 4. 待 profiling 问题

- [ ] compact+sort 内部拷贝 vs 排序的拆分(需仪器化或 nsys)
- [ ] D5H 当年失败的矩阵集与 dup 分布(确定正确的门参数)
- [ ] Ocean precise 工作流在 ocean337 上的实际占比(可从其 config/log 反推)

## 5. D5H 首轮 profiling(09-02,交替 ×2,响应"先 profiling 再策略")

| 阵 | D5H=1 | 默认 | 判定 |
|---|---|---|---|
| **F2** | **6.34**(hash_count 2.5+直写,cmp 消失) | 40.0 | **−84% = 6.3×;6.34 vs Ocean 7.5 = 反超!** |
| 333SP | 64.0 | 58.9 | +8.5%(矩阵级 dup<8 门太粗 → 333SP 边缘高 dup 付贵 counting) |
| nd24k | 275.8 | 275.9 | wash(有 dense 行 → 互斥封印;长尾阵全被封的证据) |

**结论**:precise 工作流做对 = F2 样板(counting 便宜的两遍直写完胜 est+compact);
门的粒度错误(矩阵级 → 应行/bin 级 dup 门);互斥封印挡住全部带 dense 行的长尾阵。
17 阵长尾电池跑批中(bet2ezkag)→ 出 D5H 赢家集 + 与 dup 分布对照定门参数。

## 6. D5H 长尾电池(17 阵,进行中已 8 阵)——precise 工作流收益面实锤

| 阵 | D5H=1 | 默认 | vs Ocean |
|---|---|---|---|
| F2 | 6.34 | 40.0(−84%) | **0.85× 反超** |
| TSOPF_RS_b2383_c1 | 13.1 | 76.6(−83%) | 1.28× |
| pre2 | 14.8 | 190.4(−92%) | **0.97× 反超** |
| Si34H36 | 6.96 | 54.5(−87%) | **0.62× 反超** |
| dielFilterV2real | 43.1 | 247.9(−83%) | **0.60× 反超** |
| road_usa / road_central / Ge87 | wash | | |

**方法论沉淀(用户钦定五步法,入 memory)**:数据流对照先行 → 封印机制先探(D5H 三重封印
就在库里!)→ 小口子量化(无 dense 行阵)→ 电池对照 → 发现即入档。docs/69-70 = 第 1 步产物,
D5H 电池 = 第 2-3 步产物,位图 dense_count(已写待编)= 第 5 步工程。

## ⚠ 7. §6 全表作废(09-02 深夜,nnz 红旗第三次立功)

D5H 快跑全部在下溢检查误回退(d5h_nnz_sum 计算块被 if(false) 封存致恒 0,D5H 行输出
全记 hash 侧)→ 死管线部分相位被当 TOTAL → −83~92% 全为假象。修复(重启用求和块,
commit 见 git log)后真实 D5H:F2 +2% / pre2 +32% / Si34H36 +10% —— **当前实现的
D5H = 中性偏负,不是发现**。

**保留的真实资产**:①数据流对照结论(compact 拷贝税存在)不受影响;②d5h_nnz_sum
bug 修复本身有价值;③教训第三次实证:**任何"惊人变快"先查 Result C/nnz 再谈别的**
(merge3 假象 → D5H 假象,同一模式:不完整管线计时偏小)。位图 dense_count(BCNT,
已写待测)成为 dense 改造的下一个真候选。
