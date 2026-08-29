# docs/61 c-big 类攻坚设计(下窗实施蓝图;23 大差阵的架构战场)

**日期**:2026-08-30 | **状态**:设计稿,未实施 | **目标**:c-big 4.86×(291 vs 60)/ Cube_Coup 3.95× 等 23 阵

## 1. 现状事实(全部已证)

- c-big(n=345k,cnnz=448M):hash accumulate 267ms = 86%;**Ocean kernel 级 ~10× 差距**
- Ocean 把 26453 行(7.7%)路由到 dense;我们只 88 行(0.03%)—— **双维路由差距**
- 双维路由(dense_bin ≤ hash_bin → dense)曾实现过:编译过、目标阵 DNF、**代码已丢失**(树/stash 均无),根因未查
- prime 表(Ocean HASH_BIN_SIZES {331,673,1327,2719,5449} 消 clustering):mask→% 侵入大,待做
- 今日教训(必须带进实施):①交替测量 ②nnz 对 scipy ③正确性优先

## 2. 三板斧(按 ROI 排序)

### A. 双维路由复活(先行,~100 行)
1. `compute_bucket_kernel` 加 dense_bin 梯(span ∈ {451,906,1818,3641,7284,∞} 六档);
   hash_bin = 现有 est 梯。
2. 路由:`dense_bin ≤ hash_bin 且 span≤n/4` → BIN_DITER(与 v4 门 AND,防 F2 类回归);
   现有 d_span_len 已在 device(无需新 kernel)。
3. **DNF 防复发**:上次死因未知 → 本次加(a)分档上界断言(dense 行 span ≤ 梯值 ×1.5)
   (b)首验目标阵直接对比 cnnz vs scipy(c)BIN_DITER 行数 dbg 打印(观测路由量级)。
4. 验证:单阵交替 A/B(c-big/Cube_Coup/F2 回归)+ 全量 refresh。
   预期:c-big 的 7.7% dense 行吃掉 accumulate 的长尾(hash 大表行)。

### B. prime 表(消 primary clustering,治 c-58 类探测长尾)
- 封装 `hash_slot(j, ht_size)`:ht_size 为 prime → `%`,pow2 → `&`。侵入点 = mask 引用的
  所有 kernel(hash_spa 族+heavy+retry)。分两步:①hash_spa_kernel 主路径 ②retry/heavy。
- 风险:% 比 & 慢 ~20 cycles —— 只对 ht≥2048 的 bin 用 prime(小表 clustering 无所谓)。

### C. per-bin 模板拆分(launch_bounds 家族,最后做)
- docs/44 已否决单 kernel 强制 launch_bounds(spill 崩);安全路径 = 拆 3-4 个模板实例。
- 收益上限 = Ocean 满占用保证;工作量最大。

## 3. 顺序与判定

A(1 天)→ 若 c-big < 2.5× 则 B 上马;A 无效则先诊断 accumulate 相位 nsys(探测长尾分布)
再定 B/C。全程 nnz 对 scipy、交替测量、refresh 期间双静默。
