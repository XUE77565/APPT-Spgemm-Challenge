# 03 · MinHash Estimator 数学修复(本系列最大单项收益)

> commit `877b0df`。改一行公式,治三块矩阵类(exdata_1 737→84ms、TSOPF 回退变 5× 胜、3Dspectralwave2 重回 hash)。

## 1. 背景:est 决定一切

hash 路径的每行 distinct 列数估计 `est` 同时决定:
- **tmp buffer 槽位**(gapped 布局,Σest = 总 buffer);
- **SMEM hash 表大小**(snap 到 2 的幂,>HASH_CAP 走全局表);
- **bin 路由**(batched / SMEM 梯 / heavy / ultra)。

est 偏大 → 内存墙 + 表满载争用 + 误入 heavy 全局表(慢 10×);est 偏小 → 表满溢出(旧行为:整阵回退 merge3)。**Ocean 用 HLL + avg_product 双工作流,我们用 MinHash(partition bottom-k sketch:每行对 128 个 partition 各存最小完整 hash 值)。**

## 2. 病灶:调和均值被 Jensen 不等式系统性高估

旧公式(sketch 全 128 个 partition 非空时):

```
E = 2³² · Σ_j (1/min_j) − m        ← 调和式
```

推导它的人的直觉:min_j ≈ 2³²/n_j(该 partition 内 n_j 个均匀 hash 的最小值),
所以 1/min_j ≈ n_j/2³²,求和 ≈ D/2³²。**错在期望不可交换**:

```
E[1/min] ≥ 1/E[min]   (Jensen,1/x 凸)
E[1/min] = (n/2³²)·H_n ≈ (n/2³²)·ln n
```

→ 旧式期望高估 **~ln(n_j) ≈ ln(D/128) 倍**:D=2000 时高估 ~2.7×,D=20000 时 ~5.1×。
**实测**:exdata_1 est 虚高 17.85×、TSOPF ~20×、3Dspectralwave2 ~6.7×,与理论吻合
(ln 因子 × EST_EXPAND 1.5 × pow2 取整的复合)。

## 3. 修复:算术均值(无偏)

min_j/2³² ~ Exp(rate = n_j/2³² = D/(m·2³²)) ⇒ **E[min_j] = m·2³²/D**(指数分布期望,精确)。
用样本均值 Σmin_j/m 估计 E[min]:

```
D ≈ m·2³² / (Σmin_j / m) = m²·2³² / Σmin_j     ← 算术式(无偏)
```

```cpp
// mh_merge_kernel,改动仅此一处(sum_inv → sum_min,公式换)
E = (double)MH_M * MH_M * 0x100000000LL / sum_min;
```

**调试插曲**:首版我漏了因子 m(写成 m¹·2³²/Σmin),低估恰 128 倍——exdata_1 est 总量
254k vs 真实 C_nnz 11.3M,一炮定位。修复后 est 普降到 1.5-2× over-alloc。

## 4. 防线:est = min(估计, 每行精确 flop)

count_flop(每行乘积数,精确上界,D ≤ flop 恒成立)现在无条件先算(0.75ms,兼作
avg_product 门),est 取 `min(snap(E), flop)`:封顶高估尾部,且**不引入新的低估**
(flop ≥ D,取 min 只会收紧到真上界)。

## 5. 收益链

| 矩阵 | est over-alloc 前→后 | 直接后果 |
|---|---|---|
| exdata_1 | 17.85× → 1.49× | 重行回归 SMEM bin,737→46ms |
| TSOPF_b39 | ~74× → 4.6× →(1.15 expand)~2× | heavy arena 从 82GB(超熔断)降回可行,hash 跑通即胜 Ocean 5× |
| 3Dspectralwave2 | 6.7× → 1.91× | 62ms → 13.6ms 反超 Ocean |

配合 EST_EXPAND 1.5→1.15(04 行级重试兜底后才敢收)。

## 6. 论文表述建议

> Our MinHash cardinality estimator originally used the harmonic mean of per-partition
> minima, which Jensen's inequality inflates by a factor ≈ ln(D/m). We derive the
> unbiased arithmetic form D ≈ m²·2³²/Σ min_j (min_j/2³² ~ Exp(D/m)), cutting worst-case
> over-allocation from 17.9× to 1.5×.
