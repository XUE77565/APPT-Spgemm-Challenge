# 18 · 方案 1(预排序二分寻址)A/B 测试:持平,未达预期(2026-08-26)

> HSMU(HPCA'25)思路在我们场景下的实测。8 阵 A/B 对比,BSEARCH=0/1 切换。

## 实现

两阶段 kernel:
- `bsearch_symbolic_kernel`: hash 表收集列集合 + count-rank 排序(同现有 batched 但只写 key,val=0)
- `bsearch_numeric_kernel`: 加载已排序列到 SMEM → 每积二分定位 → `atomicAdd`(零 CAS)

路由:bin0(est≤64 且 rk≤32)替代 `hash_spa_batched_kernel`,`BSEARCH=1` 启用。

## A/B 结果(8 阵)

| 矩阵 | 基线 | 方案1 | 差 | 特征 |
|---|---|---|---|---|
| bcsstk30 | 8.82 | 8.86 | +0.5% | 中等,dup 19× |
| pwtk | 29.43 | 30.52 | +3.7% | 中等,FEM |
| 333SP | 58.03 | 59.92 | +3.3% | 超稀疏图 |
| germany_osm | 49.05 | 52.23 | +6.5% | 超稀疏图 |
| exdata_1 | 22.00 | **21.06** | **-4.3%** | 稠密(47%) |
| TSOPF | 123.53 | 123.20 | -0.3% | 高 dup 76× |
| Ga3As3H12 | 118.29 | 119.61 | +1.1% | 重行 |
| offshore | 17.96 | **17.80** | **-0.9%** | 带状 |

## 分析:为什么没有预期的大幅加速

预期:CAS 消除 → accumulate 1.6× 提升。

现实:**几乎持平,个别阵微退**。原因:

1. **symbolic + numeric 两遍的总工作量 ≈ 原来一遍 hash 的工作量**
   - symbolic: 仍然用 hash 表(CAS 收集列) + count-rank 排序
   - numeric: 二分查找 O(log d) + atomicAdd
   - 原来: CAS(1 步) + atomicAdd = O(1)
   - 方案1: hash(1 步) + 排序 O(d²) + 二分 O(log d) + atomicAdd
   - **多了一遍遍历 + 排序的开销,CAS 节省被抵消**

2. **atomicAdd 仍然存在**——只去掉了 CAS,没去掉 Add
   - 高 dup 阵(TSOPF 76×)的瓶颈是热列的 atomicAdd 串行化
   - CAS 只在"首次插入"发生,dup 的后续积只做 Add(不打 CAS)
   - 所以 CAS 消除省的是"首次插入"的开销 ≈ 1/D 的比例

3. **HSMU 的场景与我们不同**
   - HSMU 是通用的 SpGEMM 库(C=A·B,B 不一定等于 A)
   - 它需要 symbolic pass 来知道 B 的结构(无法预估)
   - 我们已经用 MinHash 估计了 est → hash 表大小已知 → 不需要精确 symbolic
   - **HSMU 的二分寻址是为"不知道列集合"设计的——我们已经知道了(est 就是估计)**

## 结论

方案 1 在我们的架构下**不产生净收益**——symbolic 的开销吃掉了 CAS 消除的收益。
根本原因:我们的 hash 表负载因子 ~50%,CAS 的平均探测链 ~1.2 步,本来就很快;
省掉的 CAS ≈ 每积 1 次原子操作,但新增的排序 + 二分 ≈ 每行 O(d²) + 每积 O(log d)。

**HSMU 的二分寻址适用于"必须做 symbolic"的场景(通用 SpGEMM),
我们已有精确的 est(flop 上界 + MinHash)因此可以跳过 symbolic——这正是我们的优势。**

## 与基线的正确对比

| | 我们(hash+est) | HSMU(hash+symbolic+二分) |
|---|---|---|
| 阶段数 | est(1 遍) + accumulate(1 遍) = 2 | symbolic(1 遍) + numeric(1 遍) = 2 |
| accumulate 原子 | CAS(首次) + Add(每次) | Add(每次) |
| 前置开销 | MinHash ~0.4ms | symbolic hash ~same |
| 排序 | count-rank(融合在 extract) | count-rank(在 symbolic) |
| **实质** | **相同结构,原子操作数相同** | |

→ 两者的计算量几乎一样,差异只在实现细节——所以我们打平是合理的。
