# 25 · 三项实验裁决:hybrid-tier 否决 / Fix2 有 bug 默认关 / merge3 无生态位(2026-08-27)

refresh8(Phase A v4 门)= **2.195×**(337 阵全量,vs Ocean)。基线 `methods_cmp_v8_v4gate.csv`。

## 1. 全 bin Hybrid Value:❌ 否决(默认关,HYB_TIER=1 可开)

设计:ht≥4096 的 bin,keys 留 SMEM(12→4B/槽,理论 occupancy ×3)+ values 全局 L2 atomicAdd。
实测(高 dup 目标阵):

| 阵 | hyb off | hyb on | 结论 |
|---|---|---|---|
| Ge99H100 | 74.3 | 86.2 | +16% 恶化 |
| **Ga3As3H12** | 89.3 | **257.8** | **2.9× 恶化** |
| pwtk/bin0 | 8.7 | 8.7 | 无感 |

**根因:每乘积一趟全局 L2 原子的往返延迟成本 > SMEM 争用成本**(即使 dup 76×)。
与 docs/23(bin0-only -0.8%)相互印证:Ocean 的 HYBRID_HASHMAP 优势建立在它的负载均衡/
组结构上,裸搬 value 池无效。c-58 -8% 的收益来自测试噪声区间,不足翻案。

## 2. Fix2 compact 原地化:🐛 正确性 bug,默认关(FIX2=1 可开)

守卫 ∀j: A(j)≤E(j) 数学上覆盖跨行写读(推导见 docs/21),rajat16 实测 min_margin=0 通过,
但 **sorted check 从 0 变 294411 违规** → 某紧凑路径存在未识别的自覆盖(疑 csort 的
gapped 区间二次读,或 retry 区与 E_off 前缀的交互)。修复待查;恢复后 0 违规 ✓。
内存收益(省 12B×C)仅鲸鱼阵需要,而鲸鱼当前已能跑 → 优先级降。

## 3. merge3 在 ocean337:无生态位(0/337 选择正确)

| 阵 | merge3 | hash(Auto v4) | 差距 |
|---|---|---|---|
| c-58 | 501.3 | 12.0 | 42× |
| bloweya | 926.4 | 19.3 | 48× |
| email-Enron | 101.4 | 8.2 | 12× |
| soc-Slashdot0902 | 445.7 | 24.4 | 18× |
| F2 | 75.5 | 19.9 | 3.8× |
| pre2 | 318.3 | 80.3 | 4.0× |

中尺寸簇 = 稠密输出 = dense/hash 主场,值域 K 桶 merge 无优势;合成带状阵(band_n32000 35×)
的优势在真实套件无对应结构。**B 线在 ocean337 关闭**;merge3 保留给带状 PDE 输入域
(paper 叙事:dispatcher 的第三臂,first100 类套件)。

## 4. 当前剩余差距结构(2.195×)

| 簇 | 代表 | 比值 | 根因 | 下一步 |
|---|---|---|---|---|
| 高 dup hash | Ge99H100 | 4.4× | Ocean hash 有我们没有的结构优势(负载均衡组) | Ocean localLoadBalance 移植 |
| compact 税 | pre2 | 6.3× | 75% 时间在 compact+sort | 方案5 直接写终态(需精确计数 pass) |
| 鲸鱼 | c-73/rajat | 2-5.5× | 同上 + 内存带宽 | 同上 |
| 中段 | 大多数 | ~2× | accumulate 通用差距 | Phase B v2(自适应组+游标窗口) |
| 超稀疏巨图 | wb-edu | ~1000× | Ocean usparse 1.5ms vs 我们 1504ms | ultra/merge 路径排查(独立调查) |

*复现:`HYB_TIER=1 ./spgemm_test ...`、`FIX2=1 ...`、`METHOD=merge3 ...`。*
