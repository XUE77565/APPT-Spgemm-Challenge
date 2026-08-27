# 24 · Phase A 按行 dense-iter:门控三版演进 + refresh7 全量(2026-08-27)

数据:refresh7(`refresh_auto_v5.log`,Phase A v1 门)+ v2-v4 门控 A/B(测量轮,profile_top_losers.py)。

## 1. refresh7 全量(Phase A v1 门 + DNF 修复)

**geomean vs Ocean:2.261 → 2.204×(-2.5%),且 16 头鲸鱼(2-5×)首次计入**(此前 DNF 不在 geomean 内)。
同口径 321 阵对比:-5.8%(12.907→12.153ms)。

- 51 大赢(-65~-85%):c-58 12.2/c-62 12.8/Enron 8.2/soc 24.0/bloweya 17.1…
- **51 回归(+21~+217%)**:F2 53.5(+217%)/laminar_duct3D 16.2(+174%)/bbmat/pkustk13/sme3Dc/consph…
  = FEM 族被 v1 门(flop≥4096, span≥n/16)错误吸入旧慢窗口内核
- 16 鲸鱼全部出数:cage15 153ms(2.0×)/audikw 253(2.65×)/c-73 344(4.55×)/rajat 族 ~170(3.7-4.6×)/
  mouse_gene 1181(**胜 Ocean 1726**)/wb-edu 1504(Ocean 1.5ms,天差,单独议题)

## 2. 门控演进(核心发现:判别特征不是密度,是 hash 侧速度)

| 版 | 门 | 结果 |
|---|---|---|
| v1 | flop≥4096 ∧ flop≥n/16 ∧ dup<8 | 中尺寸簇大胜,但 FEM 族 51 阵 +21~217% |
| v2 | span 换真实跨度(B 首末列,O(nnz) kernel) | F2/pkustk 未治本(它们的 span≈n 非带状) |
| **v4(定型)** | **flop≥4096 ∧ est≥2048 ∧ 16×flop≥span ∧ dup<8** | FEM 全修复(laminar **1.11×**/bbmat 2.08×/consph/cant/gyro ≤+1%),赢家全保(c-58 2.21×/soc 2.87×/bloweya 3.93×/a5esindl 3.67×/Enron 2.57×) |

**为什么 est(表大小)是真判据**(CPU 精算 + GPU 实测):
- flop/span 密度分布**完全重叠**(输家 pkustk13 中位 0.61 > 赢家 c-58 0.16)——密度假说被否决
- 真区别在 hash 对该行的速度:**F2 的行 est~1-2k(小表高占用)→ hash 9.7 G/s 本来就快**,
  送窗口反而亏 8×;**c-58 的重行 est 3-10k(大表)→ 我方 hash 0.96 G/s**,窗口大胜
- ⇔ 我们 hash 的结构短板 = 大表行(低占用);Ocean 的 hash 无此短板(其 c-58 轻行 0.44ms)
  → **下一主攻 = 全 bin 的 Hybrid Value(值走全局 L2)治 hash 大表行**(docs/23 只挂了 bin0)

## 3. Phase B v1(否决记录)

固定 64 线程组 + W=8192 + 1024 线程:F2 更糟 2.6×(139.9ms)、a5esindl 2×。
教训:**组大小必须按 B 切片长度自适应**(Ocean localLoadBalance 的本质),短切片 64 线程组的
冗余 lower_bound 主导成本。已 git stash;v2 方向 = 自适应组 + 线性扫描免 lower_bound + 游标窗口。

## 4. 下一步(按预期收益)

1. **全 bin Hybrid Value**(治 hash 大表行 = 中段 2-5× 主体 + Ge99/web-Google/road 类)
2. **compact 直接写终态**(pre2 75% 税 + 鲸鱼内存峰值;Ocean symbolic 仅 0.4-3.3ms)
3. Phase B v2(自适应组 + 游标窗口)把 c-58 从 2.2× 推向 ~1.2×
4. wb-edu 类超稀疏巨图的 merge/ultra 路径(Ocean 1.5ms vs 我们 1504ms)

*门控 A/B 复现:`DITER_MIN_FLOP=0` 关闭;`scripts/profile_top_losers.py F2 c-58 ...`;CPU 精算见本文件 §2。*
