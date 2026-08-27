# hash 对齐 Ocean 优化全程记录(2026-08-25 ~ 08-26)

> 目标:DATE'27 论文前把我们的 hash SpGEMM 从 4.04× 落后追平/反超 Ocean(2025 SOTA)。
> 本目录 = 每项改动的【机制 + 根因分析 + 数学 + 实测】完整记录,兼作论文素材与接手文档。
> 代码:commit `01be5b5`(安全根治)→ `28da5b8`(夜班收官)共 12 个;全部改动在
> `src/spgemm_kernel_hash.cu`(hash 主路径)与 `src/spgemm_merge.cu`(64 位安全)。

## 总战报(2026-08-26 Step3 后,同机同条件背靠背直测,`scripts/profile_us_vs_ocean.py`)

| 矩阵 | 优化前 | **优化后** | Ocean(直测) | 前比值 | **后比值** |
|---|---|---|---|---|---|
| 3Dspectralwave2 | merge3 回退 | **16.5ms** | 18.65 | 回退 | **🏆 0.88× 胜** |
| pwtk | 8.4ms | **8.8ms** | 6.66 | 1.25× | **1.31×** |
| exdata_1 | ~76ms | **14.5ms** | 10.29 | ~7× | **1.41×**(dense v1) |
| 333SP | 38.5ms | **9.6ms** | 6.16 | 6.2× | **1.56×** |
| bcsstk30 | 2.7ms | **3.7ms** | 2.27 | 1.2× | **1.61×** |
| TSOPF_FS_b39_c7 | merge3 回退 | **~36ms** | 20.9 | 回退 | **1.72×**(dense 窗口,Step3) |

> ⚠ 勘误(2026-08-26):早先记录的 "TSOPF 胜 5×" 基于 ocean337 CSV 的 Ocean=400ms ——
> 同条件直测仅 20.8ms(CSV 值疑为通宵跑批时竞争污染),已用直测口径全面修正。
> 六阵 C_nnz 全程精确 + 0 行内乱序;3Dspectralwave2 为干净反超。

## 文档索引(按阅读顺序)

| 文档 | 内容 | 对应 commit |
|---|---|---|
| [01-safety.md](01-safety.md) | 安全根治:64 位化 + loop cap + alloc sanity(三次 GPU wedge 的教训代码化) | `01be5b5` |
| [02-heavy-global.md](02-heavy-global.md) | heavy 全局表路径:est>HASH_CAP 的行不再回退 merge3(含三个连环 bug 的验尸) | `7dad171` |
| [03-estimator.md](03-estimator.md) | **MinHash estimator 数学修复**:Jensen 高估 → 算术无偏式(本系列最大单项收益) | `877b0df` |
| [04-row-retry.md](04-row-retry.md) | 行级重试:欠估行 flop 定表重跑(Ocean out_overflow_row_ids 同款,含去重门陷阱) | `de4abe7` |
| [05-batched-family.md](05-batched-family.md) | 批量 kernel 族:warp-per-row 累加 + 融合 extract + 批量 copy + 聚合原子 | `a72f51a`→`96e633e` |
| [06-roadmap.md](06-roadmap.md) | 剩余差距归因 + 超越 Ocean 的新优化方向 | — |
| [20-bottleneck-analysis.md](20-bottleneck-analysis.md) | **全量瓶颈分析(337 阵)**:差距双层结构 + Ocean 解剖 → 四条战线(dense 家族扩建/直接写终态/hash 结构差/DNF) | — |
| [21-dnf-root-cause-and-fixes.md](21-dnf-root-cause-and-fixes.md) | **16 阵 DNF 根因**:subwarp8 遮蔽 heavy bug + retry 泄漏 + flop 无 n 封顶;Fix0-4 设计 | 待提交 |
| [22-dense-iter-port-design.md](22-dense-iter-port-design.md) | **Ocean 机制移植设计**:按行 dense-iter 分流(重行占 89-100% flop 实测)+ 内核升级 B1-B5 | 待提交 |
| [23-hybrid-ablation.md](23-hybrid-ablation.md) | Hybrid 消融 | — |
| [24-phaseA-gate-evolution.md](24-phaseA-gate-evolution.md) | **Phase A 门控 v1→v4**:判别特征=hash 侧速度∝表大小(非密度);v4 门定型 | `ffc6e16` |
| [25-hyb-tier-merge3-fix2.md](25-hyb-tier-merge3-fix2.md) | **三项裁决**:全 bin Hybrid 否决 / Fix2 默认关 / merge3 无生态位;refresh8 2.195× 定型;剩余差距结构 | `c0ef2b0` |
| [26-timing-audit.md](26-timing-audit.md) | **口径审计**:CSV Ocean 列漏 symbolic(真实 ~2.0×);其余逐项对齐 | `06ddefa` |
| [27-ocean-dataflow.md](27-ocean-dataflow.md) | **Ocean 端到端数据流解剖**:四工作流判据+数据流、逐相位对照表、可移植机制清单(方案5=抄 type1 直写终态/localLoadBalance/溢出换算法/bin 级 stream) | `914f549` |
| [28-lit-round3.md](28-lit-round3.md) | **文献第三轮**:8 新机制(MH-SpGEMM bitmap-symbolic+bitonic⭐/Robin Hood/bin 2^k/warp-spec 正负结果)+ 20 条已查无新东西清单 | — |
| [29-direct5-impl.md](29-direct5-impl.md) | **方案5 实现**(DIRECT5 默认关):count pass+直写终态+tmp 缩容;LLB 按行动态 G;成功路径泄漏修复;A/B 预案 | `3bfed54` |

## 一页纸:为什么我们曾经慢(差距的完整因果链)

```
MinHash 调和式估计器(Jensen 高估 ~ln 倍)
  ├─→ est 虚高 → 表/buffer 膨胀 → 内存墙 + 满载原子争用(exdata_1 17.85×、TSOPF 20×)
  ├─→ Σest 爆 int → 偏移绕负 → 野指针写(TSOPF 147 亿教训)
  └─→ 旧版钳 HASH_CAP 掩盖 → 重行直接溢出回退 merge3(ocean337 上 67 阵 28.5× 灾难)
每行固定开销(1 CTA × 256 线程伺候 ~6 个积)
  └─→ 333SP 型(370 万行)accumulate 5.4× 差距 → 批量 kernel 族治
全局原子争用(binning/compact 的 12 计数器 3.7M 次 atomicAdd)
  └─→ warp 聚合原子治(5.9→0.24ms)
```
