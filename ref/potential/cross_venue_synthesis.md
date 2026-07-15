# 跨顶会 SpGEMM 文献融合分析:SC/ICS/HPCA/CGO 2025-2026

> 日期:2026-07-15
> 范围:SC(超算)、ICS(超算)、HPCA(体系结构)、CGO(代码优化),2025-2026 年。
> 目的:从这些顶会论文里抽取可融合到本项目 SpGEMM(ESC sort → hash/merge)的技术点。
> 论文摘要:见 `papers/` 子目录。

---

## 检索到的关键论文(7 篇)

### SpGEMM 直接相关(稀疏×稀疏)

| # | 论文 | 会议 | 核心贡献 | 与你的关联 |
|---|---|---|---|---|
| 1 | **Ocean** | ICS'26 | HLL 估计 + 混合累加器 + indirect sort | 已部署对比;sort 瓶颈的解法 ★★★ |
| 2 | **Trident** | ICS'26 | 层次感知分布式 SpGEMM(CA) | 多 GPU 扩展时参考 |
| 3 | **矩阵重排序** | SC'25 | 行聚类 + cluster-wise 计算 | blocked Gustavson 的理论支撑 ★★ |
| 4 | **HSMU-SpGEMM** | HPCA'25 | 高 shared mem 利用率(binary search 累加器) | shared mem 累加器设计参考 |
| 5 | **SparseX** | CGO'26 | 协同多个 GPU 库自适应选择 | 库级调度思想 |

### GEMM/SpMM 可借鉴(稀疏×稠密 / 稠密)

| # | 论文 | 会议 | 核心贡献 | 与你的关联 |
|---|---|---|---|---|
| 6 | **KAMI** | SC'25 | 单 GPU 内 CA GEMM(寄存器+shared mem+tensor core) | 寄存器级累加器思路 ★★ |
| 7 | **Swift** | HPCA'26 | SpMM 矩阵加载优化 | H2D 优化参考(次要) |

---

## 核心发现:三个可融合的技术方向

### 方向 1:Ocean 的 indirect sort + 你的 pinned/arena IO ★★★ 最直接

**出处**: Ocean (ICS'26) §4.2
**融合方式**:

你现在 ESC 的 sort 排 **64-bit key + 32-bit val = 12 bytes/元素**。Ocean 的 indirect sort:
- 把 `key + ptr` 打包成 **32-bit 整数**(key 占高位,ptr ≤14 bit 指向 value)。
- radix 排序时忽略 ptr 位 → 只排 key → **4 bytes/元素**。
- 排完后按 ptr gather value → 写回 CSR。

**预期效果**: sort 内存流量减 3× → sort 快 ~3×(radix 是带宽 bound)。这是**改动最小、见效最快**的优化。

**与你的融合点**:
- 保留你的 ESC 框架(expand → sort → reduce → finalize)。
- 只改 sort 的输入打包(12B → 4B)和输出解包(gather)。
- 正确性不变(C_nnz 不变)。
- 叠加你的 pinned/arena IO → 端到端仍最优。

---

### 方向 2:矩阵重排序 + blocked Gustavson ★★ 你的融合方案 2 的支撑

**出处**: Islam et al. (SC'25) + 你的 `ref/formulation_fusion_ideas.md` 融合方案 2

**SC'25 论文证明了**:
- 对 A 做行重排序 + 层次聚类 → 相似行(共享 k 的行)聚到一起。
- cluster-wise 计算 → **A[k,:] 只读一次**(从 global memory),多个行共享。
- 平均加速 **1.39×**(110 矩阵),预处理成本低(<20× 单次 SpGEMM)。

**融合方式**:
1. 预处理:对 A 的行做层次聚类(按 Jaccard 相似度 / graph partitioning)。
2. 运行时:每个 block 负责一个**行簇**(而非单行)。
3. block 内:遍历 k ∈ ∪A[i,:] for i in cluster;A[k,:] 只读一次到 shared memory,所有行共享。
4. 每行用 hash/ESC 累加器,shared-mem 内去重。

**与 KAMI (SC'25) 的呼应**: KAMI 证明了"寄存器做存储 + shared memory 做通信"的单 GPU CA 有效。blocked Gustavson 正是 SpGEMM 版本的 CA:shared memory 缓存 A[k:] = "通信介质",各行累加器 = "本地存储"。

**预期效果**: A[k,:] 读取量减半(取决于聚类质量) + 免 sort(如果用 hash)。

---

### 方向 3:KAMI 的寄存器级 CA 思想 + spECK hash 累加器 ★★ 深层融合

**出处**: KAMI (SC'25) + Register-Aware (NPC'18) + spECK (PPoPP'20)

**KAMI 的核心洞察**:
- GPU 的 **寄存器(1 cycle)比 shared memory(20+ cycle)快 20×**。
- 传统 GPU SpGEMM 把累加器放 shared memory(hash 表)→ 20+ cycle/访问。
- 如果把累加器放**寄存器**(register-based hash / dense)→ 1 cycle/访问。

**Register-Aware (NPC'18)** 已经证明了:对短行,register hash 比 shared mem hash 快得多(因为短行的 hash 表可以塞进寄存器)。

**融合方式**:
- 短行(中间项 < 64):**register dense/hash 累加器**(不用 shared mem,不用 global)。
- 中等行:**shared mem hash 累加器**(spECK 式)。
- 长行:**global + shared 混合 hash**(Ocean 式,index 在 shared,value 在 global)。
- 这是 Ocean 的混合累加器 + KAMI 的寄存器级优化的结合。

---

## 各技术对你当前瓶颈(sort)的影响

| 技术 | sort 影响 | IO 影响 | 端到端影响 | 改动量 |
|---|---|---|---|---|
| **Indirect sort(Ocean)** | sort -3× | 无 | 计算总 -40% | 小 |
| **矩阵重排序(SC'25)** | 无直接 | A 读取减少(缓存) | 计算 -10-20% | 中(预处理) |
| **Blocked Gustavson(融合 2)** | 可去 sort(hash) | A 读取减半 | 计算可能 -50% | 大 |
| **KAMI 寄存器级(融合 3)** | 可去 sort(register hash) | 无直接 | 计算可能 -60% | 大 |
| **你的 pinned/arena(已做)** | 无 | IO -90% | 端到端 -50% | 已完成 |

---

## 融合路线建议(分步)

```
第 0 步(已完成):pinned A + arena C → IO 最优
第 1 步(最小改动):indirect sort → sort 快 3× → 计算总降 40%
第 2 步(中等改动):矩阵重排序预处理 → A 读取缓存命中 → 计算再降 10-20%
第 3 步(大改动,目标):per-row hash 累加器(短行 register / 中行 shared / 长行 global)→ 免 sort → 计算降 70%+
```

**第 0+1+3 步叠加的预期端到端**:
- IO: 0.3ms(你的 pinned/arena)
- 计算: ~0.3ms(hash + indirect sort,接近 Ocean)
- 端到端: **~0.6ms**
- vs 你现在: 1.79ms → **快 3×**
- vs Ocean: 3.15ms → **快 5×**(因为 Ocean IO 糟糕)

---

## 论文清单 + 链接

### SpGEMM
1. **Ocean**(ICS'26): [arXiv:2604.19004](https://arxiv.org/abs/2604.19004) | [代码](https://github.com/CornellHPC/Ocean-SpGEMM) | 详见 `ref/ocean_deep_dive.md`
2. **Trident**(ICS'26): [arXiv:2603.21444](https://arxiv.org/abs/2603.21444) | 分布式 CA SpGEMM
3. **矩阵重排序**(SC'25): [arXiv:2507.21253](https://arxiv.org/abs/2507.21253) | 行聚类 + cluster-wise
4. **HSMU-SpGEMM**(HPCA'25): [代码](https://github.com/wuminqaq/HSMU-SpGEMM) | binary search 累加器
5. **SparseX**(CGO'26): [IEEE](https://www.computer.org/csdl/proceedings-article/cgo/2026/11395201/2elbYVE2cNy) | 库协同

### GEMM/SpMM(可借鉴)
6. **KAMI**(SC'25): [PDF](https://www.ssslab.cn/assets/papers/2025-wang-KAMI.pdf) | 单 GPU CA GEMM
7. **Swift**(HPCA'26): [链接](https://2026.hpca-conf.org/) | SpMM 加速

### 之前已有
8. **Hash 多阶段 SpGEMM**(arXiv:2512.12036): [链接](https://arxiv.org/abs/2512.12036) | hash + near-HBM
9. **Register-Aware**(NPC'18): [PDF](https://pacman.cs.tsinghua.edu.cn/npc2018/papers/register-aware.pdf) | sort/merge/hash 对比

---

## 关键洞察(一句话)

> **SC'25 的矩阵重排序 + ICS'26 Ocean 的 indirect sort + 你的 pinned/arena IO = 三层优化(数据局部性 + 排序效率 + 传输效率),覆盖了 SpGEMM 端到端的所有瓶颈。** 最终方案应该是:重排序后的 blocked Gustavson + per-row hash 累加器(register/shared/global 三级)+ indirect sort(hash 输出)+ pinned/arena IO → 预期端到端 ~0.6ms(比当前快 3×、比 Ocean 快 5×)。

相关文档:
- `ref/ocean_deep_dive.md`(Ocean 解读)
- `ref/formulation_fusion_ideas.md`(四公式融合)
- `ref/sort_innovation_directions.md`(sort 优化方向)
- `ref/spgemm_papers_2025.md`(第一轮文献)
- `worklog/profiling_analysis.md`(瓶颈分析)
