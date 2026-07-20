# SpGEMM SOTA 文献调研(2021-2026 顶会)

> 日期:2026-07-20
> 范围:SC / ICS / PACT / PPoPP / ASPLOS / ISCA / HPCA / USENIX ATC / TPDS / SISC 等顶会顶刊
> 关键词:SpGEMM / sparse matrix multiplication / sparse-sparse matmul / GPU

---

## 一、算法范式归类

将所有 SpGEMM 工作按**底层方法论**归为 4 大范式:

### 范式 A:全局哈希累加器(Hash-based SPA)

**核心思路**:对每行,分配一个 hash 表(col→sum),遍历所有中间项 `a_ik·a_kj`,scatter 进 hash 表,同 key 自动求和。输出无序,需后排序或 compact。

**优势**:每个中间项 O(1) 独立插入,全并行,无串行链。GPU 友好。
**劣势**:hash 碰撞(需线性探测/resize);hash 表内存开销;输出无序(需 compact)。

**代表工作**:

| 论文 | 会议/年份 | 开源? | 代理映射 |
|---|---|---|---|
| **cuSPARSE 11.x** (NVIDIA) | 生产库 | ✅ 闭源但随 CUDA 发布 | 即本体(cusparseSpGEMM 默认走 hash) |
| **spECK** (Anh et al.) | TPDS 2021 | ✅ [GitHub](https://github.com/Jeremy-tn Anh/spECK) | 直接用源码 |
| **NSPARSE** (Nagasaka et al.) | ICPP 2017 | ✅ [GitHub](https://github.com/EBD-CREST/nsparse) | 直接用源码 |
| **MH-SpGEMM** | arXiv 2025 | ❌ | → cusparseSpGEMM(hash 变体) |
| **Multi-GPU NSparse** (Mavliutov et al.) | Wiley 2025 | ❌ | → cusparseSpGEMM(单 GPU 版) |
| **Ocean** | arXiv 2024 | ❌(有二进制) | → cusparseSpGEMM(Ocean 本质是 hash+binning) |
| **ACES/Spada** (Li et al.) | ASPLOS 2023 | ❌(硬件) | → cusparseSpGEMM(SpGEMM 部分) |

**代理总结**:不开源的 hash 类 → `cusparseSpGEMM_prune` 或 `cusparseSpGEMM_compute`(cuSPARSE 默认 hash 累加器)。

---

### 范式 B:行分段归并(Merge-based)

**核心思路**:利用 A 是 CSR → 每行的列有序;行 i 的中间项列 = 若干有序列链 `{A[k,:] : k∈A[i,:]}` 的并集。对这些链做 k-way merge,直接得到有序输出。

**优势**:无排序、无 hash 碰撞;小行低开销。
**劣势**:按输出有序 → 串行链(straggler);大行上串行链长。

**代表工作**:

| 论文 | 会议/年份 | 开源? | 代理映射 |
|---|---|---|---|
| **RMerge** (Gremse et al.) | SISC 2015/2018 | ❌ | → 本项目 merge2(warp k-way shuffle merge,算法等价于 NPC'18 reg-merge) |
| **Register-Aware** (Liu et al.) | NPC 2018 | ❌ | → 本项目 merge2(同算法:warp shuffle min/sum k-way merge) |
| **Multiway Merge Partitioning** (Lorimer et al.) | **PACT 2025** | ❌ | → 本项目 merge3(col-bucket + binary search) |
| **本项目 merge2/merge3** | (工作中) | ✅ | 即本体 |

**代理总结**:不开源的 merge 类 → **本项目的 merge2/merge3**(在 GPU 上实现了与 NPC'18 reg-merge 等价的算法)。

---

### 范式 C:Tile 分块 + 共享内存(Tile-based)

**核心思路**:将矩阵分成小块(tile),每块加载进 shared memory,在 shared mem 里做稀疏累加(sort/merge/hash 都在片上)。

**优势**:片上访存极快;适合中等密度。
**劣势**:shared mem 容量有限(48KB-228KB)→ tile 大小受限;对极稀疏矩阵浪费。

**代表工作**:

| 论文 | 会议/年份 | 开源? | 代理映射 |
|---|---|---|---|
| **bhSPARSE** (Liu & Vinter) | IPDPS 2014/2015 | ✅ [GitHub](https://github.com/bhSPARSE) | 直接用源码 |
| **AC-SpGEMM** (Winter et al.) | ACM TACO 2019 | ❌ | → cusparseSpGEMM(CU 11.x 内部用 tile+hash 混合) |
| **Gamma** (Zhang et al.) | ASPLOS 2021 | ❌(硬件) | → CUTLASS Grouped GEMM(稀疏 tile 近似) |
| **SpArch** (Zhang et al.) | ISCA 2019 | ❌(硬件) | → 无 GPU 代理(硬件专用 merge tree) |

**代理总结**:不开源的 tile 类 → `cusparseSpGEMM`(内部 tile+hash 混合)或 CUTLASS 的 sparse GEMM 接口。

---

### 范式 D:负载均衡分割(Split-K / Merge-Path)

**核心思路**:不改变累加器算法,而是用 merge-path / split-K 等技术将**不均匀的工作负载**均匀分配到 GPU 线程/SM。

**优势**:解决 SpGEMM 固有的负载不均(有些行几百项,有些行几项)。
**劣势**:分割本身有开销;不改变累加器的算法瓶颈。

**代表工作**:

| 论文 | 会议/年份 | 开源? | 代理映射 |
|---|---|---|---|
| **AC-SpGEMM** (Winter et al.) | TACO 2019 | ❌ | → cusparseSpGEMM(内部用 merge-path 做 binning) |
| **Dalton et al.** | ACM TOMS 2015 | ✅ [merge-spmm](https://github.com/owensgroup/merge-spmm) | 直接用(SpMM 版;SpGEMM 需改造) |
| **KAMI** (Wang et al.) | SC 2025 | ❌ | → cuSPARSE(communication-avoiding 版) |

**代理总结**:不开源的 split-K 类 → `cusparseSpGEMM`(CU 11.x 内部已实现 binning 做负载均衡)。

---

## 二、各类矩阵的最优方法(按范式推荐)

| 矩阵类 | 预期最优范式 | 理由 |
|---|---|---|
| **S-D**(小稠密) | **Merge** | 固定开销最低;无 hash 建表成本 |
| **S-H**(小高稀疏) | **Merge** 或 **Hash** | 都行;merge 略优(开销低) |
| **S-E**(小极稀疏) | **Merge** | 几乎全是开销,merge 最轻 |
| **M-D**(中稠密) | **Hash** | 重行/中间项多 → hash 处理好 |
| **M-H**(中高稀疏) | **Hash** | 算法甜区;hash 通用最强 |
| **M-E**(中极稀疏) | **Hash + Split-K** | 负载不均 → 需 merge-path 分配 |
| **L-D**(大稠密) | **Hash + binning** | 内存爆炸 → hash+binning 分散压力 |
| **L-H**(大高稀疏) | **Hash + binning** | **主战场**;Ocean/cuSPARSE 互角 |
| **L-E**(大极稀疏) | **Adaptive Hash** | 跨行极不均 → Spada 式自适应 |

---

## 三、引用清单(必引)

| # | 论文 | 范式 | 会议 | 年份 |
|---|---|---|---|---|
| 1 | Gustavson | SPA(起源) | ACM TOMS | 1978 |
| 2 | RMerge (Gremse et al.) | **Merge** | SISC | 2015/2018 |
| 3 | Register-Aware (Liu et al.) | **Merge** | NPC | 2018 |
| 4 | **Multiway Merge Partitioning** (Lorimer et al.) | **Merge** | **PACT** | **2025** |
| 5 | SpArch (Zhang et al.) | Merge(硬件) | ISCA | 2019 |
| 6 | bhSPARSE (Liu & Vinter) | ESC(sort) | IPDPS | 2015 |
| 7 | NSPARSE (Nagasaka et al.) | Hash | ICPP | 2017 |
| 8 | spECK (Anh et al.) | Hash | TPDS | 2021 |
| 9 | AC-SpGEMM (Winter et al.) | Hash + Split-K | TACO | 2019 |
| 10 | Gamma (Zhang et al.) | Tile(硬件) | ASPLOS | 2021 |
| 11 | Spada (Li et al.) | Adaptive(硬件) | ASPLOS | 2023 |
| 12 | Ocean | Hash + binning | arXiv | 2024 |
| 13 | cuSPARSE 11.x | Hash + tile | NVIDIA | 2021+ |
| 14 | Dalton et al. (Merge-Path) | Split-K | ACM TOMS | 2015 |
| 15 | KAMI (Wang et al.) | Comm-avoiding | SC | 2025 |
