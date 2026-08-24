# Merge-based SpGEMM 创新点:列域分桶行内并行

> 对应代码:`src/spgemm_merge.cu::spgemm_self_product_merge3`(列域分桶)+ 设计点② 收尾 C(`MRG3_FLOP_UB`,flop_ub 估计式 sizing)。
> 口径:H100 PCIe,double,cudaEvent compute-only。
> 定位:把本项目**全局创新**(`innovation_points.md`:自适应调度 / 框架 / HLL sizing / profiling)之外,**merge 这条算法线本身**相对文献的独立贡献讲清楚。
> 日期:2026-07-25

> ⚠ **2026-08-24 查重更新**:发现 **MMSpGEMM(PACT 2025, GT HPArch, 开源)**同样做了"行内切开
> merge 累加以治重行 straggler"(rank 域等大小切分 + 块内 radix sort),**早于本文发表**。
> 下文"文献里没有任何一个在行内切开并行"/"我们补上第四个维度是首个"一类表述**失效**,
> 须按 `mmspgemm_comparison.md` §5 收窄为"值域切分 vs rank 切分"的差异化主张并引用它。

---

## 0. 一句话:我们的 merge 新在哪

> 文献里的 merge-based SpGEMM,**并行维度只有三个**:① 跨行(Gustavson 行归并,行间并行);② 按行长分桶做负载均衡(Liu & Vinter,仍是行间);③ 沿 K 条链用归并树降单步比较(SpArch,O(K)→O(log K))。**没有任何一个在「行内沿列域」把单行的 merge 切开并行。** 我们的 merge3 提出第四个维度——**列域分桶(column-domain bucketing)**:把一行的列域 `[0,n)` 切成 K 个不相交桶,每个 `(row,bucket)` 一个 warp 独立归并,桶按序拼接即列有序 CSR。这把困扰 merge 家族几十年的**重行串行 straggler** 从根上消掉,同时保住 merge「免全局 sort」的核心红利。

---

## 1. 文献脉络:为什么 merge 在 GPU 上是「冷门」

SpGEMM 两族对偶算法里,merge(Gustavson 行归并)的卖点是**输出天然列有序、免 sort**;致命弱点是 k-way merge 的「比较 K 个链头→取最小→前进」是**串行依赖**——重行(k 大、链长)被一个执行体串行处理,成为 straggler。正因为此,近 10 年 GPU 上的 fast SpGEMM **几乎都弃 merge 走 hash 或 ESC**(`ref/sort_innovation_directions.md` 方向 A 原文:「GPU 上 merge-based SpGEMM 极少见,文献里要么 hash 要么 sort」)。

文献里的 merge 家族(本次调研覆盖 2014–2025):

| 工作 | 会议/年 | 并行轴 | merge 是否行内并行 | 与我们的关系 |
|---|---|---|---|---|
| **Gustavson** | TCS 1978 | 行间 | ❌ 每行单执行体串行 k-way merge | 我们的原型;straggler 的根源 |
| **bhSparse(Liu & Vinter)** | IPDPS'14 | 行间(按行长分 **38 桶**负载均衡) | ❌ 短行一线程 heap-merge;长行 gather/compress+merge path(仍是行级) | 最直接的 GPU 前辈;但它分桶只为**跨行负载均衡**,单行 merge 仍串行 |
| **SpArch** | ISCA'19/HPCA'20 | K 链(merge-tree 二叉归并树) | ⚠️ 沿 K 链维降比较 O(K)→O(log K),**但行内不切列域** | 硬件加速器,outer-product 公式;与我们的列域轴**正交** |
| **MatRaptor / Sparm / SpAda** | HPEC'20 / ASAP'24 / ASPLOS'23 | 数据流(IP/OP/row-wise) | ❌ 硬件合并网络 | 专用硬件,算法/数据流是天花板参考,非 GPU 可跑 |
| **AC-SpGEMM** | PPoPP'19 | A 非零元均匀切分 | ❌ 多轮 ESC(展开-排序-压缩),**已弃 merge** | ESC 路线;把 merge 换成了片上 sort |
| **nsparse / spECK / opSparse / Ocean / HSMU** | '17–'25 | 中间积(hash/二分) | — **已弃 merge** | hash/bitmap SPA 路线 |
| **Segmented Merge** | SSSLab'21 | 分段原语 | 部分(分段归并原语替全局 sort) | 一个免 sort 的归并原语,非完整 SpGEMM |
| **Register-Aware** | NPC'18 | 累加器层级(register sort/merge/hash) | 对比三种累加器 | 证明短行 merge 有效,但未解行内并行 |

**关键空白**:这些工作要么并行「行间」(Gustavson/bhSparse),要么并行「K 链」(SpArch merge-tree),要么干脆「弃 merge」(hash/ESC)。**没有一个在「行内沿列域」切开并行**——而这正是重行 straggler 的命门:一行的列一旦多,单执行体扛不住。我们的 merge3 补上这第四个维度。

---

## 2. 核心创新:列域分桶 = 行内并行的新维度

### 2.1 做法(`spgemm_self_product_merge3`)

行 `i` 的中间项列 = 若干有序列链 `{A[k,:] : k ∈ A[i,:]}` 的并集(因 A 是 CSR,每条链列有序)。merge3 把列域 `[0, A_cols)` 均分成 **K=5 个不相交桶** `[blo_b, bhi_b)`,每个 `(row, bucket)` 起一个 warp(32 线程):

1. **二分定位**:对每条链 `A[k_p,:]`,用 `dev_lower_bound` 在 `O(log nnz)` 内找到它落入本桶的子区间 `[seg_ptr_p, seg_end_p)`。这把每条链**只切出本桶相关的段**,代价极低。
2. **桶内 warp-merge**:32 线程协作对本桶内各链子区间做归并——扫各链头、`__shfl_xor` 取列最小、同列求和(`__shfl_xor` 归约)、前进。每步吐出一个本桶列有序的输出项。
3. **桶按序拼接**:K 个桶互不相交且列区间有序 → 直接拼接即行内**列有序 CSR**,**免全局 sort**。

### 2.2 两轴并行(2D 并行网格)

merge3 的并行是 **(行 × 列桶)** 的二维网格:
- **行级轴**:不同行天然独立,`grid = (A_rows, K)`。
- **列桶轴(新)**:同一行的 K 个列桶由 K 个 warp **同时**处理 → **重行不再由单执行体扛**,straggler 被切成 K 段并行。
- 桶内再叠 **warp 内 32 线程协作归并**(沿 K 链轴取最小)。

这三层(行 × 列桶 × warp)是正交的并行来源;文献此前只用前两个 + K 链,我们从「列桶轴」拿到增量并行度。

### 2.3 为什么列域分桶是「对的」切法

- **桶不相交 ⇒ 输出有序拼接**:这是 merge 免 sort 红利能**在并行下存活**的关键。沿 K 链切(SpArch 式)或沿中间积切(hash 式)都拿不到这个性质——hash 出来无序还得排,merge-tree 是 outer-product 失去行归并结构。列域切是唯一能既并行又保序的。
- **lower_bound 让切分廉价**:把一条有序链裁到本桶子区间只需 `O(log nnz)` 二分,无需扫整条链。
- **天然适配 CSR**:A 的列有序是免费前提,不需要额外建 CSC(对比 outer/colw/inner 各建一次 CSC)。

---

## 3. 次要创新

### 3.1 flop_ub 估计式 sizing 替精确 count-merge(C,−32%)

文献里 merge/ESC 的 symbolic 阶段普遍走**精确两遍式**(先 count 再 compute),或 bhSparse 式的「短行上界 + 长行渐进」混合。原版 merge3 的 `bucket_count_kernel` 也为拿**精确** per-bucket distinct,**把整遍 warp-merge 又跑了一次(只数不写)= 双 merge**,占 merge3 总时间 ~37%(bcsstk30 count 6.1ms / merge 10.4ms)。

我们的收尾 C(`MRG3_FLOP_UB`,默认开):用**确定性 flop 上界**替精确 count——`bucket_flop_kernel` 对每 `(row,bucket)` 只做 `lower_bound + Σ(seg_end−seg_ptr)`(无 merge 迭代),得到该桶 distinct 的上界;merge 写到按 flop 预留的 gapped 区并记真实数;scan 真实数 + `bucket_compact_kernel` 压紧。

- **性能**:count 6.1ms → flop+fscan+rscan+compact ≈ 0.87ms;merge3 compute-only **bcsstk30 16.6→11.3ms(−32%)、bcsstk32 7.37→5.61(−24%)、bcsstk08 1.11→0.84(−24%)**。
- **正确性**:flop path 的 C_nnz == 精确 == 参考(can_24 336 / bcsstk08 305612 / bcsstk30 8946070 等,全对);compact 保序 → 输出仍列有序。
- **取舍**:gapped buffer = Σflop × 12B(大阵 ~2GB,H100 可容);但 dispatcher 在大阵选 hash,merge3 实际服务小/中阵(flop 小 → buffer 小)。

> 这是「估计式 sizing 替精确 symbolic count」思路在 merge 侧的对应(	hash 侧对应见 [[hash_innovation_kmv]]):不为精确去重付两遍 merge 的代价,只求一个安全上界定槽位,真实数留到 write 阶段顺带记录。

### 3.2 免全局 sort 红利(在并行下保留)

hash 路径(nsparse/spECK/opSparse/Ocean)的累加器输出无序,必须末尾全局 sort(或间接 sort);merge 路径靠有序链归并天然列有序。merge3 的关键在于:**列域分桶后这个红利仍在**——桶内有序 + 桶间有序拼接 = 全行有序。这是 merge 家族相对 hash 的结构性优势,我们把它扩展到了并行场景。

---

## 4. 与 SpArch 的正交定位(`worklog/potential.md` §2.3)

SpArch(merge-tree)与我们的列域分桶是**两个正交的并行轴**:

| SpArch 思想 | 轴 | 我们的对应 | 可借鉴度 |
|---|---|---|---|
| Merge tree(K-way 二叉归并树,O(K)→O(log K)) | K 链轴(降单步比较) | 我们已用 warp-shfl 取 K 链头最小,单步 O(log K);**K=5 时 merge-tree 收益边际** | 低 |
| 行重排复用(重排 A 行,相邻输出行共享 B 行 → 降内存项 β) | 行序轴(跨行复用) | **未做**(每行独立读 A 的 CSR 子行) | 中高(正交,未来工作) |
| 列域分桶并行 | 列域轴(行内切并行度) | **merge3 主创新** | — |

- K=5(当前)下 SpArch 的 merge-tree 几乎不增值(我们 shfl 已达 log K);**若未来 K 调大(重行多桶),merge-tree 才显价值**,可与我们的列域分桶叠加。
- SpArch 的**行重排复用**攻击的是内存项 β(读 A 的随机 CSR),与我们偏 compute/merge 的优化正交,是 merge3 仍可叠加的未来方向(预处理:按共享 B 行的图聚类重排 A 行序)。
- 注意 SpArch 是 outer-product + 专用硬件;移植到通用 GPU 的 Gustavson 行归并要**保留「有序链免 sort」红利**,不能照搬其 outer-product(那会失去免排序)。

---

## 5. 创新点对比总结(我们 vs 文献)

| 维度 | Gustavson'78 | bhSparse'14 | SpArch'19 | hash 系(nsparse/spECK/Ocean/HSMU) | **我们 merge3** |
|---|---|---|---|---|---|
| 行间并行 | ✅ | ✅(38 桶负载均衡) | —(硬件) | ✅ | ✅ |
| **行内列域并行** | ❌ | ❌ | ❌ | —(hash 不归并) | ✅ **(新)** |
| K 链降比较 | O(K) 串行 | O(K) 串行 | O(log K) merge-tree | —(O(1) atomic) | O(log K) warp-shfl |
| 免全局 sort | ✅(天然有序) | ✅ | ❌(outer-product) | ❌(需 sort/间接 sort) | ✅ **(并行下仍有序)** |
| sizing | flop 上界 / 两遍 | 短上界+长渐进 | 硬件 | HLL 估计 | **flop_ub 估计式(−32% count)** |
| 可跑 GPU | ✅ | ✅ | ❌(ASIC) | ✅ | ✅ |

**我们的独立贡献**:在「行间并行」「K 链并行」之外,补上文献缺失的**「行内列域并行」维度**,让 merge 既保住免 sort 红利、又治掉重行 straggler——使 merge 这条「冷门」路线在 GPU 上重新可与 hash 路线竞争(并经自适应调度器与 hash 互补,见 `innovation_points.md` 创新点 1–2)。

---

## 6. 性能(compute-only,double)

| 阵 | serial v1(1 线程/行) | warp v2(1 warp/行) | merge3(列域分桶) | merge3+C(flop_ub) |
|---|---|---|---|---|
| bcsstk30 | 远慢(straggler) | ~16 ms | 16.6 ms | **11.3 ms(−32%)** |
| bcsstk32 | — | — | 7.37 ms | **5.61 ms(−24%)** |
| bcsstk08 | — | — | 1.11 ms | **0.84 ms(−24%)** |

> 注:merge3 与 serial v1 / warp v2 不是简单「更快」——它们**算法等价**(展开的是同一批中间项),区别只在并行度。serial v1 单线程/行、warp v2 单 warp/行,都在重行上 straggler;merge3 的列域分桶把重行切成 K 段并行后才把 merge 拉到可与 hash 竞争的区间(大阵仍输 hash,因 merge O(中间积)、hash O(products) 去重更省;靠 dispatcher 路由,见全局创新点 1)。

---

## 7. 可引用英文段落(Paper-ready)

> **Column-Domain Bucketing for Intra-Row Merge Parallelism.** Classical merge-based SpGEMM (Gustavson) parallelizes only across rows; within a row, the k-way merge of sorted lists is a serial dependency that turns heavy rows into stragglers—precisely why recent fast GPU SpGEMMs abandon merge for hash accumulators or ESC. Prior merge work adds either cross-row load balancing by row length (bhSparse) or a K-chain merge tree that reduces per-element comparison from O(K) to O(log K) (SpArch). We introduce a fourth, orthogonal axis: **column-domain bucketing**, which partitions a row's column range `[0,n)` into K disjoint buckets, assigns one warp per `(row, bucket)`, and confines each bucket's merge to the per-chain sub-range located by an O(log nnz) binary search. Because buckets are disjoint and ordered, their concatenation is column-sorted—**the merge family's no-sort dividend is preserved under parallelism**, a property unattainable by hash (output unordered) or SpArch's outer-product (loses row-merge structure).

> **Estimation-Based Sizing Replaces Exact Count-Merge (−32%).** Exact symbolic counting in merge/ESC costs a second full merge pass (≈37% of merge3 on bcsstk30). We replace it with a deterministic flop upper bound per bucket—computed by a single `lower_bound + sum` pass without any merge iteration—write into a gapped flop-sized buffer while recording the true count, then scan and compact. This cuts merge3 compute by 24–32% (bcsstk30 16.6→11.3 ms) with identical C_nnz, mirroring the estimation-over-exact-count idea on the merge side.

> **Positioning vs SpArch.** SpArch's merge tree and our column-domain bucketing are orthogonal: the former lowers comparison cost along the K-chain axis, the latter splits parallelism along the column axis. At our K=5, warp-shuffle already achieves O(log K), so the merge tree adds little; the column axis is the higher-value lever on GPUs. SpArch's row-reordering for B-row reuse attacks the memory term β and remains future work orthogonal to our compute-side optimizations.

---

## 8. 文献清单(本次调研覆盖 2014–2025)

**Merge / row-merge SpGEMM(GPU + 硬件)**:
- Gustavson, *Two Fast Algorithms for Sparse Matrices*, ACM TOMS 1978 — 行归并原型。
- Liu & Vinter(bhSparse), *An Efficient GPU General SpGEMM for Irregular Data*, IPDPS 2014 — 按行长 38 桶负载均衡,短行 heap-merge / 长行 merge path。
- SpArch(Qi/Jia et al.), *Efficient and Accurate SpGEMM*, ISCA 2019(扩展 HPCA 2020)—— 层次化 merge-tree 加速器,outer-product。
- MatRaptor(Zhang et al.), *Row-Wise Product SpGEMM Accelerator*, MICRO 2020 — row-wise product 数据流。
- Sparm(NUDT), ASAP 2024 — IP/OP/Gustavson 三数据流 + 正则化合并(K_ID Merge Manager)。
- SpAda, ASPLOS 2023 — 自适应数据流选择,把 IP/OP/row-wise 统一成「窗口」模板。
- Segmented Merge(SSSLab)2021 — 分段归并原语替全局 sort。
- Register-Aware SpGEMM(Tsinghua), NPC 2018 — register 级 sort/merge/hash 累加器对比。

**对照(hash / ESC 路线,已弃 merge)**:
- AC-SpGEMM, PPoPP 2019(多轮片上 ESC);nsparse ICPP 2017、spECK PPoPP 2020、opSparse 2022、Ocean ICS 2026、HSMU HPCA 2025(hash/bitmap SPA)。

**综述**:[A Systematic Survey of SpGEMM](https://dl.acm.org/doi/10.1145/3571157)。

> 调研依据见 `ref/spgemm_acceleration_paper_survey.md`、`ref/SpGEMM_Survey.md`、`ref/sort_innovation_directions.md`、`ref/formulation_fusion_ideas.md`、`worklog/potential.md` §2.3。novelty 表述按「to our knowledge」口径——文献里未发现 GPU merge-SpGEMM 用列域分解做行内并行的先例(2026-07 检索确认)。

---

## 附:本文件与其他 inno 文档的关系

- `innovation_points.md`:全局创新(调度公式 / 自适应框架 / HLL sizing / cudaEvent profiling)——merge3 是其中「自适应框架」的 merge 一翼,本文件聚焦 merge 算法线本身的相对文献创新。
- `engineering_details.md`:工程优化(accumulate G、compact+sort 融合等)——本文件的列域分桶是算法级创新,不在 engineering_details 之列。
- `hash_innovation_kmv.md`(任务 2):hash 算法线——与本文件共同构成「merge + hash 双翼 + 自适应调度」的三点论文叙事。
