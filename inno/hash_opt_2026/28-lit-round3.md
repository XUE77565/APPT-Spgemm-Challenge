# 28 · 文献第三轮(2026-08-27,跑批期间网络侧)

## 调研范围与方法

纯网络侧(WebSearch + WebFetch,零 GPU/本地重计算)。六个方向:累积阶段负载均衡、bitmap 累积器、
免排序输出、memory-efficient(避免物化 O(flop))、多 GPU、2024-2026 新 SpGEMM 全量扫(arXiv API
"SpGEMM" 按提交时间倒序 + PPoPP'26/CGO'26/HPCA'26/SC'25 议程翻查 + 顺 Ocean/spECK/nsparse 引用链)。
已归档论文(Ocean/ggMAGNUS/ACM 3774654/Wang TACO/spECK/SaSpGEMM/bhSparse/MMSpGEMM/HSMU/MAGNUS/
TileSpGEMM/nsparse SC'21/ACSpGEMM/tSparse/IA-SpGEMM/AiSpGEMM)不重复调研,只在关联处提一句。
共约 20 组检索 + 15 次页面抓取;查重过的一篇(PACT'25 Multiway Merge Partitioning)与本地
`inno/mmspgemm_comparison.md` 核对后确认=已档的 MMSpGEMM,未重复计入新发现。

## A. 新发现机制

### A1. MH-SpGEMM(ICCD 2025)— bitmap-symbolic + 重行 bitonic + 细粒度均衡 ⭐ 本轮最相关
- **出处**:ICCD 2025,DOI [10.1109/ICCD65941.2025.00119](https://doi.org/10.1109/ICCD65941.2025.00119)
  (DBLP conf/iccd/YangWLPY25;[SemanticScholar](https://www.semanticscholar.org/paper/76cb7f804c0fc797b6b651da49ade1c6c4e3cd79)、
  [ResearchGate](https://www.researchgate.net/publication/399217530),2025-11)。HSMU(HPCA'25,已档)同国同方向的后续,对比对象
  HSMU/OpSparse/nsparse/cuSPARSE,自称全面占优(数值未验,ICCD 档次中等,谨慎)。
- **机制**:三件针对 hash 管线三阶段的武器:(1) **symbolic 阶段给 B 建 mask(位掩码)存储格式,用按位 OR
  算结果 nnz**——不用 hash 探测也不用估计器,C 行的列集合 = OR over k∈A[i,:] of B[k,:] 掩码;
  (2) **numeric 阶段对"多非零行"改用 bitonic sort**(替代 radix 排序);(3) 更细粒度的负载均衡策略。
  摘要级确认(全文在付费墙后),bitonic 具体在 SMEM 还是 global 原地未读原文。
- **我们缺什么**:三处全打在我们的缺口上——compact+sort 税(8-75%):bitonic 若在 SMEM 内做可省
  cub 分段排序的 global 往返(我们 gapped 写出→read-sort-compact 三次过显存);est 阶段:我们
  MinHash/HLL 是概率法,mask-OR 是**精确计数**且比 hash symbolic 便宜;窗口化后内存可控(见下)。
- **移植成本**:中。bitmap symbolic 直接照抄不可行(全宽 mask = n·m/8 bytes,百万维阵爆炸;其场景是
  AMG 中小阵)。但我们已有**行跨度窗口**(B 首末列 O(nnz) 求 span),把 mask 限定在 span 内 =
  span/8 bytes/行,正是我们 roadmap 方案 3(Bitmap dense)的思想,这篇给了外部引用支点。
  重行 bitonic:改 heavy bin 的写出路径,SMEM 装得下的行(≤~4k 项)走 bitonic+原地 compact 融合。
- **预期收益**:中(compact+sort 是最大单一税种;但 bitonic 只对装进 SMEM 的行有效,特大行仍需
  分段 radix,收益上限受行宽分布约束)。

### A2. nsparse 多 GPU 扩展的单卡部分(CPE 2025)— 2^k 细粒度 bin + chunk kernel
- **出处**:Wiley CPE 2025,[10.1002/cpe.70313](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.70313)
  (Padova 组,Leonardo 上验证)。多 GPU 部分见 A7;此处只取**单卡**改进:在 nsparse 基础上把 bin 细化为
  **16-256 非零数、2 的幂边界**,hash 探测改 while-loop 找空槽,超 SMEM 行走 chunk kernel——单卡即得
  ~2× over nsparse。
- **我们缺什么**:我们 13 个 est bin 的边界是手定的,未必与子组线程数(16/32/64/128/256,2^k)对齐。
  bin 边界对齐 2^k 后,"每 bin 固定子组大小"的分配(重行多线程协同插一行)零浪费。
- **移植成本**:低(只调 bin 边界表 + 子组大小映射)。
- **预期收益**:低-中(对重行 4× 类有边际改善;这是 Ocean localLoadBalance 同空间,天花板有限)。

### A3. AsyncSparse(arXiv 2604.17834)— Hopper 异步机制上稀疏核的系统结论
- **出处**:[arXiv 2604.17834](https://arxiv.org/html/2604.17834v1)(H100,SpMM 家族,但结论是架构级的)。
  消融:WGMMA +0.34×、TMA +1.14×、**warp specialization(producer/consumer + mbarrier 三级循环缓冲)
  +2.75×**,合计约解释 98% 提升。两个**负结果**同样值钱:**persistent kernel 在稀疏负载 -1.68×**
  (PID swizzling 也救不回,负载不均把常驻循环打穿);thread block cluster + TMA multicast 也回退。
- **我们缺什么**:我们的累积循环是同步式——同批 warp 既拉 B 行(gather,间接寻址)又做 CAS 插入,
  HBM 延迟不藏。producer/consumer 分工(producer warp 用 TMA/cp.async 批量搬 A 行段与 B 行,
  consumer 专做插入)是没试过的维度。
- **移植成本**:高(B 行是间接 gather,TMA 只能覆盖规则段(A 行、B 的 csr 元数据);WGMMA 对
  double 精度基本无用武之地,只搬数据机制)。
- **预期收益**:存疑-中(方向 1 的"重行负载不均"另有一条路=不等持久化,先把数据流异步化)。
  另:persistent kernel 的负结果是重要证据——**别在累积阶段上常驻核**,除非配偷取。

### A4. Robin Hood / 子组分桶探测(GPU 哈希表线的现成轮子)
- **出处**:Ashkiani et al., "Data-Parallel Hashing Techniques for GPU Architectures", TC 2020
  ([CSDL](https://www.computer.org/csdl/journal/td/2020/01/08765787/1bLyuQUYtEY));可复用实现
  [aterenin/GPURobinHoodHashing](https://github.com/aterenin/GPURobinHoodHashing)(header-only,
  bucketed + sub-warp + Robin Hood 开放寻址)。2024-2026 无更新之作(检索确认),但机制未被我们用过。
- **机制**:Robin Hood 位移让每个 key 的探测代价趋于均匀(劫富济贫),消除长尾链;分桶 + warp 相邻
  线程协作保访存连贯。**适用面恰好是我们最痛的工况:高负载因子的 SMEM 表**(正是触发溢出重试的行)。
- **我们缺什么**:线性探测 + CAS,负载因子逼近 1 时探测代价长尾化;重试(表开更大重插)也在烧。
- **移植成本**:低-中(改探测与插入逻辑,数据结构不动;SMEM 内无 atomic 问题照旧)。
- **预期收益**:中(直接攻重行插入成本;有现成参考实现,可先做微基准验证再上)。

### A5. SC'25 行聚类/重排提升 B 局部性(arXiv 2507.21253)
- **出处**:[arXiv 2507.21253](https://arxiv.org/abs/2507.21253),SC 2025。**注意:平台是 CPU**(OpenMP
  Gustavson+hash,EPYC 7763),非 GPU——机制可迁移,实证不可直接搬。
- **机制**:对 A 的行做结构相似度聚类(评了 10 种重排:RCM/AMD/ND/METIS GP/PaToH HP/Gray/Rabbit/
  Degree/SlashBurn),相似行相邻处理 → 同一 B 行在 cache 里被整个簇用掉。层次聚类平均 1.39×(70% 阵
  受益,最高 4.68×);HP/GP 重排 1.77× 但预处理 100+ 次迭代才摊销;轻量聚类(定长/变长)约 40-45%
  阵受益、预处理 ≤20× 单次 SpGEMM、~20 次摊销。Acc-SpMM(PPoPP'25,[arXiv 2501.09251](https://arxiv.org/html/2501.09251v1))
  在 GPU tensor-core 侧用了同款"结构相似行聚类重排"提 TC 块密度——同一机制两个平台佐证。
- **我们缺什么**:B 行流的时间局部性完全没经营(按行号顺序发射 block,B 行 reuse 全凭运气撞 L2)。
  轻量版 = 按行跨度/列重叠排序发射(我们的行跨度窗口数据已经在线拿到,顺路)。
- **移植成本**:中(发射序重排 + 每套件单次计时的口径问题)。
- **预期收益**:**低(赛制)/中(论文叙事)**——单次计时下任何 20× 预处理都不摊销,只有"免费顺路"
  的排序(跨度排序,O(n log n) 微秒级)才可能白捡;中段 2× 的差距里 B 带宽占比未测,先 profile 再动。

### A6. DiggerBees(PPoPP'26)— GPU 层次化块级 work stealing 的现成蓝图
- **出处**:PPoPP'26,[PDF](https://www.ssslab.cn/assets/papers/2026-niu-DiggerBees.pdf)(BSC 等)。
  为 DFS 设计:两级栈(SMEM+global)+ 层级偷取 + 细粒度 CPU-GPU 偷取。
- **我们缺什么**:重行 bin 内仍是静态分配。SpGEMM 行独立,行粒度偷取(块尾原子领取行区间)实现
  极易;但 Ocean localLoadBalance 类静态均衡已把方差压得不错,而 A3 的 persistent 负结果提示
  "常驻+偷取"必须成对出现才有意义。
- **移植成本**:中(改发射循环为常驻 + 双层队列)。
- **预期收益**:存疑(无 SpGEMM 实证;先做 A4/A1 的确定性收益更划算)。

### A7. 多 GPU / 分布式一揽子(CPE'25 nsparse-MP、CombBLAS-GPU semiring、CoLa、NVSHMEM)
- CPE 2025 [10.1002/cpe.70313](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.70313):1D 行分布,
  MPI 换 B 行(segmented CSR 拼接),512 GPU 近理想扩展,负载 CV<5%。
- McFarland/Bellavita/Guidi, [arXiv 2504.06408](https://arxiv.org/abs/2504.06408)(DOI 10.1145/3676151.3719365):
  CombBLAS 上的 GPU 分布式 SpGEMM,任意 semiring,按消息大小动态切换 host/host 与 dev/dev 通信路径,
  比 CPU CombBLAS 2×。
- CoLa(ICS'25,[PDF](https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-65.pdf)):
  通信高效分布式 SpMM(GNN 向)。RDMA/NVSHMEM 路线([arXiv 2311.18141](https://arxiv.org/abs/2311.18141))。
- **评估**:全部创新在**通信侧**(B 行交换、消息路径切换),单卡内核机制与我们的差距结构无关。
  与我们单卡叙事的关系:论文里作为"正交维度"引用一段即可,不构成威胁也不提供可搬机制。
  **成本高(无关)、收益无**——不建议投入。

### A8. (关联,一句话)其余带机制的近邻
- **VDHA**(PPoPP'26,清华):SpMSpV 的 vector-driven hash aggregation——哈希聚合 + 向量化的
  近亲,机制细节在 SpMSpV 语境,可翻其 warp 内聚合手法。收益存疑。
- **Bit-GraphBLAS**([arXiv 2201.08560](https://arxiv.org/abs/2201.08560)):B2SR 位块格式,boolean
  SpGEMM 6555×;double 场景只能借"位块+intrinsics"思想,与我们 roadmap 方案 3 同族,作引用。

## B. 查过但无新东西的(防第四轮重复)

- **Balanced Hashing**(IACS'16,[10.1145/2925426.2926273](https://dl.acm.org/doi/10.1145/2925426.2926273)):
  行分组完美均衡 = nsparse 前身,binning 已覆盖。
- **MMSpGEMM = "Multiway Merge Partitioning for Sparse-Sparse MM on GPUs"**(PACT'25,
  [CSDL](https://www.computer.org/csdl/proceedings-article/pact/2025/829500a160/2cuFE7q33Es)):
  与本地 `inno/mmspgemm_comparison.md` 核对为同一篇,已在档。
- **ASA**(SC'06/07 时代,[eScholarship](https://escholarship.org/content/qt33d5d816/qt33d5d816.pdf)):
  列向 SpGEMM 稀疏累积,老论文。
- **RMerge**(2015,Gremse et al.):iterative row merging,merge3 同族更早。
- 硬件加速器类(软件不可移植,引一句即可):**SpArch**(MICRO'20)、**InnerSP**(PACT'21)、
  **SPLIM**([arXiv 2311.03826](https://arxiv.org/abs/2311.03826),PIM,自称 275× vs A6000)、
  **近-HBM AIA**([arXiv 2512.12036](https://arxiv.org/abs/2512.12036),hash 多相 + 自定义近存单元)、
  SparseZipper、MatRaptor、DMSA。
- **SparseX**(CGO'26,[页面](https://2026.cgo.org/details/cgo-2026-papers/51/)):SpMM 库自动选路
  (cuSPARSE/Sputnik/CLASP/Jigsaw × CUDA/TC/STC),预测模型选库——我们 dispatcher 线的 related work,
  无单卡新机制。
- SpMM/SDDMM 家族(tensor-core 或 dense-B 设计,对 double SpGEMM 无直接机制):
  **Swift**(HPCA'26)、**HR-SpMM**(ICS'25)、**HC-SpMM**(ICDE'25,逻辑回归预测 GPU 核分配——
  页面付费墙未验证细节,机制疑似并发核 SM 切分)、**RSH-SpMM**([arXiv 2603.08734](https://arxiv.org/html/2603.08734v1))、
  **ASM-SpMM**(PPoPP'26,Arm SME)、**SPIDER**(PPoPP'26,strided 稀疏 TC stencil)、
  **Acc-SpMM**(PPoPP'25)、**Libra**([arXiv 2506.22714](https://arxiv.org/abs/2506.22714),TC+CUDA core
  2D 感知分工)、**GE-SpMM**、**AsyncSparse 的 SpMM 内核本体**。
- **Long-vector SpGEMM**(ICS'23,[10.1145/3588195.3593000](https://dl.acm.org/doi/10.1145/3588195.3593000)):
  SX-Aurora,H-SPA(t)/H-HASH(t) 层次分块累积器——分块降表尺寸与 13-bin 同思路。
- **DCU SpGEMM**(J Supercomputing 2024,[10.1007/s11227-024-06234-2](https://link.springer.com/article/10.1007/s11227-024-06234-2)):
  nsparse 移植 + 四处修补,无新机制。
- **Masked SpGEMM accumulators**(SPAA'22,[arXiv 2111.09947](https://arxiv.org/pdf/2111.09947)):
  hash/masked-SPA/masked-compressed 四种累积器对比——mask 语义我们无对应场景;masked compressed
  思想已被 count-then-place 覆盖。
- **Nagasaka Pascal(ICPP'17)**:memory-saving 经典(全档在案);2024-2026 无新续作
  (CPE'25 多 GPU 即其延伸线)。"Scaling SpGEMM on CPU-GPU nodes"(Xia/Jiang)页面无内容,
  **未确认,留待第四轮**。
- **PPoPP'26 全议程翻查**:无直接 SpGEMM 论文(最接近的 VDHA 是 SpMSpV);"Waste-Efficient Work
  Stealing"是通用偷取理论,无 GPU 稀疏实证。
- **GPU 并发哈希表 2024-2026**:无新论文(Ashkiani 2017/2020 仍是最新,见 A4)。
- **SC25 议程/TPDS/JPDC 2026 扫描**:除 2507.21253(A5)外未见 SpGEMM 新作。

## C. 结论:top-3 移植建议(按预期收益/成本)

1. **A1 · MH-SpGEMM 两件套——重行 bitonic 排序 + 窗口化 bitmap 精确计数**(成本中/收益中):
   直接打 compact+sort 税(8-75%):SMEM 装得下的行走 bitonic+原地 compact 融合,省 gapped 写出→
   分段 radix→compact 的两趟显存;span 限宽的 mask-OR 精确计数可给重行提供零重试的下界
   (与 roadmap 方案 3 合流,ICCD'25 给了引用)。先在 top-loser 阵上微基准。
2. **A4 · Robin Hood + 子组分桶探测改造 SMEM 插入**(成本低-中/收益中):高负载因子行的探测
   长尾是重行 4× 类差距的可疑成分,有 header-only 参考实现,可当天验证。
3. **A2 · bin 边界 2^k 对齐 + 子组大小映射**(成本最低/收益低-中):CPE'25 单卡 2× 的组成部分,
   一次边界表修改,对重行 bin 的子组协同零浪费。

落选说明:A3(异步化/warp specialization)机制最强但成本高、且 WGMMA 对 double 无益,建议排在
A1 验证之后;A5(行聚类局部性)单次计时不摊销,只做论文叙事与"免费顺路"的跨度排序;A6(persistent+
偷取)有 A3 的负结果在前,无实证不动。
