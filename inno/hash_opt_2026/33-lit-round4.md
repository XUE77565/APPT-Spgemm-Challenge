# 33 · 文献第四轮(2026-08-28,定向三问)

## 调研范围与方法

纯网络侧(WebSearch + WebFetch,零 GPU/本地重计算)。本轮不再全量扫 SpGEMM,而是按三个定向问题检索:
(1)小/中矩阵 launch 与 phase 开销压缩(kernel fusion / mega-kernel / CUDA Graphs / persistent 多相);
(2)SMEM hash 探测效率(线性 vs double vs Robin Hood 高负载实测 / warp 协作插入 / SMEM-global 混合);
(3)每行 CTA 数自适应(多行打包进一个 CTA / grid-stride vs 1 block/row)。
已归档(docs/12/17/28 + memory)的 Ocean/spECK/nsparse/opSparse 代码实测/HSMU/MMSpGEMM/MH-SpGEMM/
Robin Hood TC'20/warp-spec +2.75×/persistent -1.68× 不重复调研,只在机制关联处引一句。
共 **25 组检索 + 16 次页面抓取(12 次成功;ACM 3774654 正文 403、NSF PAR PDF 两次 socket 断、
opSparse PDF 抓回二进制无文本)**;检索语言中英混合,时间窗 2021-2026 为主。

## A. 新发现机制

### 方向一:小/中矩阵的 launch 与 phase 开销压缩

#### A1. Programmatic Dependent Launch(PDL,sm_90 原生)— 我们管线最便宜的边界压缩 ⭐
- **出处**:NVIDIA CUDA Programming Guide, [Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html);
  CUDA 12.3+ 可与 CUDA Graphs 边(`cudaGraphDependencyTypeProgrammatic`)联用。
- **机制**:主 kernel 内调 `cudaTriggerProgrammaticLaunchCompletion()` 提前放行,从 kernel 在同 stream 里
  **先启动、prologue(取指/SMEM 初始化/索引计算)与主 kernel 尾部重叠**,再 `cudaGridDependencySynchronize()`
  等数据可见。要求 compute capability **≥9.0(Hopper,我们 H100 正好)**;同 stream、机会主义(不保证并发)。
- **我们缺什么**:12 个 phase 的依赖链(count_flop→MinHash×2→est_scan→binning→accumulate 家族→compact)
  全部走"上一 kernel 全排干→launch→ramp"的串行边界。Stanford 实测(见 A3)H100 上**即便有 CUDA Graphs,
  每 kernel 边界仍 ~2.1µs**;我们 11 个边界 ≈ 23µs + 各 kernel 尾部排空/爬坡,占 0.7ms 小阵的 3-6%。
- **移植成本**:低(每个被依赖 kernel 加 trigger,每个从 kernel 开头加一次 device-wide sync + launch attribute;
  不改算法、不改数据结构、不改计时口径)。
- **预期收益**:中低但确定(小阵 3-6%;与其它手段叠加)。是三问里 ROI 最高的第一步。

#### A2. CUDA Graphs 全家桶:成本模型已量化,单发矩阵不划算,重复调用才摊销
- **出处**:
  - KTH, [Kernel Batching with CUDA Graphs](https://arxiv.org/html/2501.09398v1)(arXiv 2501.09398,2025):
    手工建图成本 **~4.2µs/节点 + 160-420µs 截距**;最优 batch 50-100 kernel;>2500 节点反而变慢;小负载最多 **1.4×**;
    只做静态图,明确不碰动态更新。
  - NVIDIA, [Constant-Time Launch for Straight-Line CUDA Graphs](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/):
    重复发射 CPU 开销从 O(n)(2µs+200ns/节点)降到近常数(**2.5µs+~1ns/节点**,CUDA 12.0+/12.6 实测;
    1000 节点 200µs→2.5µs);但** instantiation 是一次性大头**(1025 节点建图 1.5ms 量级)。
  - NVIDIA, [Device Graph Launch](https://developer.nvidia.com/blog/enabling-dynamic-control-flow-in-cuda-graphs-with-device-graph-launch/)(CUDA 12.0+):
    kernel 内 `cudaGraphLaunch`(fire-and-forget / tail-launch 两种语义),**设备侧发射延迟比 host 侧低 2× 以上且与图结构无关**;
    [Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)
    (API 12.4 起,博客口径 12.8+Blackwell 才完整支持——**H100 能否用条件节点未确认,查不到明确说法**)。
- **机制**:把 N 个 launch 打包成一个图节点 DAG,重复发射只付一次 CPU 提交;设备侧图发射让"下一相怎么走"
  的决策留在 GPU 上做,不必回 host。
- **我们缺什么 / 判断**:每矩阵**单发**(竞赛口径一次计时)下,12 节点建图 ≈ 50-200µs,直接吃掉全部收益
  → **CUDA Graphs 对我们的单发口径是负资产**;只有把 API 改成"同一矩阵重复乘"(warmup 建图、正式计只付 2.5µs)
  才成立,但那是改赛道口径(Ocean 同样可以这么改),有公平性风险。真正可搬的是 **Device Graph Launch 的思想**:
  est→binning→accumulate 的分流判据现在要 D2H 读回再由 host 发射,可以改成设备侧算好的谓词/子图发射,
  省掉 2-3 次 D2H 同步(每次 5-20µs)。
- **移植成本**:图本体=中(不推荐);**设备侧分流(谓词 kernel 或 device graph launch)=低-中**。
- **预期收益**:图本体:单发口径下无/负;设备侧分流:小阵 1-3%(消 D2H 同步)。
- **查不到**:`cudaGraphExecUpdate` 的 µs 级公开实测(所有来源只说"partial re-upload"),无法为"逐矩阵改参复用"定价。

#### A3. Mega-kernel 正例(2025-2026 密集出现,但全在 LLM 域):机制蓝图可借
- **出处**:
  - Stanford Hazy Research, [Look Ma, No Bubbles! Designing a Low-Latency Megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)(2025-05):
    Llama-1B 前向融成单 kernel,H100 <1ms、78% 带宽利用率,1.5× vs SGLang。
  - [MPK: Mega-Kernelizing Tensor Programs](https://arxiv.org/pdf/2512.22219)(arXiv 2512.22219,CMU Mirage/Zhihao Jia 组):
    编译器自动 megakernel 化,**端到端最多 1.7×** 低于 kernel-per-operator 系统。
  - [Ada-MK](https://arxiv.org/html/2605.11581v1)(arXiv 2605.11581,2026):自适应 megakernel——但"自适应"=
    **离线 profiling 把调度定死在编译期**,bs=1 提升 23.6%,无运行时偷取。
- **机制**(Stanford 版最可搬):① 常驻 kernel + 每个 SM 执行**预调度的指令序列**;② **跨 SM 计数器同步**
  (完成计数 + 等待目标值)替代 kernel 边界,粒度比 PDL 细(可等 4 个 chunk 而非整个 kernel);
  ③ **SMEM 分页**(H100 228KB 切 13×16KiB 页,指令显式申请/释放,前一算子释放页的瞬间下一算子开始搬权重)。
  关键量化:**H100 上即使有 CUDA Graphs,每 kernel 边界仍 ~2.1µs**——这就是我们 11 个边界的单价。
- **我们缺什么**:不是缺"融合"概念(docs/12 已有融合,`inno/kernel_fusion_detail.md` 两处融合已在),
  而是缺**跨相的设备侧同步原语**——现在每个相边界=排干+launch+读回。pre-accumulate 链
  (count_flop→MinHash 两遍→est_scan→binning/scan)同构性强(可统一 launch 配置),可用 cooperative launch
  的 grid.sync() 或 Stanford 计数器法融成 1-2 个 kernel;accumulate 家族(各 bin SMEM/block 异构)无法单发射融合,
  除非上"解释器式" megakernel——而 AsyncSparse 的 persistent -1.68× 负结果(已档 docs/28 A3)警示:
  负载不均的累积阶段常驻化会打穿,除非像 Stanford 那样静态排程+分页可控。我们小阵 sizing 链是规则的、
  类 LLM decode 的 memory-bound 小核,恰是 megakernel 思路的适用面。
- **移植成本**:pre-accumulate 链融合=中(改同步方式+统一模板);全管线 megakernel=高(不推荐)。
- **预期收益**:小阵(sizing/binning 固定开销占比 48% 的 bcsstk30 类、0.7ms 级小阵 2-30% 差距)预链条融合
  预期砍掉 3-6 个边界 + 2-3 次读回 ≈ 小阵 5-15%;全管线 megakernel 无 SpGEMM 实证,风险高。

### 方向二:SMEM hash 探测效率

#### A4. HKV:单桶圈禁 + 8-bit digest 一条 cache line SIMD 扫描 ⭐ 本轮机制最强
- **出处**:HierarchicalKV, [arXiv 2603.17168](https://arxiv.org/html/2603.17168v1)(2026-03;页面自称 SIGMOD 接收,
  NVIDIA Merlin/Meta 系作者)。推荐系统 embedding 存储,非 SpGEMM——机制可搬,场景不同。
- **机制**:每个 key **只属于一个 128 槽桶**(无二级探测链/cuckoo 迁移/溢出链);每槽存 **8-bit digest**
  (Murmur3 位 32-39),**128 个 digest 恰好一条 128B L1 cache line**,一次访存事务载入,32 个 `__vcmpeq4`
  SIMD 并行比对全部 128 个候选;digest 命中才做全 key 比较(假阳 ~1/256)。探测代价=**固定 1 次访存**,
  与负载因子无关。λ=1.0 时 find 3.37-3.40 B-KV/s(**载入 0.50→1.00 波动 <1%**);对照:WarpCore -90%、
  BGHT -31%、cuCollections(线性探测开放寻址)**塌到 ~0**。
- **我们缺什么**:SMEM 线性探测在负载因子逼近 1 的重行上探测链长尾化 + retry 风暴,正是中阵 1.5-3× 差距的
  可疑成分(docs/28 A4 已提 Robin Hood 思路,但 HKV 给了更强的实测背书和更具体的布局)。**128 槽 digest 行 = 128B
  = SMEM 的 4 次 32-lane 向量读**,无 bank 冲突,digest 扫描在 SMEM 里同样成立;插入=digest 位图 ballot 找
  空槽/命中槽,一次搞定。代价:cache 语义(允许逐出)与我们的精确集合语义不同——我们需要桶满走 retry,
  但"探测 O(1) 化"这个核心收益不受影响;桶内不均衡需要好哈希+可能的二级小桶。
- **移植成本**:中(表布局重排:digest 行 + key 行分离,插/查循环改写;retry 路径保留)。最小化版本见 A5。
- **预期收益**:中(直击高负载探测长尾与 retry;若中阵差距中探测占 1/3-1/2,可望吃回一截)。

#### A5. WarpSpeed(ALENEX'26):高负载并发表实测 —— double hashing + fingerprint 是赢家组合
- **出处**:WarpSpeed, [arXiv 2509.16407](https://arxiv.org/html/2509.16407v2)(ALENEX 2026,开源库,8 种设计:
  Iceberg/P2C/Cuckoo/Double/Chaining + 各自 metadata 变体;A40;负载 90%)。
- **机制/实测**:全局内存表,**插入在载入 >35% 后 DoubleHT(double hashing)最快(峰值 1.76B ins/s)**,
  查询 >25% 载入后 DoubleHT 最快(3.96B q/s);**16-bit fingerprint metadata** 使正查询探测 4→2.5、
  负查询 8→2(载入 >60% 后 metadata 版全面占优);**tile 协作**:cooperative groups 把 warp 切成 8/4 线程的
  小组,**一组线程同时探一个桶的多个槽**。删除最快是 Cuckoo(低关联度)。
- **我们缺什么**:两个可直接偷的件——① **每槽 fingerprint 早退**(8 或 16-bit,插/查先比 digest 再比 key,
  高载下探测减半;这是 A4 的最小化版本,不动表结构,成本最低);② **小组协作探测**(8 lanes 并行探一段
  开放寻址窗口,替代单线程 while-loop 爬链)。
- **移植成本**:低-中(fingerprint 版=每槽加 1-2B + 比较次序改写;协作探测=探测循环 warp 化)。
- **预期收益**:fingerprint:中(高载行探测减 2-4× 的文献实证);协作探测:中低。
- **注意**:WarpSpeed/BGHT 全部是 **global-memory 表**;**SMEM 专属的 线性 vs double vs Robin Hood 实测对比,
  2021-2026 没有找到**(Robin Hood 线 docs/28 A4 已档 TC'20;Owens 组 NSF PAR 那篇是 2012-14 老文)。
  诚实结论:SMEM 场景只能外推(冲突代价结构不同:SMEM 无 cache line 概念、bank 冲突主导)。

#### A6. cuCollections 协作探测(NVIDIA 官方,H100 实测)
- **出处**:NVIDIA Developer Blog, [Maximizing Performance with Massively Parallel Hash Maps on GPUs](https://developer.nvidia.com/blog/maximizing-performance-with-massively-parallel-hash-maps-on-gpus/)(cuco::static_map)。
- **机制**:开放寻址+线性探测;**一个 key 交给一组连续线程**(cooperative groups),一次合并访存预取相邻桶窗口,
  `__ballot_sync`/`__shuffle_sync` 协作选出候选槽。H100 实测:4 线程组在高负载因子下 **insert +13%、find +40%**
  (87.5 GB/s insert / 134.6 GB/s find)。
- **我们缺什么**:**"整 warp 合并插一行 B"的 SpGEMM 正例没有找到**(查不到;最近邻=本条 per-key 协作探测 +
  spECK 的每 block 32 行共享一张 SMEM 表(已档))。cuCollections 的数字证明探测的 warp 协作化本身有 10-40% 空间,
  且机制(窗口预取+ballot)与我们的 SMEM 探测兼容。
- **移植成本**:低-中。
- **预期收益**:低-中(叠加在 A5 fingerprint 之上;两者都作用于探测内循环)。

#### A7. BGHT(PVLDB'22 线):静态建表的方案排名 —— 对我们主要是"排除法"价值
- **出处**:Better GPU Hash Tables, [arXiv 2108.07232](https://arxiv.org/abs/2108.07232)(Awad/Ashkiani/Owens,
  2021-12 修订;BGHT 是 WarpSpeed 的前身)。**bucketed cuckoo(3 哈希)在载入 0.99 下插入均 1.43 次探测、
  正/负查询 1.39/2.8 次**,优于 power-of-two 与 iceberg。
- **判断**:cuckoo 的收益在"建一次、查多次"的静态表;我们的行表**每行每次乘法重建**,cuckoo 插入的重排成本
  不摊销 → 不建议 cuckoo,支持走 A4/A5 的 confinement+fingerprint 路线。列出供 dispatcher 叙事引用。
- **移植成本/收益**:作为选型证据,零成本。

#### A8. TACO'25(Wang/Lin/Wei/Gao/Ji):逐阵调 hash 负载因子与乘数 —— 最便宜的自适应旋钮
- **出处**:[Optimizing General SpGEMM on the GPU](https://dl.acm.org/doi/10.1145/3774654),ACM TACO 2025(摘要级确认,
  正文 403 抓不到;同组 2018 many-core 前作同款表述)。机制四件:ML 选 sizing 方法;**kernel 按组发射最大化
  SMEM 利用**;**启发式逐阵选 hash load factor 与 hash multiplier(降碰撞)**;symbolic 阶段线程缩减提块内并行。
- **我们缺什么**:表尺寸由 est 隐式决定,负载因子与哈希乘数是**全库固定值**;高 dup 阵(重试多)与低 dup 阵
  (探测浅)最优 LF 不同。这把我们 bin 边界/A2(2^k 对齐)的讨论再往下推一层:**LF 与乘子也该进 dispatcher 特征**。
- **移植成本**:低(两个标量参数进已有 dispatcher;可先离线扫 top-loser 阵做表)。
- **预期收益**:低-中(对 1.5-3× 中阵差里的碰撞份额;与 A4/A5 正交且互补——它们改探测结构,这里改参数)。

### 方向三:每行 CTA 数自适应(我们 1 CTA/行)

#### A9. opSparse Kernel0:多行/块 + 4 线程/行的混合指派(已接入基线,机制细节此前未入档)
- **出处**:opSparse, [arXiv 2206.07244](https://arxiv.org/abs/2206.07244)(IEEE Access 2022;V100;比 spECK 2.04×)。
  PDF 正文抓取失败,关键句来自检索片段:"combines two thread assignment methods. **Kernel0 computes multiple
  rows in one thread block. We use 4 threads to …**";摘要另给"通过**设恰当的 binning range** 调 hash 碰撞率与
  硬件利用率的折中"(与 A8 同思想)。
- **机制**:两套线程指派——轻行:一个 block 装多行、每行 4 线程(行内 4-lane 协作);重行:另一套(1 block/行
  大线程数)。即"按行工作量把 2-8+ 行打包进一个 CTA"的直接先例。
- **我们缺什么**:轻行 bin 目前 1 CTA/行(ultra 是单线程/行,已轻;但 32-256 线程档的轻行 bin 存在
  block 多而小、launch 尾部长的浪费)。2D 模板(R 行 × T 线程)的 bin 化正是 docs/24 Phase A 门控之外的
  另一个正交维度。
- **移植成本**:低-中(bin→(R,T) 映射表 + kernel 模板参数,dispatch 框架已在)。
- **预期收益**:低-中(小/中阵轻行 bin 的 SM 占用更满、块数更少;对 0.7ms 级小阵的 accumulate 相位有感觉)。

#### A10. TACO'25 的分组发射 + spECK/HR-SpMM/老证据的行打包谱系
- **TACO'25**(A8 同篇):"**Group 1 processes multiple small rows per thread block to avoid resource
  under-utilization**",block 1024 顶配 + symbolic 线程缩减(检索片段;正文未读)——小行打包的直接表述。
- **spECK**(PPoPP'20,已档):hash 累积器**每 block 最多 32 行**,Hopper 上 SMEM 232KB 红利使其小阵
  launch 开销最低(memory 档案实测 bcsstk08 0.13ms 全场最低)——**行打包+大 SMEM 表 = 小阵强**的活证据。
- **HR-SpMM**(ICS'25,[PDF](https://hpcrl.github.io/ICS2025-webpage/program/Proceedings_ICS25/ics25-31.pdf)):
  自适应行分区(短行阈值 τ→CUDA cores,长行→另一路)——但**每行仍独立 thread block**,是分流不是打包;
  SpMM(dense B)场景,机制可引不可直接搬。
- **老证据**(2017,窗口外,作引用):ResearchGate "On improving performance of SpGEMM on GPUs":
  **同质矩阵上 >1 行/线程最高 2.4×**。
- **GeneralSparse**(USENIX ATC'25,剪枝 LLM 的 SpMM):多行处理 + row-nonsplit/split 维度优化——近邻一句话引用。

#### A11. grid-stride 行循环 vs 1 block/row 的实测对比:**查不到**
- 检索多轮(SpGEMM/SpMV/通用)没有找到 2021-2026 公开的逐点对比数据;只有 NVIDIA 2013 年 grid-stride
  pro tip 的通用理由(任意规模/持久兼容)和 A3 Stanford 博客对"straggler block 尾部排空"的量化
  (~2.1µs/边界 + 尾部不均)。我们自己的 docs/16(multi-stream 只能 0-2%,SM 不能凭空造)是同结论的本地版。
- 可用的间接证据链:1 block/row 的尾部问题 → 行打包(A9/A10)在**不引入常驻循环**的前提下缓解;
  grid-stride 常驻行循环=persistent 家族,受 AsyncSparse -1.68×(已档)与负载不均制约,**无偷取不正动**。

## B. 查过但无新东西 / 明确查不到的(防第五轮重复)

- **PDL/PDL+Graphs 边**:官方文档无 µs 级收益数字(只说"can provide benefits"),收益需自测——但机制门槛/限制已全(A1)。
- **`cudaGraphExecUpdate` 开销实测**:公开来源(含论坛)只有"partial re-upload"定性说法,**查不到 µs 数据**。
- **CUDA Graphs 条件节点在 H100(sm_90)的可用性**:NVIDIA 博客口径 12.8+Blackwell;H100 是否回移**未确认**。
- **CUDA Graphs 用于 SpGEMM/稀疏库的论文**:没有找到专门论文;只有论坛实践(cuSPARSE+graph capture 帖)与
  HC-SpMM(ICDE'25)在 SpMM 语境提 launch overhead(docs/28 已列,未获机制细节)。
- **SMEM-resident 哈希的方案对比实测(线性 vs double vs Robin Hood,2021-2026)**:**查不到**;现有实测
  (BGHT/WarpSpeed/HKV/cuCollections)全部是 global/L1 场景,只能外推(注意 SMEM 无 cache-line 概念、bank 冲突主导)。
- **整 warp 协作合并插一行 B 的 SpGEMM 正例**:**查不到**;最近邻=cuco 4 线程组 per-key 协作探测(A6)
  与 spECK 每块 32 行共享表(已档)。
- **SMEM hash vs global hash 混合调度**:无新论文(我们 Hybrid Value 与 Ocean keys-SMEM/vals-global 均已档);
  HKV 的 L1-digest 设计可视为"分层思想"在 global 侧的最新形态,但不是 SMEM-global 混合调度。
- **persistent+偷取在 SpGEMM 的正例**:**查不到**;正例都在 LLM megakernel(静态排程,无偷取,Ada-MK 明说
  调度编译期定死)与 DiggerBees 蓝图(已档 docs/28 A6)。AsyncSparse 负结果(已档)继续有效。
- **NSF PAR "Analyzing and Implementing GPU Hash Tables"**:Owens 组 2012-14 老文(BGHT 前身),窗口外,PDF 两次抓取断连,不再追。
- **opSparse / TACO'25 / ACM 3774654 正文**:403 或 PDF 无文本,细节停在摘要+检索片段级(A8/A9 标注了置信度)。
- **MGG(OSDI'23)**:intra-kernel 计算通信流水(GNN 多 GPU),非 phase 开销压缩主线,一句话引用即可。
- **Kernel Batching(arXiv 2501.09398)**:静态图、要求迭代数整除 batch、手工建图难维护——对我们单发口径直接否。

## C. 结论:top-3 移植建议(按预期收益/成本)

1. **A4+A5(合流)· SMEM 表探测改造:fingerprint 早退起步,桶圈禁 + digest 行 SIMD 扫描为目标态**
   (成本:起步版低 / 完整版中;收益:中)。三篇独立实测背书同一结论——高负载下开放寻址长尾是主要退化
   (cuCollections λ→1 塌、WarpCore -90%),而 **digest 元数据 + 桶圈禁把探测变成固定 1 次向量扫**
   (HKV λ=1.0 波动 <1%;WarpSpeed 探测 4→2.5/8→2)。最小版本(每槽 8-16bit fingerprint,插/查先比 digest)
   当天可试,直接打中阵 1.5-3× 的 hash 探测份额与重行 retry;A8 的 LF/乘子自适应(成本最低)与其正交,顺手带上。
2. **A1 · PDL 把 11 个相边界重叠掉**(成本:低;收益:小阵 3-6%,零口径风险)。sm_90 原生、
   Stanford 实测每边界 2.1µs(有图也免不掉)是我们差距里确定存在的固定税;A2 的设备侧分流
   (谓词 kernel / device graph launch 消 D2H 读回)作为第二刀,同样低成本。CUDA Graphs 本体判负:单发口径下
   instantiation(50-200µs/12 节点)吃掉收益,只在做"重复调用 API"叙事时再启。
3. **A9+A10 · 轻行 bin 的 (R 行 × T 线程) 2D 模板化**(成本:低-中;收益:低-中,小/中阵 accumulate 相位)。
   opSparse Kernel0(多行/块+4 线程/行)、TACO'25 Group-1(小行进同 block 防 under-utilization)、spECK
   (32 行/块 + Hopper 大 SMEM = 我们套件里小阵最强活证据)三处先例;我们 bin dispatcher 已就位,加一层
   R 映射即可。注意与 docs/24 Phase A 门控正交:dense-iter 管"重行怎么省 est",这里管"轻行怎么省 block"。

落选说明:A3 全管线 megakernel(成本高 + persistent 负结果在前,只取"pre-accumulate 链 cooperative 融合"
子集,并入建议 2 的后续);A7 cuckoo(建表成本不摊销,选型证据而已);A11 grid-stride 常驻行循环
(无 SpGEMM 实证、与 persistent 同风险);CUDA Graphs 静态批(单发口径负资产)。
