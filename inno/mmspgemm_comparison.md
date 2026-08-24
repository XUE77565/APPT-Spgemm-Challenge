# MMSpGEMM(PACT 2025) vs 我们的 merge3(列域分桶)— 查重比对

> 对象:**MMSpGEMM**, "Multiway Merge Partitioning for Sparse-Sparse Matrix Multiplication on GPUs",
> PACT 2025(2025-11 刊出,IEEE doc 11282921),Georgia Tech HPArch 组,开源
> `github.com/gthparch/MMSpGEMM`(源码已拉到 `external_sota/MMSpGEMM/`,2026-08-24)。
> 我方:`src/spgemm_merge.cu::spgemm_self_product_merge3`(列域分桶 warp-merge,`inno/merge_innovation.md`)。
> 日期:2026-08-24。**结论先行:思想同轴(同一空白、同一动机,它先发表,必须引);机制差异大
> (切分域/计算引擎/重复处理/阶段结构全都不同),不构成实现级撞车;我们的 novelty 主张需收窄并引用。**

---

## 1. MMSpGEMM 是怎么工作的(从源码读出,两阶段)

**阶段 1 `split`(离线预处理,写 `lb_data.bin`/`lb_block_ptrs.bin` 到盘)**:
1. `compute_partial_row_sizes`:每输出行 i 的工作量 = Σ_k len(B[k,:])(**与我们的 flop_ub 同式**);
2. `scan_gen_blocks`:对全部行的中间积做**全局前缀扫描**,切成**等大小 2048 项/块**(BLOCK_SIZE=2048);
   块可**跨行**——小行被顺路打包进同一块;若切点落在行后半段,该行 **reverse 反向切**(优化);
3. `row_splitter`:对每块,在 m 条有序链(B 的行)里用 **tournament tree kth-smallest/largest**
   (注释明言 = **Varman partitioning algorithm**)找**精确 rank-p 多路归并断点** `b[i]`,
   使 Σb[i]=p 且断点前都 ≤ 分割值;同值元素跨块 → `carry_out` 标记。

**阶段 2 `compute`(读回元数据,真正计算)**:
1. `cuda_build_thread_splits`:块内 128 线程 × 16 项(IPT),每线程预计算入口 {a_row, bp, b_col};
2. `cuda_load_block_coop`:每线程**流式走完**自己 16 项(跨 k/跨行推进),生成
   key=((row_local)<<column_bits)|col, val=a·b —— **这是 expand,不是 merge**;
3. `cub::BlockRadixSort` 对全块 2048 项**排序**;
4. 线程内 + warp(shfl/ballot)+ 块内**分段求和去重**;块边界同列靠 carry/atomic_p 拼和;
5. 写出 + `kernel_compact`。
   内存:预分配 `MAX_MATRIX_SIZE 600000000`,README 自注"没有 symbolic pass,可以改进"。

**一句话定性:名字里的 "Multiway Merge" 只在分区阶段(merge-path/Varman rank 断点);
计算内核是 块内 ESC(expand→BlockRadixSort→segmented-reduce),不是 k-way merge。**

## 2. 我方 merge3(`bucket_merge_flop_kernel`,src/spgemm_merge.cu:636)

- (row, bucket) 一块,32 线程 warp;K=5 个**等宽列值域桶** `[blo,bhi)`;
- 每条链 `dev_lower_bound` 二分裁出本桶子区间(在线、单遍);
- 桶内**真 k-way warp-merge**:各链头 shfl 取列最小 → 同列 shfl 求和 → 前进 → 产出列有序项,**全程无排序**;
- sizing = flop_ub 上界(同它阶段 1 公式)→ gapped 写 + `bucket_compact_kernel` 压紧;
- 桶间列值域不相交 ⇒ **同一列永远落同一桶,跨桶重复结构性不存在**。

## 3. 逐机制对比

| 维度 | 我方 merge3(列域分桶) | MMSpGEMM(PACT'25) | 撞? |
|---|---|---|---|
| 动机/问题 | 重行 merge straggler | 中间积负载均衡(同一问题) | **同** |
| 行内切分依据 | **列值域**(K=5 等宽区间) | **精确 rank**(多路归并第 p 小,Varman tournament tree) | 异 |
| 块粒度/均衡 | 固定 K 桶/行,桶大小随列分布**不均衡** | 全局**等大小 2048 项/块**,跨行打包小行,完美均衡 | 异(它优) |
| 链定位 | lower_bound O(log n) 二分,**在线** | tournament-tree 精确 rank 选择,**离线 split 阶段写盘** | 异 |
| 计算引擎 | **真 k-way warp merge**(shfl 最小+求和+前进),零排序 | **expand + cub::BlockRadixSort + 分段归约**(块内 ESC) | **异(核心)** |
| 跨块同列重复 | **结构性不可能**(同列必同桶),免 carry | rank 切分必跨块 → carry_out/reverse/atomic 边界机制 | 异(我简) |
| 输出有序性 | 桶序拼接 = 列序 | 块内排序 + 块序拼接 | 同目标异手段 |
| sizing/内存 | flop_ub 上界 + gapped + compact(有界) | 预分配 600M,自认无 symbolic pass | 异 |
| 阶段结构 | 单遍在线(切分在 kernel 内) | 两阶段离线(split→bin 文件→compute) | 异 |
| 数值精度 | double | float(compute.cu 全 float) | 异 |
| 场景/系统 | C=A·A、A·Aᵀ;自适应 dispatcher 的 merge 支 | 通用 A·B sparse-sparse;standalone | 异 |

## 4. 相似度判定

- **思想层(高相似,≈同轴)**:"把单行累加切开并行,治重行 straggler / 均衡中间积"这一空白,
  两者都填。PACT'25(2025-11)**早于**我方工作(2026-07)公开发表 ⇒ 属于**在先工作,必须引用**。
  `merge_innovation.md` §1 的"文献里的 merge 家族没有任何一个在行内切开并行"一句**失效**,需改写。
- **机制层(低相似)**:切分域(value vs rank)、计算引擎(真 merge vs 块内排序)、
  重复处理(结构性免 carry vs carry/atomic)、结构(在线单遍 vs 离线两阶段)全部不同;
  逐函数对照无对应关系,非实现级相似。
- **类比**:同为"重行 straggler"的两个家族解法,关系类似 hash-SPA vs bitmap-SPA——同题不同构。
- 唯一公式级重合:flop_ub 工作量估计 Σ_k len(B[k,:])(其 `compute_partial_row_sizes`)——
  这是 bhSparse 时代就有的经典上界,双方都非首创,无碍。

## 5. 论文对策

1. **引用定位**:related work 表加一行——"MMSpGEMM (PACT'25):rank 域等大小切分(Varman 多路归并
   分区,块内 radix sort);与本文的列值域分桶正交"。
2. **claim 收窄**(替换"第一个行内并行 merge"):
   > 行内并行存在两条切法:**按 rank 等大小切分**(MMSpGEMM,承 Varman/merge-path)与
   > **按列值域切分**(本文)。值域切分独有三性质:①同列必同桶——跨分区重复结构性不存在,
   > 免 carry/边界原子;②桶内真 k-way merge——免块内排序;③切分在线单遍——免预处理阶段。
   > 代价是固定 K 桶在列偏斜下不均衡,由 dispatcher 将重列偏斜阵调往 hash 路径兜底。
3. **审稿攻击点预案**:"为何不等大小切分/为何不比 MMSpGEMM?"
   - 可答:MMSpGEMM 等大小均衡的代价是 carry 机制 + 块内排序 + 离线预分割;
     我们用 dispatcher(全阵层面自适应)替代行内精确均衡,小/中阵 launch 开销主导时更优。
   - 实验上如需正面对比:代码已就位 `external_sota/MMSpGEMM`(make 需 moderngpu submodule,
     CUDA12;注意口径:它 float+预分配 600M+两阶段,应对齐 compute 阶段计时)。
   - 自适应桶边界(按列分位而非等宽)列为 future work,正面回应均衡劣势。
4. **可引用的它方自认弱点**:无 symbolic pass(预分配 600M)、两阶段预分割。

## 6. 信息源

- 代码:`external_sota/MMSpGEMM/`(gpu/split.cu 492-609、gpu/compute.cu 68-299、partitioner.py 头注)
- 论文页:ieeexplore.ieee.org/document/11282921(CSDL/IEEE 反爬,摘要经搜索结果转引:
  "perfectly partitions the partial products into equal-size blocks… outperform state-of-the-art
  besides AC-SpGEMM");PACT'25 program: pact2025.github.io/program
- "Varman partitioning" 出处:其 partitioner.py/split.cu 注释自述
