# 27 · Ocean 端到端数据流解剖(2026-08-27 夜,refresh_ocean_sym 跑批期间完成)

> 目的:系统借鉴 Ocean(arXiv 2604.19004,H100 sm_90 构建)的端到端工作流。本文是其
> `run()` 主干 + 四工作流 + 全部 host 编排的完整解剖,并逐相位对照我们管线,落到"我们缺什么"。
> 内核级(numericBinning 六档 / denseNumericIterKernel 游标窗口 / localLoadBalance)此前已解剖
> (docs/20 §尾、docs/25),本文整合并补齐 host 侧数据流、epilogue、溢出重做、stream/内存纪律。

## 0. 全局参数(H100 构建,`ocean/include/Common.h`)

- `NUM_SM=114`,`MAX_THREADS_PER_SM=2048`,**`SHARED_MEMORY_KB=128`** —— 刻意只用 128KB/SM
  (H100 上限 228KB),注释明言"太高会把 L1 压太小"。→ **SMEM 档位是吞吐/缓存折中,不是越大越好**。
- 占用梯度 `CONCURRENT_BLOCKS_PER_SM = {32,32,16,8,4,2,2}` ⇒
  `BLOCK_SIZES = {64,64,128,256,512,1024,1024}`,
  `SHARED_MEM_PER_BLOCK = {4K,4K,8K,16K,32K,64K,64K}`(B)。
- est→hash bin 边界 `HASH_NUMERIC_BIN_SIZES = {–,331,673,1327,2719,5449,16253}`(double,槽=idx4B+val8B)。
- span→dense bin 边界 `DENSE_NUMERIC_BIN_SIZES = {236,451,906,1818,3641,7284,∞}`
  (=SMEM/(8B val+1B flag)−4;bin0 特殊 =SMEM/(2·8+1)−4,值+前缀双数组)。
- 20 stream、200 event;`LARGEST_KERNEL_NUM_BUFFERS = 3×NUM_SM = 342`(全局 value 池/hybrid)。

## 1. run() 主干与四工作流(`SpGEMM.cuh:208`)

```
prologue → analysis(Ana1 分流)
  ├─ ana1_type=0 → numeric_kernel_ultrasparse            (type0, ESC)
  ├─ ana1_type=2 → numeric_kernel_fat                    (type3, 整阵稠密化)
  └─ ana1_type=1 → sample_kernel(HLL 构造+采样) → Ana2 分流
       ├─ ana2_type=0 → symbolic_kernel → numeric_kernel_precise   (type1)
       └─ ana2_type=1 → est_kernel → numeric_kernel_est            (type2)
```

### 1.1 Ana1(analysis,`SpGEMM.cuh:420`,一遍 kernel `Analysis.cuh:analysisKernel`)

**一遍 warp 归约 kernel 同时产出 5 个逐行量**(threads_per_row = block/CONCURRENT_ROWS 模板化):
`num_products`(=flop/2)、`max_b_row_len`、`avg_b_row_len`、`b_col_idx_left/right`
(**=该行输出的列跨度 span**,B 首末列)。cub scan/reduce 后 host 读回 3 个标量。

分流判据(全部 host 侧、零额外 GPU 成本):
- **fat(type3)**:`cols_A/100 > rows_A ∧ rows_B/100 > cols_B ∧ cols_B < 7284`。
  **方阵上 `cols/100 > rows` 恒假 ⇒ ocean337 全套件 fat 永不触发**(它是给极扁矩阵的)。
- **ultrasparse(type0)**:`avg_product = total_products/rows ≤ 64`(esc)。附 spark kernel
  分档 {16,32,64,128} + 超长行 outlier 通道。
- 其余 ana1=1:按 `total_products/nnz_A`(输入膨胀率)选 HLL 精度,<48 用 log2p=5(扩 2.0×),
  否则 log2p=6(扩 1.5×)。

### 1.2 Ana2(sample_kernel,`SpGEMM.cuh:524`;判据 `Analysis.cuh:decideMatrixType`)

B 全阵 HLL 构造(d_hll_b,分 block SMEM 聚合)→ 采样行(3%,夹在 [648,10800] 行)
`hllSamplingMerge` 得两个压缩比:compaction1 = 采样积/采样 nnz(flop 侧)、
compaction2 = HLL 估计均值(distinct 侧)。**est 判据(全 ≥ 才走 type2 est)**:
```
input_expansion = total_products/nnz_A ≥ 8
compaction1 ≥ 8   ∧   compaction2 ≥ 8      (estimation_*_threshold = 8)
```
即"输入膨胀大且输出压缩大"(dup 高)才免 symbolic;否则 type1 precise(付 symbolic 买精确偏移)。
**与 memory 总纲的"膨胀率≥8 且双 compaction≥8"一致,此处补齐了分子分母定义。**

sample_kernel 顺带产出(供 type1 用):
- `safe_estimated_avg_compaction = (avg − 2.53·σ)/2`(z=2.53=99% 置信,除 2"勿过度压缩",
  再夹 ≤0.8·avg)→ ≥1.5 时 **use_estimated_symbolic**:symbolic 的 binning 用
  `est=flop/safe_compaction` 缩放后的积数代替裸 flop(表更贴实际,少 overflow)。
- `dense_query_bitmap = compaction1≥2 ∧ compaction2≥2`(CR>2 才让 dense 内核走"查询再置位"的
  bitmap 写,省 atomicOr)。

## 2. 四工作流数据流(核心)

### type1 precise(symbolic 先行,直写终态)—— **结构上最值得抄**

1. **symbolic**(binning<EXPAND=1.25> on num_products[±compaction 缩放] → 14 bin):
   hash 侧 7 bin(`hashSymbolicKernel`,只插 key 不存值,occupancy 即精确行计数)+
   dense 侧(`denseSymbolicKernel`,bitmap ballot 扫 span)→ `d_num_outputs_row` = **精确行 nnz**。
   两族并发在 13 个 stream 上(streams[0..6] hash / streams[7..12] dense)。
2. scan → **精确 C_nnz,一次 alloc,C->row_offsets 直接复用 scan 输出指针**(零拷贝)。
3. numericBinning<EXPAND=1.5>(对**精确计数**再分 bin)→ hash/dense launcher:
   - hash 行:短行(occupancy≤673=bin2 边界)**内核内 `compactAndSort`(O(n²) rank sort)
     直接有序写终态**;长行乱序写进 C 的自己行槽,留给 epilogue。
   - dense 行:窗口序天然有序,直写终态。
4. epilogue 仅剩 `outputSortingDynLauncher`:对"长 hash 行"**原地** per-row BlockRadixSort
   (in==out 同指针),短行路径退化为自拷贝。
   ⇒ **无 gapped 中转、无独立 compact 相位:sort 就是 compact,且发生在终态数据上。**
   fused 键技巧 `sortOutputFused`:(pos<<IDX_BITS)|col 打包 32B 单键排序,值经 SMEM
   gather 复位(32 位 index 高位空余 = prologue 的 bits_needed 检查所保证)。

### type2 est(我们 refresh8 的同族,但细节全面不同)

1. est_kernel:HLL merge 得 d_est(带 1.5/2.0× 扩张系数);偏斜阵(max/avg 行长超阈)用
   双 kernel 分治合并(streams 并行)。
2. numericBinning on d_est(**无逐行膨胀!行槽 = est 本身**)→ psum_est →
   **malloc = Σest × 1.3**(全局 30% 冗余,即溢出尾区)。
3. hash/dense numeric:按 psum_est gapped 写;**表将满(nearFull)即弃行**:
   记 overflow 行、在共享尾区 atomicAdd 预留 `min(flop, n)` 槽、行号写 −1 防下游重复处理。
4. **溢出行重做 = `denseNumericIterKernel`(最大档游标窗口内核,1 block/行)写进尾区** ——
   不是重建大 hash 表!窗口迭代保证 SMEM 恒定、必然成功。
5. 实际计数 scan → 精确 C → epilogue 三路:**dense 行 + 短 hash 行 copy-only(已有序)**;
   overflow 行从尾区 copy;长 hash 行 sortOutputFused **读 gapped、写终态一遍完成**。

### type0 ultrasparse(avg_product≤64)

展开全部积进 estik(=total_products 槽)→ ESC(expand-sort-compress)内核按 spark 尺寸分档
{16,32,64,128} + outlier 行(超 128)单独通道(hash 复算)→ 计数 scan → **纯 copy 收尾**
(行内排序已在压缩时完成)。无 HLL、无 symbolic、无 sort epilogue。

### type3 fat(方阵永不触发,略)

整阵 `rows×cols` 稠密化 + 按 B 平均行长选每 A 元素线程数 → 计数 scan → 纯 copy。

## 3. 逐相位对照(我们 vs Ocean)

| 相位 | 我们(refresh8) | Ocean(type1/type2) | 差距判词 |
|---|---|---|---|
| 行分析 | count_flop + row_span 两遍 | **一遍 fused warp 归约出 5 量** | 我们多一遍 O(nnz);量级 μs~ms |
| 行长估计 | MinHash(m²·2³²/Σmin,est 1.15×) | HLL(1.5/2.0×)+ 采样双 compaction 决定要不要 est | est 精度同族;**Ocean 敢在 dup≥8 时整个跳过精确计数,我们永远走 est+溢出重试** |
| 分 bin | 13 bin 单维(est) | **14 bin 双维**(est×span,`dense_bin_id ≤ hash_bin_id` 则走 dense 族) | 我们的 Phase A v4 门(est≥2048∧flop≥span/16∧dup<8)是它的**绝对阈值近似**;Ocean 是**双梯相对比较**,自动适配梯形几何 |
| 累积器 | SMEM hash(CAS+探测)全 bin | hash 短行 + **dense 直址(span≤7284 即用)** + largest hybrid | 我们 Phase A/dense_win 已补,dense 覆盖由绝对门控制 |
| 负载均衡 | 固定 G∈{4,8,16,32} by ht_size | **localLoadBalance:按 (a_row_len, products, max_b_len) 逐行动态 2^k 子组**,排除最长 B 行的均值起步+双向夹逼 | 逐行 vs 全 kernel 统一 —— Ge99 类 4.4× 的主因(docs/25) |
| 溢出处理 | flop×2 pow2 定表 hash 重试 | **溢出行改走 denseNumericIterKernel 窗口迭代重做**(SMEM 恒定必成)+ 尾区预留 min(flop,n) | 我们重试要分配大表;Ocean 重试是"换算法"不是"扩表" |
| 输出 | gapped(Σest×1.15)→ cub 分段 sort → **独立 compact 相位再拷一遍** | type1:**精确偏移直写终态,sort 原地**;type2:短/dense 行 copy-only,长行 sort 读 gapped 写终态一遍 | **我们的 compact 税(8-75%)在此;Ocean 没有"compact 相位"这个概念** |
| 排序 | cub 分段 sort(gapped 上) | per-row BlockRadixSort 按行分档(SORT_ITEMS_PER_THREAD 梯度)+ fused 32B 键 | 同族;我们的行没分档 |
| stream | 4 stream(收益 0-2%) | **每 bin 一 stream(13)+事件栅栏**;bin 间真并行(bin 内 1 block/行无依赖) | 我们的 stream 切在"相位"上,Ocean 切在"bin"上 —— **bin 级才是天然并行域** |
| 内存纪律 | 手动 dev_alloc/dev_free | **cudaMallocAsync + 默认池 releaseThreshold=UINT64_MAX**:迭代间同尺寸免释放,跨尺寸由池缓存吸收 | 我们 warmup 5 轮的 alloc/free 抖动与碎片全在计时内 |

## 4. 可移植机制清单(按我们五步路线对号)

1. **方案5(直接写终态)的最短路径 = 抄 type1 epilogue 结构**,而非发明新计数 pass:
   - 我们的 est 已够准(1.15×+重试兜底),缺的是**"精确计数 pass"**(=Ocean symbolic,
     只插 key 不存值,成本约为 accumulate 的 40-60%)→ 得精确行偏移 →
     **accumulate 直写终态 + 长行原地 sortOutputFused + 短行内核内 compactAndSort**。
   - fused 32B 键(pos<<IDX_BITS|col)前提:行内元素数 < 2^(32−idx_bits);套件 n<2^19 时
     idx_bits≤19,行 ≤ 8191 才可用 32B 键 —— 超过走 sortOutputDyn(双数组)。
   - **预期消掉 compact 相位(8-75% 税)+ tmp 双份内存(鲸鱼 DNF 余量)**。
2. **Phase B v2 = localLoadBalance 逐行子组**(本文件 §3 行"负载均衡"):起步均值用
   `avg_ops=(products−max_b_len)/(a_len−1)`(**排除最长 B 行**,它由后续 while 夹逼吸收),
   `2^k` 夹逼后 clamp 到 [lbound,ubound]。移植成本:低(纯 device 函数,我们 hash_spa_kernel
   的 G 选择处替换);治 Ge99 类 4.4×。
3. **溢出重做换算法**(est 工作流):我们重试=flop pow2 大表;Ocean=换 denseNumericIterKernel
   (窗口迭代,SMEM 恒定,写尾区 min(flop,n) 槽)。对我们:重试行改走 dense_win/Phase A 窗口
   内核而非 hash_global 大表 —— 移植成本中,收益=TSOPF 类 heavy 表分配+探测风暴。
4. **bin 级 stream 并行**:我们 4-stream 切相位收益 0-2% 的根因 = 相位间有数据依赖;
   Ocean 切 bin(不同 bin 的行互不相干,1 block/行)。对我们:N_BINS=14 → 每 bin 一 stream,
   hash/dense 两族并发,事件栅栏只在族边界。移植成本:中(重构 dispatch 的 launch 顺序)。
5. **双维 binning 的相对判据**(`dense_bin_id ≤ hash_bin_id`):比我们 v4 绝对门的
   自适应性好(梯形几何自动校准)。可作为 dispatcher 线的特征工程引用(方案:CPU 精算
   span/est 双梯分布后直接套该公式对比 v4)。
6. **一遍 fused 行分析 kernel**(5 量一次出):省我们一遍 row_span O(nnz);μs 级,顺手做。
7. cudaMallocAsync+池纪律:对我们计时口径内 alloc 抖动的根治;成本=全局替换 dev_alloc。

## 5. 待跑验证(post-refresh,按序)

- [ ] 全量 Ocean 列刷新完成后:geomean 含/不含 symbolic 双口径(docs/20/23/24/25 表更新)。
- [ ] 路由分布表:跑一遍 `ocean/spgemm` 抓 stats.json 的 `decision.workflow` + `binning_1/2`
      (run_ocean 现已丢弃该字段;可加 `--capture-routing` 小脚本,或 CPU 精算 avg_product
      定 ana1 + 采样定 ana2)。产出:ocean337 四工作流占比 + 14 bin 行分布热图。
- [ ] localLoadBalance 移植 A/B(Ge99H100/c-58/bloweya,profile_top_losers.py 三坑注意)。
- [ ] 方案5 原型:精确计数 pass(先只对 hash bin 行)+ sortOutputFused 直写,pre2/鲸鱼验收。

## 6. 源码坐标速查

- 主干/四工作流:`ocean/kernels/SpGEMM.cuh:208`(run)、`:325`(prologue)、`:420`(analysis)、
  `:524`(sample)、`:667`(est)、`:734`(symbolic)、`:900`(fat)、`:1027`(ultrasparse)、
  `:1184`(precise)、`:1370`(est numeric)
- 分流判据:`Analysis.cuh:10`(decideMatrixType);阈值:`include/Utils.h:22-53`
- binning 双维:`Analysis.cuh:110`(binning)、`:214`(numericBinning,MODIFY_DENSE_KERNEL_EST_VALUE)
- 负载均衡:`AccumulatorCommon.cuh:67`(localLoadBalance);全局 value 池 `:8-63`(bitmap 取还)
- hash 内核:`AccumulatorHash.cuh:437`(compactAndSort O(n²) rank sort)、`:487`(compactOnly)、
  `:510`(hashNumericKernel:overflow 记录+min(flop,n) 尾区预留)
- dense iter 游标:`AccumulatorDense.cuh:695-750`(start_map atomicMin 窗口推进 + QUERY_BITMAP)
- epilogue:`Epilogue.cuh:68`(sortOutputDyn)、`:127`(sortOutputFused 融合键)
- stream/事件栅栏:`SpGEMM.cuh:179/190`(syncMainToStreams/syncStreamsToMain);
  bin→stream 映射:`Wrappers.cuh`(hash bins→streams[0..6],dense bins→streams[7..12])
