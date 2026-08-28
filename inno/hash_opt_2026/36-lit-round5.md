# 文献轮 5:多行/CTA 打包 + SMEM 游标窗口(实施级细节)

日期:2026-08-28。方法:纯 WebSearch/WebFetch/webReader,深度阅读 21 个来源(另 5 个仅搜索摘要级,已标注)。范围:A 线(轻行×小块打包、SMEM 直接寻址累加器布局)、B 线(SMEM 游标窗口内核先例、SMEM vs 全局原子开销)、C 线(低优先,launch invalid argument 案例)。已档 docs/12/17/28/33 不重复。

---

## A 线:多行打包进一个 CTA(轻行 × 小块)

### A1. spECK Block Merge 算法 —— "多行打包"的最直接可抄实现 ⭐⭐⭐

**出处**:Parger, Winter, Mlakar, Steinberger. "spECK: Accelerating GPU Sparse Matrix-Matrix Multiplication through Lightweight Analysis." PPoPP'20. DOI 10.1145/3332466.3374521。全文已读(TU Graz PDF);代码 github.com/GPUPeople/spECK。

**机制细节**(原文级):
- 6 档 kernel 配置:最大档 48KB SMEM + 1024 线程(Titan V),逐档 SMEM 与线程数减半;第 6 档用 96KB(双倍 SMEM,占用减半)。artifact 里写明 `spECK_STATIC_MEM_PER_BLOCK=49152`、DYNAMIC: pre-Volta 49152 / Turing 65536 / Volta 98304。
- **最小 bin 的块合并(Algorithm 2,逐字)**:
  ```
  for i ← 0 to 5:
    k ← 0; step ← 2^i
    while k ≤ n:
      if b[k] + b[k+step] < mem_min:
        b[k+step] ← b[k] + b[k+step]
      k ← k + 2*step
  ```
  6 轮邻接合并(哈希累加器最多 **32 行/块**),while 体在 block 内并行(prefix-sum 式)。只合并相邻块(保持 CSR 行序)。**最坏保证利用率 ≥50%**:两邻块若不能合并,则各自平均利用率必 >50%。
- 局部负载 g(每行 B 分到的线程数)启发式:g 从"块内引用行平均行长"起步;`iter_max = elements_max / g`,`n_rows = NNZ_A / k`;若 `iter_max > 2*n_rows` 则 `g_new = g * iter_max/(2*n_rows)`;若 `n_rows > 2*iter_max` 则 `g_new = g * iter_max/n_rows`;g round 到 2 的幂。**动态 g 比固定 32 最多快 8×**(nsparse 固定 32 线程/行,在 stat96v2 上利用率仅 9%)。
- 全局 LB 开/关阈值(自动调优结果,m_max/m_avg 与 rows_C):symbolic 39.2 / 28000(大 kernel 档 6.0 / 5431);numeric 10.5 / 23006(大档 1.3 / 1238)。小阵/均匀阵不开 LB,省 2×。

**能直接抄的**:① Algorithm 2 原样搬,把 `mem_min` 语义改成我们的 span 预算;② 32 行/CTA 上限与我们"小块 64-128 线程 × 32 块/SM"兼容;③ g 的 2× 失衡修正公式可直接用于子 bin 内线程分配;④ "m_max/m_avg 低于阈值且行数少就不做 binning"的门,与我们 v4 门控同型。

**置信度:高**(全文 + 开源代码)。

### A2. Greb & Daga SC'14 CSR-adaptive —— 行块打包的调参范围与分流规则 ⭐⭐

**出处**:"Efficient Sparse Matrix-Vector Multiplication on GPUs Using the CSR Storage Format." SC'14(computermachines.org PDF,已读)。

**机制细节**:行块(rowblock)= 连续行打包到目标字节数;`rpBlock`(行/块)在 1–1024 内二分搜索按矩阵调优(以平均行长估目标块字节);实测有效值 **8 / 16 / 32 行/块**,矩阵相关,选错损失 **1.2–2×**;块含 1–2 行 → CSR-Vector,含更多行 → CSR-Stream(整块流式,SMEM 缓存 x 向量);尾块 = 剩余行(等大小切块的最后一块)。spECK 引它作同类 bin-packing 的"串行 CPU 预处理"反例(我们要 GPU 端,取 spECK 的并行版即可)。

**能直接抄的**:rpBlock 的搜索流程 = 我们"span 档 × 行数"联合选择的现成调参脚本结构;"1–2 行/块走 vector kernel、更多行走 stream"对应"1 行/CTA 大 span vs 多行小块"分流;尾块处理模式。

**置信度:高**。

### A3. Bell & Garland SC'09 —— 轻行密度门槛公式 + persistent warp ⭐⭐

**出处**:"Implementing Sparse Matrix-Vector Multiplication on Throughput-Oriented Processors." SC'09(NVIDIA PDF,全文已读)。

**机制细节**:CSR-vector(1 warp/行)行长 <32 nnz 时利用率骤降;行数 < 最大共存 warp 数(GTX285 为 960)也不满。HYB 启发式(原文):假设 ELL≈3× COO 速度,则**当且仅当 nnz ≥ K 的行数 ≥ max(4096, M/3) 时才值得给 ELL 加第 K 列**。persistent warp-oriented 风格:固定 W 个 warp 各处理 O(M/W) 行;COO 段规约的 carry-out 直接作为下一段 carry-in,不落 DRAM。Baskaran/Bordawekar 变体:half-warp(16 线程)/行 + 行 pad 到 16 的倍数保对齐。

**能直接抄的**:① 密度门槛公式改造为"span∈{s} 档行数 ≥ max(4096, M/3) 才开该子 bin"(直接给我们 Phase A 子 bin 的合法性判据);② persistent warp 的寄存器 carry-over 与 B 线游标窗口同构;③ 短行 pad 对齐技巧。

**置信度:高**。

### A4. SELL-C-σ —— 行/块参数 C 的跨架构定值 ⭐

**出处**:Kreutzer et al., arXiv 1307.6209(ar5iv 已读)。

**机制细节**:chunk 高 C:**GPU 推荐 C=32(warp size, K20 实测)**;AVX C=4;MIC C=16;异构统一取 max。短行零填充到 chunk 内最长;长行拆到多个 chunk;总行数 pad 到 C 倍数。σ = 行长降序排序的窗口(1≤σ≤2^17),σ=C² 时最坏矩阵 chunk 占用 β≈1;性能模型 P̄ = b·β/6 bytes/flop。

**能直接抄的**:span 档内"短行 pad、长行跨 bin 拆"+ σ 窗口排序平滑块内不均(与我们的行合并兼容,可挂在 Phase A 前处理)。

**置信度:高**。

### A5. nsparse —— 轻行 partial-warp 档 + SMEM hash 表数组布局(源码级)⭐⭐

**出处**:Nagasaka, Nukada, Matsuoka. ICPP'17;代码 github.com/EBD-CREST/nsparse(`cuda-c/src/kernel/kernel_spgemm_hash_template.cu`、`spgemm_hash_kernel_gen.c`,两文件均已读)。

**机制细节**(源码引用):
- 三类 bin:**GTR**(hash 表在全局,最长行)、**TR**(SMEM hash 表/块,中档)、**PWR**(partial-warp,**PWARP=4 线程/行**,最短行)——"轻行×小块"的最小档就是 4 线程/行。
- SMEM 表 = 两个平行数组:`__shared__ int check[IMB_PW_SH_SIZE]`(key,初始 -1)+ `real shared_value[B_PW_SH_SIZE]`(值,初始 0)。**表容量 = 48KB / (sizeof(int)+sizeof(real))**(double 时 4096 项);`B_SH_SIZE = table_size × (blockDim/32)`。
- hash:`(bcol * HASH_SCAL) & (SH_ROW-1)`(乘大常数 + 2 幂掩码),线性探测;**每处理一行显式清 `check[]` 为 -1,没有 generation/pref 戳**;数值累加 `atomic_fadd(shared_value + adr, aval*bval)`。
- 行-线程映射:PWR `rid=i/PWARP; tid=i%PWARP`;TR:1 warp/行;TB 档:块内 `wnum` 个 warp 轮转共享一行(`for j = arpt[rid]+wid; j < arpt[rid+1]; j += wnum`)。
- 生成器里的硬件参数:`shared_tb=48KB, shared_sm=64KB, tb_max=1024, tb_sm=2048, max_tb_num_sm=32`(→ 32 块/SM 与我们 Ocean 对齐口径一致)。

**能直接抄的**:① PWARP=4 作为最轻档(对应我们 span=256 档的线程下限);② "每表项成本 = 4B key + 8B val" 的 SMEM 预算算式;③ check/value 分离双数组布局(我们的 dense 直接寻址累加器 val+flag+pref 同理分立);④ 若 dense 累加器 flag 改 1bit/entry 的原子 bitmask(spECK symbolic 500k entries/2.4 万 hash entries,32× 密度优势),SMEM 能换 32× 窗口宽度。

**置信度:高**(源码)。**诚实标注**:nsparse 无 pref 戳(它显式清表);"val+flag+pref 三数组交错 vs 分离对 bank conflict 的量化对比"公开文献未查到(见 A7 空白)。

### A6. GNNAdvisor OSDI'21 —— 行组打包 + 小块的量化调参(SpMM,稠密 B)⭐⭐

**出处**:Wang et al., OSDI'21(USENIX PDF 全文已读)。

**机制细节**:2D 管理 = 行向 neighbor group(参数 ngs)+ 列向 dimension worker(dw);每个 group 恰好一个 warp(warp-aligned mapping);SMEM = (tpb/32)×Dim×4B,约束 WPT≈1024、SMEM≤48–96KB。量化:ngs 扫 1→512,**最佳 ≈32(artist 数据集,过阈值后变差)**;dw 16 vs 32 差别甚小;**小块(1–4 warps,32≤tpb≤128)提高 warp 调度灵活度、避免 tail effect,占用与吞吐更高**(引 Yang/Buluç/Owens Euro-Par'18 设计原则);块级优化(warp 对齐 + SMEM 部分和)**平均减原子操作 47.85%、DRAM 访问 57.93%**;leader-warp 独享 SMEM→全局冲刷。重排序判据 √AES > ⌊√N/100⌋。

**能直接抄的**:① 32 项/行组 ≈ 我们 span 512–1024 档行组大小的经验起点;② "小块 32–128 线程"的实证背书(正好接 Ocean 32 块/SM 口径);③ leader-warp 冲刷模式:多行部分和驻 SMEM、单 warp 聚合写回,直接可用于我们窗口写回阶段压原子。

**置信度:高**。

### A7. SMEM 直接寻址累加器布局与 bank conflict(部分证据,有空白)⭐

已确认证据:
- **Hopper 微基准**(arXiv 2402.13499,已读):SMEM load 延迟 **29–30 cycles**(H800/A100/RTX4090),吞吐 ~128 B/clk/SM;DSM(SM-to-SM)180 cycles,比 L2 低 32%。无原子专项数据。
- **RTX4090 SMEM 原子微基准**(girl.surgery 博客,已读):int32 atomicAdd 无争用 **<10 cycles**(ATOMS.POPC.INC);争用拐点 int32 在 32 线程同址、float32 在 4 线程;重争用下 float32 比 int32 慢 ~50×;float-max 用 int32 重编码比 CAS 循环快 ~10×。
- **SMEM 原子队列模型**(arXiv 2503.17893,已读摘要+HTML):SMEM 原子 = 单服务器队列,服务时间 S(n,e,c) 随 warp 数 n、active lanes e(bank/同址争用)、CAS 混比 c 变化;**n>32 后吞吐饱和**;Volta 64 / Ampere 48 warps/SM;直方图两内核 30% 差异可由模型解释。
- **histogram padding 反例**(foxboxxx/fastest-histogram-cuda,搜索摘要级,未入内页):对 SMEM histogram 做 padding 消 bank conflict "测得无改善" —— 冲突由数据分布驱动时布局 padding 无用。

**能直接抄的(推理级)**:① 窗口推进的游标/计数一律 int32 原子(无争用 <10 cyc,32 线程同址才到拐点);② float 原子只做最终累加且避免同址并发(或改 warp 私有 + leader 冲刷);③ val(double)+flag+pref 若交错为 struct-of-3,bank 上表现为 3 路跨步,无公开量化 —— **需要我们自己 A/B,这是本轮诚实标注的空白**。

**置信度**:数字中–高(2402.13499、2503.17893 为论文;girl.surgery 为个人博客,方法可信但非同行评审)。

### A8. AC-SpGEMM PPoPP'19 —— 不看行边界的均匀切块 + 倒序消费 ⭐⭐

**出处**:Winter, Mlakara, Zayer, Seidel, Steinberger. PPoPP'19(MPI-INF PDF 已读)。

**机制细节**:全局 LB = **按 A 的 nnz 均匀切**(Algorithm 1:每块固定 NNZ_PER_BLOCK,写 `blockRowStarts[]` 辅数组),不看行边界,准备成本 O(1)/块。局部 work distribution:`placework/size/receivework` 三原语 —— 对每个 A 元素的积数做块内 prefix sum,`receivework(N)` 每线程领 N=8 个临时积,**从行尾倒序消费**,被切开的 B 行"当作更短的行"进入下一轮本地 ESC;~4000 临时元素/块常驻 SMEM,多轮 ESC 直到产出完整行 chunk 再落全局。负载均衡在极稀疏阵可占 hash 方案(nsparse)至 30% runtime(转引 Nagasaka)。

**能直接抄的**:① "按 nnz 而非行数切块" 与我们 span 窗口调度同构,可作为子 bin 内的第二级均衡;② "倒序消费被切开的行" 是游标窗口的重要变体:窗口右端切中某 B 行时,只消费到窗口内部分,剩余段游标留给下一窗口,天然免重扫。

**置信度:高**。

### A 线补遗:HR-SpMM(ICS'25)/ RSH-SpMM(arXiv 2603.08734)
- HR-SpMM:长短行二分,**<64 nnz → CUDA core(subwarp tiling + residue unroll),≥64 → Tensor Core;长行段上限 256 nnz,超长行拆段后用 atomic add 合并**;预处理 O(N) 仅一个辅助数组(成本 ≈ Sputnik 的 12%)。平均 2.05× vs cuSPARSE。可抄:64/256 的段上限经验值、拆段+原子合并的收尾模式。置信度高(PDF 已读)。
- RSH-SpMM:行按 τnnz/τinc 双阈值分类进 TC 窗口(W 行)或 CUDA residual;阈值 4–6 后收益饱和(自限制在小平台区)。可抄:双阈值(行长 × 增量重叠)分类思想,与我们"flop ≥ span/16 + dup<8"门同型。置信度中–高(HTML 已读,rows/CTA 未明说)。

---

## B 线:SMEM 游标窗口内核

### B1. spECK dense 累加器的列窗口迭代 —— 与 denseNumericIterKernel 同语义的直接先例 ⭐⭐⭐

**出处**:同 A1(PPoPP'20 §4.3 "Dense Rows of C")。原文(逐字):
> "If the range from minimum to maximum column index in the resulting row does not fit in scratchpad memory, the dense accumulator needs multiple iterations on different column ranges, successively progressing through the output row... **To efficiently advance through the rows in B, we store the positions of the last element that could be processed in the current iteration for each row.**"

**机制细节**:每个 B 行一个游标(本迭代处理到的最后位置);窗口 = [start_offset, start_offset+span);symbolic 用原子 bitmask(一窗口 50 万列),numeric 每窗口 prefix-sum 紧凑后把部分行写回 C,重置累加器、推进 offset。门:density >18% 且窗口数 ≤3 才走 dense;dense vs hash-only >60% 提升;行超最大 SMEM hash 时最高 40×(208bit 矩阵)。

**能直接抄的**:与我们 denseNumericIterKernel(start_map 游标跨窗口)完全同构,可作为论文里的先例引用;**差异点也明确**:spECK 单行 × 全 SMEM 窗口,我们多行 × 子 bin span{256,512,1024,2048} × 小块 —— 多行打包维度是增量;它的"每窗口 prefix-sum 紧凑写回 + 重置"循环体可照搬;18%/≤3 窗口阈值是我们 v4 门的对照锚点。

**置信度:高**(原文引用)。

### B2. Merge path —— 全局/CTA 两级游标的经典实现 ⭐⭐

**出处**:dumerrill/merge-spmv(README 已读 + SC16 preprint 在仓库)+ arXiv 2404.06047 综述。思想:CSR row-offsets 与自然数序列做"逻辑合并",合并路径按 CTA 均分;每 CTA 起点 = 对角二分搜索(2D 网格上找决策路径穿越点);CTA 内两指针合并,游标(running row-id / running nnz-id)驻寄存器逐元素推进;向下走 = 累加点积,向右走 = flush 该行;跨 CTA 的行由 fix-up(reduce-by-key)合并。**诚实标注**:preprint PDF 未能文本化,伪码细节来自 README 与综述二手描述,未逐行核实。

**能直接抄的**:① 窗口游标"每线程寄存器 + 两指针推进"的实现模式;② 跨窗口部分行 = merge 的跨界行,fix-up 归约对应我们多窗口写回同一输出行的拼接逻辑。

**置信度:中–高**(机制高,伪码级二手)。

### B3. Stream-K —— 跨 tile 游标 + fixup 的当代范本 ⭐⭐

**出处**:arXiv 2301.03598(ar5iv 已读)。

**机制细节**:每 CTA 领 `iters_per_cta = ⌈total_iters/g⌉` 个 MAC 迭代;`while iter < iter_end: tile_idx = iter/iters_per_tile; ...; iter ← tile_iter_end`(游标跳到 tile 边界,部分 tile 用 `min(iter_end, tile_iter_end)`);拥有 k=0 迭代的 CTA 负责 fixup:`for cta: Wait(flags[cta]); accum += LoadPartials(partials[cta])`(reduction lock 模式)。量化:4-SM 例子利用率 75%(data-parallel 9 tile)→ 90%(更小 tile)→ **Stream-K ~100%**;32824 个 GEMM 实测平均 1.63× vs CUTLASS data-parallel(峰值 14.7×);**通信/同步/全局存储开销与问题规模无关,只随 CTA 数 g 缩放**。

**能直接抄的**:① 游标窗口调度骨架(tile 对齐 + 边界 min());② fixup 的 flags+partials 模式可用于 span 窗口间部分和拼接;③ "开销只随 CTA 数缩放"的论证句式可直接用于我们论文的开销分析。

**置信度:高**。

### B4. SMEM 原子 vs 全局原子:开销实测拼图 ⭐⭐

| 证据 | 数字 | 来源级别 |
|---|---|---|
| SMEM load 延迟(H800/A100/4090) | 29–30 cycles;吞吐 ~128B/clk/SM | 论文(arXiv 2402.13499) |
| SM-to-SM(DSM)vs L2 | DSM 180 cycles = L2 −32% → L2 ≈ 265 cycles | 同上 |
| int32 SMEM atomicAdd 无争用 | <10 cycles;拐点 32 线程同址 | 博客微基准(girl.surgery,4090) |
| float32 SMEM atomicAdd | 比 int32 慢 ~10×;重争用慢 50×;拐点 4 线程 | 同上 |
| SMEM vs 全局原子(Maxwell+) | SMEM ≈ 2× 全局(实测帖);全局原子在 L2 解析、fire-and-forget | NVIDIA 论坛(Robert Crovella) |
| 反例 | 老帖测 SMEM atomicAdd 比全局慢 15–25%;Turing+ 问题帖无定论 | 论坛 |
| 饱和点 | SMEM 原子单服务器队列,n>32 warps 饱和;e(active lanes)↑ 服务时间↑ | 论文(arXiv 2503.17893) |

**结论(可直接用)**:① 游标推进/计数用 int32 原子;② SMEM 原子优势成立的前提是占用足(饱和前),占用不足时序列化反而输给全局原子 fire-and-forget —— 解释了我们此前"device pool / 全局原子"实验的摇摆;③ float 部分和别用原子,用 warp 私有累加 + leader 冲刷(GNNAdvisor:原子 −47.85%)。**查不到**:Hopper 同址 SMEM 原子吞吐专项公开微基准(2402.13499 无原子节;girl.surgery 是 Ada 架构)——诚实标注,需本地 5 分钟 bench 补。

### B5. 游标表的等价物:GNNAdvisor neighbor-group 元数据 ⭐

(id, target node, (start,end) CSR 区间)三元组 = 预计算游标表,均匀工作单元 —— 与我们 start_map 同构的成熟先例;证明"游标预计算 + 均匀单元"是标准做法而非我们的私设。置信度高。

---

## C 线(低优先):launch 返回 invalid argument 但参数表面合法

1. **动态 SMEM > 48KB 未 opt-in**(最常见实锤):launch 返回 invalid argument,需 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size)`;静态超限则在 ptxas 编译期报 "Entry function uses too much shared data"。来源:NVIDIA 论坛帖 251770(A100)、帖 317406(解决方案同)、Lei Mao 博客(sm_75 动态上限 64KB 等)。置信度高。
2. **SMEM 用尽但表象是"参数合法"**(论坛帖 71031):Robert_Crovella 的方法 = 把 `<<<gridSize, blockSize, sharedMemBytes>>>` 逐项对照 Programming Guide 限制表打印;用户 printf 后自答 "I used up the shared memory"。置信度高。
3. **错误源在别处、延迟上报**(论坛帖 188236,cudaMemset 11.4):cudaErrorInvalidValue 可能来自更早的 host 栈踩坏等,报错点 ≠ 出错点;用 cuda-memcheck / compute-sanitizer 定位。置信度高(机制)。
4. **未证实(诚实标注)**:用户提示的 "L1 carveout 冲突 / API 版本 / 驱动" 线索,本轮检索只找到 cudaFuncSetAttribute 在 set 阶段对非法枚举值返回 invalid value 的 API 语义文档,以及 vLLM/ROCm 侧 invalid argument issue,均非"launch 时参数表面合法却报错"的 carveout/驱动案例。若我们复现的是这种,优先自查 1/3 两类。

---

## 本轮没查到(空白清单)

1. val+flag+pref 三数组"交错 vs 分离"布局的 bank conflict 量化对比 —— 无公开数据;需本地 microbench(建议:同一 kernel 换布局跑 3 组)。
2. Hopper(H100)同址 SMEM 原子吞吐专项微基准 —— 只有 Ada(girl.surgery)与队列模型(arXiv 2503.17893,Volta/Ampere);H100 数字需本地补。
3. merge-spmv 的逐行伪码(preprint PDF 二进制未能文本化)—— 机制描述来自 README+综述。
4. carveout/驱动版本导致 launch invalid argument 的实锤案例。
5. Ocean denseNumericIterKernel 本身细节 —— 已档 docs/12,本轮未重查。

## 检索来源清单(深度阅读 21)

spECK PPoPP'20 PDF(tugraz);GPUPeople/spECK;EBD-CREST/nsparse(README + kernel_spgemm_hash_template.cu + spgemm_hash_kernel_gen.c + git tree);AC-SpGEMM PPoPP'19 PDF(MPI-INF);dumerrill/merge-spmv(README + preprint + git tree);Bell-Garland SC'09 PDF(NVIDIA);SELL-C-σ arXiv 1307.6209(ar5iv);Greb-Daga SC'14 PDF;SpMV survey arXiv 2404.06047;Stream-K arXiv 2301.03598(ar5iv);GNNAdvisor OSDI'21 PDF(USENIX);HR-SpMM ICS'25 PDF;RSH-SpMM arXiv 2603.08734;Hopper microbench arXiv 2402.13499;SMEM 原子队列模型 arXiv 2503.17893;girl.surgery/shmem_atomic;NVIDIA 论坛 71031 / 215755 / 220300;Lei Mao 博客(CUDA Shared Memory Capacity)。
搜索摘要级(未入内页):论坛帖 251770 / 317406 / 188236、Medium(ngocson2vn)、foxboxxx/fastest-histogram-cuda。
