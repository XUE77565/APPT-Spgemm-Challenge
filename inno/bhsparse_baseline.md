# bhSparse = merge 家族 SpGEMM 的开源比对对象(调研 2026-08-24)

> 结论:**bhSparse CUDA 版(IPDPS'14 + JPDC'15 官方源码)是最佳 merge 比对基线**,已拉到
> `external_sota/bhSparse/`(github.com/weifengliu-ssslab/Benchmark_SpGEMM_using_CSR)。
> 默认 double、单参数跑 C=A·A(与挑战口径对齐)、分阶段计时输出(STAGE 1-4 + total)。

## 1. 为什么是它

- **文献地位**:就是我们 `merge_innovation.md` 定位的"最直接 GPU merge 前辈"(Liu & Vinter)。
- **算法齐**:CUDA 版 `bhsparse_cuda.h` 是个算法框架,按行长分桶调度多变体
  (`compute_nnzC_Ct_cuda`,bhsparse.h:483):
  - 短/中行:`_2heap_noncoalesced`(heap) / `_bitonic`(bitonic 排序);
  - **重行:`compute_nnzC_Ct_mergepath` = EM(Expand-Merge)变体**——
    `EM_mergepath_global`(bhsparse_cuda.h:2272):**一个 CTA 归并一行**,线程间用
    `mergepath_partition` 对角线切分、`mergepath_serialmerge` 各归并各的 9 项切片,
    迭代 2-way(累加器逐链并入),SMEM 一级(c_buffsize)+ global 二级缓冲,超长行
    经 `d_queue` 多轮 kernel 续算(2016-04 修过 long-row bug)。
- 对照盘点(为什么别的不行):nsparse 本地副本只有 hash 变体(C++ 版仅 SpMV,
  ICPP'17 论文的 memory-saving merge 版未开源);merge-spmm(owensgroup)是 SpMM 非 SpGEMM;
  MMSpGEMM 已就位但它计算是块内 ESC(见 `mmspgemm_comparison.md`);SpArch 是硬件模拟器。

## 2. ⚠ novelty 连带影响(比 MMSpGEMM 更早的行内并行 merge)

bhSparse 的 EM 变体对**重行**已经是"一个 CTA 多线程在行内做 merge-path 并行归并"
= **行内(rank 域)merge 并行 2014/15 就存在**,不是 MMSpGEMM 首创。
`merge_innovation.md` 的 claim 需在 MMSpGEMM 之后再收窄一层。我们仍然独有的差异:

| | bhSparse EM(重行) | MMSpGEMM(PACT'25) | 我们 merge3 |
|---|---|---|---|
| 行内切分 | merge-path 对角线(rank) | 等大小 2048 rank 块 | **列值域 K 桶** |
| 归并方式 | **迭代 2-way**(链逐条并入,缓冲区反复读写) | 不归并(块内 radix sort) | **单遍 k-way warp merge**(每项只碰一次) |
| 跨切分重复 | 边界处理 | carry + atomic | **结构性不存在** |
| 结构 | 按行长分桶选变体 | 离线两阶段(split 写盘) | 在线单遍,dispatcher 整阵选路 |

差异化主句(建议):"已有行内并行 merge 均沿 **rank 域**(merge-path 对角线: bhSparse EM、
MMSpGEMM),且或为迭代 2-way(bhSparse)或退化为块内排序(MMSpGEMM);本文提出**列值域**切分,
同列必同桶、免跨块 carry,桶内**单遍 k-way warp 归并**免排序免重复数据移动。"

## 4. "有没有更近的 merge 开源?"——调查结论(2026-08-24):**没有**

真·执行 merge 的开源 GPU SpGEMM,**最近的仍是 bhSparse(2014/15)**。逐个排查:

| 候选 | 年份 | 实际算法(代码实锤) | 是 merge? |
|---|---|---|---|
| bhSparse EM | IPDPS'14/JPDC'15 | 迭代 2-way merge-path 归并 | ✅ 真执行 |
| nsparse | ICPP'17 | 论文有 memory-saving merge 版,**未开源**(repo 只有 hash) | ❌ |
| TileSpGEMM | PPoPP'22 | 官方源码已拉 `external_sota/TileSpGEMM`:主内核 = **nsparse hash 表**,按 tile 密度四档(TNY32/SML48/LRG160/DNS256)选累加器,grep 无 merge | ❌(密度自适应 hash/dense) |
| tSparse | 2020 | Tensor Core 化 SpGEMM | ❌ |
| ACSpGEMM | PPoPP'19(GPUPeople/ACSpGEMM 开源) | 多轮 ESC(sort) | ❌ |
| MMSpGEMM | PACT'25 | merge-path 只用于**分区**,计算=块内 radix sort | ◐(思想复兴,非执行) |
| Wang TACO'22(4) | 2025 | SMEM hash + ML 选 sizing 估计器 | ❌(但与 dispatcher 叙事相关,可引) |
| GAMMA merger | 2021 | **Verilog 硬件** merge 组件(GitHub 唯一 "spgemm merge" 活跃 repo) | ❌(非 GPU) |

GitHub 全量扫(2026-08)近期 SpGEMM repo:sketch-spgemm(2026,Rust)、LeSpGEMM/SparseOps(2026,
locality)、HSMU/Ocean(已有)、DeltaSparse(HiPC'23 多 GPU)——**全是 hash/sketch/sort 路线,无一 merge**。

**论文价值**:这个空白本身是叙事素材——GPU 上 merge 家族自 bhSparse(2015)后休眠十年,
2025 年 MMSpGEMM 重拾 merge 思想时计算阶段已退化为块内排序;**值域切分 + 单遍 k-way warp 归并
是十年来第一次让"真 merge 累加"回到 GPU SpGEMM 竞争序列**(比较对象自然就是两代:bhSparse
历史代表 + MMSpGEMM 现代代表)。

### 4.1 第二轮跨站补查(2026-08-24,GitHub 多关键词×8 / DBLP 200 篇 / Semantic Scholar / arXiv / GraphBLAST 源码)

| 新查证对象 | 结论 |
|---|---|
| **AiSpGEMM**(DATE'25) | **FPGA**,"Intra-row Parallel Merging"——行内并行 merge 在**硬件线**延续(SpArch→MatRaptor→GAMMA→AiSpGEMM);未找到公开代码;非 GPU,related work 引 |
| **SaSpGEMM**(ICPP'24) | **多核 CPU**,链表累加器(sorted insertion)免排序——2024 年"免排序保序"在 CPU 上仍有人做;非 GPU |
| **IA-SpGEMM**(PPoPP'19 系) | NN 选格式+算法的 auto-tuning(CPU/GPU,COO/DIA/ELL,TF1.4)——**与 dispatcher 叙事直接相关,必引**;非 merge |
| **GraphBLAST**(GPU GraphBLAS) | 读源码定案:标准 mxm **直接包 cuSPARSE**(`cusparseXcsrgemm2Nnz`+`Scsrgemm2`),自定义 kernel 仅 masked 点积(二分+warp 归约)——参考库自己都不实现 SpGEMM |
| bhSPARSE 库 repo | archived,同为 2015 时代,无更新 merge |
| GitHub "gustavson"/"spmspm" 扫描 | 仅 CPU MPI/OpenMP、VHDL、PIM、hash 动态调度——无 GPU merge |
| CSUR'23 系统综述 | "A Systematic Survey of General SpGEMM"(ACM Computing Surveys 2023, 10.1145/3571157)——**taxonomy 必引**,其 merge 分类亦止于 bhSparse/SpArch 一线 |

**结论不变且更强:真·执行 merge 的开源 GPU SpGEMM,最近仍是 bhSparse(2014/15)。**
行内并行 merge 的"近年动作"全部发生在硬件线(FPGA/ASIC)或 CPU,恰印证 GPU merge 空白。

## 3. H100/CUDA 12.8 移植 —— ✅ 已完成(2026-08-24,编译干净+正确性全过)

实际改动(全在 `external_sota/bhSparse/SpGEMM_cuda/`,加/改 6 个文件):
1. **剥 CUSP**:新 `mmread.h`(自写 MatrixMarket reader:general/symmetric/skew + real/pattern/complex取实部,行内排序+重复项求和);重写 `ref_spgemm.h`(cusp::multiply → **串行 host Gustavson 参考**,flops>3e8 自动跳过;csr_sort_indices 保留);重写 `main.cu`(去 cusp/gaussian poisson;**去掉原版"随机数覆写矩阵值"**,保留文件值可对拍;poisson 选项 1-4 报错提示用 .mtx)。
2. **helper 头替换**:新 `cutil_compat.h`(checkCudaErrors + StopWatchInterface/sdk*Timer chrono shim,**签名用原 cutil 的 `**` 风格**);`common.h` 换 include。
3. **shfl 修正**:bhsparse_cuda.h 4 处 `__shfl_up` → `__shfl_up_sync(0xffffffff,…)`(全满 warp scan,安全)。
4. **Makefile**:`/usr/local/cuda/bin/nvcc -O3 -m64 -std=c++14 -arch=sm_90`(c++14 兼容 2014 代码)。

**正确性(全部对串行 host 参考逐值校验)**:内置 4×6 toy ✓(nnzC=6);cage4 ✓(81);**bcsstk30 ✓ nnzC=8946070 与本项目 7 月管线(cuSPARSE/hash/merge3 三方一致)精确吻合**;333SP(371万阶/2222万nnz,pattern+symmetric 路径)✓ nnzC=71961733。
**计时输出**:`STAGE 1..4 time` + `[ CUDA ] SpGEMM time: X ms. Gflops = Y` + `nnzC = N`。口径 = 4 stage host 计时(含 stage 3 的 Ct 重新分配轮次,bcsstk30 出现 nnzCt_new 两次),输入 h2d / 输出 d2h 在计时外;对照参考(compute-only, double):bcsstk30 bhSparse 8.12ms(S1 0.088/S2 0.477/S3 7.27/S4 0.25)。
**用法**:`./spgemm -cuda -spgemm <A.mtx>`(B=A);`BH_CHECK=0` 跳过参考校验。
**⚠ 待办**:first100 矩阵集不在本机(`data/` 只剩 `data/ocean/square/` 337 个 Ocean 基准阵,其中含 bcsstk30/35/36/39、cage13-15、333SP 等)——要出 compare 列须先恢复矩阵集(在 186/187 或重新下载)。

1. **剥 CUSP**:仅 main.cu(cusp io/poisson)与 ref_spgemm.h(cusp multiply 参考校验)依赖;
   换成我们自己的 mtx reader / cuSPARSE 校验即可。
2. `__shfl`→`__shfl_sync(0xffffffff,…)` 等隐式 warp 同步修正(nsparse 同款坑)。
3. Makefile:`/usr/local/cuda/bin/nvcc`(12.8)、`-arch=sm_90`、`-std=c++17` 视情况。
4. volatile/scan/oddeven 老式 kernel 逐个过;先 `./spgemm -cuda -spgemm cage4.mtx` 冒烟。
5. 口径:计时取 STAGE1-4(含输出 C 拷贝),h2d/d2h 在 stage 外,与 compare compute-only 对齐
   同 opSparse 处理;输出 `SpGEMM time: X ms. Gflops = …` 好解析。
6. 接入 `compare_methods.py` 加 `bhSparse` 列(仿 run_opsparse;REFRESH=bh 重跑)。
