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

## 3. H100/CUDA 12.8 移植清单(参考 nsparse/opSparse 配方)

1. **剥 CUSP**:仅 main.cu(cusp io/poisson)与 ref_spgemm.h(cusp multiply 参考校验)依赖;
   换成我们自己的 mtx reader / cuSPARSE 校验即可。
2. `__shfl`→`__shfl_sync(0xffffffff,…)` 等隐式 warp 同步修正(nsparse 同款坑)。
3. Makefile:`/usr/local/cuda/bin/nvcc`(12.8)、`-arch=sm_90`、`-std=c++17` 视情况。
4. volatile/scan/oddeven 老式 kernel 逐个过;先 `./spgemm -cuda -spgemm cage4.mtx` 冒烟。
5. 口径:计时取 STAGE1-4(含输出 C 拷贝),h2d/d2h 在 stage 外,与 compare compute-only 对齐
   同 opSparse 处理;输出 `SpGEMM time: X ms. Gflops = …` 好解析。
6. 接入 `compare_methods.py` 加 `bhSparse` 列(仿 run_opsparse;REFRESH=bh 重跑)。
