# HSMU-SpGEMM: High Shared Memory Utilization
- HPCA 2025; Wu et al., U. Hunan + Jeff Zhang
- 论文:IEEE HPCA 2025, pp. 1452–1466
- 代码:https://github.com/wuminqaq/HSMU-SpGEMM

## 核心

传统 hash 累加器在 shared memory 利用率和 hash 碰撞之间有 trade-off → shared mem 利用率低。HSMU 用**新累加器设计**(非 hash)来最大化 shared mem 利用率,并配套新的 symbolic stage。

## 要点(从 GitHub README + 引用)

- **新累加器**:不用 hash,改用 **binary search on pre-generated column-index array**。消除了 hash 的 load factor,shared mem 利用率提升 ~1.5×。
- **symbolic stage 改造**:为这个累加器设计了专门的 symbolic stage(生成 column-index array)。
- 两阶段(symbolic + numeric),同 spECK/opSparse 框架。
- 代码:支持 .mtx 输入,提供 18 代表矩阵 + 338 矩阵全集测试脚本。
- 在 Ocean(ICS'26)的对比中:HSMU 在 A100 上 0/337 best,avg 32.1 GFLOPS(比 spECK 46.2 低)。

## 对你的启示

- **非 hash 的 shared-mem 累加器**思路(binary search on column array)是一个替代方案——避开 hash 碰撞,充分利用 shared mem。
- 但 Ocean 的对比显示它不如 spECK(hash-based)和 Ocean 本身——说明 **hash 累加器目前仍是 GPU SpGEMM 的性能 sweet spot**。
- 代码可读(shared-mem accumulator 实现细节),做方向 D(融合单 kernel)时可参考。
- **对应你的方向 D**(融合单 kernel / shared-mem 累加器)。

## 代码结构(关键)

```
evaluation/
  18matries/           # 18 代表矩阵
  338MatrixSet/        # 全集下载脚本
  script/              # 编译+测试脚本
    make               # → 生成 test 可执行
    ./test <path>      # 运行
    test_threshold_matrix.sh    # Figure 6
    test338matrices.sh          # Table 3,4 + Fig 9
    test_peak_memory.sh         # Fig 11
```

每个矩阵输出:加载时间、格式转换时间、各阶段耗时(对应 Fig 13)、中间项数、C nnz、压缩比、GFLOPS。

## 引用

```
@inproceedings{wu2025hsmu,
  title={HSMU-SpGEMM: ...},
  booktitle={HPCA 2025}, pages={1452--1466}, year={2025}
}
```
