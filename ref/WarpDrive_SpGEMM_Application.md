# WarpDrive 方法论迁移到本项目 SpGEMM

> 本文档基于精读 [WarpDrive(IEEE HPCA 2025,蚂蚁集团)](ref/WarpDrive_GPU-Based_Fully_Homomorphic_Encryption_Acceleration_Leveraging_Tensor_and_CUDA_Cores.pdf) 后,把它的 **GPU kernel 优化方法论**迁移到本项目的 [src/spgemm_kernel_manual.cu](src/spgemm_kernel_manual.cu),给出具体可操作的优化建议。
>
> WarpDrive 本身是**全同态加密(FHE)**的 GPU 加速,与 SpGEMM 不同领域。本文档抽取的是其通用 GPU 优化思想。

---

## 0. 大前提:能搬什么,不能搬什么

| 类别 | 内容 | 能否迁移到 SpGEMM |
|------|------|------------------|
| **算法** | NTT、CKKS、Montgomery/Barrett 模归约、bootstrapping | ❌ FHE 特有,不能搬 |
| **方法论** | kernel 融合、并行维度扩展、warp 级任务划分、Tensor+CUDA 并发 | ✅ 通用 GPU 优化,可搬 |

> 一句话:**WarpDrive 的算法搬不过来,但它的"融合、复用、别重算、别串行"这几条 GPU 优化铁律正好命中本项目代码的几个低效点。**

---

## 1. 当前代码诊断(从 WarpDrive 视角)

本项目有两个 SpGEMM kernel,各自的问题:

### 1.1 `spgemm_transpose_product_manual`(A×Aᵀ)

- [count_full_nnz_kernel](src/spgemm_kernel_manual.cu#L45) 用**暴力内积**:每个线程处理一行 i,对所有 j∈[0,N) 算 `row_i · row_j`。
  - **O(N²) 个点积**,每个点积 O(nnz),完全没利用稀疏性。N=10000 就是 1 亿次点积。
- [fill_full_result_kernel](src/spgemm_kernel_manual.cu#L73) **把同样的 N² 个点积又算了一遍**(只是这次把值填进去)→ **算力直接 2× 浪费**。

### 1.2 `spgemm_self_product_manual`(A×A)

- [count_self_nnz_hash_kernel](src/spgemm_kernel_manual.cu#L247) 用共享内存哈希累加,一个 block 处理一行 —— **方向对(这其实是 Gustavson row-wise)**。
- [fill_self_result_hash_kernel](src/spgemm_kernel_manual.cu#L298) **又把哈希累加全过程重做一遍**(累加循环与 count 一模一样)。
- 收集阶段 **thread 0 串行扫描整个 4096 哈希表**([L341–355](src/spgemm_kernel_manual.cu#L341)),排序用 **O(n²) 冒泡**([L359–372](src/spgemm_kernel_manual.cu#L359),代码注释自承"简化为冒泡")。

---

## 2. WarpDrive → SpGEMM 迁移映射

| WarpDrive 的思想 | 它解决什么 | 本项目命中点 |
|-----------------|-----------|-------------|
| **① Kernel 融合**(5 个 NTT kernel → 1 个 warp 级) | 消除 kernel 切换、显存中转、重复计算 | count/fill **重算两遍** |
| **③ PE Kernel**(单 kernel 处理大粒度、扩展并行维度) | 提高并行度、数据复用 | A×Aᵀ 的"一线程一行" O(N²) 暴力 |
| **① 的细节**(warp 级 + 寄存器/共享内存精细划分) | 避免 thread 0 串行瓶颈 | fill 阶段串行收集 + 冒泡排序 |
| **② Tensor+CUDA 并发**(4+4 warp) | 两类核心同时跑 | 稠密块用 Tensor Core,稀疏部分用 CUDA Core |
| **① 更深分解让数据进共享内存** | 中间数据落片上 | 长行中间乘积分块进共享内存(≈AC-SpGEMM chunk) |

---

## 3. 三个推荐迁移(按性价比排序)

### 🔴 迁移 1:消除 count/fill 重算(对应 WarpDrive "kernel 融合")

**问题**:count 和 fill 把点积/哈希累加**完整算了两遍** —— 这正是 WarpDrive 最痛恨的"重复计算 + 显存中转"。

**改法**(三选一,改动量从小到大):

1. **缓存中间结果**(最小改动):count 阶段把每行累加完的哈希表(或压缩后的列号+值)写到一块预分配的 global memory 缓存,fill 阶段直接读缓存填 C,**不再重算**。用空间换 2× 算力。
2. **上界法单阶段**(WarpDrive 式彻底融合):count 阶段不精确数 nnz,而用上界 `U_i = Σ_{A[i,k]≠0} nnz(A[k,:])` 一次过分配 C,然后**单 kernel 一遍算完**数值,最后紧凑化压缩。整个 SpGEMM 只算一遍。
3. **两阶段但共享 kernel 模板**:至少把 [count_self_nnz_hash_kernel](src/spgemm_kernel_manual.cu#L247) 和 [fill_self_result_hash_kernel](src/spgemm_kernel_manual.cu#L298) 的累加逻辑抽成同一个 `__device__` 函数,避免维护两份相同代码。

> ⚠️ 诚实说明:WarpDrive 的 NTT 输出大小已知,所以能彻底融合成单 kernel;**SpGEMM 输出 nnz 未知,经典两阶段有"先 count 再分配"的鸡生蛋问题**,无法像 NTT 那样完全单 kernel。但"缓存中间结果"或"上界法"能消除重算,这是能拿到的对应收益。

### 🟠 迁移 2:A×Aᵀ 从暴力内积改成 Gustavson(对应 WarpDrive "PE Kernel")

**问题**:[count_full_nnz_kernel](src/spgemm_kernel_manual.cu#L45) 的 O(N²) 暴力内积是性能黑洞。

**改法**:学**已经写对的** A×A kernel 的思路 —— Gustavson row-wise product。A×Aᵀ 的第 i 行:
```
C[i,:] = Σ_k A[i,k] · A[k,:]      (Aᵀ 的列 = A 的行)
```
即对 A 第 i 行每个非零 `A[i,k]`,取 A 第 k 行,缩放累加。这和 A×A kernel 几乎一样(只是 B 换成 A 本身),复用 A 的行、避免 N² 暴力。WarpDrive 的 PE Kernel 思想就是"**扩展并行维度、一个 kernel 处理更大粒度来复用数据**"。

### 🟡 迁移 3:fill kernel 的并行收集 + 真排序(对应 WarpDrive "warp 级并行")

**问题**:[fill_self_result_hash_kernel](src/spgemm_kernel_manual.cu#L341) 的 thread 0 串行收集 + 冒泡排序,一个 block 里 255 个线程干等。

**改法**(WarpDrive 的 warp 级思想):
- **并行收集**(stream compaction):每个线程负责哈希表的一段,非空槽标 1,共享内存 prefix sum 算输出位置,并行 gather。这是 spECK/OpSparse 的 gather 步骤标准做法。
- **真 bitonic sort**:共享内存里 bitonic sort 是 O(n log²n) 且完全并行,远优于冒泡。可直接用 CUB 的 `cub::BlockRadixSort`。

---

## 4. Tensor Core 的谨慎评估(对应 WarpDrive 标题亮点)

WarpDrive 标题最大卖点是 Tensor+CUDA 并发。但对本项目要**分情况**:

- **Tensor Core 适合稠密块**:如果是**块稀疏/结构化稀疏**(BCSR、2:4),把稠密子块 reshape 成矩阵喂 `wmma`/`mma` 指令,收益巨大 —— WarpDrive 的"4 warp Tensor + 4 warp CUDA"并发策略直接可用。
- **纯非结构化稀疏**(当前 CSR + 哈希):Tensor Core 很难喂饱,每次只有零星乘积,凑不满 16×16 矩阵;强行用反而被比特拆分/填充拖累(这正是 WarpDrive 批判 Tacker 的点)。

> 结论:**矩阵有稠密子结构**(DNN 剪枝、有限元)→ Tensor Core + 并发策略是金矿;**图分析式极不规则稀疏** → 先做迁移 1–3 更实在。

---

## 5. 建议优先级

```
迁移1(消除重算)        ← 最立竿见影,改动可控,直接 2× 算力
迁移2(A×Aᵀ Gustavson)  ← 算法层,收益最大(可能几十倍),工作量也最大
迁移3(并行收集+真排序)  ← 中等,消除 thread 0 瓶颈
Tensor Core            ← 视矩阵稀疏结构而定,进阶
```

---

## 6. 与 ref/ 论文的对应关系(串联)

迁移 1–3 其实都能在 ref/ 的论文里找到对应解法 —— WarpDrive 的价值是用一个极致例子(5 kernel → 1)强化了通用铁律:

| 迁移 | 对应 ref/ 论文 |
|------|---------------|
| 上界法单阶段 | Liu & Vinter(IPDPS'14)的混合预分配 |
| Gustavson row-wise | MatRaptor(HPEC'20)、Sparm/SpAda 的 Gustavson 数据流 |
| 并行 gather + sort | spECK(2017)、OpSparse 的 numeric 收尾 |
| 共享内存哈希优化 | spECK、HSMU(2025,进一步用排序数组+二分消除冲突) |
| 长行分块进片上 | AC-SpGEMM(PPoPP'19)的自适应 chunk |

> 详细论文综述见 [SpGEMM_Survey.md](SpGEMM_Survey.md)。

---

## 附:当前哈希累加函数(优化起点)

用户当前关注的 [hash_insert_or_add](src/spgemm_kernel_manual.cu#L202)(A×A 版本):

```cpp
__device__ int hash_insert_or_add(int *hash_keys, float *hash_vals, int key, float val) {
    int slot = key & (HASH_SIZE - 1);            // 位运算取模(HASH_SIZE 需为 2 的幂)
    for (int attempt = 0; attempt < HASH_SIZE; attempt++) {
        int old_key = atomicCAS(&hash_keys[slot], HASH_EMPTY, key);  // 无锁占位
        if (old_key == HASH_EMPTY || old_key == key) {
            atomicAdd(&hash_vals[slot], val);    // 同 key 累加(边插边去重)
            return slot;
        }
        slot = (slot + 1) & (HASH_SIZE - 1);     // 线性探测
    }
    return -1;
}
```

这正是 spECK 范式的实现(线性探测 + atomicCAS + 共享内存哈希)。已知短板:
- 固定 `HASH_SIZE=4096` 对极长行会溢出(返回 -1 未处理)、对极短行浪费共享内存 → 可借鉴 spECK 的**按行长分组配不同哈希表大小**;
- 哈希冲突固有(鸽巢原理)→ 若要消除,参考 **HSMU 的排序数组 + 二分查找**;
- 该函数在 count/fill 里被各调用一遍 → 见**迁移 1**消除重算。
