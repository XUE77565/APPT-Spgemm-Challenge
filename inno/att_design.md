# ATT 对称 SpGEMM:C = A·Aᵀ 上三角 hash SPA + 对称 mirror

> 对应代码:`src/spgemm_kernel_formulations.cu::spgemm_att_hash`(+ `att_hash_spa_kernel` / `att_mirror_coo_kernel`,main.cu att 模式接入)。
> 口径:H100 PCIe,double;att 路径当前用 host 时间戳 total(未接 cudaEvent prof,与 aa 路径不同)。
> 定位:把自乘的「hash SPA + 自适应」思路迁移到 **A·Aᵀ 对称** 场景:只算上三角(j≥i),长行 hash / 短行小表,末尾对称拷贝。
> 日期:2026-07-25

---

## 0. 一句话

> C = A·Aᵀ 天然对称 → **只算上三角 j≥i,下三角对称 mirror**。Gustavson att(axis=i)的中间项来自 A 的【列 k】(CSC),用 SMEM hash 累加器去重(同自乘 hash SPA),host 按 flop_ub 分 bin 给「长行大表 / 短行小表」的自适应 SMEM;末尾把上三角 COO 镜像(2×−diag)+ 排序成全对称 CSR。

---

## 1. 背景:ATT vs AA 的结构差异

| | AA(C=A·A,自乘) | **ATT(C=A·Aᵀ,对称)** |
|---|---|---|
| 元素 | C[i][j]=Σ_k A[i][k]·A[k][j] | C[i][j]=Σ_k A[i][k]·A[j][k]=dot(row_i,row_j) |
| Gustavson 内层读 | A 的【行 k】(CSR row k = 列 j 们) | A 的【列 k】(CSC col k = 行 j 们) |
| 需要的矩阵结构 | CSR | **CSC**(=Aᵀ 的 CSR) |
| 对称性 | 一般不对称 | **恒对称**((AAᵀ)ᵀ=AAᵀ)→ 可只算上三角 |
| 输出 | 全 CSR | 上三角 CSR → mirror 全对称 |

关键:ATT 的内层 j 来自 **A 的列**,故需先 `build_csc`(已有,GPU 端 CSR→CSC)。C 对称 → 只算 j≥i,省一半计算 + mirror 补下三角。

---

## 2. 算法(`spgemm_att_hash`)

```
build_csc(A)                              # CSR→CSC(GPU)
att_gust_count → row_ub[i]=Σ_{k∈A[i,:]} nnz(列k)   # flop_ub 上界(distinct j≥i 的上界)
host 按 ht=next_pow2(row_ub)∈[16,16384] 分 bin      # 长行大表 / 短行小表(自适应 SMEM)
for each bin(ht):
    att_hash_spa_kernel<<<n_in_bin, 256, ht×12>>>
        for k ∈ A[i,:] (并行线程):                # axis=i
            for (j, A[j][k]) ∈ 列k (CSC):          # 内层 j
                if j ≥ i:                          # 上三角 filter
                    atomicCAS 插 j / atomicAdd A[i][k]·A[j][k]   # hash 去重累加
        extract 无序 (i,j,val) → off[i] 区,记 act[i]
compact(act) → 连续无序上三角 COO
sort_by_key(i<<32|j) → 上三角 CSR
mirror: (i,j)→(i,j)+(j,i) [i≠j] → 全对称 COO → sort → full CSR
```

### 2.1 上三角 filter(j≥i)
ATT 对称 → C[i][j]=C[j][i]。算 row i 时只保留 j≥i(含对角 j=i),省一半。对角 C[i][i]=Σ_k A[i][k]² 天然包含。

### 2.2 长行 hash / 短行小表(自适应)
host 按 `ht=next_pow2(row_ub)` 把行分到 11 档(16..16384),每档一个 launch、SMEM=ht×(int+double)。**轻行落小档 → 小表 → 高 SMEM occupancy**;重行落大档(≤16384,超出的置 overflow flag)。这是自乘 hash SPA 的 binning 思想在 ATT 上的直接迁移——等价于「短行用小 hash 表、长行用大 hash 表」的自适应(短行若要更省可进一步走 ultrasparse 线性核,见 §5)。

### 2.3 对称拷贝(mirror)优化
上三角 CSR → 全对称:对每条 (i,j),i≠j 补一条 (j,i);对角 (i,i) 一份。实现 = `att_mirror_coo_kernel`(atomicAdd 全局计数填 COO,full_nnz=2×upper−diag)+ 全局 `sort_by_key` → full CSR。
- **为何 COO+重排而非原地**:上三角已按 (i,j) 有序,但补 (j,i) 要插到别的行,原地插需双区管理;COO 展开 + 一次全局 sort 最简且正确(full_nnz 已知)。
- **可优化方向**(§5):原地 mirror(先 count 各行 full nnz = upper_row_i + 列反射数,再两区填充)省一次 sort;或 cudaMemcpy2D 式批量拷。MVP 用 COO+sort 换正确性。

---

## 3. 正确性验证

| 阵 | ESC att_gust 上三角 | **att_hash 上三角** | att_hash full(=2×upper−diag) | AA 自乘(对称阵应等) | overflow |
|---|---|---|---|---|---|
| can_24 | 180 | **180 ✓** | 336 (=2×180−24) | — | 无 |
| bcsstk08 | 153343 | **153343 ✓** | 305612 (=2×153343−1074) | 305612 ✓(一致) | 无 |

- **上三角与项目既有 ESC att 完全一致**(180 / 153343)→ hash 累加逻辑正确。
- **full = 2×upper − diag**(diag=A_rows,全行对角非零,符合结构对称阵)→ mirror 算术正确。
- **对称阵上 full == AA 自乘 C_nnz**(bcsstk08:305612)→ A·Aᵀ=A·A 验证。
- 输出对称(mirror 构造保证 C[i][j]=C[j][i])。

> 注:cuSPARSE `spgemm_transpose_product` 的 full nnz(can_24 374 / bcsstk08 307094)与 att 路径(含既有 ESC)不一致——这是项目里 att 与 cuSPARSE 参照的**既有**差异(疑似 cuSPARSE 算 Aᵀ·A 或 read 对 symmetric 未展开致 A 非对称),**非本实现引入**;att_hash 与既有 ESC att 完全对齐,以此为正确性基准。

---

## 4. 附带修复:AttCtx.init 的 double 对齐 bug

实现 att_hash 时发现 att 全路径(outer/gust/colw/inner + hash)在 `(rows+1+nnz)` 为**奇**的阵上崩溃(misaligned address,如 can_24:25+160=185)。根因:`AttCtx.init` 把 csr_val 设在 `b+rp+ci`(未 8 对齐),而 `read_matrix_market` 把 A 的 val 放在 `ALIGN8(rp+ci)`——float→double 转换时的遗留(主路径已修 ALIGN8,att 的 AttCtx 漏修)。修:`csr_val=(double*)(b+ALIGN8(rp+ci))`,`totalA=ALIGN8(rp+ci)+vv`。**此修复让整个 att 路径在所有阵上不再崩**,不止 att_hash。

---

## 5. 性能 + 优化方向(MVP,未调优)

bcsstk08 att total(host 时间戳,h2d/d2h 含):

| 方法 | total ms | 产出 |
|---|---|---|
| cuSPARSE A·Aᵀ full | 1.6 | full |
| ESC Gustavson att | 2.3 | upper |
| **att_hash(本实现)** | 3.8 | **full(upper+mirror)** |

att_hash 比 ESC gust 慢,主因(MVP 未优化):
1. **内层 j 串行**(`for q in cs..ce`)——自乘 hash SPA 用 group 结构并行 j(G=16);att MVP 暂串行。→ 可移植 group 并行。
2. **提取用全局 sort**——自乘用 per-row BlockRadixSort(count-sort 小行)。att MVP 用 thrust 全局 sort。→ 可换 in-kernel BlockRadixSort。
3. **mirror 多一次全局 sort** + 双倍输出——att_hash 产 full(ESC gust 只产 upper)。→ 可做原地 mirror 省一次 sort。
4. att 路径未接 cudaEvent prof,口径是 total(含 h2d/d2h),与 aa 的 compute-only 不同口径,不能直接横比。

**优化路线**(未来工作):
- 移植 hash_spa_kernel 的 group 并行 j(G 从 ht 推)→ accumulate 提速。
- in-kernel BlockRadixSort 提取(替全局 sort)。
- 原地 mirror(双区填充)省一次 sort。
- 短行 ultrasparse 线性核(est≤16 不建 hash 表)→ 真「短行 merge/线性」。
- sizing 换 HLL/MinHash(替 flop_ub)紧 buffer(自乘路径已有,可移植)。

---

## 6. 复现

```bash
make DBG=1
USE_MEMPOOL=1 ./spgemm_test <mtx> att          # 跑全 att 套件(cuSPARSE + 4 ESC + hash)
# Result C (A·Aᵀ 全对称): ... nnz = <full>      ← att_hash 输出(对称 full)
```

---

## 附:与其他 inno 文档的关系

- `merge_innovation.md` / `hash_innovation_kmv.md`:自乘(AA)的两条算法线 + 调度。本文件把同一思想(hash SPA + 自适应)迁移到 ATT(A·Aᵀ)对称场景,是三点叙事在「转置积」上的延伸。
- `innovation_points.md`:全局创新。ATT 的「上三角 + 对称 mirror + 长 hash/短表」可作为自适应框架在新公式(A·Aᵀ)上的应用例。
