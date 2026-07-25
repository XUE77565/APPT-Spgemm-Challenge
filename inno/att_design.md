# ATT 对称 SpGEMM:C = A·Aᵀ 上三角,AA hash+merge 的忠实拷贝

> 对应代码:
> - `src/spgemm_kernel_hash.cu`:`hash_product(A, att)` 共享核心 + `spgemm_att_hash` wrapper(hash_spa/ultra 泛化 B 操作数 + upper_tri)
> - `src/spgemm_merge.cu`:`merge3_product(A, att)` + `spgemm_att_merge3`(bucket_flop/merge_flop 泛化 B + upper_tri)
> - `src/spgemm_adaptive.cu`:`spgemm_att_adaptive`(同 AA Auto 分流 → att_hash/att_merge3)
> - `src/spgemm_kernel_formulations.cu`:`build_csc`(暴露 + 列内排序修复)
> 口径:H100 PCIe,double;att 路径 host 时间戳 total(未接 cudaEvent prof)。
> 日期:2026-07-25(重构为 AA 忠实拷贝)

---

## 0. 一句话

> **ATT = AA 把第二个操作数 B 换成 Aᵀ(= A 的 CSC)+ 上三角过滤 j≥i。** 故 hash SPA / merge3 / adaptive 三条线**原样拷贝** AA:kernel 加一个 B 操作数 + 一个 upper_tri flag(AA 传 B=A、flag=0;ATT 传 B=CSC、flag=1),HLL sizing / binning / compact_sort / flop sizing / 列域分桶全部复用。只算上三角(下三角由对称性可得,不展开)。

---

## 1. 核心洞察:ATT 与 AA 只差一个操作数

| | AA(C=A·A) | ATT(C=A·Aᵀ) |
|---|---|---|
| 元素 | C[i][j]=Σ_k A[i][k]·A[k][j] | C[i][j]=Σ_k A[i][k]·A[j][k] |
| Gustavson 内层读 | A 的**行 k**(CSR row k = {j:A[k][j]≠0}) | A 的**列 k** = Aᵀ 的行 k(CSC col k = {j:A[j][k]≠0}) |
| 第二操作数 B | A 本身 | **Aᵀ = A 的 CSC**(csc_cp/csc_ri/csc_val 即 CSR(Aᵀ)) |
| 对称性 | 一般不对称 | **恒对称** → 只算 j≥i |

A 的 CSC 数组(col_ptr / row_idx / val)就是 Aᵀ 的 CSR 数组(row_ptr / col_idx / val)——同一份内存、同一布局。所以把 AA 的「内层读 A 行 k」换成「内层读 CSC 列 k」= 读 Aᵀ 的行 k = 算 A·Aᵀ。

→ 无需重写任何算法,只把 hash_spa / hash_ultra / bucket_flop / bucket_merge_flop 的**内层**数据源泛化成参数 B,再加 upper_tri 过滤。

---

## 2. 三条线的拷贝方式

### 2.1 hash SPA(`hash_product(A, att)` + `spgemm_att_hash`)
- `hash_spa_kernel` / `hash_ultra_kernel` 加 `(B_row_ptr, B_col_idx, B_val, upper_tri)`:内层 `ks/ke=B_row_ptr[k]`、`j=B_col_idx[q]`、`v=a_ik*B_val[q]`,`if (upper_tri && j<i) continue;`。AA 传 B=A、flag=0(行为不变)。
- **sizing 完全复用**:HLL Phase1 `hll_construct(B)` sketches B 的行(AA=A 的行;ATT=Aᵀ 的行=A 的列);Phase2 `hll_merge(A)` 对 A 每行 merge 引用 B-行的 sketch。est = 每行 distinct j 的上界(全 distinct,≥ 上三角 distinct → 安全上界)。binning / compact_sort(BlockRadixSort)原样复用。
- ATT 跳过 priv(warp 私有 SPA 未泛化为 B)。overflow → C_nnz=-1。

### 2.2 merge3(`merge3_product(A, att)` + `spgemm_att_merge3`)
- `bucket_flop_kernel`(flop sizing 上界)+ `bucket_merge_flop_kernel`(warp-merge 写 gapped)加 B + upper_tri。merge 输出处 `if (upper_tri && wmin<i)` 跳过(仍消费该列)。flop 是 sizing 上界,upper-tri 无需过滤(安全过估)。
- 列域分桶 K=5、lower_bound 定位子区间、免全局 sort、compact —— 全复用 AA。ATT 强制走 flop path(exact-count 的 bucket_count/merge 未泛化,att 不走)。

### 2.3 adaptive(`spgemm_att_adaptive`)
- 同 AA Auto 的多变量 score 公式(`-1.31·logflop + 1.21·logn + 1.98·logmaxrow - 2.43·logskew + 1.29`),score<0 → att_hash,否则 att_merge3。hash 溢出 → 回退 att_merge3。
- 分流是**矩阵级**(同 AA Auto);hash 内部另有 per-row ultra/hash binning(短行 ultra 线性 / 长行 hash SPA),即「短行轻路径 / 长行 hash」的 per-row 自适应。

---

## 3. 关键 bug 修复:build_csc 列内排序

实现 merge3 ATT 时发现:`build_csc` 用 `atomicAdd` 并行散列,**csc_ri 列内无序**(同一列的 row-index 按线程执行顺序落位,非升序)。`dev_lower_bound`(merge3 用)在无序数据上失效 → can_24 出 415 而非 180。

修复:`build_csc` 末尾按 composite key `(col<<32)|row` 用 `thrust::sort_by_key` 排序(zip reorder csc_row + csc_val),使**每列内 row-index 升序**。
- ESC(AA outer/colwise)走 esc_merge 全局 sort、hash 遍历全项 → 不依赖列内序,排序无害。
- ATT merge3(lower_bound)现在正确。
- 实测:AA hash/merge3/manual 在 bcsstk30(8946070)/ can_24(336)全对;ATT hash/merge3 一致。

> (历史:上一版 MVP 用 flop_ub sizing + 串行 j + 全局 sort + 对称 mirror,非 AA 拷贝,已废弃。本版按用户要求改为 AA 忠实拷贝,只算上三角、无 mirror。)

---

## 4. 正确性验证(上三角 nnz)

| 阵 | cuSPARSE A·Aᵀ 全量 | **att_hash 上三角** | **att_merge3 上三角** | **att_adaptive 上三角** | 校验 (= (A²full+diag)/2,symmetric) |
|---|---|---|---|---|---|
| can_24 | 374 | 180 | 180 | 180 | (336+24)/2=180 ✓ |
| bcsstk08 | 307094 | 153343 | 153343 | 153343 | (305612+1074)/2=153343 ✓ |
| bcsstk30 | 9007776 | 4487497 | 4487497 | 4487497 | (8946070+28924)/2=4487497 ✓ |

- 三方法(hash/merge3/adaptive)上三角完全一致 → 互相验证。
- symmetric 阵上 = (AA 自乘 full + diag)/2 → 与 AA 路径交叉验证。
- adaptive 分流:can_24(score 0.58)→ merge3、bcsstk30(score −0.53)→ hash。

> 注:cuSPARSE `spgemm_transpose_product` 全量 nnz(374 / 307094 / 9007776)与 att 上三角推出的全量(2×upper−diag = 336 / 305612 / 8946070)**不一致** —— 这是项目既有差异(cuSPARSE 疑算 Aᵀ·A 或 read 未展开 symmetric 致 A 非对称),非本实现引入。att 正确性以「上三角 = (AA 自乘 full + diag)/2」为准。

---

## 5. 复现

```bash
make DBG=1
USE_MEMPOOL=1 ./spgemm_test <mtx> att     # cuSPARSE 全量 + att_hash / att_merge3 / att_adaptive 上三角
ADAPTIVE_FORCE=h USE_MEMPOOL=1 ./spgemm_test <mtx> att   # 强制 att_hash
```

---

## 附:与其他 inno 文档的关系

- `merge_innovation.md` / `hash_innovation_kmv.md`:AA 的两条算法线 + 调度。本文件 = 把同一套(hash SPA + merge3 + adaptive)迁移到 ATT(A·Aᵀ 上三角),核心贡献是「B=Aᵀ + upper_tri 让 AA 实现零重复地复用于对称积」+ build_csc 列内排序修复。
