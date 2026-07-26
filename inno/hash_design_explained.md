# Hash SPA 两个创新点详解:MinHash sizing 与 in-place dedup

> 对应代码:`src/spgemm_kernel_hash.cu`。口径:H100 PCIe,double,cudaEvent compute-only。
> 配图:`fig/hash_minhash_design.{png,pdf}`、`fig/hash_dedup_design.{png,pdf}`(`scripts/plot_hash_mech_detail.py`)。
> 关联:`inno/hash_innovation_kmv.md`(MinHash sizing 的论文级论述)。
> 日期:2026-07-26

---

## 0. 背景:hash SPA 的两个阶段 = 我们要加速的两处

hash-based SpGEMM 的 compute 分两段(对每行 C[i,:]):

1. **symbolic(sizing)**:估计该行 **distinct 列数**(= C 的行 nnz 上界),用来同时定 ① tmp buffer 每行槽位、② SMEM hash 表大小。估松了浪费内存/压 occupancy,估紧了 overflow 回退。
2. **accumulation**:把该行的所有中间积 `(i, j, v)` 累加进 hash 表,**去重**成 distinct 的 `(j, Σv)`。

baseline `opSparse`(2022)在这两段花掉绝大部分时间(bcsstk30:**symbolic 27% + accumulation 52% = 79%**,见 `fig/opsparse_bar`)。我们的两个创新分别砍这两段:

| 阶段 | baseline(opSparse)的问题 | 我们的方案 | 原语 | 收益 |
|---|---|---|---|---|
| **symbolic** | 精确 hash-count,慢(0.82ms) | **per-partition MinHash** 概率估计 | `atomicMin`(留最小 hash) | **3.3× 快** |
| **accumulation** | materialize 全部 product 再 sort+reduce(72% runtime) | **in-place dedup** | `atomicCAS`+`atomicAdd` | **11–16× 便宜** |

---

## 1. MinHash sizing —— 加速 symbolic

### 1.1 问题

sizing 要给每行一个 **distinct 上界**。三条路:

- **精确 count**(opSparse / nsparse):遍历所有 product,hash-count 每列出现次数 → 准但慢。bcsstk30 symbolic = **0.82 ms**。
- **HLL**(Ocean / 我们此前):概率 sketch,O(nnz) 两阶段,快而紧(0.21 ms)。但 **Ocean 用的就是 HLL** → 直接抄无原创性。
- **目标**:换一个 (a) 非-HLL、(b) O(nnz) 可两阶段 merge、(c) GPU 友好、(d) 给安全上界的估计法。

### 1.2 机制:per-partition MinHash sketch

核心一句话:**把 hash 值域切成 m 个 partition,每个 partition 只留它见过的最小 hash 值**。

1. **hash 每个列号 `j`**(murmur3)→ 一个 `[0, 2³²)` 上**均匀分布**的 32-bit 值 `h(j)`。
2. **分 m 个 partition**(用 `h` 的低位)。每个 `j` 的 hash 落进某个 partition。
3. 每个 partition **只保留它见过的最小 hash 值**(`atomicMin`)。这 m 个最小值 = 该行的 **sketch**。
4. **为什么能估 distinct**:若一个 partition 见到了 `k` 个 `[0,2³²)` 上的均匀值,它的最小值期望 ≈ `2³²/(k+1)`,于是 `1/min ≈ k/2³²`。对 m 个 partition 求和:
   
   $$\sum_j \frac{1}{\min_j} \;\approx\; \frac{n}{2^{32}} \quad\Longrightarrow\quad \hat n = 2^{32}\sum_j \frac{1}{\min_j} - m$$
   
   (`n` = 该行 distinct 列数。`−m` 是对"每个 partition 自己那个最小值"的偏置修正。)
5. **小范围修正**:若有 partition 全空(说明 `n` 小),改用 linear counting `E = -m·ln(1-V/m)`(`V` = 非空 partition 数),偏保守上界,安全。
6. **重复列免费**:同一个 `j` 的多次出现 hash 值相同,**永远不会改变** partition 的 min → 不需要先去重就能估 distinct。这正是 sizing 想要的(中间积有大量重复)。

### 1.3 例子(见 `fig/hash_minhash_design`)

列号流:`5, 12, 5, 8, 12`(其中 `5`、`12` 重复),distinct = {5, 8, 12} = **3 个**。设(示意性 hash):

| j | h(j) | partition = h&3 |
|---|---|---|
| 5 | 0xB2A0 | 0 |
| 12 | 0x4F19 | 1 |
| 8 | 0x9C44 | 0 |

- **P0** 见 {0xB2A0(j5), 0x9C44(j8), 0xB2A0(j5 重复)} → 留 **min = 0x9C44**(重复的 0xB2A0 被忽略)
- **P1** 见 {0x4F19(j12), 0x4F19(j12 重复)} → 留 **min = 0x4F19**
- **P2、P3** 空

V=2 < m=4 → linear counting:`E = -4·ln(1-2/4) = -4·ln(0.5) ≈ 2.77 ≈ 3` ✓(真值 3)

### 1.4 结果与定位

- **O(nnz) 一遍,不 count**;symbolic 0.23 ms vs opSparse 0.82 ms → **3.3× 快**(bcsstk30)。
- `mh_merge` 用单 warp vectorized `uint4` 做 partition-min 合并,吞吐对齐 HLL 的 `__vmaxu4`(0.17 ≈ 0.14–0.16 ms)。
- **与 Ocean HLL 机制对偶、正交**:HLL 留"前导零的最大值"(max-leading-zero,1 字节寄存器,`__vmaxu4` merge);我们留"完整 hash 的最小值"(min-value,4 字节,逐 partition `min`)。同一 pipeline、不同代数 → 不撞车。
- 唯一代价:over-alloc 稍松(3.0–3.14× vs HLL 2.88×)→ compute +8–19%(可接受,只影响 sizing-bound 的大阵)。

---

## 2. in-place dedup —— 加速 accumulation

### 2.1 问题

一行的中间积里有**大量重复列号**——dup factor 在结构大阵上可到 **19×**(bcsstk30:173M 中间积 → 8.95M distinct)。

- **sort-based(ESC/Gustavson)**:先把全部 O(flop) 中间积 materialize(expand),再 sort,再 reduce → 这是它 **72% runtime** 的根因(`fig/esc_runtime_share`)。
- **目标**:在累加的**同时**就把重复消掉,绝不 materialize 那 19× 的流。

### 2.2 机制:per-row SMEM hash table(原子去重)

每个输出行 C[i,:] 配一个 SMEM hash 表(大小由上面的 MinHash 估出来)。每个中间积 `(i, j, v)` 流入时:

1. `hash(j)` → 定到表里某个 **slot**。
2. **`atomicCAS`**:slot 空 → **插入 `j` 一次**(占位);`j` 已经在 → **什么都不做**(它是重复)。
3. **`atomicAdd`**:把 `v` 累加进该 slot 的 value。
4. 于是:**第一次**见到 `j` 才建条目;**之后所有重复**只把值加进去,永远不产生新条目。重复在到达的瞬间就**塌缩**了。
5. 累加完,表里只剩 **distinct `(j, Σv)`** = C[i,:]。extract → compact+sort → 列有序 CSR。

> 关键:`atomicCAS` 保证每个列号只插一次,`atomicAdd` 把所有重复的值汇总 → **去重和累加同一步完成**,不需要先展开、后排序。

### 2.3 例子(见 `fig/hash_dedup_design`)

中间积:`(5,2), (5,3), (8,1)`(`j=5` 出现两次 = 重复)。

| 流入 | 定位 | atomicCAS | atomicAdd | slot 状态 |
|---|---|---|---|---|
| (5,2) | slot[5] | 空 → 插入 5 | v = 2 | {5: 2} |
| (5,3) | slot[5] | 5 已在 → 跳过 | v = 2+3 = **5** | {5: 5} |
| (8,1) | slot[8] | 空 → 插入 8 | v = 1 | {8: 1} |

结果:distinct = {(5,5), (8,1)}。**3 个中间积 → 2 个 distinct**(重复的 j=5 塌缩成一条)。真实矩阵上 19× 的重复同理被压成 1×。

### 2.4 结果

- 只写、只排 **distinct(C_nnz)**,绝不 materialize 19× 的中间积流。
- accumulate 阶段比 ESC 的 expand+sort+reduce **便宜 11–16×**(bcsstk30:2.2ms vs 34.7ms)。
- 溢出兜底:某行 distinct 超过表容量 → 置 overflow_flag → 上层 dispatcher 回退 merge3(所以 MinHash 估得稍松是安全的)。

---

## 3. 一句话总结

- **symbolic 慢** → 用 **MinHash**(每 partition 留最小 hash,`1/min` 之和即基数)替代精确 count,**3.3× 快**,且与 Ocean HLL 正交。
- **accumulation 慢** → 用 **in-place hash dedup**(`atomicCAS` 插一次 + `atomicAdd` 汇总)让 19× 重复在到达时即塌缩,**11–16× 便宜**。

两者一前一后:MinHash 先把表的大小定准(给 in-place dedup 备好合身的 hash 表),dedup 再在累加时把重复压掉——共同把 opSparse 占 79% 的两段大幅缩短。

---

## 4. 复现

```bash
make DBG=1
# MinHash sizing(默认):
METHOD=hash USE_MEMPOOL=1 ./spgemm_test data/first100/bcsstk30.mtx 2>&1 | grep -E "EST_METHOD|mh_construct|mh_merge|accumulate"
# opSparse 各阶段(对照):
external_sota/HSMU-SpGEMM/other_spgemm_code/OpSparse/opsparse data/first100/bcsstk30.mtx 2>&1 | grep -iE "symbolic|numeric"
# 出图:
.venv/bin/python scripts/plot_hash_mech_detail.py     # 两张设计图
.venv/bin/python scripts/plot_opsparse_bar.py         # opSparse 占比柱
.venv/bin/python scripts/plot_hash_prof.py            # symbolic / accumulation profiling
```
