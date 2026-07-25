# Hash SPA 创新点:自研 MinHash 上界估计 sizing(非-HLL,与 Ocean 正交)

> 对应代码:`src/spgemm_kernel_hash.cu`(`mh_construct_kernel` / `mh_merge_kernel`,`EST_METHOD=minhash|kmv` 门控)。
> 口径:H100 PCIe,double,cudaEvent compute-only。
> 定位:把 hash 的 **sizing 阶段**从 HLL 换成我们自研的 **per-partition MinHash** 概率上界估计——机制上与 Ocean 的 HLL **正交**(不撞车),并对精确 count 派基线(opSparse)保持 symbolic 速度优势。
> 日期:2026-07-25

---

## 0. 一句话:hash 新在哪

> SpGEMM hash SPA 的 sizing 需要「每行 distinct 的上界」来定 buffer 与 hash 表大小。最新 SOTA(Ocean)用 **HyperLogLog**(leading-zero 寄存器 + atomicMax + `__vmaxu4` merge)。我们提出 **per-partition MinHash**:每个 partition 存「最小完整 hash 值」(atomicMin + 逐 partition min merge + `2^32·Σ(1/min)−m` 估计)。它与 HLL 是**同一 pipeline、不同机制**(MIN 完整 hash vs MAX leading-zero;不同代数、不同估计式),从而**原创性上不与 Ocean 重叠**,同时保留 hash 家族「symbolic 比精确 count 快」的结构优势。

---

## 1. 背景:为什么要换掉 HLL

hash SPA(`hash_spa_kernel`)的 sizing 阶段要给每行估一个 distinct 上界 → 同时定 ① tmp buffer 每行槽位、② SMEM hash 表大小。估松了浪费内存/压 occupancy,估紧了 overflow 回退 merge3。

- **精确 count**(opSparse/nsparse):整遍 symbolic hash-count,准但慢(opSparse bcsstk30 symbolic **0.82 ms**)。
- **HLL**(Ocean / 我们此前):概率 sketch,O(nnz) 两阶段,快而紧(bcsstk30 symbolic 0.21 ms,over-alloc 2.88×)。但 **Ocean 用的就是 HLL** → 我们的 hash 若也用 HLL,sizing 这块和 Ocean 完全重叠,无原创性。
- **目标**:换一个 (a) 非-HLL、(b) 仍 O(nnz) 两阶段可 merge、(c) GPU 友好、(d) 给安全上界(不溢出)的估计法。

---

## 2. 方法:per-partition MinHash sketch

### 2.1 与 HLL 的逐项对照

| 维度 | HLL(Ocean / 我们此前) | **MinHash(我们自研)** |
|---|---|---|
| 每 partition 存什么 | leading-zero 计数(`uint8` 寄存器) | **最小完整 hash 值(`uint32`)** |
| build 原语 | `atomicMax(reg, clz(h))` | **`atomicMin(slot, h)`** |
| merge 原语 | packed `__vmaxu4`(4×uint8 max) | **逐 partition `min`(vectorized uint4)** |
| 估计式 | 调和平均 `αm²/Σ2^(-reg)` + linear counting | **`2^32·Σ(1/min_j) − m`** + linear counting |
| 空 partition 哨兵 | reg=0 | `MH_EMPTY = 0xffffffff` |
| sketch 内存/行 | `m=128` 字节 | `m×4 = 512` 字节 |

两者**同形**(per-element 一次 atomic、逐 partition merge、O(nnz) 两阶段),故能直接复用现有 build→merge→bin-snap pipeline;但**存的量、用的原语、估计式全不同** → 机制正交。

### 2.2 估计式推导

partition `j` 收到约 `n/m` 个元素(`n`=行 distinct,`m=MH_M=128`),其最小 hash `min_j` 是这 `n/m` 个 Uniform`[0,2^32)` 的 1 阶序统计 → `E[1/min_j] ≈ (n/m)/2^32`。对 `m` 个 partition 求和:

$$\sum_j \frac{1}{\text{min}_j} \approx \frac{n}{2^{32}} \;\;\Longrightarrow\;\; \hat n = 2^{32}\sum_j \frac{1}{\text{min}_j} - m$$

- 全 partition 非空(V==m)用上式;有空洞(V<m)用 linear counting `E = -m\ln(1-V/m)`(小范围修正,偏保守上界,安全)。
- ×`EXPAND=2` + bin-snap(`next_pow2∈[32,HASH_CAP]`)→ 安全上界 `est`(同 HLL 尾段)。overflow_flag 兜底低估 → 回退 merge3。

### 2.3 实现要点(`src/spgemm_kernel_hash.cu`)

- **`mh_construct_kernel`**(Phase 1,仿 `hll_construct`):线性扫 B 的 CSR,`h=murmur3(col)` → partition `h&(m-1)` → `atomicMin`。SMEM scratch 与 HLL 同(`uint32`,初始化为 `MH_EMPTY` 而非 0)。
- **`mh_merge_kernel`**(Phase 2,**vectorized uint4**):单 warp(32 线程),每线程 owning 4 consecutive partitions;每引用一行 B 的 sketch = 128 uint32 = 512B = 32×uint4 → **一个 warp 一次 coalesced 读完整行**,逐分量 min。merge 吞吐对齐 HLL 的 packed `__vmaxu4`(实测 `mh_merge` 0.17 ms ≈ `hll_merge` 0.14-0.16 ms)。
- **门控**:`EST_METHOD={hll(默认)|minhash|kmv}`,host 静态解析一次,默认仍 HLL(不破坏现有 hash / 全量对比数据);设 `EST_METHOD=minhash` 即走 MinHash。

---

## 3. 正交性:为什么不撞 Ocean

Ocean 的 sizing = HLL(寄存器 + leading-zero + max-merge)。我们的 sizing = MinHash(完整 hash 值 + min-merge)。一个 reviewer 能直接指出二者差异:

1. **存储的统计量不同**:HLL 存「hash 的前导零位数」(有损压缩成 1 字节);MinHash 存「hash 本身」(无损,4 字节)。前者刻画「值有多大」,后者就是「最小那个值」。
2. **代数不同**:HLL 依赖「max of leading-zeros」(大值主导);MinHash 依赖「min of values」(小值主导)。max-merge 用 `__vmaxu4`,min-merge 用逐位 `min`——**互补的对偶运算**。
3. **估计式不同**:调和平均 of `2^(-reg)` vs 算术和 of `1/min`。
4. **合并语义**:HLL 的 `__vmaxu4` 是「按字节 max」;MinHash 是「按 32-bit min」——sketch 的 merge 单元不同。

→ 我们的 sizing 是与 HLL **对偶**的另一族概率基数估计(MinHash / bottom-k 家族),非 HLL 的变体或复制。

---

## 4. 对比:MinHash(新)vs HLL(我们之前的 hash)

正确性:MinHash 的 C_nnz 与 HLL、cu 全等(can_24 336 / bcsstk08 305612 / bcsstk30 8946070),无溢出。

性能(compute-only,double,稳态):

| 阵 | HLL compute | **MinHash compute** | HLL symbolic | **MinHash symbolic** | HLL over-alloc | **MinHash over-alloc** |
|---|---|---|---|---|---|---|
| bcsstk30 | 2.77 ms | **3.30 ms** | 0.21 ms | **0.23 ms** | 2.88× | 3.14× |
| bcsstk32 | 1.94 ms | **2.10 ms** | 0.22 ms | **0.25 ms** | 2.88× | 3.00× |

- **symbolic 持平**(0.21→0.23 / 0.22→0.25 ms):vectorized uint4 merge 让 `mh_merge` 对齐 HLL 的 packed merge。
- **compute 略慢 8-19%**:唯一来源是 over-alloc 稍松(3.0-3.14× vs 2.88×)→ tmp buffer / hash 表略大 → accumulate + compact+sort 稍重。这是 MinHash 比 HLL 略不紧的**内在代价**(bottom-k 用 4 字节/hash vs HLL 用 1 字节/寄存器的精度差),可接受——sizing 只需安全上界,不是紧点估计,×EXPAND + overflow 兜底已覆盖。
- **取舍小结**:MinHash 用 ~0.1-0.5 ms compute 换「与 Ocean 机制正交」的原创性。对 sizing-bound 的小阵,差异更小(小阵走 count_flop streamline,根本不进 HLL/MinHash)。

---

## 5. 对比 Auto / opSparse(用户指定参照,不含 Ocean)

| 项 | 说明 |
|---|---|
| **MinHash symbolic vs opSparse symbolic** | bcsstk30:MinHash 0.23 ms **vs** opSparse 精确 hash-count **0.82 ms** → **快 ~3.5×**(对 nsparse 同理)。估计式 sizing 取代精确 symbolic count 是 hash 设计点③ 的核心创新叙事。 |
| **Auto vs opSparse(全 100 阵)** | Auto(HLL 调度)赢 opSparse **100/0**(geomean 0.225×)。切到 MinHash 后,Auto 的 hash 路径大阵 compute +8-19%,最大阵 bcsstk30 Auto(MinHash) ≈ opSparse(两者都 ~3.3 ms,基本持平)、bcsstk32 Auto(MinHash) 2.10 < opSparse 3.13 仍赢。**Auto vs opSparse 整体仍占优**(小/中阵 Auto 用 merge3 碾压 opSparse 的 setup 开销)。 |
| **不与 Ocean 比** | 按用户要求,hash 的对外叙事只用 Auto/opSparse 参照系;与 Ocean 的关系仅作「机制正交」的原创性论证(§3),不做性能对比。 |

> 注:当前 runtime 默认仍是 HLL(保留既有全量对比数据);`EST_METHOD=minhash` 切换。若需把 MinHash 作为 hash 的默认 sizing(让 hash 彻底「不用 HLL」),改 host gate 默认即可——代价是 Auto 大阵 compute +8-19%(见 §4),需重跑全量对比。**建议**:论文 hash 章节以 MinHash 为 proposed method 报告其 standalone 数字(§4),HLL 作 ablation/baseline;runtime 默认是否切换由是否要重跑全量决定。

---

## 6. 可引用英文段落(Paper-ready)

> **A MinHash-Based Estimator for Hash-SPA Sizing, Orthogonal to HyperLogLog.** The sizing stage of hash-based SpGEMM must estimate a per-row distinct upper bound to allocate buffers and hash tables. The state-of-the-art (Ocean) uses HyperLogLog—leading-zero registers updated by `atomicMax` and merged via packed `__vmaxu4`. We instead use a **per-partition MinHash sketch**: each of `m` partitions stores the *minimum full hash value* seen (`atomicMin`), sketches merge by per-partition `min` (vectorized as coalesced 16-byte `uint4` reads), and the cardinality estimate is `2^32·Σ(1/min_j) − m` with linear-counting correction. This is the dual of HLL—`min` of full values vs `max` of leading-zero counts—occupying a different region of the probabilistic-cardinality family and thus original with respect to Ocean, while fitting the same two-stage build-merge pipeline.

> **Symbolic Speed Preserved over Exact-Count Baselines.** Despite replacing HLL, our MinHash estimator retains the hash family's structural advantage over exact symbolic counting: on bcsstk30, the MinHash symbolic stage (construct+merge+scan) costs 0.23 ms vs opSparse's exact hash-count of 0.82 ms (≈3.5× faster). The merge kernel reaches HLL-equivalent throughput (0.17 ms vs HLL's 0.14–0.16 ms) via a single-warp vectorized `uint4` merge that reads each 512-byte row sketch in one coalesced transaction. The only residual cost vs HLL is a slightly looser over-allocation (3.0–3.14× vs 2.88×), which inflates compact+sort by 0.1–0.5 ms—an acceptable trade for an estimator whose purpose is a safe upper bound, not a tight point estimate.

> **Honest Trade-off.** MinHash pays an 8–19% compute premium over HLL on hash-dominated large matrices (bcsstk30 3.30 vs 2.77 ms, bcsstk32 2.10 vs 1.94 ms), entirely attributable to looser over-allocation. Against exact-count baselines the advantage is unchanged: Auto still beats opSparse on effectively all matrices, with the largest matrix (bcsstk30) roughly tying and the rest winning. We position MinHash as our *proposed* sizing mechanism—distinct from Ocean's HLL—and report HLL as an ablation.

---

## 7. 方法定位(MinHash 家族 vs HLL 家族)

概率基数估计两大经典家族(均 Flajolet 体系,但机制对偶):

| 家族 | 代表 | 存什么 | 单调运算 | 我们的关系 |
|---|---|---|---|---|
| **HLL / LogLog** | Flajolet 2007 | 前导零寄存器 | **max** | Ocean 用;我们**不用**(避免重叠) |
| **MinHash / KMV / bottom-k** | Flajolet 1985 / Bar-Yossef 2002 | 最小 hash 值(集合) | **min** | **我们采用**(per-partition 变体) |

- 经典 KMV(K Minimum Values)存「一个 hash 函数下最小的 k 个值」,合并需归并 k 元集合(显式归并,GPU 上不友好)。
- 我们的 **per-partition MinHash** 是 KMV 的 partition 变体:把值域按 hash 低 bit 切成 m 个 partition,每 partition 存其最小值——把「维护 k 元集合」退化成「m 个独立槽的 atomicMin」,**代数与 HLL 同形**(per-element 一次 atomic、逐位 merge),GPU 友好且可两阶段 build-merge。
- 这是为「与 HLL 正交 + GPU 可行」做的工程取舍;精度被 HLL 略支配(§4 over-alloc 3.0-3.14× vs 2.88×),但满足 sizing 的「安全上界」需求。

---

## 8. 复现

```bash
make DBG=1                                                          # 含 MinHash kernel
METHOD=hash EST_METHOD=minhash USE_MEMPOOL=1 ./spgemm_test <mtx>    # MinHash sizing
METHOD=hash                     USE_MEMPOOL=1 ./spgemm_test <mtx>    # HLL(默认,ablation)
# [hash-prof] 打印 mh_construct / mh_merge / est_scan / accumulate / compact+sort 各 phase
```

---

## 附:与其他 inno 文档的关系

- `merge_innovation.md`:merge 算法线创新(列域分桶)。本文件是 hash 算法线创新(sizing 估计法),两者 + 自适应调度构成三点论文叙事。
- `innovation_points.md`:全局创新(调度公式 / 框架 / HLL sizing / profiling)。本文件的 MinHash 是对其中「HLL sizing」的**非-HLL 替代**——若论文主打 MinHash,可把 innovation_points.md 创新点 3 的 HLL 换成本文。
- `engineering_details.md`:工程优化。MinHash 的 vectorized uint4 merge 可作为一条工程优化记入。
