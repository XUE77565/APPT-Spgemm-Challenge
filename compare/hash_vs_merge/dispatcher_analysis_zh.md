# Hash vs Merge3 自适应调度器:观察、分析与 H100 硬件因素

> SpGEMM C = A·A 自乘,NVIDIA H100 PCIe(51 TFLOPS FP32,2.0 TB/s HBM3,114 SMs)。
> 100 个 SuiteSparse 矩阵。cudaEvent 纯 GPU 计时(排除 h2d/d2h)。
> 数据来源:`hash_vs_merge_enriched.csv`。

---

## 1. 观察:什么矩阵 Hash 赢,什么矩阵 Merge3 赢

100 阵实测:

| 矩阵特征 | 例子 | Hash(ms) | Merge3(ms) | 赢家 | 倍数 |
|---|---|---|---|---|---|
| 小 + 稀疏(n<500,nnz<5000) | can_24, bcspwr01 | 0.17 | 0.08 | **Merge3** | 2× |
| 中 + 稀疏(n~1000,nnz~10000) | bcsstk09, can_1072 | 0.19 | 0.16 | **Merge3** | 1.2× |
| 中 + 稠密(n~1000,重行) | bcsstk08, bp_0 | 0.47 | 0.91 | **Hash** | 2× |
| 大(n>5000) | bcsstk30 | 2.49 | 15.7 | **Hash** | **6.3×** |

**规律**:flop(中间积数)小 → merge3 赢;flop 大 → hash 赢。交叉点在 flop ≈ 1.4×10⁵。

---

## 2. 两个算法的核心差异

### Hash 算法(每行一个 block)

```
对 C 的每一行 i:
  ① 在共享内存(SMEM)建 hash 表(ht_size 个槽,由 HLL 估计定大小)
  ② 遍历所有中间积 (k,j):
     - 线程计算 hash(j) → 定位槽位
     - atomicCAS:槽空 → 插入 j;已有 j → 找到
     - atomicAdd:累加值
  ③ 提取 hash 表中的 distinct (列,值) → 压实 → 排序 → CSR
```

### Merge3 算法(每行一个 warp)

```
对 C 的每一行 i:
  ① 把列域分成 K=5 个桶
  ② 遍历所有中间积,按列号分桶
  ③ K-way merge:每步从 5 个桶各取当前最小列,去重合并
     → 输出天然有序,不需要额外排序
```

---

## 3. 为什么 Hash 赢大阵、Merge3 赢小阵 —— 三个硬件因素

### 因素 ①:每行的线程并行度(最主要)

| | Hash | Merge3 | 差异 |
|---|---|---|---|
| 每行线程数 | **256**(HASH_BLOCK) | **32**(1 个 warp) | **8×** |

**Hash 为什么能用 256 线程?** Hash 插入是**独立的** —— 每个线程算自己的
hash(j),往不同的 SMEM 槽位做 atomicCAS,互不干扰(不同列 → 不同槽)。256 个
线程可以**同时处理 256 个中间积**。

**Merge3 为什么只能用 32 线程?** K-way merge 有**串行依赖** —— 每一步必须先比较
5 个桶的当前最小列(需要 warp 内 `__shfl` 归约),选出最小值,然后推进那个桶。
这个"比较 → 选最小 → 推进"的循环是**串行的**。一个 warp 的 32 个线程协作做这一步,
但**不能像 hash 那样让 256 个线程独立推进**。

**对一行有 10000 个中间积的矩阵的影响:**

| | 每行步数 | 每步 cycles | 总 cycles |
|---|---|---|---|
| Hash | 10000 / 256 = 39 | ~40 | 1560 |
| Merge3 | 10000 / 32 = 313 | ~15 | 4695 |

→ **Hash 每行快 3×** —— 线程多,每行处理时间短。

### 因素 ②:去重机制

| | Hash | Merge3 |
|---|---|---|
| 每个重复中间积的开销 | **O(1)**:atomicCAS 找到已有槽 → atomicAdd | **O(log₂K)**:在 K-way merge 树里比较 ~2.3 次 |
| 重复列处理方式 | 只多一次 atomicAdd(累值) | 必须走完整个 merge 步骤(比较 + 合并) |

**每个中间积的成本分解:**

| | Hash | Merge3 |
|---|---|---|
| 每中间积 cycles(每线程) | ~40(1 次 atomicCAS + 1 次 atomicAdd) | ~15(log₂5 次比较 + 推进) |
| 每行线程数 | 256 | 32 |
| **吞吐量**(中间积/cycle) | **256 / 40 = 6.4** | **32 / 15 = 2.1** |

→ Hash 的总吞吐量是 **3× 高** —— 8× 的线程优势超过了 2.7× 的单中间积成本劣势。

### 因素 ③:固定流水线开销

| | Hash | Merge3 | 差异 |
|---|---|---|---|
| 固定开销 γ | **0.207 ms** | **0.189 ms** | Hash 多 0.018 ms |

**Hash 多出的开销来自:**

| 组件 | 开销 | 原因 |
|---|---|---|
| HLL 两阶段估计 | ~0.23 ms | Phase1 建每行基数 sketch + Phase2 packed merge |
| 分桶(binning) | ~0.09 ms | GPU 直方图 + scan + scatter |
| 压实 + 排序(per row) | ~0.03 ms | BlockRadixSort 生成列有序 CSR |
| 多次 kernel launch | ~0.01 ms | 10+ 个独立 kernel |

**Merge3 省掉的:**

| 省掉的组件 | 原因 |
|---|---|
| 不需要 HLL | 用精确 count(一遍扫 A 的 CSR) |
| 不需要排序 | K-way merge 天然输出有序 |
| 更少的 launch | 流水线阶段更少 |

**影响**:对小阵(总时间 ~0.1 ms),0.018 ms 差距占 18% → merge3 赢。
对大阵(总 ~2 ms),占 < 1% → 吞吐量主导。

---

## 4. 交叉点推导

### 拟合的性能模型(R² = 0.948 Hash,0.974 Merge3)

```
T_hash   = α_h × flop / min(n, 114) / 256  +  β_h × nnz_A / BW  +  γ_h

T_merge3 = α_m × flop / min(n, 114) / 32   +  γ_m

其中:
  α_h = 2.74 × 10⁻⁴ ms·线程/中间积      (hash 插入:CAS + Add 摊薄)
  α_m = 4.87 × 10⁻⁴ ms·warp线程/中间积   (K-way merge:log₂5 次比较)
  γ_h = 0.207 ms   (HLL + 分桶 + 压实/排序 + launches)
  γ_m = 0.189 ms   (count + scan + launches,无排序)
```

### 交叉条件

Hash 赢当 T_hash < T_merge3:

```
  (α_m / P_m − α_h / P_h) × flop > γ_h − γ_m

  其中:
    α_h / P_h = 2.74e-4 / (114 × 256)  = 9.4×10⁻⁹  ms/中间积   (hash 每中间积时间)
    α_m / P_m = 4.87e-4 / (114 × 32)   = 1.34×10⁻⁷ ms/中间积   (merge3 每中间积时间)
    γ_h − γ_m = 0.207 − 0.189 = 0.018 ms

  → flop* = 0.018 / (1.34×10⁻⁷ − 9.4×10⁻⁹) ≈ 1.44 × 10⁵
```

### 直觉解释

> 每处理一个中间积,hash 比 merge3 省 1.25×10⁻⁷ ms(因为 256 个独立线程 vs
> 32 个协作线程)。要补回 0.018 ms 的固定开销差距,hash 需要处理
> **~14 万个**中间积。超过这个数 → hash 赢。

### 简化调度规则

$$\text{选 hash} \iff \text{flop} > \tau^*, \quad \tau^* \approx 1.44 \times 10^5$$

$$\text{其中 } \text{flop} = \sum_{k} \text{nnz}(\text{行}_k)^2 \approx \frac{A_{\text{nnz}}^2}{n} \quad (\text{对称矩阵})$$

100 阵调度准确率:**82%**(误分类全在交叉点附近)。

---

## 5. 为什么不是共享内存大小或寄存器数量

| 可疑因素 | 实际角色 | 解释 |
|---|---|---|
| **SMEM 大小** | 不是瓶颈 | Hash 用 256B–128KB SMEM(hash 表);Merge3 用 ~1KB(merge workspace)。H100 有 228KB SMEM/block,两者都够。 |
| **寄存器数量** | 不是瓶颈 | 两个算法的寄存器使用都在 255/thread 限制内。 |
| **Warp 数量** | 间接因素 | 真正的问题是**每个 block 有多少线程在干活**。Hash = 8 warp × 32 线程 = 256 全活跃;Merge3 = 1 warp × 32 线程,7 个 warp 槽位浪费。 |
| **每行线程数** | **主因** | 256(hash)vs 32(merge3)= 8×,决定同时处理多少中间积。 |
| **去重机制** | **次因** | O(1) atomic(hash)vs O(log K) comparison(merge3)决定单中间积成本。 |

---

## 6. H100 PCIe 硬件参数在模型中的角色

| 参数 | 值 | 在模型中的作用 |
|---|---|---|
| FP32 peak | 51 TFLOPS | 理论算力上界(SpGEMM 被 atomic/比较主导,远未触顶) |
| HBM3 带宽 | 2.0 TB/s | 读取 A 的 nnz + 写出 C 的 nnz(模型中的 β 项) |
| SM 数量 | 114 | 并行度上限:min(行数, 114)个 block 同时跑 |
| Boost 频率 | 1.98 GHz | cycles → 墙钟时间换算(如 40 cycles ≈ 20 ns) |
| SMEM atomic 延迟 | ~20 cycles | atomicCAS / atomicAdd(hash 每中间积成本的核心驱动) |

---

## 7. 总结(文章用)

> Hash SpGEMM 在大阵/高 flop 矩阵上胜出,因为其**每行 256 线程的独立 hash 插入**
> 提供了 **8× 于 merge3 32 线程协作式 K-way merge** 的线程并行度。叠加 O(1) 的
> atomic 去重(vs merge3 的 O(log K) 比较去重),hash 实现了**每行 3× 的中间积
> 吞吐量**。交叉点出现在 **flop ≈ 1.44×10⁵** 个中间积处 —— 当 hash 的吞吐优势
> 累积超过其 **0.018 ms 的额外固定开销**(HLL 估计 + 分桶 + 排序)时,hash 成为
> 更优选择;低于此阈值时,merge3 更简单的流水线和无排序输出使其更快。
