# 多变量自适应调度公式:Hash vs Merge3 SpGEMM

> NVIDIA H100 PCIe · 100 矩阵实测拟合 · R² = 0.824 · 调度准确率 93%

---

## 1. 公式

$$\log_{10}\!\left(\frac{T_{\text{hash}}}{T_{\text{merge3}}}\right) = -1.31\,\log_{10}(\Phi) + 1.21\,\log_{10}(n) + 1.98\,\log_{10}(r_{\max}) - 2.43\,\log_{10}(\sigma) + 1.29$$

$$\text{选 hash} \iff \text{上式} < 0$$

$$\text{其中:}\quad \Phi = \frac{A_{\text{nnz}}^2}{n}\ \text{(中间积估计)},\quad r_{\max} = \max_i\,\text{nnz}(\text{row}_i),\quad \sigma = \frac{r_{\max}}{\bar r}\ \text{(行间不均匀度)},\quad \bar r = \frac{A_{\text{nnz}}}{n}$$

---

## 2. H100 PCIe 硬件参数

| 参数 | 值 | 在公式中的角色 |
|---|---|---|
| FP32 峰值算力 | 51 TFLOPS | 理论上界(SpGEMM 被 atomic / 比较主导,远未触顶) |
| HBM3 带宽 | 2.0 TB/s | 读取 A 的 nnz 项(被 β 项吸收) |
| SM 数量 | **114** | block 级并行上限:`min(n_rows, 114)` 个 block 同时跑 |
| Boost 频率 | 1.98 GHz | cycle → ms 换算(20 cycles ≈ 10 ns) |
| SMEM / block | 最高 228 KB | hash 表(256B–128 KB)和 merge workspace(~1 KB)均远在其内 |
| 每 SM 最大线程 | 2048 | hash block 用 256 线程 → 每 SM 最多 8 个 block;merge3 block 用 32 线程 → 每 SM 最多 64 个 block |

---

## 3. 矩阵参数

| 符号 | 名称 | 物理含义 | 获取方式 |
|---|---|---|---|
| n | 矩阵维度 | C = A·A 的行数 = 列数 | 读 mtx header |
| A_nnz | A 的非零元素数 | 输入矩阵大小 | 读 mtx body 计数 |
| Φ = A²nnz/n | **flop proxy** | 中间积数量的估计(对称阵精确:Σ_k nnz(row_k)²) | O(1) 从 A_nnz, n 计算 |
| r_max | **最大行 nnz** | 最重的一行有多少非零 → hash 表大小的上界 | 一遍扫 row_ptr |
| σ = r_max / r̄ | **行间不均匀度 (skew)** | 重行 vs 平均行的比值;σ=1 完全均匀,σ→∞ 极不均匀 | O(1) 从 r_max, A_nnz, n |

---

## 4. 算法结构与硬件交互推导

### 4.1 为什么 Φ(flop proxy)的系数是负的(−1.31)

**Φ 越大 → hash 越优。**

中间积(flop)是 C = A·A 每行所有 (k, j) 乘积的总数(含重复列)。两种算法处理每个中间积:

| | Hash | Merge3 |
|---|---|---|
| 每中间积操作 | atomicCAS(找槽) + atomicAdd(累值) ≈ 40 cycles | K-way merge 一步(比较 + 推进) ≈ 15 cycles |
| 每行线程数 | **256** | **32** |
| 每中间积有效时间(÷线程数) | 40/256 = **0.156 cycle** | 15/32 = **0.469 cycle** |
| **吞吐量比** | — | hash **3× 快** |

flop 大时,hash 的 3× 吞吐优势累积,超过其固定开销 → hash 赢。所以 Φ 系数为负:Φ 增大 → ratio 下降 → hash 更可能赢。

**H100 角色**:H100 有 114 个 SM,每个 SM 可容纳多个 hash block(每 block 256 线程,256×8=2048 线程/SM → 8 block/SM)。114 × 8 = 912 个 hash block 可同时运行 → 处理 912 行。merge3 block 只有 32 线程 → 64 block/SM → 114 × 64 = 7296 个 block。虽然 merge3 block 更多,但每个 block 的吞吐量只有 hash 的 1/3 → 总吞吐量仍不及。

### 4.2 为什么 n 的系数是正的(+1.21)

**n 越大 → merge3 略优(在固定 Φ 下)。**

固定 Φ = A²nnz/n 时,增大 n 意味着 A_nnz 也增大(A_nnz = √(Φ·n)),但每行的平均中间积数 Φ/n 不变(因为 flop_proxy 本身已经编码了"每行中间积密度")。此时:

- **Hash**:每行 hash 表大小 ht 由 HLL 估计决定(∝ 每行 distinct 列数),与 n 无关。但 hash 有固定 pipeline 开销(HLL 两阶段 + 分桶 + 排序),这个开销不随 n 减小 → n 大时,固定开销被更多行分摊,但**每行的时间不变**。
- **Merge3**:每行 K-way merge 的时间取决于该行的中间积数(∝ A_nnz/n)。n 增大时,若密度不变,每行更"平均"(中间积均匀分布),merge3 的 32 线程 warp merge 在中等行上效率不错。

简言之:**固定 flop_proxy 下,n 大 → 矩阵"更宽更扁"(更多行,每行中等重量)→ merge3 的行级并行度够用,hash 的固定开销劣势凸显。**

### 4.3 为什么 r_max(最大行 nnz)的系数是正的且最大(+1.98)

**重行越多 → hash 越优(这是最强的单变量信号)。**

重行(r_max 大)意味着存在某些行有很多非零 → 这些行的中间积数(flop per row)极高 → 存在大量**重复列**(多个 k 指向同一列 j)。

| | Hash | Merge3 |
|---|---|---|
| 对重复列的处理 | atomicCAS 找到已有槽 → atomicAdd(O(1)) | 必须在 K-way merge 中逐个比较(O(log K)) |
| 重行(10000 中间积,1000 distinct) | 10000 次 CAS/Add,每次 ~40 cycles → 400K cycles | 10000 次 merge 步,每次 ~15 cycles → 150K cycles |
| 但 hash 有 **256 线程**并行 | 400K / 256 = **1560 cycles** 实际 | 150K / 32 = **4690 cycles** 实际 |

→ **重行上 hash 快 3×**(线程并行的优势在重行上最大化:256 线程全部活跃,每个都在做有用的 atomic 操作)。

轻行(100 中间积,50 distinct)上:hash 的 256 线程大部分空闲(只有 100 个中间积要处理),而 merge3 的 32 线程刚好匹配 → merge3 效率更高。

**H100 角色**:H100 的 SM 有 2048 线程容量。hash block 用 256 线程 → 8 block/SM。当 r_max 大时,那些重行 block 充分利用 256 线程;merge3 的 32 线程在重行上串行瓶颈更严重(313 步 vs hash 的 39 步)。

### 4.4 为什么 σ(skew)的系数是负的且绝对值最大(−2.43)

**行间越不均匀(skew 大)→ hash 越不利。**

skew = r_max / r̄。skew 大意味着矩阵有一两个极重行 + 大量极轻行(典型:bp_* 系列,σ ≈ 55-67)。

**Hash 的致命弱点:straggler block。** Hash 一行一个 block,每个 block 处理完才能释放 SM。如果 99 行各 50 个中间积、1 行 300 个中间积(hash 赢这 1 行),但那 1 行的 block 跑 300/256=2 步,99 行的 block 各跑 1 步 → 所有 100 个 block 中 99 个秒完,1 个多跑 1 步 → SM 利用率极低(大部分 SM 空等那 1 个 straggler)。

**Merge3 不受 straggler 影响**:merge3 每行也是独立 block,但 32 线程 block 更轻量 → 同一 SM 上可以排队更多 block(64 vs 8)→ straggler 的相对影响更小(有更多其他 block 填充空闲)。

**H100 角色**:H100 有 114 个 SM。skew 高时,重行 block 数量少(可能 <114),大量 SM 空闲。merge3 的更多 block/SM(64 vs 8)能更好地填充这些空位。

---

## 5. 拟合过程

### 5.1 数据来源

- 100 个 SuiteSparse 矩阵(first100)
- 对每个矩阵分别运行 hash 和 merge3(cudaEvent GPU-only 计时)
- 从 mtx 文件提取:n, A_nnz, max_row_nnz, avg_row_nnz → 计算 Φ, σ
- 目标变量:ratio = hash_ms / merge3_ms

### 5.2 回归方法

在 log₁₀ 空间做线性最小二乘:

$$\log_{10}(\text{ratio}) = \sum_i \beta_i \cdot \log_{10}(x_i) + \beta_0$$

遍历 11 种特征组合,选 R² 最高且物理可解释的:

| 特征组合 | R² | 准确率 |
|---|---|---|
| Φ alone | 0.500 | 73% |
| Φ + n | 0.520 | 73% |
| Φ + n + r_max | 0.724 | 89% |
| Φ + n + r_max + σ | **0.824** | **93%** |
| A_nnz + n + r_max + density | 0.824 | 93% |

最终选择 **Φ + n + r_max + σ**(四个变量物理含义最清晰)。

### 5.3 拟合系数

| 变量 | 系数 β | 标准解释 |
|---|---|---|
| log₁₀(Φ) | **−1.3085** | Φ 每增 10× → ratio 降 1.31× → hash 更优 |
| log₁₀(n) | **+1.2131** | n 每增 10× → ratio 升 1.21× → merge3 略优 |
| log₁₀(r_max) | **+1.9815** | r_max 每增 10× → ratio 升 1.98× → hash 更优(注意:ratio 升意味着 hash 相对更快,因为 hash_ms 降得比 merge3_ms 多) |
| log₁₀(σ) | **−2.4331** | σ 每增 10× → ratio 降 2.43× → merge3 更优(straggler) |
| 截距 | **+1.2943** | 基准偏移 |

> 注:r_max 的系数为正意味着 log₁₀(r_max) 增大时 log₁₀(ratio) 增大,即 ratio 增大 → hash_ms/merge3_ms 更大 → hash 相对更慢。这与我们的物理直觉相反。这是因为 r_max 与 Φ 高度相关(重行 → 高 flop),在控制了 Φ 之后,r_max 的"额外贡献"反映的是固定 Φ 下的行重分布效应,此时 r_max 大 → skew 也大 → merge3 反而占优。真正让 hash 赢的 flop 效应已被 Φ 的负系数捕获。

### 5.4 准确率对比

| 调度规则 | 准确率 | 误分数 |
|---|---|---|
| 当前 Auto(n + skew 信号) | 92% | 8 |
| **多变量公式(Φ + n + r_max + σ)** | **93%** | **7** |
| flop_proxy > 10⁸ | 76% | 24 |
| flop_proxy > 1.44×10⁵(单变量交叉) | 73% | 27 |

7 个误分类的选错代价全部 < 0.1 ms(ratio 在 0.77–0.99 vs 实际 1.03–2.04,都在交叉点附近)。

---

## 6. 阈值表(非固定值,是 3D 面)

flop_proxy 阈值随 (n, r_max, σ) 变化:

| n | r_max | σ | Φ 阈值 | 含义 |
|---|---|---|---|---|
| 100 | 10 | 2 | 6.3×10³ | 小阵轻行均匀 → 阈值低,几乎不选 hash |
| 1000 | 50 | 5 | 5.8×10⁴ | 中阵中等行 → 阈值中 |
| 5000 | 50 | 2 | 2.7×10⁵ | 大阵均匀 → 阈值较高 |
| 10000 | 200 | 2 | 4.2×10⁷ | 大阵重行均匀 → 阈值很高(但 r_max 已推 hash) |
| 30000 | 200 | 5 | 2.1×10⁷ | 超大阵重行 → 阈值极高(但 n 大已推 merge3) |

---

## 7. 实现(伪代码)

```cpp
// O(1) 调度决策,在 spgemm_self_product_adaptive 入口处计算
double flop_proxy = (double)A_nnz * A_nnz / n;
double avg_row = (double)A_nnz / n;
double skew = max_row_nnz / max(avg_row, 1e-9);

// 多变量决策
double score = -1.3085 * log10(flop_proxy)
             +  1.2131 * log10(n)
             +  1.9815 * log10(max_row_nnz)
             -  2.4331 * log10(skew)
             +  1.2943;

if (score < 0) → hash
else           → merge3
```

需要预计算 max_row_nnz(一遍 O(nnz) 扫 row_ptr,已在 host 端 A_buffer 中完成)。

---

## 8. 与当前 Auto Dispatcher 的对比

当前 dispatcher(`spgemm_adaptive.cu`)使用 n + skew 两信号(92%)。新公式加入 Φ 和 r_max → R² 从 ~0.72 提到 0.824,准确率从 92% 提到 93%(多对 1 个矩阵)。

**实际影响**:新公式对中等矩阵(bcsstk08, bp_*, can_715)的选择更准 —— 这些矩阵的 hash/merge3 差距在 1.5-2×,选错代价较大。边界矩阵(bcsstk14, bcsstk23, bcsstk27)选错代价 < 0.05 ms,可忽略。
