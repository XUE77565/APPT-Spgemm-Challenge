# 创新点

> 这些是本项目对 SpGEMM GPU 领域的**独立创新主张**,可作为论文/竞赛的核心贡献。

---

## 创新点 1:多变量自适应调度公式

### 公式

$$\frac{A_{\text{nnz}}^2}{n} > \tau \implies \text{hash}, \qquad \frac{A_{\text{nnz}}^2}{n} \leq \tau \implies \text{merge3}$$

$$\tau = \frac{\Delta\gamma}{\frac{1}{\pi_{\text{merge3}}} - \frac{1}{\pi_{\text{hash}}}}, \qquad \pi = \frac{N_{\text{SM}} \times B \times f_{\text{clock}}}{c}$$

**左边 = 矩阵工作量(O(1) 从矩阵头部算出);右边 = GPU + 算法确定的交叉阈值。**

### 拟合精度

100 阵实测,多变量公式(4 变量:flop + n + max_row_nnz + skew):

$$\text{hash} \iff -1.31 \cdot \log\Phi + 1.21 \cdot \log n + 1.98 \cdot \log r_{\max} - 2.43 \cdot \log\sigma + 1.29 < 0$$

- R² = 0.824(单变量 flop_proxy 只有 0.500)
- 调度准确率 **95%**(旧规则 92%,单变量 73%)

### 创新性

- **不是 Ocean 的做法**:Ocean 用 HLL 估计 + bin-snnap 定 hash 表大小,不做 hash vs merge 方法选择。我们做的是**跨方法自适应**(hash vs merge3),基于多变量公式而非单阈值。
- **不是 cuSPARSE/SparseX 的做法**:它们做库级调度(选库),我们是算法级调度(同框架内选方法)。
- **公式含 GPU 参数**:τ 中的 π 包含 N_SM、threads/block、clock、atomic 延迟 → **GPU 架构感知**。换 GPU(H100→A100)只需改参数,不需重新拟合。

### 文章引用价值

> "We propose a multi-variable dispatch formula that selects between hash-based and merge-based SpGEMM accumulators based on matrix properties (flop proxy, dimension, max-row-nnz, skewness) and GPU hardware parameters (SM count, thread parallelism, atomic latency). The formula achieves 95% dispatch accuracy on 100 SuiteSparse matrices, improving from 92% (rule-based) and 73% (single-variable). The crossover threshold τ is analytically derivable from the throughput gap π_hash − π_merge3, which stems from hash's 8× higher thread parallelism per row (256 vs 32)."

---

## 创新点 2:hash/merge3 自适应框架(O(1) 指标 + 自动回退)

### 框架

```
入口:spgemm_self_product_adaptive(A)
  │
  ├─ O(1) 计算指标:flop_proxy = A_nnz²/n, skew = max_row_nnz / avg
  │
  ├─ 多变量公式 → score
  │
  ├─ score < 0 → hash SPA(HLL 估计 + SMEM hash + compact+sort)
  │              │
  │              └─ hash 溢出(distinct > HASH_CAP)→ 自动回退 merge3
  │
  └─ score ≥ 0 → merge3(K-way column-bucket merge)
```

### 创新性

- **集成进源码**,不是外部脚本(Ocean 的 dispatcher 是库级,SparseX 是多库协同)。
- **O(1) 指标**:flop_proxy 从 A_nnz²/n 算出,零额外 kernel。max_row_nnz 从一遍 O(n) host 扫描(极廉价)。
- **自动回退**:hash 溢出 → merge3 兜底,保证正确性。

### 文章引用价值

> "Our adaptive dispatcher runs in O(1) at the entry point (no extra kernel launches), selects between two complementary algorithms based on a fitted formula, and falls back automatically on hash overflow."

---

## 创新点 3:HLL + bin-snap 统一 sizing

### 做法

HLL 两阶段估计(P7, m=128,Ocean 架构对齐)→ hll_merge 输出 bin-snapped est → **同时确定**:
1. tmp buffer 的 per-row 槽位(row_off = est 的 prefix scan)
2. hash 表大小(compute_bucket 直接读 est)

一个 HLL 估计信号驱动整个 pipeline 的内存分配,替代了传统的 flop_ub(松 19×)和独立 count(多一个 kernel)。

### 创新性

- Ocean 用 HLL 但**不做方法选择**(它只走 hash 路径)。我们用 HLL 做**跨方法 + 同方法**的双重 sizing。
- bin-snap(向上取整到 hash 表桶大小)在 hll_merge kernel 内完成(output 阶段),不需要额外 pass。
- 小阵 streamline(count 替 HLL)是 bin-snap 框架的扩展(按 A_nnz 选估计器,count_intermediates_par_kernel 就位)。

### 文章引用价值

> "We unify buffer sizing and hash-table sizing through a single HLL estimate signal, bin-snapped to power-of-2 hash table sizes. This eliminates the separate flop_ub count pass (saving 1 kernel) and the redundant buffer sizing pass, achieving 2.9× tighter buffer allocation than deterministic upper bounds."

---

## 创新点 4:cudaEvent 同口径 profiling 方法论

### 做法

用 RAII HashProf(cudaEvent)包住每个 GPU phase → compute-only = TOTAL(GPU) − h2d − d2h。内部控制流 D2H(h_bin_count 等)纳入对应 phase → 与 Ocean 的 stats.json 同口径(都是 GPU stream 时间,排除边界 h2d/d2h)。

### 创新性

- **发现并修正了 host 时间戳的系统偏差**:gem5 CPU 争用让 host 时间戳膨胀(小阵 +125%),导致之前的性能分析归因错误(compact 被误认为 7ms straggler,实际 0.8ms)。
- **对齐 Ocean 的计时边界**:确认 Ocean 的 timing.* 不含 h2d/d2h(在 timing 循环外),我们的 compute-only 同样排除 → 可直接对比。

### 文章引用价值

> "We establish a same-methodology timing framework using cudaEvent (GPU-stream timing, excluding boundary h2d/d2h) for fair cross-method comparison. This corrected a systematic bias in host-timestamp measurements caused by CPU contention, which had inflated small-matrix timings by up to 125%."

---

## 创新点对比总结

| 创新点 | 类型 | 独立性 | 论文定位 |
|---|---|---|---|
| 多变量调度公式 | 算法 + GPU 架构感知 | Ocean/cuSPARSE 都没做跨方法多变量公式 | **核心贡献** |
| hash/merge3 自适应框架 | 系统设计 | 集成进源码 + O(1) + 自动回退 | **系统贡献** |
| HLL + bin-snap 统一 sizing | 内存管理 | Ocean 用 HLL 但不做跨方法 | **优化贡献** |
| cudaEvent 同口径 | 方法论 | 修正测量偏差 | **实验贡献** |
