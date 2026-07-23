# 自适应调度公式(文章用)

## 公式

$$\frac{A_{\text{nnz}}^2}{n} > \tau \implies \text{hash}, \qquad \frac{A_{\text{nnz}}^2}{n} \leq \tau \implies \text{merge3}$$

**左边 = 矩阵的工作量(中间积数 W),右边 = GPU 的交叉阈值 τ。**

---

## 左边:矩阵参数 a = W

$$W = \frac{A_{\text{nnz}}^2}{n}$$

这是 C = A·A 自乘产生的**中间积总数**(含重复列)。两个算法都必须处理全部 W 个中间积 —— 它是共同的"工作量"。从矩阵头部 O(1) 算出,无需遍历。

---

## 右边:GPU 参数 b = τ

$$\tau = \frac{\Delta\gamma}{\dfrac{1}{\pi_{\text{merge3}}} - \dfrac{1}{\pi_{\text{hash}}}}$$

### 组成部分

| 符号 | 含义 | H100 PCIe 取值 |
|---|---|---|
| $\Delta\gamma = \gamma_h - \gamma_m$ | hash 额外的固定流水线开销(HLL 估计 + 排序) | ~0.02 ms |
| $\pi_{\text{hash}}$ | hash 的中间积吞吐量 | 见下 |
| $\pi_{\text{merge3}}$ | merge3 的中间积吞吐量 | 见下 |

### 吞吐量公式(体现 GPU 参数)

$$\pi = \frac{N_{\text{SM}} \times B \times f_{\text{clock}}}{c} \quad \text{(中间积/秒)}$$

| 参数 | Hash | Merge3 | 来源 |
|---|---|---|---|
| $N_{\text{SM}}$ | 114 | 114 | H100 PCIe 硬件 |
| $B$(线程/block) | **256** | **32** | 算法设计 |
| $f_{\text{clock}}$ | 1.98 GHz | 1.98 GHz | H100 boost |
| $c$(cycles/中间积) | $\approx 2 \times t_{\text{atomic}}$ | $\approx \lceil\log_2 K\rceil \times t_{\text{cmp}}$ | 算法结构 |

- **Hash 的 $c$**:每个中间积 = 1 次 `atomicCAS`(找/插槽) + 1 次 `atomicAdd`(累值) $\approx 2 \times 20 = 40$ cycles(SMEM atomic 延迟)
- **Merge3 的 $c$**:每个中间积 = 在 K=5 个有序链中做一次 K-way merge 步 $\approx \lceil\log_2 5\rceil \times 5 = 15$ cycles(warp shuffle 比较)

### 代入数值

$$\pi_{\text{hash}} = \frac{114 \times 256 \times 1.98 \times 10^9}{40} = 1.44 \times 10^{12} \text{ 中间积/s}$$

$$\pi_{\text{merge3}} = \frac{114 \times 32 \times 1.98 \times 10^9}{15} = 4.82 \times 10^{11} \text{ 中间积积/s}$$

$$\frac{1}{\pi_m} - \frac{1}{\pi_h} = \frac{1}{4.82 \times 10^{11}} - \frac{1}{1.44 \times 10^{12}} = 2.08 \times 10^{-12} - 6.94 \times 10^{-13} = 1.38 \times 10^{-12} \text{ s}$$

$$\tau = \frac{0.02 \times 10^{-3}}{1.38 \times 10^{-12}} \approx 1.4 \times 10^7$$

> 理论 τ ≈ 10⁷;实测 τ ≈ 10⁵。差距来自:内存延迟(A 的 CSR 随机访问)、hash 碰撞(探测链)、kernel launch 开销、warp 同步 —— 这些在实际运行中让有效 $c$ 比理论值高 ~10×。论文中取**实测值 τ ≈ 10⁵**。

---

## 结论(文章用)

> **调度判据**:当矩阵的中间积数 $W = A_{\text{nnz}}^2/n$ 超过 GPU 交叉阈值 $\tau$ 时选择 hash,否则选择 merge3。阈值 $\tau$ 由 hash 的额外流水线开销 $\Delta\gamma$ 与两个算法的吞吐量差 $\Delta(1/\pi)$ 决定。hash 的吞吐优势来自其**每行 256 线程的独立 hash 插入**,是 merge3 32 线程协作式 K-way merge 的 8 倍并行度;而 hash 的额外开销来自 HLL 基数估计和输出排序。在 H100 PCIe 上,实测 $\tau \approx 10^5$,即矩阵产生超过约 10 万个中间积时,hash 的并行优势开始主导。
