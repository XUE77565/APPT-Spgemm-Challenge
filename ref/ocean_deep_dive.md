# Ocean 深度解读:思路、技术细节与对你 SpGEMM 的启示

> 日期:2026-07-15
> 论文:Ocean: Fast Estimation-Based SpGEMM on GPU(ICS 2026)
> 作者:Yifan Li, Giulia Guidi(Cornell HPC)
> 代码:https://github.com/CornellHPC/Ocean-SpGEMM(CUDA C++, 支持 A100/H100)
> 摘要文档:`ref/papers/ocean_estimation_spgemm.md`
> 关联:`ref/sort_innovation_directions.md`、`ref/formulation_fusion_ideas.md`、`ref/spgemm_papers_2025.md`

---

## 〇、Ocean 是什么

Ocean 是目前 GPU SpGEMM 的 SOTA(ICS 2026):
- 比 spECK 快 **1.4×**、HSMU 快 **2×**、opSparse 快 **2.6×**、cuSPARSE 快 **18.8×**(A100, 337 方阵)。
- 在 87% 的矩阵上最优(294/337 best)。
- 开源(CUDA C++),支持 A100/H100。

核心贡献:用 **HyperLogLog 估计**替代精确 symbolic pass + **混合累加器**(hash/dense/ESC per-row)+ **间接排序**(indirect sort)。

---

## 一、Ocean 要解决什么问题

当前 GPU SpGEMM 的标准流程是 **two-pass**(symbolic + numeric):

```
Pass 1 (symbolic):  用 Gustavson 遍历所有中间项 → 算出每行输出 nnz
                    ← 目的:指导 shared memory 分配 + binning(分桶)
Pass 2 (numeric):   实际乘法,用 hash/dense 累加器累加
```

Ocean 发现两个瓶颈:

| 问题 | 占比 | 原因 |
|---|--:|---|
| **Symbolic pass 太贵** | ~28% runtime | 它和 numeric 一样遍历所有中间项,但只输出一个数字(每行 nnz) |
| **累加器对短行/长行都不好** | — | 短行(<32 个非零):warp 利用率低;长行:超出 shared mem → 退化为全局 hash(暴跌) |

---

## 二、创新 1:HyperLogLog 替代 symbolic pass ★核心

### 动机

Symbolic pass 的目的是算"C 的第 i 行有多少个**不同的** (i,j)"——这是一个**基数估计**(cardinality estimation)问题。

HyperLogLog(HLL)是已知**最高效**的基数估计算法:
- 空间:O(m) registers(m 通常 32-64,每个 1 byte),**与数据量无关**。
- 更新:每个元素一次 `atomicMax`(无碰撞、无 CAS)。
- 精度:相对误差 ~1.04/√m(32 registers → ~18%;64 → ~13%)。

Ocean 首次将 HLL 用于稀疏线性代数。

### 怎么做(construct-and-merge)

**Step 1:对 B 的每行建 HLL sketch**(预处理,一次)

```
B 的第 k 行有若干 col index:j₁, j₂, j₃, ...
对每个 j,hash(j) → 更新 sketch[k] 的对应 register(取 leading-zeros max)
→ sketch[k] 能估计"B 的第 k 行有多少个不同元素"
```

sketch 只有 32-64 bytes/行(对比:hash 累加器需要 ~12 bytes × nnz/row)。

**Step 2:按 A 的每行 merge 对应的 sketch**

C 的第 i 行 = A[i,:] 涉及的 B 行的**并集**。HLL 的 merge = **element-wise max**(只要用了同一个 hash 函数):

```
C_sketch[i] = max over k in A[i,:] of sketch[k]    ← 逐 register 取 max
```

**Step 3:从 merged sketch 估计 C[i,:] 的 nnz**

用 HLL 标准公式(harmonic mean + bias correction)。

### 为什么比 symbolic 快

| | Symbolic(传统) | HLL estimation(Ocean) |
|---|---|---|
| 做什么 | Gustavson 遍历所有中间项,用 hash/dense 累加器去重,输出 nnz | merge sketch(max 操作),从 sketch 估计 |
| 数据访问 | **不规则**(irregular,随 B 的行结构变) | **规则**(sketch 固定大小,连续访问) |
| 原子操作 | hash 插入有碰撞(atomicCAS) | atomicMax(无碰撞) |
| 耗时 | 28% runtime | **4% runtime** |

### 精度与 overflow

| registers/行 | 平均相对误差 | overflow 率(估计偏小) |
|--:|--:|--:|
| 32 | 13% | 1.2% |
| 64 | 10% | 0.3% |
| 128 | 7% | <0.1% |

Overflow(hash 表分配不够大)时:启动一个 **fallback kernel**(最大 dense 配置,逐行扫描)。overflow 率 <1.2% → fallback 对总性能影响很小。

---

## 三、创新 2:工作流选择(ER/CR 驱动)

HLL 不是万能的。对某些矩阵,estimation 反而比 symbolic 更贵。Ocean 用两个指标动态选择:

### ER (Expansion Ratio) = 中间项数 / nnz(A)

衡量 symbolic 遍历的中间项量相对 A 本身的大小。ER 小 → A 稀疏 → 中间项少 → symbolic 便宜 → **用 symbolic**。

### CR (Compression Ratio) = 中间项数 / nnz(C)

衡量中间项被压缩多少。estimation 的代价是输出后需要 **compaction**(重排到连续 CSR)。CR 小 → 输出 ≈ 中间项 → compaction 贵 → **不用 estimation**。

### 选择规则

| 条件 | 工作流 | 原因 |
|---|---|---|
| 中间项/行 < 64 | **upper-bound**(跳过 symbolic 和 estimation) | 中间项太少,估计固定开销不值;直接用中间项数做上界 |
| ER ≥ 8 且 sampled CR ≥ 8 | **estimation**(HLL) | symbolic 贵 + compaction 便宜 → estimation 收益大 |
| 否则 | **symbolic**(传统) | estimation 不划算 → 回退到精确 symbolic |

**sampled CR**:不全算,采样 3% 的行(随机选 A 的行 → merge sketch → 估 CR)。analysis step 占 7% runtime,HLL sampling 只占 2%。

---

## 四、创新 3:混合累加器(hash + dense + ESC)

Ocean 对不同长度的行用不同累加器:

### Enhanced hash accumulator(长行)★★

传统:整个 hash 表(col_index + value)放 shared memory → 长行超出 → 退化为全局 hash(慢)。

**Ocean 的发现**:**value 可以放 global memory,性能影响很小**:

```
传统:  hash 表 = [col_index (shared)] + [value (shared)]   ← 全挤 shared → 长行溢出
Ocean: hash 表 = [col_index (shared)] + [value (global)]   ← 只 index 在 shared,value 在 global
```

为什么 value 放 global 不慢:
1. **index 操作复杂**(read → compare → swap),必须快 → shared mem。
2. **value 只做 atomicAdd**,GPU 上 **fire-and-forget**(单条 SASS 指令 `RED.E.ADD`,不等返回值)→ global 延迟被流水线隐藏。
3. **FP64 shared-mem atomicAdd 不是原生的**(编译成 CAS 循环),反而比 global 的原生 `RED.E.ADD.F64` 慢!

结果:hash 表可处理 **3× 长的行**,不全挤 shared mem。

### ESC accumulator(短行)

极短行(中间项 < 64):hash 表的固定开销不划算。

Ocean 用 **ESC(Expand-Sort-Compact)**——就是你现在的 sort 路线!优势:
- **不需要知道输出大小**(不像 hash 需要预分配表)→ 不依赖 estimation 精度。
- 一个 block 可并行处理**多个短行**(2 或 4 行/block)。
- 短行中间项少 → sort 便宜。

### Dense accumulator(中间行)

Dense 数组 indexed by col,带 bitmap 辅助:
- bitmap 记录哪些位置有非零(1 bit/col)。
- 写入前先 query bitmap:如果已存在 → 只 atomicAdd value;如果空 → 写 col + value + set bitmap。
- CR 高时(输出 << 中间项)bitmap query 大幅减少 shared mem 写。

### 选择策略

每行选**最小资源需求**的配置。Dense vs hash:资源相同时选 dense(更快)。ESC 仅在 upper-bound workflow 中使用。

---

## 五、创新 4:间接排序(Indirect Sorting)★ 最直接适用于你的 ESC

hash 累加器输出后,每行的 col 是**无序的**(hash 插入顺序)→ 需排序满足 CSR 格式。

**传统**:排 key-value 对 = 32-bit col + 64-bit val = **12 bytes/元素**。

**Ocean 间接排序**:

```
Step 1: 生成 key+ptr 对
  key  = col_index          (排序键)
  ptr  = 指向 value 的位置   (≤14 bit,on-chip 内存空间小)
  packed = (key << 14) | ptr  ← 打包成一个 32-bit 整数

Step 2: radix 排序 packed 值
  begin_bit = 14          ← 跳过 ptr 位(低 14 位)
  end_bit   = 32          ← 只排 key 位(高 18 位)
  → 排序后 packed 按 key(col)升序

Step 3: 按 ptr gather value
  for each sorted packed:
      col = packed >> 14
      val = value_buffer[packed & 0x3FFF]    ← ptr 低 14 位
      write to output CSR
```

**效果**:
- 排序数据从 12 bytes/元素 → **4 bytes/元素**。
- Radix sort 是**内存带宽 bound** → 流量减 3× → **sort 快 ~3×**。
- 寄存器压力降低(32-bit vs 96-bit)。

**对你的 ESC 直接可用**:你现在排 64-bit key + 32-bit val = 12B/元素。改用 indirect:32-bit (key+ptr packed) = 4B/元素 → sort 内存流量减 3×。

---

## 六、Ocean 的完整流程

```
┌─────────────────────────────────────────────────┐
│ 1. Analysis (7% runtime)                        │
│    ├─ 算 ER:  O(nnz_A)                           │
│    ├─ 对 B 每行建 HLL sketch: O(nnz_B)           │
│    └─ 采样 A 的行(3%)→ 估算 sampled CR          │
├─────────────────────────────────────────────────┤
│ 2. Workflow selection                            │
│    ├─ 中间项/行 < 64 → upper-bound               │
│    ├─ ER≥8 且 CR≥8 → estimation (HLL)            │
│    └─ 否则 → symbolic (传统 Gustavson)            │
├─────────────────────────────────────────────────┤
│ 3. Size prediction (4% if est / 28% if symbolic) │
│    estimation: merge B sketches → 估 C 每行 nnz   │
│    symbolic: 精确 Gustavson 遍历                  │
│    upper-bound: 中间项数做上界                    │
├─────────────────────────────────────────────────┤
│ 4. Binning: 按 predicted nnz 分桶 + 选累加器       │
│    hash / dense / ESC, 多种配置                   │
├─────────────────────────────────────────────────┤
│ 5. Numeric computation (main cost)               │
│    per-row accumulator:                          │
│    ├─ hash: shared(index) + global(value)         │
│    ├─ dense: bitmap-assisted                     │
│    └─ ESC: expand → sort → compact (短行)         │
├─────────────────────────────────────────────────┤
│ 6. Post-processing (~8%)                         │
│    ├─ hash 输出: indirect sort (32-bit key+ptr)   │
│    └─ compaction: 重排到连续 CSR                  │
└─────────────────────────────────────────────────┘
```

---

## 七、性能数据(ICS'26, A100, 337 方阵)

| 方法 | 最优矩阵数 | avg GFLOPS | vs spECK |
|---|--:|--:|--:|
| **Ocean** | **294/337(87%)** | **63.7** | **1.4×** |
| spECK | 22 | 46.2 | 1.0× |
| HSMU | 0 | 32.1 | 0.69× |
| opSparse | 16 | 24.2 | 0.52× |
| cuSPARSE | 0 | 3.39 | 0.07× |

Ocean 的优势随**中间项增多(矩阵变大)**而增大(estimation 的相对开销摊薄 + enhanced hash 处理长行)。

---

## 八、Ocean 的"符号"阶段是什么(对照 Gustavson)

| | Gustavson(你的 ESC) | Ocean |
|---|---|---|
| **符号阶段** | count(中间项数)+ scan(行偏移) | analysis(中间项数 + ER/CR 统计)+ estimation(HLL 预测输出 nnz) |
| **需要知道什么** | 每行**中间项数**(展开多少条 COO) | 每行**输出 nnz**(hash 表开多大) |
| **不算什么** | 输出 C 的精确结构(交给 sort+reduce) | —(HLL 已经给了估计) |
| **开销** | ~60µs(count+scan) | ~60µs(analysis)+ ~60µs(estimation,仅大矩阵) |

核心计算一样(遍历 A,累加每行的中间项数)。差别在后面:Gustavson 算完就停(后续靠 sort);Ocean 额外做 HLL 估计(为 hash 表分配)。

---

## 九、我们在 H100 PCIe 上的实测对比

### 计算-only(不含 IO,100 矩阵均值)

| 方法 | 计算均值(ms) | vs Gustavson |
|---|--:|--:|
| cuSPARSE | 0.72 | 0.53× |
| **Gustavson(你的 ESC)** | **1.35** | 1.0× |
| **Ocean(hash)** | **0.30** | **4.5× 快** |

### 端到端(各用自己的 IO,100 矩阵均值)

| 方法 | 端到端均值(ms) | vs Gustavson |
|---|--:|--:|
| cuSPARSE | 1.23 | 0.69× |
| **Gustavson(pinned+arena)** | **1.79** | 1.0× |
| **Ocean(pageable IO)** | **3.15** | **1.76× 慢** |

Ocean 计算快 4.5×,但 IO 慢 10×(pageable + 3 次分开传 + 无 arena)→ 端到端反而慢。

### 按规模分桶(3 矩阵均值,计算-only)

| 规模 | Gustavson | Ocean | 加速 | sort 占 Gustavson |
|---|--:|--:|--:|--:|
| 小(C<1K) | 228µs | 114µs | 2.0× | 21% |
| 中(1K-10K) | 283µs | 140µs | 2.0× | 39% |
| 大(10K-100K) | 567µs | 250µs | 2.3× | 34% |
| **巨大(C>100K)** | **11035µs** | **1000µs** | **11.0×** | **73%** |

---

## 十、对你最有价值的三个技术

| Ocean 的技术 | 对你的意义 | 改动量 | 预期效果 |
|---|---|:--:|---|
| **★★★ indirect sorting(§4.2)** | 把你的 sort 从 12B/元素降到 4B/元素 → sort 内存流量减 3× | 小(sort 前打包+sort 后 gather) | sort 快 ~3× |
| **★★ ESC 用于短行(§3.3)** | 你的 ESC 对短行已最优;长行考虑切 hash → 混合策略 | 大(hash kernel) | 自适应最优 |
| **★ value 放 global(§3.3)** | 如果做 hash 路线,value 放 global 可处理 3× 长行 | 中(hash 设计) | 长 hash 表不再溢出 |

---

## 十一、Ocean 代码结构(可直接参考)

```
ocean/
├── kernels/
│   ├── AccumulatorHash.cuh    ← hash 累加器(shared index + global value)
│   ├── AccumulatorDense.cuh   ← dense 累加器(bitmap-assisted)
│   ├── AccumulatorESC.cuh     ← ESC 累加器(短行,block-level sort)
│   ├── HLL.cuh                ← HyperLogLog sketch(construct + merge + estimate)
│   ├── Epilogue.cuh           ← 间接排序(CUB BlockRadixSort,key+ptr packed 32-bit)
│   ├── Analysis.cuh           ← ER/CR 统计 + 工作流选择
│   ├── SpGEMM.cuh             ← 主流程(analysis → prediction → binning → numeric → epilogue)
│   └── Hashmap.cuh            ← hash 函数 + 碰撞处理
├── include/
│   ├── Common.h               ← NUM_SM / SHARED_MEMORY_KB / 累加器配置
│   └── CSR.h                  ← CSR / cuCSR(H2D 在此)
├── src/main.cu                ← 入口(H2D → run → D2H)
├── config/
│   ├── bench.json             ← 标准 benchmark(不输出统计)
│   ├── bench_detail.json      ← 带阶段计时(track_stage_time=true)
│   └── analysis.json          ← 所有参数可见
└── utils/convert.cpp          ← .mtx → .csr 格式转换
```

---

## 十二、一句话总结

> Ocean 用 **HLL 估计**替代精确 symbolic(28%→4%)、用 **混合累加器**(hash为主+dense+ESC)适配不同行长度、用 **indirect sort**(4B/元素)加速 hash 输出排序。计算比你的 ESC 快 4.5×(均值),大矩阵快 11-13×(sort 爆炸)。
>
> 但 Ocean **IO 糟糕**(pageable + 无 arena)——你的 pinned+arena IO 比它快 10× → **端到端你反而比 Ocean 快 1.76×**。
>
> 最优组合:**Ocean 的 hash 计算 + 你的 pinned/arena IO** → 预期端到端 ~0.6ms,比你现在(1.79ms)快 3×,比 Ocean(3.15ms)快 5×。
