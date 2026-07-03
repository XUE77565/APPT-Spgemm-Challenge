# WarpDrive 精读总结

> 本文档是对 [WarpDrive(IEEE HPCA 2025)](ref/WarpDrive_GPU-Based_Fully_Homomorphic_Encryption_Acceleration_Leveraging_Tensor_and_CUDA_Cores.pdf) 的精读总结。
>
> 注意:这篇属于**全同态加密(FHE)的 GPU 加速**领域,与 ref/ 中其他 8 篇 SpGEMM 论文不同主题。
>
> 关于把 WarpDrive 的方法论迁移到本项目 SpGEMM 代码的实践建议,另见 [WarpDrive_SpGEMM_Application.md](WarpDrive_SpGEMM_Application.md)。

---

## 1. 基本信息

- **标题**:WarpDrive: GPU-Based Fully Homomorphic Encryption Acceleration Leveraging Tensor and CUDA Cores
- **作者 / 单位**:Guang Fan、Mingzhe Zhang(通讯)等;**蚂蚁集团**(杭州)+ 中科院大学密码学院 + 中科院信工所(IIE)
- **发表**:**IEEE HPCA 2025**(国际高性能计算机体系结构顶级会议),pp. 1187–1200
- **DOI**:10.1109/HPCA61900.2025.00091

---

## 2. 研究背景与动机

### 2.1 什么是 FHE,为什么又贵又重要
**全同态加密(FHE)** 允许在**不解密**的情况下直接对密文做计算,数据全程保密,适合云计算、隐私机器学习等场景。但代价极大 —— **FHE 比明文计算慢 4 个数量级以上**。

FHE 代际演进:
- 第二代:**BGV、BFV**(提升效率与噪声管理);
- 第三代:**FHEW、TFHE**(提升 bootstrapping 效率,但不支持 batching);
- 第四代:**CKKS**(擅长近似/浮点运算,隐私机器学习主流,**本文重点**)。

### 2.2 主要计算瓶颈
FHE 核心运算是海量多项式乘法,**数论变换(NTT)** 把多项式乘转成点乘,是最耗时的原语;而密文乘(HMULT)、密钥切换(KeySwitch)、旋转(HROTATE)、重缩放(Rescale)、**自举(Bootstrapping)** 又反复调用 NTT/INTT。

### 2.3 现有 GPU 方案的三大痛点
1. **Tensor Core 方案内存优化不足**:即便用了 Tensor Core,pipeline stall 严重,算力发挥不出来(TensorFHE 的 NTT 五阶段中,内存相关停顿占 54%~99.5%);
2. **只用单一计算单元**:GPU 的 CUDA Cores 和 Tensor Cores 是分离且**可并发**的,但现有方案只用其一;直接套用其他领域的并发方案(如 Tacker)对 FHE 不适用;
3. **依赖大批量密文**:以往靠喂 GPU 大量并发密文填满资源,但 CKKS 本身已支持单密文 SIMD,真实场景给不出那么多密文 → 单密文下硬件利用率很低(INTT 利用率 <25%,KeySwitch 多数 kernel <61%),且大批量会加剧 FHE 本就紧张的显存压力。

---

## 3. 三大核心创新(重点)

### 创新 ①:高效 Tensor Core-Based NTT
- **Warp 级任务分配**:把以往 5 个独立 kernel(split → GEMM → 中间 → GEMM → merge,各自要走显存中转)**融合成单个 warp 级 kernel**,用 WMMA 指令在 warp 粒度直接调用 Tensor Cores,**避免频繁片上/片外数据交换** → 消除长延迟停顿;
- **更深的 2-level 分解**:基于 4-step NTT 把 N 迭代分解两层,旋转因子矩阵从 2^16 降到 **2^8(可放进共享内存)**,矩阵乘运算量降到原来的 **1/8**;
- **用寄存器减少共享内存访问**(借鉴对 Tensor Core 线程寄存器分配的逆向工程结果);
- **模归约**:NTT 内用 Montgomery(比 Barrett 快约 10%),其他运算用 Barrett;测了 Karatsuba 乘法但无收益,弃用。
- **成效**:相比 SOTA Tensor Core 方法,**指令数减少 73%,流水线停顿减少 86%**。

### 创新 ②:Tensor Cores 与 CUDA Cores 并发使用(标题亮点,首次实现)
本文最大卖点 —— **首次让两类核心在同一 kernel 内真正并发**。建立统一框架 WarpDrive-NTT,提供多个变体:
- **WD-Tensor**:Tensor Core 主导(基线);
- **WD-CUDA**:让 CUDA Cores **直接做 32 位 GEMM**(不像 Tacker 那样做低精度比特拆分);
- **WD-BO**:用**蝶形运算(butterfly)**而非 GEMM 做内层 NTT(高基 radix-16/8/4);
- **WD-FUSE(默认)**:融合 WD-Tensor + WD-BO,比 WD-Tensor 再提升 **4%~7%**,首次证明"Tensor+CUDA 并发"能超越任一单一单元。

**关键分配策略**:跨 block 把任务绑到固定 SM 做不到稳定,但**同一 block 内所有 warp 必落在同一 SM**。于是每个 NTT block 内用 **4 个 warp 跑 CUDA-NTT + 4 个 warp 跑 Tensor-NTT**,正好覆盖一个 SM 内全部处理单元,让两类任务重叠执行。

### 创新 ③:Parallelism-Enhanced Kernel(PE Kernel)—— 挖掘密文内并行
- 一条密文含多个多项式(HMULT 涉及 a/b;KeySwitch 展开成 dnum 个),RNS 又引入维度 L。以往以"单个多项式"为 kernel 输入,只用了 N、L 两维;
- WarpDrive 让**单个 kernel 处理整条密文的所有多项式**,既减少显存压力,又能和 batching 叠加降低所需 batch size;
- **效果**:KeySwitch 所需 kernel 数从 59/90/109 降到 **11(减少 81%~90%)**;计算吞吐利用率最高提升 **1.87×**,内存吞吐利用率最高提升 **2.12×**。

---

## 4. 实验结果

**平台**:NVIDIA **A100-PCIE-80G**;对比 TensorFHE、Cheddar、Liberate.FHE、100x、GME 等。

| 指标 | 结果 |
|------|------|
| **NTT 吞吐** | 相比 **TensorFHE 加速 9.7×~13.4×**;相比 CPU 加速 **1305×~1692×** |
| **停顿分析** | 整体周期数减少 **86%**,内存相关停顿从 70% 降到 21.2% |
| **同态操作延迟**(vs 100x opt) | HMULT 1.30–1.82×、HROTATE 1.30–1.88×、RESCALE 1.64–1.93× |
| **HMULT 吞吐** | 相比 TensorFHE 加速 **1.37×~3.46×**;相比 CPU 加速 **260×~726×** |
| **Bootstrapping** | **97 ms**(BS=16)vs TensorFHE 250 ms(BS=128) |
| **HELR** | 78 ms/it(BS=16)vs TensorFHE 220 ms/it(BS=64)→ **2.82×**,且 batch 小得多 |
| **ResNet-20** | 4.77 s vs TensorFHE 4.94 s |
| **AES-CRT 转加密** | 3.5 分钟,比 48 核 CPU 快 **31.6×** |

> 核心亮点:**在小 batch / 单密文条件下优势最明显**(这正是以往 GPU 方案的软肋)。

---

## 5. 局限与适用场景

**局限**
- 相比 **GME**(改造 GPU 微结构的专用加速器)仍较慢 —— WarpDrive 的优势在于**纯软件、可部署于商品 GPU**;
- 精度策略:只用 INT32 + Tensor Cores,放弃 FP32(尾数 24 位精度不足)与 FP64(数量少、转换有开销)。

**适用场景**
- 隐私机器学习推理(ResNet、HELR)、密文自举、AES-CRT 转加密等需要高性能 CKKS 的 GPU 部署;
- 尤其在**单密文 / 小 batch** 条件下相对以往 GPU 方案优势最大;
- NTT 与 kernel 设计可迁移到 BGV/BFV/TFHE 等其他 FHE 方案。

---

## 6. 一句话总结

> **WarpDrive = 用"warp 级 kernel 融合 + 2-level NTT 分解"榨干 Tensor Core,首次让 Tensor Cores 与 CUDA Cores 在同一 kernel 内并发(4+4 warp 分配),并用 PE Kernel 把整条密文的并行挖满,从而在单密文/小 batch 下把 CKKS 加速到新 SOTA(NTT 快 ~13×、Bootstrapping 97ms)。** 它的哲学是 —— **不靠堆更多硬件,而靠更精细的任务划分与资源并发调度来压榨现有算力**。

---

## 相关文档
- [SpGEMM_Survey.md](SpGEMM_Survey.md) —— ref/ 中 8 篇 SpGEMM 论文的综述
- [WarpDrive_SpGEMM_Application.md](WarpDrive_SpGEMM_Application.md) —— 把 WarpDrive 的 GPU 优化方法论迁移到本项目 SpGEMM 代码的实践建议
