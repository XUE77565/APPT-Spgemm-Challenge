# Kernel Fusion 实现详解(小白版)

> 对应图:`fig/launch_fusion_design.png`(直观设计图)、`fig/launch_fusion_detail.png`(前后对比)。
> 这份文档讲 **kernel fusion 在代码里到底是怎么合并的**(两处融合 + D2H 合并)。
> 风格跟 `inno/engopt_beginner_qa.md`(pool 那份)一致,每个词从头解释。

---

## 0. 先懂几个基础概念

| 词 | 通俗解释 |
|---|---|
| **kernel** | 一段丢给 GPU 并行跑的函数。 |
| **launch(启动)一个 kernel** | CPU 喊一声"GPU,去跑这个 kernel",GPU 才开始执行。**每次喊都有固定开销**(像每次打电话都要拨号等接通),跟这个 kernel 干多少活无关。 |
| **launch overhead(启动开销)** | 上面那个"固定开销"。小矩阵计算量≈0,但每个 kernel 照样付这份开销 → 时间全花在这。 |
| **histogram(直方图/计数)** | 数"每个桶里有多少个东西"。比如数每个 hash 桶里有几行。 |
| **atomicAdd(原子加)** | 多个线程同时给同一个计数器 +1 时,保证**一个一个加、不丢不乱**(像收银台一次只接待一个客人)。 |
| **scan / prefix sum(前缀和)** | 把一个数组变成"前缀累加和"。如 `[3,1,2,4]` → `[3, 3+1, 3+1+2, 3+1+2+4]` = `[3,4,6,10]`。用来算"每行/每桶的起始偏移"。 |
| **shared memory(SMEM,共享内存)** | GPU 一个 block 内所有线程共享的快速小内存,比全局显存快很多。 |

---

## 1. 为什么 fusion(回顾)

small 矩阵计算量≈0,但 pipeline 里要启动**很多个 kernel**(分桶、数桶、扫描……),每个 kernel 付一份固定 launch 开销 → **时间 = 一堆 launch 开销叠起来**。

**fusion 的核心:** 把"能合并的 kernel 合成一个、少启动几次"→ launch 开销(图里的红块)从 ~8 个降到 3 个。

代码里做了**两处融合** + **一处 D2H 合并**:

---

## 2. 实现①:binning 融合(compute_bucket + bucket_count → 1 个 kernel)

### 背景:binning(分桶)是干嘛
hash 路径一开始,要把每一行按"它的 hash 表该多大"分到不同的**桶(bin)**:
- 短行 → 小桶(小 hash 表),长行 → 大桶。
- 这样每个桶里的行用**同样大小的 hash 表**,启动一个专门配好的 kernel 处理。

### 融合前(2 个独立 kernel)
```
kernel A "compute_bucket":  每行算出自己属于哪个桶 → 写 bucket_id[row]
kernel B "bucket_count":    数每个桶有几行 → 写 counts[bin](直方图)
```
2 次 launch + 还要 memset 清零 counts。各自跑一遍 A 的所有行(扫两遍)。

### 融合后(1 个 kernel,我们的做法)
**让 kernel A 在算出 bucket_id 的同时,顺手 atomicAdd 进直方图**——一遍搞定:
```cuda
// 融合后的 compute_bucket_kernel(每行一个线程)
int bid = 算出这行属于哪个桶;      // 原 kernel A 的活
bucket_id[row] = bid;             // 写桶号
atomicAdd(&counts[bid], 1);       // ← 顺手 +1 进直方图(原 kernel B 的活,合并进来)
```
- 一次遍历、一次 launch,**数桶这件事被"夹带"进了分桶**——不用单独的 kernel B,也不用 memset。

> **类比:** 融合前 = 先给每个人发号码牌(kernel A),再单独点名数每队几人(kernel B),跑两趟;融合后 = 发号码牌的**同时**在对应队的计数器上按一下(atomicAdd),一趟搞定。

### 配套:D2H 合并(2 次 sync → 1 次)
分完桶,host(CPU)要知道每个桶几行,好决定给每个桶启动多大 grid。
- **前:** 两次单独的 `cudaMemcpy`(拷 counts、拷 offsets),**各跟一次 sync**(等拷完)。
- **后:** 两次都改成 `cudaMemcpyAsync`(异步发起),然后**只 sync 一次**(一次性等两份都到)。

**这处一共省了:** 1 个 kernel launch + 1 次 memset + 1 个 sync 点。

---

## 3. 实现②:scan 融合(thrust → 单 block Hillis–Steele scan)

### 背景:scan(前缀和)是干嘛
分桶/数完之后,要把"每个桶几行"转成"每个桶的行从全局哪个位置开始"——这就是 **scan(前缀和)**。sizing 阶段也要 scan(`est_scan` 算每行估值的累计、`cnnz_scan` 算每行 nnz 的累计→row_ptr)。

### 融合前(用 thrust)
`thrust::inclusive_scan` 是个通用库函数。问题是:**它对小数组(几百个元素)也照样启动好几个 kernel**(它内部有通用流程,不分大小)。小矩阵行数少,这种"杀鸡用牛刀"反而被 launch 开销拖累。

### 融合后(自写单 block Hillis–Steele scan)
对小矩阵(**A_rows ≤ 1024**),用**自己写的一个 kernel** 做前缀和——**1 个 block、1 次 launch** 就算完。

**Hillis–Steele scan 怎么做(并行前缀和):**
- 把数据放进 **shared memory**(快)。
- 第 1 步:每个位置加上"前 1 个"位置的值。
- 第 2 步:每个位置加上"前 2 个"位置的值。
- 第 3 步:加上"前 4 个"……
- 一直翻倍,log₂(n) 步完成。每步之间 `__syncthreads()`(等所有线程同步)。

```
数组:     [3, 1, 2, 4]
第1步(+1):[3, 4, 3, 6]     每个加前1个
第2步(+2):[3, 4, 6, 10]    每个加前2个  → 完成 = 前缀和 [3,4,6,10]
```
- 1024 个元素只要 ~10 步,全在一个 block 里,**1 次 launch**。
- 对比 thrust 的小数组要 3+ 个 kernel,这里只要 1 个。

> **类比:** thrust 像请搬家公司(流程全、但对小活计也要派好几辆车、手续费贵);Hillis–Steele 单 block = 自己一个工人一趟搬完(小活反而快)。

---

## 4. 代码层面小结(两处融合)

| | 融合前 | 融合后(我们的) | 省了 |
|---|---|---|---|
| 分桶 | compute_bucket + bucket_count(2 kernel)+ memset | 1 个 kernel(分桶时 atomicAdd 进直方图) | 1 launch + 1 memset |
| D2H | 2 次 memcpy,各 1 sync | 2 次 async memcpy + 1 sync | 1 sync 点 |
| scan | thrust(小数组也多 kernel) | 单 block Hillis–Steele(1 kernel) | 把 3+ kernel 压成 1 |

合计:**~8 次 launch → 3 次 launch**(见图)。

---

## 5. 为什么大矩阵不回退(gate)

单 block scan **最多 1024 个元素**(一个 block 最多 1024 线程)。所以代码里**加了 gate**:
```
if (A_rows <= 1024)  用单 block Hillis–Steele scan;   // 小阵:1 kernel
else                 还是用 thrust;                    // 大阵:thrust 更合适,不强行单 block
```
- 小阵(占 first100 的 78%)走单 block scan → 省 launch。
- 大阵(行数 >1024)继续用 thrust → 不回退、不损失。
- bcsstk30(大阵)实测 3.24 → 3.11ms,**无回退**。

> 这是"按规模分档"的做法:小阵用轻量单 kernel、大阵保留通用库,各取所长。

---

## 6. 结果 + 一句话

- 小阵 bp* 跟 cuSPARSE 的差距:**0.03–0.08 ms → 0.008 ms**(基本打平)。
- 大阵 bcsstk30:3.24 → 3.11 ms(无回退)。

> **一句话:** kernel fusion = 把分桶的两个 kernel 合成一个(用 atomicAdd 把"数桶"夹带进"分桶")、把 2 次 D2H 的两次等待合成一次、把小阵的 thrust scan 换成单 block Hillis–Steele scan——**少启动几次 kernel、少等几次,小阵的 launch 开销(图里的红块)从 ~8 块降到 3 块**;大阵 gate 保护、不回退。
