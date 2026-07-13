# SpGEMM 五法 profiling 分析:谁最优、瓶颈在哪、怎么加速

> 数据来源:`profile_aa.py` 解析 `results/aa/first100_aa/log/` 的 100 个矩阵日志(`run_aa.sh` 跑 `C = A·A`)。
> 明细:`profile_aa.csv`(每矩阵×方法 分阶段)、`profile_aa_summary.csv`(每矩阵汇总)。
> 测试机:H100 PCIe,`-O3 -arch=sm_90`,DBG=1 分阶段打桩,WAIT=0/WRITE_MTX=0。
> 生成日期:2026-07-13(用 `CU_REF=1` 的二进制重跑,见末尾「数据复现」)。

---

## 0. 测的是什么

5 种 SpGEMM 算 `C = A·A`,每种**独立**做一次 H2D(上传 A)→ 计算 → D2H(下载 C),互不共享缓冲区(为了公平分别计时):

| tag | 方法 | 说明 |
|---|---|---|
| `cu` | **cuSPARSE** | 库基线(`cusparseSpGEMM` workEstimation→compute→copy) |
| `gust` | **Gustavson** | ESC,外层 i,走 CSR 行 + CSC 列 |
| `outer` | 外积 | ESC,外层 k,读 CSC 列 |
| `colw` | 列向 | ESC,外层 j,读 CSR 行 j + CSC 列 k |
| `inner` | 内积 | ESC 符号 + 逐元素归并点积(多一遍 numeric) |

ESC = Expand–Sort–Compress:`count → scan → expand → [compact] → sort_by_key → reduce_by_key → finalize`。

> **方法论提醒**:100 个矩阵是 SuiteSparse id 1..100,**以小矩阵为主**(中位 C_nnz 约 3.5K)。所以"均值"是小矩阵主导的;大矩阵结论看 bcsstk 尾部(§4.3)。

---

## 1. 谁最优

### 1.1 逐矩阵胜场

| 方法 | 胜场 / 100 |
|---|--:|
| cuSPARSE | **52** |
| Gustavson | **48** |
| outer / colw / inner | 0 |

接近平手,但 **outer/colw/inner 从没赢过**——要麼 cuSPARSE 赢,要麼 Gustavson 赢。

### 1.2 按输出规模分桶(均值 ms)

| C_nnz 桶 | # | cuSPARSE | Gustavson | outer | colw | inner | 最优 |
|---|--:|--:|--:|--:|--:|--:|---|
| 小 <1K | 13 | 1.31 | **1.21** | 1.31 | 1.26 | 1.30 | Gustavson |
| 中 1K–10K | 29 | 1.25 | **1.22** | 1.31 | 1.28 | 1.31 | Gustavson |
| 大 10K–100K | 40 | **1.53** | 1.64 | 1.77 | 1.74 | 1.78 | cuSPARSE |
| 巨大 >100K | 18 | **11.23** | 13.45 | 13.59 | 13.64 | 14.04 | cuSPARSE |

### 1.3 按稀疏类别(均值 ms)

| class | # | cuSPARSE | Gustavson | outer | colw | inner |
|---|--:|--:|--:|--:|--:|--:|
| Dense | 6 | 1.3 | 1.3 | 1.4 | 1.3 | 1.4 |
| Mildly sparse | 45 | 1.8 | 2.0 | 2.1 | 2.1 | 2.1 |
| Highly sparse | 39 | 4.9 | 5.7 | 5.8 | 5.8 | 6.0 |
| Extremely sparse | 10 | 3.7 | 3.9 | 4.1 | 4.0 | 4.1 |

### 1.4 结论

- **手写法里 Gustavson 一致最快**,且在**小/中矩阵(C_nnz<10K)上击败 cuSPARSE**。
  原因:cuSPARSE 有固定的 `workEstimation` 开销(~0.7ms),小矩阵上摊不开;Gustavson 没有 cuSPARSE 那套 estimate/compute/copy 三段式。
- **大矩阵(>10K)cuSPARSE 反超,越大赢得越多**。原因见 §2——手写法的 `sort_by_key` 随中间项数膨胀。
- **Gustavson 在手写里最快**的两个原因:① 不建 CSC(符号阶段 0.089ms vs outer/colw/inner 的 0.19ms);② inner 还多一遍数值归并(+0.15ms)。
- outer/colw 性能几乎相同(只是 axis 不同,访存模式接近),都略慢于 Gustavson(建 CSC 的开销)。

---

## 2. 瓶颈在哪

### 2.1 各方法阶段构成(100 矩阵均值,ms)

| 方法 | h2d | d2h | **传输%** | 符号 | 展开(真计算) | 合并 | 数值 | 打包 | 总ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cuSPARSE | 0.24 | **2.06** | **75%** | – | 0.72¹ | – | – | 0.03 | 3.04 |
| Gustavson | 0.20 | **1.87** | 61% | 0.09 | 0.13 | **1.09** | – | 0.04 | 3.42 |
| outer | 0.19 | 1.89 | 60% | 0.19 | 0.11 | 1.04 | – | 0.04 | 3.46 |
| colw | 0.20 | 1.86 | 60% | 0.20 | 0.13 | 1.02 | – | 0.03 | 3.44 |
| inner | 0.20 | 1.87 | 57% | 0.20 | 0.12 | 1.03 | 0.15 | 0.04 | 3.60 |

¹ cuSPARSE 的"计算"列 = workEstimation + compute + copy。

### 2.2 关键观察

1. **d2h 是绝对第一瓶颈**(每法 ~1.9ms,占总时间 **57–75%**)。
   - h2d 很小(~0.2ms,A 小)——**瓶颈全在下载 C,不在上传 A**。
   - d2h 慢在两件事:`cudaMallocHost`(给 host 内存锁页)+ `cudaMemcpy` D2H。PCIe 传输本身已到 ~24GB/s 峰值(Gen4×16),**慢的是反复锁页**。详见 `transfer_optimization.md`。
2. **手写法的第二瓶颈是「合并」,不是「计算」**。
   - Gustavson 合并 1.09ms(占 32%),其中 **`sort_by_key` 占合并的 69%**(0.75ms)。
   - 真正的计算 `expand` 只有 0.13ms(**4%,可忽略**)。
3. **"真正干 SpGEMM 的活"对比**:
   - cuSPARSE workEst+compute+copy = **0.72ms**
   - Gustavson expand = **0.13ms**(比 cuSPARSE 还快!)+ sort 0.75ms = 0.88ms
   - → **手写法在 expand 上其实更快,全输在 sort**。

### 2.3 Gustavson 的 ESC 内部拆解(均值 ms)

| count | scan | expand | sort | reduce | final |
|--:|--:|--:|--:|--:|--:|
| 0.04 | 0.05 | 0.13 | **0.75** | 0.14 | 0.20 |

`sort_by_key` 是 ESC 里唯一显著的开销,其余都是零头。

### 2.4 大矩阵更极端(bcsstk30,n=28924,C_nnz=8.9M)

| | 传输 | | sort | expand | 合计 |
|---|--:|--:|--:|--:|--:|
| cuSPARSE | 39.5 | **82%** | – | 8.4(计算) | 48 |
| Gustavson | 35.0 | 55% | **21.6** | 2.2 | ~63 |

- cuSPARSE:**82% 全在传输**(下载 107MB 的 C + 锁页)。
- Gustavson:d2h 32.4ms(51%)+ **`sort_by_key` 21.6ms(34%)**,expand 才 2.2ms。
- → 大矩阵上 **sort 和 d2h 几乎一样大**,而且 sort 随中间项数线性膨胀,这正是 Gustavson 大矩阵输给 cuSPARSE 的根因。

---

## 3. 假设验证

> "是不是 h2d/d2h 通讯和 esc 的开销?"

**对,但要细分:**

| 假设 | 判定 | 说明 |
|---|---|---|
| h2d/d2h 通讯是瓶颈 | ✅ **第一瓶颈** | 但**几乎全是 d2h**(下载 C + `cudaMallocHost` 锁页);h2d 可忽略 |
| ESC 有开销 | ✅ **第二瓶颈** | 但具体是 ESC 的**合并 `sort_by_key`**,不是 expand/count(那俩加起来才 ~5%) |

一句话:**你感觉到的"ESC 开销"其实是 thrust 全局排序;感觉到的"通讯开销"其实是下载 C 时的反复锁页。** 计算本身到处都不是瓶颈。

---

## 4. 加速路线(按收益排序)

| 优先级 | 杠杆 | 收益 | 影响范围 | 难度 |
|:--:|---|---|---|:--:|
| **1** | **砍 d2h** | **~60% 总耗时** | 所有方法 | 低–中 |
| **2** | **替换 `sort_by_key` 合并** | 大矩阵 ~30–40% | 手写法(翻盘 cuSPARSE 的关键) | 中–高 |
| 3 | outer/colw 复用同一 CSC | ~0.1ms/法 | outer/colw | 低 |
| – | expand/count kernel 优化 | <5% | – | 不值得 |

### 4.1 杠杆 1:砍 d2h(最大、惠及所有人)

现状:每次调用都 `cudaMallocHost` + `cudaMemcpy` D2H,而且 `WRITE_MTX=0` 时 C 下载后**立即 `cudaFreeHost` 丢弃**——纯白费。

- **方案 B `KEEP_ON_DEVICE`(见效最快)**:benchmark 路径结果不回传 host,函数返回 device 指针。d2h 直接归零。bcsstk30 的 cuSPARSE 会从 48ms → ~9ms。
- **方案 A pinned 内存池**:进程启动锁一大块 pinned arena,之后所有 d2h 从中切取——锁页只付一次。生产路径(`WRITE_MTX=1`)用这个,d2h 从 ~1.9ms → ~0.3ms(纯传输)。
- 细节见 `transfer_optimization.md`。**这一步对所有 5 法一视同仁,是最高性价比。**

### 4.2 杠杆 2:干掉 `sort_by_key`(手写法翻盘关键)

现在 ESC 用 `thrust::sort_by_key` 把 `(row<<32|col)` 64-bit key 全局排序再去重。这是手写法大矩阵输给 cuSPARSE 的唯一原因。

候选:
- **radix sort**:64-bit 整数 key 用 radix 比 thrust 的 merge/quicksort 快几倍,且随 nnz 近线性。
- **按 row 分桶归并**:同一 row 的中间项本就连续写入(见 `att_code_walkthrough.md` 的 compact),可在 expand 时直接按 row 做 reduce,免全局排序。
- **两阶段 symbolic→numeric**(cuSPARSE 路线):先数结构再填值,完全不走 sort。改动最大,但天花板最高。

> bcsstk30 的 sort=21.6ms 如果能压到 ~5ms,Gustavson 大矩阵就能追平甚至超过 cuSPARSE(因为 Gustavson 的 expand 0.13ms 远快于 cuSPARSE 的 0.72ms)。

### 4.3 杠杆 3:CSC 复用(小优化)

outer/colw/inner 现在各建一次 CSC(~0.1ms/法)。Gustavson 不建 CSC 所以最快。若一次 att/aa 运行里复用同一 CSC,可省 ~0.2–0.3ms。优先级低。

---

## 5. 数据复现

### 5.1 生成日志

```bash
make                                  # 必须确认 CU_REF=1(见下)
bash run_aa.sh                        # → results/aa/first100_aa/log/<name>.log
# att 模式: bash run_att.sh           # → results/first100_att/log/
```

### 5.2 出 profiling

```bash
.venv/bin/python suitesparse_crawl/profile_aa.py     # → profile_aa.csv(分阶段) + profile_aa_summary.csv(汇总)
.venv/bin/python suitesparse_crawl/profile_att.py    # → profile_att.csv
```

### 5.3 ⚠️ 坑:`CU_REF` 必须为 1

`src/main.cu` 里 cuSPARSE 的**计时**运行(T1 块)被 `#if CU_REF` 包着(`include/spgemm.h` 里 `#define CU_REF 1`)。
如果二进制是用 `CU_REF` 未定义/0 编的,T1 被编译掉,cuSPARSE 只在 warmup 里跑过一次(全程序首次调用),`[cu]` 桩全是首次初始化开销(h2d≈8ms、workest≈24ms),profile_aa.csv 里 cu 行会大 100 倍。

**自查**:`grep -c "T1 cuSPARSE self_product: start" results/aa/first100_aa/log/<name>.log` 应为 1。为 0 就是过期日志/CU_REF 关了 → 确认 `CU_REF 1` → `make` → 重跑。

详见 memory `cu-ref-gates-cusparse-timed-run.md`。

### 5.4 阶段桩 ↔ CSV 列对照

`profile_aa.csv` 每行是一个 (矩阵, 方法),列 = 该方法各阶段耗时(ms),由相邻 `[tag]` 时间戳相减得到:

- 传输:`h2d` + `d2h`
- 符号:`csc` + `count` + `scan`(cuSPARSE 无;Gustavson 无 csc)
- 计算:`expand`(cuSPARSE 是 `workest`+`compute`+`copy`)
- 合并:`sort` + `reduce` + `final`
- 数值归并:仅 `inner` 有 `numeric`
- 打包:`pack`

---

## 6. 一句话总结

> **d2h(下载 C + 锁页)是所有人的第一瓶颈(~60%),手写法的 `sort_by_key` 是第二瓶颈(~30%,大矩阵上 ~40%)——真正的 SpGEMM 计算(expand)只占 ~4%。**
> 加速先砍 d2h(惠及全部),再换掉 sort(手写法翻盘 cuSPARSE 的关键)。
