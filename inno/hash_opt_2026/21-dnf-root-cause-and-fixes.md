# 21 · 16 阵 DNF 根因分析 + 修复设计(2026-08-26)

数据:CPU 精算 Σflop(scipy 二值化 SpMV,`rajat16 est=904,099,445 vs 精算 904,719,545 = 0.999×`
→ **MinHash 在低 dup 时是准的,不是估计器的锅**)+ rajat16 GPU 崩溃现场。

## 1. 失败分类(全部是进程崩溃,5-46s,非 200s 超时)

| 类 | 阵 | Σflop | dup(flop/nnzA) | 根因 | 证据 |
|---|---|---|---|---|---|
| A:三份叠加 | rajat16/17/18/20/25/28, c-73/c-73b | 0.9-3.9G | ~1400-3000× | tmp(est×16B=14.5GB)+retry(flop×32B)+compact(C×12B=10.7GB) 叠加>80GB | rajat16 实测:`C_nnz=891M → compact cudaMalloc 10.7GB OOM`;同族 rajat31(n=4.7M,maxrow=1252,Σflop=1万)成功 |
| B:retry flop 槽位爆炸 | TSOPF_FS_b300_c2, mouse_gene, audikw_1, dielFilterV3real, Cube_Coup_dt0 | 7.5-54G | 59-6153× | retry 的 rk/rv/rk2/rv2 按 **d_flop** 分槽(非 est 非 n);dup 高的阵 flop≫C,任意批量重试即爆 | TSOPF_FS_b300_c2 Σflop=53.9G → 10% 行重试 = 5.4G 槽×32B=172GB |
| C:待复现 | cage15, wb-edu | 2.1G / 1.6G | 21× / 27× | tmp 上界 33/25GB 理论放得下 → 别的崩点(int 溢出?kernel 断言?) | GPU 被 refresh6 占用,完成后复现 |

对照组:c-big(Σflop 474M,33GB 上界)cu 列跑成了(C=448M),Auto hash 崩 → 介于 A/B。

## 2. 结构性根因

**我们的峰值内存 = f(flop) ≈ tmp(est) + retry(flop×4 数组) + compact(C);Ocean = f(C) 一份**
(symbolic 精确计数 + numeric 直接写终态 CSR)。

两个违背数学事实的放大器:
1. **每行 distinct ≤ min(flop_i, n)**,但 retry 定表和槽位都按裸 flop 分配
   (dup=6153× 的行,flop 是 distinct 的几千倍)
2. **C 被 materialize 多次**:tmp(gapped est 布局)→ retry 排序副本 ×2 → compact 后的 CSR

## 3. 修复设计(按工作量递增)

### Fix 1(一行级):retry 容量封顶 min(flop, n)
```cpp
// retry_prep_kernel / retry_slot_kernel 里:每行容量 = min((long long)d_flop[row], (long long)n)
```
- A 类:rajat16 重试行 29561 × min(flop_row, 94294) → 槽位从 ~2G 降到 ≤2.8G…仍偏大,但
  B 类直接救活(TSOPF:53.9G → ≤n×行数封顶);配合 Fix 2 后 A 类也活
- 风险:极少数行 distinct 真接近 n 且 flop>n 时二次溢出 → 保持行级再重试语义(容量翻倍)

### Fix 2(compact 原地化,治 A 类 + 全体内存税):消掉 12B×C 的第二份
- 现状:compact 从 tmp(Σest×16B gapped)读,写到**新分配**的 C(12B×C)
- 改法:compacted 输出**写回 tmp 自己的地址**(Σest ≥ C 恒成立,前缀写不覆盖未读数据)
- 收益:峰值内存直降 12B×C(rajat16 省 10.7GB)+ 省一次大 alloc/free;
  顺带让"运行时内存"叙事对齐 Ocean
- 注意:tmp_key/tmp_val 是交错 rank 布局,原地紧凑后 C 的 (col,val) 要拷成独立数组 →
  返回时 key 拆 col;或直接改 tmp 布局为两个数组(本就是分开的 d_tmp_key/d_tmp_val!原地各自前缀拷即可)

### Fix 3(架构级,= 方案 5 + Ocean symbolic):直接写终态
- 轻量精确计数 pass(学 Ocean symbolic_dense 0.4-3.3ms):我们已有 count_flop kernel 的
  全量访存,可同步产 exact per-row distinct(或 est+ 行级修正)
- accumulate 直接按 exact offset 写 CSR → tmp/retry/compact 三阶段全消失,内存 = C×12B
- 这是 dense 家族和 hash 家族共用的终局形态;也是 docs/20 战线 3 的完整解

### Fix 4(dense 窗口路径分流):B 类里 flop 巨大但 C 中等的阵
- mouse_gene(C 巨,Ocean 也要 1.7s)除外;TSOPF_FS_b300_c2/audikw/dielFilter/Cube_Coup
  的 Ocean 时间 90-125ms = C 中等 → dense 窗口路径天然无 tmp/retry 结构,直接覆盖

## 4. 验收
- 16 DNF 阵全部出数 + 与 Ocean 对照记录(rajat 族 Ocean 36-47ms,mouse_gene 1726ms,
  wb-edu **1.47ms**)
- 内存峰值监控:跑前后 nvidia-smi 差值记录进 docs
- 全量 geomean 重跑含这 16 阵(它们目前白送在 geomean 外)

*复现命令(GPU 空闲后):`METHOD=adaptive USE_MEMPOOL=1 ./spgemm_test data/ocean/square/cage15.mtx 2>&1 | tail -20` 逐阵抓 stderr。*
