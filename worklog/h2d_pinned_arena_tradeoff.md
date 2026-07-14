# 为什么加 pinned arena 会让 h2d 变慢:完整因果链

> 日期:2026-07-13
> 关联:`include/mempool.h` + `src/mempool.cu`(pinned host arena);`src/matrix_utils.cu`(A_buffer 改 pinned)
> 一句话:**arena 不直接拖慢 h2d,而是和「A 曾 pageable」+「大块传输」+「spgemm 上下文」三因子叠加,挤窄了 driver 的 pageable-staging 路径。把 A 改 pinned(绕开 staging)后基本消除。**

---

## 0. 现象

pool(`USE_MEMPOOL=1`)在最大的 2-3 个矩阵(bcsstk30/32)上 h2d ≈ 2× legacy,**可复现(5/5)**;小矩阵不受影响(86/100 pool h2d ≤ legacy)。h2d 代码 legacy/pool **一字不差**。

## 1. 因果链(逐步)

1. **h2d 代码没改**。pool 唯一的运行时新增 = 启动 `mempool_init` 里 `cudaMallocHost(256MB)` 锁一大块 pinned host。
2. 这块 pinned 被注册成 GPU 的 **DMA 映射**(page-locked + BAR1),**进程全生命周期持有** → 持续占用 driver 的 pinned-memory / DMA 资源,改变了 driver 的全局内存状态。
3. A 的 H2D `cudaMemcpy` 在这个被改变的状态下执行。**A 原本是 pageable**(`read_matrix_market` 用 `malloc`)→ memcpy 走 driver 的**内部 staging**(pageable → driver 内部锁页缓冲 → DMA),而非 DMA 直传。
4. staging 路径对 pinned/DMA 资源压力**敏感**。arena 占用大量 pinned/DMA 资源 + **spgemm 上下文**(cuSPARSE handle + warmup 里 5 个方法的一堆 device 分配)进一步抬高内存压力 → **大块 pageable H2D(16MB A)的 staging 被严重拖慢**(2.3→7.8ms)。小块传输 staging 量小,无感。

## 2. 这条链解释了所有观测

| 观测 | 由哪一步解释 |
|---|---|
| h2d 代码没变却变慢 | 步骤 2:arena 改变 driver 全局状态,h2d 在新状态下执行 |
| 慢在 memcpy,不在 cudaMalloc | `[cu] h2dmalloc` 拆分实测:cudaMalloc ~0.1ms 不变,memcpy 2.3→7.8ms |
| 只有最大的矩阵受影响 | 步骤 4:只有大块 pageable 传输才会耗尽 staging |
| 缩小 arena(256→128)没用 | 步骤 2/4:arena 必须 ≥ 最大 C(107MB),仍是"大块",照样挤 |
| **pin A 后基本消除** | 步骤 3:pinned A 走 DMA 直传,绕开 staging,arena 不再伤它 |
| 干净 micro-bench 复现不了 | 步骤 4:缺 spgemm 上下文这个 co-factor;micro-bench 里 pageable+arena 反而完全不慢 |

## 3. 关键证据(本会话实测)

**拆分 h2d = cudaMalloc(dA) + memcpy(A)[bcsstk30]:**
```
legacy: cudaMalloc 0.10ms   memcpy(A) 2.29ms   ← 慢全在 memcpy
pool  : cudaMalloc 0.06ms   memcpy(A) 7.75ms
```

**干净 micro-bench(隔离 arena)——复现不了:**
```
pinned  src, 无 arena: 0.620ms | 256MB arena: 0.730ms  (仅 +18%)
pageable src, 无 arena: 0.634ms | 256MB arena: 0.625ms  (无影响!)
64MB pageable:         2.42ms  | 256MB arena: 2.50ms    (无影响)
```
→ arena 单独(无论 src pinned/pageable、16/64MB)都不显著拖慢 H2D。**spgemm 上下文是必要 co-factor。**

**A 改 pinned 后(根因修复)[bcsstk30 memcpy(A)]:**
```
legacy: 2.29ms → 0.61ms   (3.8× 快:pageable staging → pinned 直传 DMA)
pool  : 7.75ms → ~1.5ms   (惩罚从 +5ms 缩到 +0.9ms)
```

## 4. 已证实 vs 推测(诚实标注)

- **已证实**(可复现实测):慢在 memcpy、只影响大矩阵、A 曾 pageable 是放大器、pin A 基本消除、需 spgemm 上下文才复现。
- **推测**(最自洽的 driver 层解释,未用 nsys 确认):"arena 的大块 pinned 映射挤窄了 driver 的 pageable-staging 路径"——与所有观测一致,但**确切的 driver 内部机制(staging pool 大小 / BAR1 / DMA 调度)需 nsys/nvprof 才能定位**。

## 5. 修复与结论

**修复(已落地):`matrix_utils.cu` `malloc→cudaMallocHost`(A_buffer 锁页),`main.cu` `free→cudaFreeHost`。**
- 这是**独立于 pool 的纯收益**:所有方法的 H2D 都更快(A 被上传 5 次,5× 省钱)。
- 顺带把 pool 的 h2d 副作用从 +5ms 压到 +0.9ms(残留是 pinned-DMA 的轻微上下文影响,小到可忽略)。

**net**:bcsstk30 上 pool 让 d2h −25ms、h2d +0.9ms(pin A 后)→ 每法净 **−24ms**。池子整体仍非常划算。

## 6. 教训

1. **结构论证("代码没改")不能凌驾于可复现实测之上**——要去找代码之外的 driver 全局副作用。
2. **"锁大块 pinned" 不是免费的**:它会改变 driver 的内存/DMA 状态,可能间接影响其它(代码上无关的)传输。为 d2h 锁页 → 可能伤 h2d,这是真实 tradeoff。
3. **pageable 源是放大器**:任何 H2D 都应优先 pin 源 buffer(直传 DMA),既快又避开这类 staging 副作用。
4. 重尾分布的 per-matrix 计时:大矩阵的异常往往是"真实副作用 × 上下文",不要轻易当噪声用统计压掉。

---

## 7. 关联发现:pin A 也大幅加速了 **legacy 的 d2h**(同根因)

排查"为什么 pin A 后,d2h 的 pool-vs-legacy 加速比从 **−96%** 掉到 **−11%**"时,用**受控对照**(同矩阵、同 session,A pinned vs pageable,legacy cu d2h)发现:

```
                 A=pinned legacy    A=pageable legacy
1138_bus d2h       0.035 ms           1.092 ms      ← A pageable 慢 31×
494_bus            0.029 ms           0.647 ms      ← 22×
ash85              0.029 ms           0.903 ms      ← 31×
bcspwr01           0.026 ms           0.866 ms      ← 33×
bcsstk30          33.3   ms          46.4   ms       ← 大矩阵只 1.4×
```

**结论**:加速比缩小**不是 pool 变慢,而是 legacy 被 pin A 治好了**。A pageable 时 legacy 的 d2h(`cudaMallocHost(C)`)在小矩阵上异常慢(~0.9ms);pin A 后恢复到 ~0.03ms。pool 的 d2h 一直是 ~0.035ms,所以差距从 −96% 收窄到 −11%。

**统一根因**:A pageable + spgemm 上下文 → 整体内存操作变慢,有两个表现:
- **h2d**:A pageable + arena → 慢 H2D memcpy(A)(本文 §1)。
- **d2h**:A pageable(连 legacy 都中招)→ 慢 `cudaMallocHost(C)`(本节)。

**pin A 拿掉"A pageable"这个共同因子 → h2d 和 d2h 一起治好**。所以 pin A 的独立收益比预期大得多:不止修 h2d,还把 legacy 的 d2h 提速 ~30×(小矩阵)。pool 的**剩余**价值主要在大矩阵(跳过大 C 的昂贵锁页;mean d2h −81%),小矩阵 pin A 后 legacy 已很快,pool 优势小。

### 又一次教训:干净 micro-bench 复现不了上下文效应

我中途曾基于"干净 micro-bench(3 次 cudaMallocHost)看不出差别"就**retract 了"pin A 让 legacy d2h 变快"**——错的。这是 micro-bench **第三次**骗我(h2d、本节):**spgemm 上下文相关的内存效应必须在 spgemm 内做受控对照才能看到,干净 micro-bench 复现不了。** 以后这类问题直接做 in-spgemm 受控对照,别再信干净 micro-bench。
