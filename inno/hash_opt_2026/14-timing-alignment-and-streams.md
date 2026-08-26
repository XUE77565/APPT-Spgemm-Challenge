# 14 · 计时口径对账 + Ocean 多 Stream 发现(2026-08-26)

## 一、计时口径:✅ 一致

| | Ocean | 我们 |
|---|---|---|
| **总口径** | analysis+estimation+numeric+epilogue+prologue | TOTAL(GPU) − h2d − d2h |
| **h2d** | convert 阶段单独计 H2D_time(不计入) | [hash-prof] h2d(减去) |
| **d2h** | main.cu 单独计 D2H_time(不计入) | [hash-prof] d2h(减去) |
| **malloc** | prologue 的 cudaMallocAsync 计入 | dev_alloc(bump arena)分摊到各相位 |

**结论:双方都是 compute-only,h2d/d2h 各自排除——公平。**

细微差异(不利于 Ocean):Ocean 的 prologue(workspace cudaMallocAsync)计入;
我们的 pool_reset 在计时外。这意味着 Ocean 的数字里含一小块 malloc 开销,我们不含。

## 二、⚠ 重大发现:Ocean 用 20 个 CUDA Stream 并行跑 bin kernel

论文说"不描述 multi-stream"→ 我误读了:**它不用 multi-stream 重叠 h2d/d2h,
但用了 20 个 stream 并行跑不同 bin 的 numeric kernel!**

```
Ocean 源码(SpGEMM.cuh:160-162):
  const int NSTREAMS = 20;
  streams.resize(NSTREAMS);  // 创建 20 个 stream
  cudaStreamCreate(&streams[i]);

Wrappers.cuh(14 个 numeric kernel 调用):
  hashNumericKernel<6,...><<<..., streams[1]>>>    ← bin1 在 stream 1
  hashNumericKernel<7,...><<<..., streams[2]>>>    ← bin2 在 stream 2
  hashNumericKernel<8,...><<<..., streams[3]>>>    ← bin3 在 stream 3
  ...
  denseNumericKernel<6,...><<<..., streams[1]>>>   ← 密集 bin1 也在 stream 1
  ...
  ESCKernelDispatcher<<<..., streams[1]>>>          ← ultra 在 stream 1

同步机制(SpGEMM.cuh:179-197):
  syncMainToStreams:  stream0 → 等待所有子 stream 完成
  syncStreamsToMain:  所有子 stream → 等待 stream0
```

**含义:Ocean 的 5-6 个 bin kernel 同时在不同 stream 上跑,GPU 并行处理所有 bin。
我们逐 bin 串行跑(单流),accumulate = Σ(各 bin 时间);Ocean = max(各 bin 时间)。**

### 影响量化(以 bcsstk30 为例)

| bin | 行数 | 预计单 bin 时间 |
|---|---|---|
| bin0(批量) | ~3000 | 0.3ms |
| bin3-5(中行) | ~15000 | 0.8ms |
| bin7-9(大行) | ~8000 | 0.5ms |
| **串行(我们)** | | **1.93ms** |
| **并行(Ocean)** | | **~0.8ms** |

→ **仅 stream 并行一项就是 ~2× 差距**,这不是算法差距,是执行模型差距!

### 修复方向

在 accumulate 的 per-bin 循环中,把每个 bin 的 kernel launch 放到独立 stream,
最后 cudaStreamSynchronize 汇合:

```cpp
cudaStream_t streams[8];  // 池化,一次创建
for (int bi = 0; bi < N_BINS; bi++) {
    // 每个 bin 用 streams[bi % 8],kernel 并行执行
    hash_spa_batched_kernel<<<grid, block, smem, streams[bi%8]>>>(...);
}
// 汇合
for (int i = 0; i < 8; i++) cudaStreamSynchronize(streams[i]);
```

⚠ 注意:我们的 retry 依赖所有 bin 完成(收集 overflow 行),汇合点在 retry 之前。
⚠ 计时:cudaEvent 记录在 stream0(启动所有 kernel)→ 汇合 → 停表 = 真实并行时间。

## 三、文献调研更新

原调研(12 文档)说"Ocean 不用 multi-stream"是**错的**——重新检查后:
- Ocean 论文确实不描述 h2d/d2h 的 stream 重叠
- 但 Ocean 代码**用 stream 并行化 bin 执行**——这不在论文里,是纯代码发现
- 这验证了"读源码比读论文更可靠"的原则

下一步:**实现 bin-level stream 并行**(这是纯执行模型改动,不改算法,预期 accumulate 2×)。
