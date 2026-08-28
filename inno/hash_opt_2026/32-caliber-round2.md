# 32 · 计时口径对齐第二轮(2026-08-28 晨,用户复核令)

> 前轮:docs/26(修 symbolic + stale 事故)。本轮:v11 管线(新增 dense_count/hash_count/
> dense_direct 相位)下的复核 + 两侧"墙 vs 相位和"实测。

## 1. 两侧计时结构(实测)

| 项 | 我们(spgemm_test) | Ocean(bench_detail.json) | 判 |
|---|---|---|---|
| 轮次 | verify + 5 warmup + **1 计时轮**(取末轮 hash-prof) | 10 warmup + **10 轮均值** | 我们更噪,无偏 |
| 指标 | TOTAL(GPU) − h2d − d2h(相位事件和) | 相位和(analysis+est+sym+numeric+epi+prologue) | ✅ 同为相位和 |
| 墙 vs 相位和 | 墙 − 相位和 = **8.2-12.6%**(F2 3.7ms/c-58 4.5/brainpc2 10.4) | 墙(=total_time)− 相位和 = **+0.0~2.1%**(4 阵实测:c-58 +2.1/TSOPF_b300 +0.0/pre2 +1.0/mult_dcop +0.8) | ⚠ **不对称利我们** |
| malloc | 我们的 dev_alloc(纯 cudaMalloc,~15 buffer/轮)在 prof 外 = **不计入** | estimation.malloc/numeric_malloc **计入**相位和(cudaMallocAsync+池 ≈ μs) | ⚠ 同上 |
| h2d/d2h | 排除 | 排除(stats 另算) | ✅ |
| 相位间隙 | 排除(事件和天然不含) | 排除 | ✅ |

**结论**:口径形态对齐(双侧都是"相位和、排传输");残余不对称集中在**我们的未计时区**
(alloc/host 逻辑,墙的 8-13%)被排除、而 Ocean 的等价物(池化 malloc)计入且其墙差仅 0-2%。
方向 = **利我们**(我们的数字比"诚实墙"乐观最多 ~10%)。论文级对齐的两种修法:
1. **改用 cudaMallocAsync + releaseThreshold=UINT64_MAX**(docs/27 §4.7):未计时区缩到 μs 级,
   不对称自然消失,且免 driver 往返可能直接提速 —— 一石二鸟,**推荐**;
2. 或把 dev_alloc/free 包进一个 prof("alloc")相位计入,保守修法。

## 2. 本次顺带的口径数据

- Ocean 4 阵:相位和 vs total_time 差 +0.0~2.1%(墙略大 = 相位间隙)→ run_ocean 用相位和
  给 Ocean 记分**不虚**(比其墙还紧 0-2%)。
- 我方墙−相位和明细含:pinned_d2h_alloc、~15 次 cudaMalloc/Free、host 编排;其中传输相关
  (pin/arena)约占一半,纯 alloc+间隙 ~4-6ms 级(中尺寸阵)。

## 2.5 修法1实测(MALLOC_ASYNC,2026-08-28)

env 门控实现(`MALLOC_ASYNC=1` → cudaMallocAsync + releaseThreshold=MAX,默认关)。A/B(墙/compute):
F2 -7.6%/-1.5%、pwtk -12.4%/+2.3%、bcsstk30 -13.9%/+17.5%、mult_dcop -2.1%/+2.1%,**但 c-58 +9.7%/+10.8%、
brainpc2 +15.0%/+3.5% 回归** —— legacy stream 上 cudaMallocAsync/FreeAsync 的隐式同步串行化反噬,
池纪律在我们管线非免费。**默认关,留 env**;论文级口径对齐改走修法 2(alloc 包 prof 计入)或
每 bin 专用 stream + async 的组合(下一班)。

## 3. 学 Ocean 的下一步(与本口径结论绑定)

1. **cudaMallocAsync 池纪律**(§4.7,上表修法 1)—— 下一个实施项,兼修口径与延迟。
2. **溢出行换 denseNumericIter**(§4.3):我们的 retry 仍是 flop 定表 hash_global(大分配);
   改路由到游标窗口内核(SMEM 恒定)→ 免重试大表,鲸鱼余量+提速。
3. TSOPF_FS 家族游标回归(+29-39%)待路由精调(avgB 单变量不足,疑需 dup/avgB 双变量;
   可先 Ocean 跑该族抓 binning_2 看它路由到哪个 dense 档)。
