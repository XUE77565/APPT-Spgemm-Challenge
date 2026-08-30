# docs/62 2026-08-30 全天 loop 战报(20 轮自主循环总账)

**基线轨迹**:v23 1.5008×/赢28 →(DCFUSE)v24[作废,字节渗漏]→(count-fix)v25 1.5007×/27-29
→(ADAPT_EXPAND)**v26 1.4307×/赢34**。单日 −4.6% geomean、+6 赢面,另有三个正确性修复。

## 1. 性能线

| 优化 | 效果 | 状态 |
|---|---|---|
| **DCFUSE**(docs/56) | brainpc2 翻面/mult_dcop −8.5%/rajat25 −9.2%/分散小赢 | 默认开 |
| **ADAPT_EXPAND**(docs/60) | bcsstk30 −17.8%/c-62 −13.4%/c-64 翻面/**Cube_Coup 3.95→1.26×(retry 风暴集体治愈,95 阵 |Δ|>8%)** | 默认开(状态机 v3 待净窗定论) |
| MHSAMP / MH_K / PIPEFUSE | 全部中性或负(docs/57/58,DVFS toll 侧写) | 否决保留 |
| DIM2/DIM2-s 双维路由 | c-big straggler 挂 v4 span 门(19491/19578),**门挡得对**,路由无解 | 基建保留默认关 |
| SPANF 旋钮 | 净窗实验待跑(固定窗并行 clear 真实开销 ~2-3×) | 备弹 |

## 2. 正确性线(今日最重要)

1. **字节渗漏**(我的,docs/59):向量化归约 `v|v>>8|...` 高字节混入低位 → c-64 +6.2% 过计;
   修复 = `__popc(v & 0x01010101)`(flag∈{0,1} 无需折叠)。
2. **TSOPF 陈年过计**(v8 起 +300,933):随 kernel 替换治愈。
3. **Hermitian 展开缺失**(第一天起,docs/62/commit 8721d0b):3Dspectralwave×2 是 complex
   Hermitian 存储,`find("symmetric")` 匹配不上 → 一直算 T·T,而 Ocean 正确展开 = 不同负载
   假赢;修复后 C=scipy 真值精确。
4. **全量 cnnz scipy 审计**:335 对/2 错(即 Hermitian 对,已修)—— **337 阵 cnnz 全数背书**,
   ocean337 上线以来首次完整正确性基线。

## 3. 认知线(论文素材)

- **DVFS toll 五证据链闭合**(docs/57/58):管线内小 kernel 冷跑(merge 40× 慢于隔离)与
  grid/工作量/字节/空档全部无关;真因 = 运行历史负载形态(h2d/d2h 走 copy engine → SM 345MHz)。
  代码侧无解;**管理员锁频 = 唯一钥匙(净比值收益 −5~7% 投影 + 更公平口径)**;Ocean 也是单发
  无 warmup(main.cu:72)→ 双方都吃,锁频对双方同条件。
- **merge3 生态位验证**:宽带(bw=1024)胜 Ocean 3.2-3.9×,窄带全败 —— 判别特征 = 带宽/高dup
  非"带状"(memory: merge3-band-niche)。
- **测量方法学**(血泪五条):交替×4 取中位 / nnz 对 scipy 真值(对拍≠校验)/ nnz 随参数变=红旗 /
  refresh 期间 GPU+CPU 双静默 / 开工前 uptime 查净。

## 4. 净窗待办队列(优先序)

1. ADAPT_EXPAND 状态机 v3 逐阵定论(v1 受害者名单复检)
2. Hermitian 双阵正确负载重测(赢面 34→32 预期)+ CSV 更新
3. SPANF=4 固定 expand 下 c-big 交替实验
4. v26 全量复验(load 11 期间的数据可信度抽查)
5. (架构)按行密度自适应窗口宽度 = c-big 真解(docs/61 改写方向)
