# docs/67 中差带三簇解剖(Ocean 移植战役收官后的新主线)

**日期**:2026-08-31 晚 | **工具**:相位分解 + DD_ROWTIME/HASH_ROWTIME 逐行周期
**目标**:v27 干净列差距榜三簇:Ga/Si/Ge-H 量子化学族(~10 阵 2.7-4.3×)/crankseg×2(4×)/
in-2004(6.84× 最大单点)

## 1. 相位分解(compute-only)

| 阵 | compute | dense_direct | dense_count | accumulate | compact | retry |
|---|---|---|---|---|---|---|
| Ga41As41H72 | ~112 | **64.1(57%)** | 20.5 | 10.4 | 14.5 | 1.0 |
| crankseg_2 | ~68 | **49.6(73%)** | 12.1 | 2.8 | 0.6 | 1.9 |
| in-2004 | ~92 | 28.4(31%) | 14.6 | 14.6 | **22.7(25%)** | 7.7 |

统一病灶 = dense_direct;in-2004 另有 compact+sort 25%(hash 域排序税)。

## 2. 路由排除(全部中性,交替 ×2-3)

- PB2_AVGB=0(全 cursor):Ga41 −2~7%(弱/单对)、crankseg 完美镜像、in-2004 wash
- 结论:Ga 族税【不在】search/cursor 路由(与 c-64/TSOPF 系结论一致:cursor 处处不劣)

## 3. Ga 族逐行定标(DD_ROWTIME + 特征 join)——病根实锤

30196 dense 行高度均匀(p50=860k cyc,max 1.87M,top1% 占 1.9%)= **吞吐病非 straggler 病**:
- 特征中位:kc=395 / flop=156k / **est=4096 / span=112,581 → 窗口占用 3.63%** / dup=38×
- 21 窗(112k/5400),每窗 42k cyc;每乘积 6.6 cyc
- **税 = O(span) 固定成本**(每窗 clear 5400×9B + warp0 串行 prefix 5400 + emit 扫描)
  vs 每窗仅 ~192 真实输出 = **~27× 管理开销/有用工作**

这就是 docs/63 touched-reset 论文的病,但在 **10 阵家族尺度**(docs/63 只见 3Dspec2/c-big
单点)。docs/63 §5 的否决只针对"块并行 prefix"单换;touched-reset 换的是 clear+prefix+emit
三件套,且当时"真税源未定位"——现已定位。

## 4. 修法阶梯(ROI 排序)与当晚结果

1. **占用轴 OCCGATE(est≥span/16,env 默认关)已实现 + 电池(交替 ×2,无 nnz 红旗)**:

| 阵 | Δ | 备注 |
|---|---|---|
| Ga41As41H72 / Si41Ge41H72 / crankseg_2 | **−6.1% / −7.8% / −8.8%** | 双复现,家族三连 |
| **3Dspectralwave2** | **−32%(137.7 vs 202.9)** | 完美双复现;占用 2.6% 病行回 hash |
| Ge99H100 / crankseg_1 / c-58 / mult_dcop | wash | |
| brainpc2 | 噪声内偏赢 | 其行全过占用门 = 纯噪声参照 |
| c-64 | ~~rep1 +17% / rep2 wash~~ → **周期终审:门正确** | 专项 ×4:occ 99.6/91.7/85.5/85.7 vs def 85.3±0.4(墙钟噪声形态);**逐行周期裁决**:仅 334/19776 行被搬,**每行 hash 便宜 3.44×**(57.8M→16.8M cyc),未搬行逐位一致,总周期 occ −0.6% |

强制全 hash(DITER_MIN_FLOP=∞)对照:Ga41 −8~17% / crankseg_2 −15% —— OCCGATE 赢面
略小 = 选择性正确的迹象。**纪律:默认不翻,净窗复验 + v28 refresh 后定**(用户明示)。

2. **自制双梯的 dense 侧成本律(120k 行 × 7 阵拟合,R²=0.896)**:
   `t_dense ≈ 30,109·窗数 + 2.55·flop` —— 每窗固定税 30k 周期(清零+串行 prefix+emit),
   每乘积 2.55。F2 +85% 复算闭合(span<窗宽地板 → 30k 税 vs 小表 hash);Ga 病 = 21 窗
   ×30k=63 万税 > 40 万有用工作。OCCGATE 即此律的零阶判据(span≤16·est ⟺ 30k·span/5400
   ≲ ~100·est)。
3. **hash 侧分段律 + 自制双梯 v0(dac89e7,LADDER env 默认关)**:
   配对采数(9 阵 hashside + dense 侧,69,213 对行):
   - oracle:86% diter 候选行 hash 更快;OCCGATE 一致率 69%(残留 1.94e9 周期)
   - t_hash ≈ 8.1·ht + 1.89·flop + 1038·kc − 50225(大表均匀总体 R²=0.926)
     —— **kc 第三轴实锤**(每输入链 1038 周期;3Dspec2 kc=117 → 12 万/行)
   - **LADDER v0 电池(交替 ×2)**:Ga41 −6.2/Si41 −4.9/crankseg_2 −8.9/**Ge99 −6.4/
     crankseg_1 −6.8(OCCGATE 漏抓的双复现赢)**/3Dspec2 −29.4/mult_dcop wash;
     **但 c-64 +23% / c-58 +7% / brainpc2 +4~9% 回归**
   - **⚠ 周期和 oracle ≠ 墙钟(并发干扰盲区)**:c-64 est-8k/占用 16% 行周期和说 hash 快
     3.44×,搬过去墙钟 +23% —— dense_direct(1024T 大块)与 hash bin(64-512T)的 SM 重叠
     结构随行群迁移而变,逐行周期和不捕捉。**入血泪纪律第八条候选:成本律 oracle 须墙钟电池
     终审;判据修法 = 边距(th<0.8·td 才搬)或 10 阵墙钟结局重标定,均待净窗。**
   - 现状:**OCCGATE 保持 v28 首选**(电池全绿无回归);LADDER 是方向验证(Ge99/crankseg_1
     证明成本律能抓 OCCGATE 漏的),修好盲区后接管。
4. touched-reset(docs/63 §2)= 梯子之后的内核工程选项(税消而非避让)。
5. Ocean 对照:同结构行 Ocean 全家 52ms(我们 compute 112)。

## 5. in-2004(次级)

compact+sort 22.7ms(25%)+ retry 7.7ms:hash 域排序税 + 欠估 retry(harness 双 expand
在 CSV 口径已部分兜底;单发口径仍有)。dense_direct 28.4 也非零。待 Ga 族修法定型后回看。
