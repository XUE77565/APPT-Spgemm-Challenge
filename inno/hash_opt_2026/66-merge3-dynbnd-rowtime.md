# docs/66 merge3 动态负载均衡 + per-row ROWTIME 采数(用户钦点双方向)

**日期**:2026-08-31 | **代码**:src/spgemm_merge.cu(bnd kernel + 消费 4 kernel)、
src/spgemm_kernel_hash.cu(hash_spa ROWTIME)| **脚本**:ab_mrg3_dynbnd.py、collect_rowtime.py

## 1. 病根(用户判断正确)

merge3 旧分桶:`blo = b*n/K` 等宽切 [0,n)。宽带阵(bw=1024, n=32000)桶宽 6400 ≫ 带宽
2048 → **全部工作落桶 0,K=5 切分在自己生态位上完全失效**,重行 warp-merge 串行全量 flop。

## 2. v1 值域二分(否决)

精确等 flop 切点:每边界对 v 二分,rank(v)=Σ_k lower_bound(B_row_k,v)。
**A/B 实测宽带 +374%/+284%、窄带 +21%** —— rank 评估 = O(num_k·log(avgB)) 次 cache
不友好访问 × log(n) 轮 ≈ merge 本身的成本。教训:**负载均衡的定价不能贵过被均衡的工作**。

## 3. v2 span 内均分(采纳,近零开销)

min/max 乘积列 = 各输入 k 的 B 行首/末列(B 行有序,O(num_k) 顺序负载,零二分)→
边界 = `min + j·span/K`,桶 0 左端收紧到 min。轻行(flop<8192,MRG3_DYN_MIN)回退等宽
= 旧行为逐位一致。门:MRG3_DYN_BND(默认 1)。

**A/B(reps=4 交替中位)**:
| 阵 | off | on | Δ | 备注 |
|---|---|---|---|---|
| band_n32000_x1024 | ~~23.34~~ | ~~24.67~~ | ~~+5.7%~~ | **废数:濒死运行(见 §2b)** |
| band_n8000_x1024 | ~~6.37~~ | ~~6.48~~ | ~~+1.7%~~ | **废数:同上** |
| band_n2000_x1024 | 53.04 | 53.02 | −0.0% | flop 2G<cap ✓ 真实 |
| band_n32000_x128 | 19.00 | 18.01 | **−5.2%** | 真实 |
| band_n32000_x16 | 0.89 | 0.90 | +1.3% | 回退路径,噪声 |
| er_n32000_x16 | 2.65 | 2.67 | +0.5% | 中性 |

## 2b. ⚠ 重大更正:x1024 阵 = SAFETY 濒死运行,宽带旧结论全体作废

x1024 两阵 flop 4.6G/20G ≫ MRG3_MAX_ENTRIES cap 3e9 → fscan 后 `[mrg3] SAFETY` exit(1),
无 TOTAL/Result C;**旧 compute_only_from_prof 在 TOTAL 缺失时把已有相位求和冒充时长**。
→ 08-30"宽带 merge3 胜 Ocean 3.2-3.9×"(23.7/6.4ms)全是这种废数;记忆已更正。

**真实三方(n8000_x1024,count 路径 MRG3_FLOP_UB=0,cnnz 三方一致 28,514,400)**:

| | Ocean | hash | merge3(count 路) |
|---|---|---|---|
| compute-only | 20.6ms | ~24.5ms | **581.9ms(输 28×/24×)** |

count/merge 相位:298→265 / 352→317(v2 双相位 −10~11%);bnd 开销 0.062ms。
宽带 flop-ub 路本机不可行(gapped = Σflop×12B = 240GB ≫ 80GB)。**merge3 现无实测生态位**
(与 dispatcher 0/337 一致);窄带败 1.9-13× 结论不变(那些跑全了)。
工具修复:compute_only_from_prof TOTAL 缺失 → None(不再求和);Result C 缺失 = 硬红旗。

**解读**:桶划分不改变总工作量(merge = distinct×num_k 恒定),只改并行度与等待结构;
count/merge kernel 在桶并行下兑现 −10.5%(每行 5 warp vs 1 warp 的延迟 hiding)。
真生态位(若存在)要用 §4 ROWTIME 在真实阵上找。

## 4. per-row ROWTIME 采数(路由判据的直接测量)

- hash 侧:`hash_spa_kernel<0/1/2>` 加 `rowtime` 参数,lane0 记 clock64 差(周期,
  DVFS 免疫);`HASH_ROWTIME=path` 落盘 `行 kcnt flop est span ovf cycles`
  (est 取 binning 后快照——dense 路径会置零;ovf 行 retry 重做过,t_hash 失真 → 剔除)
- merge3 侧:`bucket_merge_flop_kernel` 每 (row,bucket) 记周期;`MRG3_ROWTIME=path`
  落盘 `行 桶 cycles`;**行墙钟 = max over 桶**(桶间并行)
- 分析:`collect_rowtime.py` 同阵双跑 → 总量对比 + flop 十分位画像 + 门搜索
  (flop/kcnt/dup/span × 分位阈值 → net win 最大的门)
- 这是 docs/63 §6 遗留"per-row 判据需逐行计时采集"的兑现,也是 B 线(merge3)接入
  C 线(dispatcher)的数据地基

## 5. 待办

- [ ] v2b:修正则后 C-nnz 红旗复验(v2 首轮 nnz 抓错行,未验)
- [ ] ROWTIME 真实阵采数:band 族 + c-big/F2/web-Google/rajat 类 → 判据画像
- [ ] 宽带 +5.7% 损失机理(相位分解)
- [ ] 行自适应 K(重行更多桶)——straggler 证据到位后再做
- [ ] 判据画像 → compute_bucket_kernel 加 BIN_MERGE3 路由(C 线合流)
