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

## 5. per-row 判决(08-31 采数,前 4 阵;周期数 = clock64,GPU 独占下与负载无关)

| 阵 | 可比行 | Σt_mrg/Σt_hash | 全部十分位 t_mrg/t_hash 中位 | mrg 胜率 |
|---|---|---|---|---|
| er_n32000_x16(dup=1) | 31900 | 3.18×劣 | 2.7-4.0× | **0%** |
| band_n32000_x128(dup~10) | 32000 | 10.9×劣 | 11-17× | **0%** |
| skew_n32000_x24(重尾) | 30872 | 21.0×劣 | 3.2-55× | **0%** |
| F2(真实,dup 5-10) | 65737 | 19.9×劣 | 9-32× | **0%** |

**判决:merge3 对我们 hash_spa 无行级生态位**——dup=1(无 hash 红利)也输 3×;
k-way merge 的 O(distinct×num_k) 段扫描 vs hash 的 O(flop) 原子插入,全特征域溃败。
dispatcher 0/337 从矩阵级猜测升格为**逐行实测证明**。B 线(SpGEMM 调度)就此关闭;
merge3 保留价值 = ATT 分层交付物(deck design4)+ 论文对比基线。
(merge3 行时 = flop-ub 单 pass 全成本;hash 行时 = 建表+插入+extract 全成本;ovf 行已剔。)

## 6. 待办(修订)

- [x] C-nnz 红旗复验(count 路真实跑,n8000 双版 28,514,400 一致)
- [x] ROWTIME 采数 + 判决(§5)
- [x] dense 单遍直写(§7,commit b9137a6):count 5.2×/numeric 2.26×,宽带 count 路 2.87×
- [x] ~~行自适应 K / BIN_MERGE3 路由~~(生态位已否决)
- [ ] Ocean-C(Ana2 统计安全系数)= 剩余可移植机制;B 残余阀低优先
- [ ] 净窗 → 复验 36 阵 → v28

## 7. dense 单遍直写(用户钦点"两次精确 count 也改一下";commit b9137a6)

count 路径(count+numeric 两遍 merge 迭代)→ 小跨度桶 SMEM dense 直写:
- `bucket_count_kernel` flags 直计(O(flop) 幂等置位 + popcount,无原子)→ **5.2×**
- `bucket_merge{,_flop}_kernel` dense 累加 + 32 lane 连续段顺序发射(real_nnz 副产物)→ **2.26×**
- **三条实测教训内建**:①盲扩 SMEM → occupancy 14→3 blocks/SM(band128 flop 相位 0.85→2.0ms)
  → merge 模式恒保基础 SMEM + numeric 拆双 launch;②L<128 = 原子地址地板(band128 L=51
  dense 2× 劣)→ eligible = 128≤L≤dcap;③空 dense launch ~5ms 块调度税 + 双 launch 防双跑
  → bnd kernel 顺带 atomicMax(max_span),host D2H 一个 int(~10μs)得精确 dcap。
- 实测:band_n8000_x1024 count 路 597→208ms(**2.87×**);band_128/16 逐位不变(dcap 门全跳过)。
  门:MRG3_DENSE_SPAN 默认开;DYN_BND=0 → dense 自动关。

## 8. dense_direct 逐行 cursor/search 判据(DD_ROWTIME;docs/63 §6 遗留的终审)

**方法**:hash_dense_direct_kernel 逐行 clock64 + 路径模式落盘;PB2_CURSOR 1/0 双跑 +
HASH_ROWTIME 特征 join。周期数负载免疫 —— 这正是本工具的杀器(见下)。

**判决(推翻 docs/63 §6)**:cursor 六阵逐行【全胜】1.24-6.30×,无任何特征门可让 search 赢
(最优门 ≡ 全选 cursor);交替矩阵级 cursor 5/6 不劣(−2.5% ~ −12.0%,TSOPF 噪声内)。
docs/63 §6 的"search 三赢"(c-64 −11%/TSOPF −27%/3Dspec2 −14%)是 load 11-34 污染假象
—— 同场 web-Google 路由不变却 +69.6% = 噪声带实锤。**教训(入血泪纪律):凡 load>10 窗内
的 A/B 结论一律未定论,逐行周期数据优先于墙钟。**
生产含义:<1M 行阵生产态本已 cursor(无动作);唯一 >1M dup 阵 Cube_Coup(2.16M)被 dup 门
强切 search —— PB2_CURSOR=2(强制 cursor)终审中,胜则删 dup 门(commit 已加 override)。
