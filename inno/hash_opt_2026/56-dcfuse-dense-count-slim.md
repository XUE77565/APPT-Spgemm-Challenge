# docs/56 DCFUSE:dense_count 相位瘦身

**日期**:2026-08-30 | **状态**:A/B 进行中 | **门控**:`DCFUSE=1` 开(默认关),`DCF_BUDGET_GB`(默认 24)

## 1. 问题

DIRECT5(方案5)dense 路径的 `dense_count` 相位在近赢阵上占比过高:

| 阵 | compute-only | dense_count | 占比 | vs Ocean |
|---|---|---|---|---|
| brainpc2 | 10.88ms | 1.957ms | **18.0%** | 1.020×(差 0.22ms) |
| c-64 | 21.84ms | 1.974ms | 9% | 1.039×(差 0.85ms) |

原设想"count 与 direct 融合省一遍产品扫描"。**论证后否决该路线**:

- 融合的障碍是信息论的:直写终态需要 exact row_ptr,而 row_ptr 需要全部行 count 完成 → 单
  kernel 内做不到(除非 cooperative launch + 网格同步,省的只是 launch/同步不是扫描)。
- 替代路线 gapped 乱序写 + compact 善后:数据搬运 24B/entry(read+write 12B)vs count 扫描
  4B/product(只读 col)——低 dup dense 输出下搬运更贵;D5H(docs/31)的 csort 教训同向。
- Ocean 同样两遍(precise = symbolic count + numeric 直写)——两遍本身不是劣势,**相位内的
  纯开销才是**。

## 2. 相位解剖(1.957ms 里有什么)

1. 规模门 pre-pass:dense_sum kernel + **D2H 同步 #1**
2. hash_dense_count_kernel(产品扫描主体,~1.2-1.3ms)
3. dense_sum ×2 → **D2H 同步 #2、#3**(两次独立 8B 拷贝)
4. zero_est + 全行 inclusive_scan 重扫 + **D2H 同步 #4**(tmp_slots)

每次小 D2H = 一次流水线停顿(~0.1-0.2ms);4 次 + 2 个小 kernel + 重扫 ≈ 0.5-0.7ms 纯开销。

## 3. 改动(全部 DCFUSE 门控)

1. **门跳过**:矩阵级(dense_nr==A_rows 且 ≥1000)时规模门两条件平凡成立 → 免 dense_sum+D2H。
2. **合并 D2H**:count 后两次 8B 拷贝 → 一次 16B。
3. **免坍缩**:zero_est+重扫+D2H 只为把 tmp 从 total_est 缩到坍缩值(省内存,不省正确性:
   hash 行的 d_off 偏移在两种口径下都是自身 est 的前缀,dense 行不在任何 hash bin)。
   `total_est×32B ≤ 24GB`(tmp 16B/slot + csort scratch 16B 最坏)→ tmp 直接按 host 已知
   total_est 定容,鲸鱼阵超预算自动走原坍缩路径。
4. **count kernel 动态窗宽**:DENSE_CNT_W=65536 对 n=27k 阵浪费一半 SMEM(64KB→32KB =
   2→3 CTA/SM);窗宽 = min(65536, ceil4(n)) 作参数传入。
5. **向量化 flag 归约**:逐字节 LDS → 4B/iter(1×LDS.32 + `v|v>>8|v>>16|v>>24` 折叠非零
   byte 判定 + `__popc(u & 0x01010101)`),flag∈{0,1} 使该折叠成立;尾段(非 4 倍数)走旧路径。

## 4. 安全性

- 免坍缩下 d_est/d_off 的全部后续消费者(accumulate/retry/compact/heavy_prep)只访问
  hash-bin 行,这些行的两项在两种口径下逐项相等(dense 行不在任何 hash bin)——已 grep 验证。
- 下溢护栏(2796 行)`C_nnz − dense_nnz_sum − d5h_nnz_sum > tmp_slots`:tmp_slots 变大
  (total_est ≥ 坍缩值)→ 护栏更宽松,语义不变。
- cnnz 一致性:A/B 脚本逐阵核对 `Result C: nnz`。

## 5. A/B 结果(2026-08-30,交替测量定案)

**⚠ 方法学教训:本机单阵方差 ±7%(c-58 off 跑出 7.50-8.63),顺序块测量(3×off 连跑再 3×on)
会被漂移整块污染 —— 先后两次顺序测把 c-58 测成 -5.9% 又 +9.4%、bloweya +10.8%,全是假的。
交替 off/on×4 取中位才是可信口径**(scripts/ab_dcfuse_repeat.py)。

| 阵 | off | on(默认坍缩) | delta | 备注 |
|---|---|---|---|---|
| brainpc2 | 10.71 | 10.11 | **−5.6%** | count 1.85→1.23;ratio 1.020→~0.96 **翻面** |
| c-58 | 8.58 | 7.42 | **−13.5%** | 交替后最大赢家 |
| mult_dcop_03 | 19.81 | 17.85 | **−9.9%** | count 4.89→2.99 |
| bloweya | 7.18 | 6.83 | **−4.8%** | 顺序测量假回归(+10.8%)的澄清 |
| TSOPF_FS_b39_c7 | 25.38 | 24.95 | −1.7% | 距翻面还差 0.33ms |
| c-64 | 21.60 | 21.38 | −1.0% | 距翻面还差 0.60ms |
| SiO2/pre2/web-Google/exdata_1/dielFilter/Cube_Coup/webbase/ohne2/cnr | — | — | ±0.5% | 中性(n≥65536 窗宽不变) |

- **no_collapse(免坍缩)全阵净负,已默认关**(DCF_BUDGET_GB=0):pool 扩容成本 > 省的重扫
  (分解实验 compare/ab_dcfuse_decomp.log:brainpc2 10.18 vs 10.10 / c-64 21.86 vs 21.38)。
  收益来源 = count kernel 动态窗宽(n<65536 时 SMEM 缩 → CTA/SM 增)+ 向量化 flag 归约
  (4B/iter LDS + `v|v>>8|v>>16|v>>24` 折叠 + popc)+ 门跳过(矩阵级)+ 合并 D2H。
- 回归 6 阵(pwtk/333SP/3Dspec2/bcsstk30/Ga3)±0.6% 噪声内,cnnz 全对。
- **默认开**(DCFUSE=0 可关)→ refresh24 全量验证。
- 剩余:TSOPF/c-64 翻面需攻 dense_direct(15.5ms=Ocean 总时间 75%)与管线税(retry 0.6/
  compact 1.0/mh_merge 1.3ms)——下一战场。
