# docs/64 Ocean 机制移植规格(源码精读 → 我们的代码库映射)

**日期**:2026-08-31 | **来源**:ocean/kernels/{Analysis,SpGEMM}.cuh + include/Common.h 逐行精读
**定位**:中差带(97 阵@1.80)+ 大差带(21 阵@2.91)的主攻蓝图

## 机制 A:双梯相对路由(Analysis.cuh:110-211 binning / 214-303 numericBinning)

```cpp
range = b_right − b_left + 1                    // 行 span(analysisKernel 已产出,≈我们的 d_span_len)
dense_bin_id = range ∈ DENSE_BIN 梯             // {0(弃用), S1×8−32, ..., S5×8−32, 0(∞)}
bin_id       = est ∈ HASH_BIN 梯(prime {331,673,1327,2719,5441,8179?})
if (dense_bin_id <= bin_id) bin_id += BIN_NUM   // ★ 相对判据:数组不比表贵就走 dense
    // "speck style":dense 侧保留 hash bin_id 的工作量分级(bin_id+7 而非 dense_bin_id+7)
```
**与我们现状的差异**:我们是绝对阈值门(est≥2048/dup<8/16×flop≥span),它们是两条梯的
大小比较 —— 无需逐门调参,天然跨形态。**我们的全部输入已就位**(d_est/d_span_len 都在)。
DIM2 基建(docs/61)已把梯和比较写进 compute_bucket_kernel,当时败在"被 v4 门 AND 住"。

**实施**:v4 门改为 OR-退化路径——`(旧门全部命中) ∨ (dense_bin ≤ hash_bin ∧ flop≥4096 ∧
span≤n)`,即双梯只放行 v4 之外的新行;F2 类由 flop 门继续挡。逐步用实验松绑。

## 机制 B:溢出=计数上限的逃逸阀(替代我们的 retry)(SpGEMM.cuh:262-280)

```cpp
numericBinning(..., hybrid_hashmap_cnt, hybrid_hashmap_max):
  int numeric_max_bin = (use_largest && *cnt < max) ? BIN_NUM : BIN_NUM−1;
  // 最大梯(est>8179)的行 → 全局 hybrid hashmap kernel,但【全局计数封顶】,超额行留在常规梯
  if (bin_id == BIN_NUM−1) atomicAdd(hybrid_hashmap_cnt, 1);
```
**与我们 retry 的对比**:我们 = 溢出收集→flop 定表重跑→独立排序→compact 路由(0.3-3.4ms/
阵,c-64 retry 0.58/3Dspec2 3.4ms);它 = **binning 时一次性分流,超额行就地走常规路径**,
零重跑零排序。预分配 escape buffer(rows/100 × (max_product+6)),bitmap 管理。
**实施**:在 compute_bucket_kernel 加 BIN_LARGEST 的全局 atomicAdd 计数 + host 预算
(重试区容量 = f(max_product)),把 run_retry 整段替换为"超额行 → heavy 全局表"(我们已有
heavy 路径!只是现在按 est 入,改成按 est+配额入)。**这是自包含改动,ROI 最高**。

## 机制 C:Ana2 采样工作流决策(Analysis.cuh:10-29 + SpGEMM.cuh:600-660)

```cpp
ana2_type = 1(est 工作流)iff:
  total_products/A_nnz ≥ input_expansion_threshold   // 输入扩张 = avgB 大
  ∧ 采样 avg_compaction ≥ threshold ∧ avg_compaction2 ≥ threshold  // 输出压缩 = dup 高
else ana2_type = 0(precise:symbolic 先行 → 直写终态)
// 采样统计还做 var/z 值:safe = avg − z·√var,clamp 0.8×avg → est 的安全系数是【统计量】
// 不是常数!(我们 est_expand=1.15/1.4 手调对应此物)
```
**关键洞察:precise 只用于低 compaction(低 dup)矩阵** —— counting 便宜时才付两遍;
高 dup 走 est+逃逸阀。**D5H 失败的根因找到了**:我们把它用在 dup<8 的门内(高 dup 也进),
而 Ocean 只对 compaction<1.5 用 precise。
**实施**:MHSAMP 的采样机器(merge 采样 + 投影)复用为 Ana2:采样行跑 dense_count 精确计数
→ compaction 样本均值+方差 → 判 precise/est + 算 safe 系数(替代 est_expand 手调)。

## 机制 D:symbolic = numeric 的 bin 化 dry-run(SpGEMM.cuh:734-898)

symbolic 与 numeric **共用同一套 bin 化 kernel 家族**(hashSymbolic/denseSymbolic vs
hashNumeric/denseNumeric),只是 count-only。precise 流程 = 同梯 binning → symbolic 计数
→ 精确偏移 → numeric 直写(无 compact/retry)。est 流程 = estimated symbolic(est 表 +
global buffer 溢出)+ numeric。
**实施**:我们的 hash_spa MODE=1(D5H 遗产)+ dense_count 已是 count-only 版本;
缺的是 per-bin 模板化的速度(docs/61 之 C)和"只对低 compaction 用"的门。

## 实施顺序(ROI 排序)

1. **B(逃逸阀替 retry)**:自包含,~80 行,c-64/TSOPF/3Dspec2 等 retry 阵直接受益;
   纪律:nnz 对 scipy + 交替 A/B + 回归(c-big 无 retry 不动)。
2. **A(双梯放行)**:DIM2 基建改 OR-退化,逐类松绑实验(F2/333SP 回归线)。
3. **C(Ana2 采样决策)**:MHSAMP 机器复用 + 统计安全系数替换手调 expand。
4. **D+per-bin 模板**:最大工程,precise 工作流落地的前提(docs/61 之 C 合并)。

## Ocean 常数备查(H100 128KB SMEM 档)

HASH_BIN(prime):{331, 673, 1327, 2719, 5441?…}(Common.h:158 档);numeric 从 331 起
(最小 bin 不支持 numeric)。DENSE_BIN = 各档 SMEM/(val+flag 字节)−32;numeric dense =
SMEM/(8+1)−4。symbolic 溢出 buffer = rows/100 × (max_product+6),0xFF 填充。

## 5. ROI 重估(08-31,B 实现前的现状核查)

精读我们的 retry 链(accumulate 内收集 → h_ovf D2H → flop 定表重跑 hash_global → 排序 →
compact 路由)后发现:**retry 风暴的主体已被 ADAPT_EXPAND 吃掉**(v26 大胜的根源正是
expand=1.4 消灭了千行级 overflow;c-64 残余 retry 仅 0.29ms)。B 的真实残余价值 =
0.3-3ms/阵 的固定成本(3Dspec2 类)——**降级到第二优先**。
**A(双梯相对路由)升为第一**:它是结构性缺口(docs/61 双维实验证明 v4 门挡对了路但
绝对阈值无法跨形态),也是 C/D 的地基。新序:**A → B(残余 retry 阀)→ C → D**。

## 6. 机制 A 实测否决(08-31,ab_dim2_or.py 首轮)

OR-退化版(v4 全门 ∨ dim2∧flop≥4096∧e≤HASH_CAP∧dup_ok)7 阵电池(load12 单发,
nnz 全一致 ✓ 路由确定性成立):

| 阵 | Δ | diter 行变化 |
|---|---|---|
| F2 | **+85.3%** | 4449→36915(梯子放进 3.2 万低 est 行) |
| c-big | +5.3% | 88→6962(目标 straggler 只进 6874,净亏) |
| bcsstk30 | +9.2% | 0→647 |
| c-58 | −23%(+4 行不可能 = 单发噪声实证) | 7337→7341 |
| 333SP/brainpc2/mult_dcop | ±1% | 0(门未触) |

**机理:双梯只比"数组 vs hash 表"的尺寸,不模 dup 工作量不对称**——dense 付逐乘积
(∝flop),hash 只付 distinct(∝est)。梯子放进的恰是 est 1-2k 小表行(我们 hash 最快的
领域,docs/24 已证 9.7G/s),dup 2-8× 时窗口反亏。Ocean 能用双梯因它的 dense 窗口
经济学不同(数据驱动窗+满阵原子布局);**结论:A 不移植,DIM2 永久默认关**。
c-big straggler 的真解仍是 docs/63 touched-reset 类窗口经济学改造,非路由。
新序:**B(残余 retry 阀,低优先)→ C(Ana2 统计安全系数)→ D**;主线让位 merge3/
长尾(docs/66)。
