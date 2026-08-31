# docs/63 自适应窗口 → touched-list 稀疏复位(下窗主攻;两战线汇聚的架构目标)

**日期**:2026-08-30 | **状态**:设计稿 | **目标**:3Dspectralwave2 4.6×/c-big straggler 类

## 1. 病根定量(v27 实测)

3Dspec2 诚实负载:dense_direct 68.9ms 处理 59.6M nnz = **0.86G/s(c-64 的 1/11)**。
47871 diter 行 × span≈292k(全矩阵)→ 每行窗口税 = **O(span)**:clear dflag/dval 全窗 +
prefix 全窗 = 292k SMEM ops/行 ≈ 14GB 级复位流量,而有效工作只 O(flop)≈5k/行。
c-big straggler 同病(span 345k / flop 16-30k)。**税 ∝ span 与窗宽无关**(窗数×窗宽 ≈ span)。

## 2. 修法:touched-list 稀疏复位(经典 sparse-reset trick)

每窗 accumulate 时维护 touched 列表:
```cpp
// 探测首次置位,追加 touched(原子游标;容量 = 每窗产品数,上界 flop/窗)
int old = atomicExch(&dflag[j], 1);        // 或 scan 阶段记录
if (!old) touched[atomicAdd(&nt, 1)] = j;
```
窗末:
- **复位**:只清 touched 里的 dflag/dval(O(P) 非 O(W);未触列仍为初始 0 ✓)
- **有序输出**:touched 列表 csort(P 条,SMEM warp 排序)→ 排名即 dpref,免全窗 prefix
  (每窗 P≈20-100 条,csort 微秒级;或直接沿用 dpref 但只扫 touched 排序后区间)

**成本变换:每行 O(span+flop) → O(flop·logP)**;3Dspec2 估算 dense_direct 68.9 → ~8-12ms
(总 ~42ms vs Ocean 22.5 = 1.9×,从 4.6×);c-big straggler 同解后 **SPANF 门可大幅放宽**
(窗口税不再 ∝ span → v4 的 16×flop≥span 门失去存在理由 → c-big 主战线重开)。

## 3. 落点与顺序

1. `hash_dense_direct_kernel` search 路径(avgB<64)先做(3Dspec2 走此路径;结构简单)
2. cursor 路径(大 avgB)同构移植(smap 游标不受影响)
3. gate:TOUCHED_RESET=1 env 门控,默认关 → A/B(3Dspec2/c-big + 回归 c-64/brainpc2/
   mult_dcop 低跨度阵必须 ±2% 内 —— 它们 P≈W,复位成本近似,touched 追加开销小)
4. 若胜出 → SPANF 放宽实验(c-big straggler 入窗)→ 双战线收网
5. ⚠ 正确性纪律:touched 追加遗漏 = 脏 flag 跨窗残留 = 假 nnz → cnnz 对 scipy 全程;

## 4. 风险

- atomicExch/atomicAdd on dflag/nt 的争用(高密度窗 P≈W 时反而比纯写慢)→ 高密度行
  (est/span > 1/4)走旧路径,低密度行走 touched(按行二选一,免 per-window 分支)
- touched 容量 SMEM 预算:P 上界 = min(flop_in_window, W);预分配溢出 → 回退全窗 clear

## 5. 前置实验否决:块并行 prefix(08-30 实测)

docs/63 §2 的前置步骤"全块并行 prefix 替代 warp0 串行"实测**真回归**:交替新旧 binary 同环
境,c-64 direct 15.55→21.15(+36%)/brainpc2 7.13→9.24(+30%);3Dspec2 反而无感(68.7)。
**机理教训**:warp0 串行扫描时其余 31 warp 停在 barrier 上是零成本的(它们反正要等 prefix
才能 emit);块并行版引入 3 次 1024 线程全块 __syncthreads/窗 + 全线程指令 —— **大 CTA 的
sync 比闲置贵**。已回滚(工作树 = c680fe9 状态)。

**对 §2 主设计的修正**:prefix 非 3Dspec2 主税(替换它无感)→ 68.9ms 的主税在别处
(clear?accumulate 的 lower_bound?待 nsys 分解),touched-reset 的预期收益需重新评估后
再实施。低密度行 profile 应先用 nsys/相位内分解定位真税源,勿再凭模型动手。

## 6. 路由规则实验(08-31)~~:dup≥4→search 一刀切两头不讨好~~ 【§6b 复核:本节结论是污染假象】

> **⚠ 08-31 晚更正(docs/66 §8 复核)**:本节的 A/B 数字全测于 load 11-34 污染窗。铁证:
> 同场 web-Google(n<1M,两版路由【完全相同】)竟 +69.6% —— 该 session 噪声带 ±70%,
> "Cube_Coup +263%/TSOPF −27%/3Dspec2 −14%"全部无效。逐行周期数据(负载免疫)判决:
> cursor 在 c-64/TSOPF/c-58/brainpc2/mult_dcop/3Dspec2 六阵【全胜】1.24-6.30×
> (mult_dcop 6.30×/3Dspec2 4.36×/brainpc2 3.65×);交替矩阵级 cursor 5/6 不劣。
> 生产态(<1M 行)本就 cursor —— 本节"回滚"实为回到正确态,但理由错了。
> dup 门的存废待 Cube_Coup(唯一 >1M dup 阵)PB2_CURSOR=2 终审。

**原记录(留档,勿再引用)**:废除全局规则的 A/B:Cube_Coup_dt0 +262.6%/TSOPF_FS_b39_c7
−27.1%/3Dspec2 −14%。判据困境:同特征异嗜好。已回滚(工作树 = 裸 dup 规则);per-row
判据采集已由 docs/66 §8 兑现(DD_ROWTIME)。

### 6b. 逐行判据实测(08-31,DD_ROWTIME,clock64 周期负载免疫)

| 阵 | cursor 逐行优势 | 逐行中位 t_cur/t_srh | 交替矩阵级 |
|---|---|---|---|
| mult_dcop_03 | **6.30×** | 0.15 | −12.0% |
| 3Dspectralwave2 | **4.36×** | 0.23 | −7.7% |
| brainpc2 | **3.65×** | 0.26 | −9.7% |
| TSOPF_FS_b39_c7 | 1.62× | 0.61 | 噪声内 |
| c-58 | 1.35× | 0.68 | −2.9% |
| c-64 | 1.24× | 0.73 | −2.5% |

无任何特征门可让 search 赢:最优门全部 = 全选 cursor。c-64 唯一 t_cur/t_srh>1 尾部
(avgB 6211 极重行)占比极小。**判决定稿:hash_dense_direct 无 search 生态位,uc=1 恒 cursor**
(待 Cube_Coup 终审后删 dup 门)。
