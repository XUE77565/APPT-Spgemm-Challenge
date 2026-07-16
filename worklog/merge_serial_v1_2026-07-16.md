# Serial k-way Merge v1:实现 + 与 ESC sort 的交叉点

> 日期:2026-07-16,分支 `feature/optiESC`。
> 目标(见 `ref/sort_innovation_directions.md` 方向 A):用 k-way merge 替代 ESC 的 `sort_by_key`。
> 本文件是 **v1 = 最简串行版**(每行一个 block、thread 0 做 merge),目的:验证可行性、找与 CUB sort 的交叉点、定位瓶颈,为 v2(行内并行)铺路。
> **ESC baseline 完全保留**(`spgemm_self_product_manual` 不动),merge 为并列新函数。

---

## 1. 实现

新增 `src/spgemm_merge.cu`(自包含),签名同 manual:

```
count_intermediates_kernel  (复用 manual.cu)
inclusive_scan(d_ub) → row_off, total_ub
expand_serial_kernel<<<A_rows,1>>>     ★ 串行写,保证每个 k 的贡献是【连续、col 有序】一段
                                        (与 manual 的 atomic 版不同 —— 这是 merge 能工作的前提)
merge_serial_kernel<<<A_rows,1,smem>>>  ★ thread0 串行 k-way merge + dedup + sum
                                        shared: seg_cur[num_k] + seg_end[num_k]
                                        每步:扫所有段头取 min col → 求和匹配段并前进 → 输出 (col,val)
compact_kernel<<<A_rows,256>>>          merge 输出在 upper-bound slot(行间有间隙),压成连续 CSR
inclusive_scan(row_nnz) → C_row_ptr;  pack + D2H
```

- **关键**:A 是 CSR → 每行 k 贡献的 cols 有序;串行 expand 把每个 k 连续写 → 每行得到 num_k 条有序列链 → merge 直接归并,同列自然求和去重。**省掉 sort + reduce**(merge 输出每行已去重+有序,直接 CSR)。
- 接入:`main.cu` 加 T4b 计时块(warmup 也预热);`spgemm.h` 声明;`Makefile` 加源;`profile_aa.py` 加 `merge` tag / 列 / 阶段映射(merge 阶段映到「排序」列,与 gust.sort 直接对照)。
- shared mem 大小:host 扫 pinned `A_row_ptr` 得 max_row_nnz → `2*max_row_nnz*sizeof(int)`(动态);>48KB 时 opt-in(本数据集 max=339,≈2.7KB,不触发)。

## 2. 正确性

逐矩阵 merge 的 **C_nnz 与 cuSPARSE、ESC 三者完全一致**(1138_bus 11142;bp_200 13197;bp_1200 22313;bcsstk30 **8,946,070**)。
与独立 oracle cuSPARSE 在大矩阵上精确匹配 → 足证 col/merge/去重正确,无越界、无 shared 溢出。全 100 矩阵 run_aa:OK 100 / FAIL 0 / TIMEOUT 0。

## 3. 性能(全 100 矩阵,merge vs gust-ESC)

| 指标 | 值 |
|---|---|
| merge 赢(<1.0×) | **49** |
| 持平(0.95–1.05×) | 33 |
| 输(>1.05×) | 39 |
| 几何均值 merge/gust | **1.377×** |
| **排除 bp_* 重行族后(91 个)** | **1.097×**(≈ 持平) |

**按稀疏类别(均值 ms)**:
| class | # | cuSPARSE | gust(ESC) | merge | mrg/gust |
|---|--:|--:|--:|--:|--:|
| Dense | 6 | 1.7 | 1.8 | 2.0 | 1.09 |
| Mildly sparse | 45 | 2.0 | 2.2 | 3.1 | 1.25 |
| Highly sparse | 39 | 4.7 | 5.7 | 12.0 | 4.05 |
| Extremely sparse | 10 | 3.6 | 3.9 | 4.9 | 0.97 |

**分阶段(均值 ms,merge 的「排序」列 = merge 阶段本身)**:
| 方法 | 计算(expand) | 排序(sort/merge) | 合计 |
|---|--:|--:|--:|
| Gustavson(ESC) | 0.60 | 5.91(sort) | 9.02 |
| Merge | 2.20 | **22.88(merge)** | 26.88 |

## 4. 两个瓶颈(决定 v2 方向)

### (a) 重行 straggler —— bp_* 族,与总规模无关 ★主因
- bp_0..bp_1600(共 ~11 个):822×822,但有一行 **max_row_nnz≈300**;其余行才 3–5。
- 串行 merge 对那个 num_k≈300 的重行 = output×300 次串行全局读,**单个线程块就要 ~15ms**,其余 821 行微秒级 → 整个矩阵被 1 个 straggler 拖到 11–17× 慢。
- 全 first100 max_row_nnz 最大才 339(bcsstk08),无 ≥1000 的病态行,但 ~300 已足够致命。

### (b) 大矩阵 —— 每输出元素重读 num_k 个段头
- bcsstk30(28924 行,C_nnz 8.9M):merge 阶段 **78.7ms** vs ESC sort 23.3ms。
- 根因:每个输出元素都要从**全局内存**重扫 num_k(~70)个段头 → ~12 亿次全局读 → 79ms。

## 5. 结论 / v2 方向

- **v1 证实 merge 路线可行 + 正确**,且在**正常(非重行)矩阵上 ≈ 持平 ESC**,小/极稀疏矩阵上一致地赢(gust 没有的固定 sort 开销)。
- **两个瓶颈同指向一个 v2:行内并行协作 merge** —— 每步对 num_k 个段头做并行 min-reduction(而非 thread0 串行扫)。
  - 救 (a):num_k≈300 的重行由 32 线程分担 → ~10×。
  - 救 (b):多线程并发取段头,隐藏全局读延迟。
- 候选实现:warp-per-row + `__shfl` min/sum 归约(shared 放 num_k 段指针);或 block-per-row 协作。merge 的串行 expand 也应顺手并行化(当前 1-thread expand 比 ESC 的 256-thread expand 慢 ~3.6×,见分阶段「计算」列 2.20 vs 0.60)。

## 6. 产物

- `results/aa/first100_aa/log/*.log`(100 个,含 `[merge]` 阶段桩)—— 本次由 `bash scripts/run_aa.sh` 重新生成。
- `suitesparse_crawl/profile_aa{,_summary}.csv`、`charts/profile_aa.png`、`charts/profile_totals.png`。

> 注:`make clean` 会 `rm -rf results/*`。本次实现期间误跑过一次 clean,删了旧日志(已用 run_aa.sh 恢复);分析 CSV/charts 在 `suitesparse_crawl/` 下未受影响。
