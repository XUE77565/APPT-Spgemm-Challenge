# Parallel k-way Merge v2:warp 协作 → 全面翻盘 ESC

> 日期:2026-07-16,分支 `feature/optiESC`。
> v1(串行,见 `merge_serial_v1_2026-07-16.md`)摸清瓶颈:① 大矩阵散列全局重读段头;② bp_* 重行 straggler。
> v2 = **warp-per-row 协作 merge**(每行一个 block=1 warp,32 线程并行归并),直接对症。
> **serial v1 / ESC baseline 全部保留**,v2 为并列新函数 `spgemm_self_product_merge2`。

---

## 1. 实现(`src/spgemm_merge.cu`,与 v1 同文件)

新增 `merge_warp_kernel<<<A_rows,32,smem>>>` + host `spgemm_self_product_merge2`。与 v1 两处本质差异:

- **① 无 expand 阶段(Tier 0 融合)**:merge 直接读 A 的 CSR 行(段 p = `A[k_p,:]`),砍掉「写 COO 一遍 + 读回一遍」的来回 traffic。
- **② 行内并行(Tier 2)**:每行 1 warp,shared 放 `seg_ptr[num_k]/seg_end[num_k]/weight[num_k]`;每步 32 线程各自扫持有的链头 → `__shfl_xor_sync` min-归约取全局最小列 → 各线程把头部==min 的链累加前进 → sum-归约 → lane0 输出。

```
loop:
  (1) 每 lane 在自己 ⌈num_k/32⌉ 条链里找局部最小头列(直接读 A,32 路并发隐藏延迟)
  (2) warp shuffle min-归约 → wmin;  INT_MAX → break(全 warp 一致)
  (3) 每 lane:头部==wmin 的链累加 weight*A_val,前进
  (4) warp shuffle sum-归约 → wsum
  (5) lane0 输出 (wmin, wsum); out_idx++
```

正确性:展开的是同一批中间项;逐次输出全局最小列并合并同列 → 等价 sort+reduce。count/scan/compact 复用 v1。warp 内各 lane 只读写自己 stride 拥有的 `seg_ptr[p]`(无跨 lane 竞争),shuffle 隐式同步,break 全 warp 一致无发散。

## 2. 正确性

全 100 矩阵四法(cu / ESC / serial / **par**)C_nnz **完全一致**(1138_bus 11142;bcsstk30 8,946,070;bp_200 13197)。run_aa:OK 100 / FAIL 0 / TIMEOUT 0。

## 3. 性能(全 100 矩阵)

> ⚠️ 本节数据是 **legacy(`USE_MEMPOOL=0`,未开 arena)** 跑的,d2h 膨胀/抖动。开 arena 重测后更优(作准):**par vs ESC 0.747×、赢 89/输 10、4 类全赢**;并新增 **par vs cuSPARSE 0.861×、赢 83/输 15**。详见 `merge_acceleration_session_2026-07-16.md` 阶段 3。

**总览(merge vs gust-ESC)**:
| | 赢 | 平 | 输 | 几何均值 |
|---|--:|--:|--:|--:|
| **并行 par(v2)** | **85** | 3 | 12 | **0.910×** ★整体比 ESC 快 |
| 串行 ser(v1) | 52 | 7 | 41 | 1.363× |

并行 vs 串行:几何均值 **1.50× 加速**,91/100 矩阵并行更快。

**按类别(均值 ms)**:
| class | # | cuSPARSE | gust(ESC) | merge(ser) | merge(par) | par/gust | par/ser |
|---|--:|--:|--:|--:|--:|--:|--:|
| Dense | 6 | 1.2 | 1.3 | 1.5 | **1.2** | 0.88× | 1.26× |
| Mildly sparse | 45 | 1.6 | 1.9 | 2.8 | **1.6** | 0.85× | 1.35× |
| Highly sparse | 39 | 4.4 | 5.4 | 11.6 | **4.4** | 1.00× | 1.89× |
| Extremely sparse | 10 | 3.7 | 4.1 | 5.1 | **3.4** | 0.89× | 1.09× |

→ merge(par) **4 类全不输 ESC**,且与 cuSPARSE 基本打平(Mildly 1.6 vs 1.6、Highly 4.4 vs 4.4)。

**分阶段(均值 ms)** —— 证实 merge kernel 本身已快过 sort:
| 方法 | expand | 合并阶段(sort/merge) | compact/reduce/final | 合并步合计 |
|---|--:|--:|--:|--:|
| gust(ESC) | 0.13 | sort **0.75** | 0.32 | 1.07 |
| merge(ser) | 0.36 | merge 3.54 | 0.19 | 3.73 |
| **merge(par)** | —(无) | merge **0.45** | 0.13 | **0.58** |

并行 merge kernel(0.45ms)< ESC sort(0.75ms),且无 expand → 合并步 0.58ms vs ESC 1.07ms。

## 4. 仍输的 12 个(边界)

9 个 bp_* + bcsstk08(822–1074 行,**小矩阵 + 重行**,ESC sort 本就便宜,N 小压不过)+ bcspwr01(39×39,固定开销主导)+ bcsstk28(1.20× 临界)。
即:**小矩阵** regime —— sort 固定开销低,merge 的每输出工作 + kernel 开销压不过。中大矩阵 merge 一致赢(top:bcsstk24 0.66×、bcsstk29 0.68×、bcsstk16 0.70×)。

## 5. 结论

- v2 验证了并行 k-way merge 路线**全面可行且优于 ESC**:整体 0.91×、赢 85/100、merge kernel 本身快过 CUB sort。
- 两个 v1 瓶颈都被治住:bp_* 重行(straggler)从 11–17× 恶化降到 ~2×;大矩阵 merge 从 ~79ms 降到个位数 ms。
- 残留差距在小矩阵(sort 固定开销低)—— 若要再榨,可上 Tier 1(shared 缓存段头)进一步压 merge kernel,或对小矩阵走「直接 sort」的自适应派发(方向 B)。

## 6. 产物

- `results/aa/first100_aa/log/*.log`(100 个,含 `[merge]` 与 `[mrg2]` 阶段桩)
- `suitesparse_crawl/profile_aa{,_summary}.csv`(4 法)、`profile_aa.png`(阶段构成含 Merge(ser)/Merge(par))
- `suitesparse_crawl/analyze_merge.py` + 图:`merge2_vs_esc_scatter.png`、`merge2_speedup_over_serial.png`、`merge2_winloss_by_class.png`
