# 30 · Phase B v2:游标窗口内核 + 动态组(2026-08-28 凌晨,用户睡觉自主夜战)

> 目标:mult_dcop/vsp_south31 类(Ocean 走 denseNumericIterKernel,我们同族窗口内核输 4.8-6.9×)。
> 手段 = docs/27 §4 的 Ocean iter 内核三件套:**start_map 游标 + localLoadBalance 动态组 + 数据驱动窗口**,
> 装进 `hash_dense_direct_kernel`(DIRECT5 的数值 pass)。

## 1. 诊断链(为什么窗口内核慢)

- mult_dcop_03 相位:dense_direct 66.9ms 占 96%,吞吐 7.8G prod/s vs Ocean iter 54G/s = **6.9×**。
- **固定 warp-per-k 的利用率**:重行 a_len 小而 B 行长(如 a_len=5、avgB=4000)→ 5 warp 干活,
  16 warp 中 11 个闲置 = 31% 占用。→ 动态组(G 到整 block)。
- **每窗每 k 的 lower_bound 重复搜索**:G 越大冗余越多(G×2log(avgB) vs 有效功 avgB)——
  动态组单独上时 c-58 +18%/case39 +32%/gupta1 +52%/TSOPF +59% 回归的根因。→ 游标消除搜索。

## 2. 实现(hash_dense_direct_kernel)

- **游标**:每 k 一个全局 int(区 = Σ a_len 的 exclusive-scan 偏移;`smap[2×Σa_len]` 双区:活动槽 +
  每窗起点镜像)。首窗每 k 恰一次 lower_bound(block-stride 零冗余);越窗线程 `atomicMin` 回写推进;
  **每窗先读旧值进镜像 + 原槽复位 INT_MAX**(Ocean reset-to-∞ 语义),**哨兵检查**(INT_MAX = 该 k 耗尽,skip)。
- **动态组**:local_load_balance(a_len, flop, maxbl, 5, 9) → G∈[32,512];无搜索冗余后纯利用率驱动。
- **数据驱动窗口**:下一窗起点 = 全体越窗列的最小列(atomicMin 块归约;跳空窗,全处理完 = n 退出)。
- **混合路由**:avgB = flop/a_len < 64 的行走旧固定窗搜索路径(=52b89da 行为)—— 游标镜像/复位是
  每窗 O(a_len) 全局往返,开销/工作量 ∝ 1/avgB(TSOPF avgB≈16 实测吃亏)。
- PB2_W=12900 窗宽(SMEM 9B+4B/列);PB2_CURSOR=0 可整体关(对照组)。

## 3. 调试战史(三连坑,全部 cuda-gdb/SASS 实锤,防复发)

1. **"illegal instruction" ①**:smap 偏移用了 **inclusive scan + 槽位 +1 双错** → 行 chunk 右移互相
   覆盖 → 游标读到别行 B 位置 → **j<0 → SMEM 下越界写**(SASS:`ATOMS.CAST.SPIN` 自旋前无下界检查)
   → 砸内核元数据。修 = 无移位 exclusive scan(docs/21 陷阱再现)。
2. **"illegal instruction" ②**:atomicMin 的旧游标恒小于新 break 值 → **min 永不更新 → 游标永不前进**
   → 下窗重读旧位置 → col<win_lo → j<0。修 = 每窗镜像读出 + 复位 INT_MAX(Ocean 原版有此行,
   解剖时见过 `start_map[...] = -1` 但初版漏移植)。j<0 钳位实验实锤可达性后定位。
3. **"illegal memory access"**:耗尽 k 的槽留 INT_MAX → 下窗 `q = INT_MAX+my_id` 溢出为负 →
   `B_col_idx[负]`。修 = 哨兵 skip(Ocean 的 `if (b_start == -1) continue`)。
4. 混合分支窗宽误用 DENSE_MAX_N(14980)> 数组尺寸 PB2_W(12900)→ SMEM 越界,统一 PB2_W。

教训:**移植 Ocean 内核时"看起来多余"的复位/哨兵行是语义的一部分**;SMEM 下越界写在 H100 上
表现为 illegal instruction(不是 memory access),SASS 的 ATOMS.CAST.SPIN 自旋无边界检查。

## 4. A/B(vs refresh9-D5 基线,即 52b89da 行为;游标+avgB 路由)

| 阵 | 基线 | 游标版 | Δ | 备注 |
|---|---|---|---|---|
| vsp_south31_slptsk | 84.94 | **36.7-37.7** | **-56%** | 5.6×→**2.4×**(Ocean 15.2)|
| mult_dcop_01/03 | 77/69.4 | **39.7/40-41** | -43~-48% | 7.9×→**2.8×**(Ocean 14.4)|
| bloweya | 18.26 | **9.7-11.8** | -35~-47% | |
| brainpc2 | 28.40 | 21.1-21.4 | -25% | |
| case39 | 77.13 | 71.6-77.1 | -6%~0 | |
| c-58 | 11.61 | 11.2-13.0 | -3%~+12% | 单测方差 ±7%,真值待全量 |
| gupta1 | 72.97 | 72.3-73.1 | ±0.5% | |
| TSOPF_FS_b39_c7 | 31.86 | 38.4-40.4 | **+22~27%** | ⚠ 基线本身 31.9↔38.9 双模噪声;avgB<64 已路由搜索路 |
| pre2/pwtk/exdata/333SP | — | ±0.2% | 0 | 不受影响 ✓ |

cnnz 全对 ✓。**refresh10(cursor 默认开)跑批中定夺 geomean 与路由阈值**;若 TSOPF 类全量仍差,
回退手段 = PB2_CURSOR=0 或调阈值。

## 5. 待办(下一班)

- refresh10 结果 → geomean vs 1.9722×(v10);回归清单 → 路由阈值精调(avgB 分界 64 的 CPU 精算)。
- pre2(5.3×)= **hash 行的方案5**:count pass 扩到 csort bins(compact+sort 62ms 占 76%!)+
  sortOutputFused 融合键(docs/27 §4.1 完整体)。
- hash_global 的 LLB(Ge99 类)。
- 矩阵级 dense_win 直接案例(本轮仍未逮到)。
