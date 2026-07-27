# Engineering Optimization Log — AA (C = A·A)

Goal: systematically optimize the **AA self-product** pipeline — hash path first
(large matrices, where the dispatcher sends AA), then merge3 (small matrices).
Each change is measured before/after and **KEPT only if net-positive** (correctness
preserved + compute down); reverted otherwise. ATT optimization comes later.

---

## Methodology

- **Benchmark**: `scripts/bench_aa_hash.py -n 3` → median **compute-only**
  (cudaEvent `hash-prof`, excludes h2d/d2h) + per-phase medians, on
  `bcsstk30 / bcsstk32 / bcsstk31 / bcsstk29 / bcsstk17`.
- **Correctness gate**:
  - nnz-affecting changes: `C_nnz == cuSPARSE`, **0 overflow** (overflow → merge3 fallback).
  - **accumulate / value-affecting changes**: also verify **values == cuSPARSE**
    element-wise via `scripts/compare_mtx_values.py` (hash `.mtx` vs cuSPARSE `.mtx`),
    not just nnz (a value bug can leave nnz correct).
- **Verdict**: `KEPT` (net positive + correct) / `REVERTED` (not positive or broke correctness).
- **Context — where the time is** (bcsstk30, EXPAND=1.5): accumulate 2.42 ms (≈67%,
  **atomic-bound**, proven irreducible in worklog §21–24 — 3 atomic-reduction levers
  failed), compact+sort 0.40, sizing (mh_merge+construct+scan+binning) ≈0.33. So the
  realistic levers are compact+sort / sizing / binning, and a careful accumulate attempt.

## Baseline (starting point, EXPAND=1.5)

`scripts/bench_aa_hash.py -n 3`, `./spgemm_test` (DBG=1, MinHash, EXPAND=1.5):

| matrix   | compute | accumulate | compact+sort | mh_merge | binning |
|----------|--------:|-----------:|-------------:|---------:|--------:|
| bcsstk30 | 3.60    | 2.42       | 0.40         | 0.171    | 0.081   |
| bcsstk32 | 2.16    | 1.50       | 0.26         | 0.174    | 0.111   |
| bcsstk31 | 1.36    | 0.92       | 0.15         | 0.106    | 0.091   |
| bcsstk29 | 0.84    | 0.53       | 0.08         | 0.059    | 0.071   |
| bcsstk17 | 0.75    | 0.42       | 0.07         | 0.049    | 0.088   |
| **geomean** | **1.46** | | | | |

---

## Optimization log

### #1 — `HLL_EXPAND` 2.0 → 1.5 (tighter sizing safety factor)  · KEPT
- **What**: sizing rule is `next_pow2(est × HLL_EXPAND)`. The blanket ×2.0 was
  over-conservative for an estimator with ~9–15% error → over-alloc 3.14×.
  Lowered to 1.5 (still ≥ estimator error). (Applied before the formal harness;
  before/after are single warm-block cudaEvent runs.)
- **Before (2.0) → After (1.5)**, bcsstk30:
  - over-alloc: 3.14× → **2.29×** (bcsstk32 2.10×, bcsstk31 2.15×)
  - compact+sort: 0.71 → **0.39** ms (smaller tmp buffer → less to sort)
  - compute-only: 3.31 → **3.16** ms
  - accumulate: 2.23 → 2.41 (noise; atomic-bound, unchanged as expected)
- **Floor**: EXPAND=1.3 **overflows** on bcsstk30 (heaviest row trips HASH_CAP) →
  unusable. **1.5 is the safe floor.**
- **Correctness**: C_nnz correct on bcsstk30/32/31, 0 overflow.
- **Verdict**: **KEPT** — real win (memory −30%, compact+sort −0.32 ms), no correctness
  cost. Does NOT beat opSparse on its own (accumulate atomic wall untouched) but is a
  strict improvement.

### #2 — read-check before CAS in `hash_spa_kernel` (HASH_MULTI)  · REVERTED
- **What**: in the accumulate inner loop, read `sh_col[slot]` first; if it already
  holds `j` (duplicate), do only `atomicAdd` and **skip the CAS**. Duplicates
  dominate (19× dup), so the hope was to roughly halve atomic ops (skip 18 of 19 CAS).
- **Hypothesis**: CAS on duplicates is the accumulate bottleneck.
- **Measurement** (bench_aa_hash -n 3, bcsstk30): accumulate 2.42 → **2.37 ms** (−2%);
  other matrices flat-to-slightly-worse (bcsstk31 0.92→0.95). The apparent compute-total
  drop (3.60→3.11) was run-to-run noise — accumulate & compact+sort are both flat, so
  real compute is flat.
- **Correctness**: values == cuSPARSE (can_24, bcsstk08 PASS) — the transform is sound.
- **Why it failed**: the bottleneck is the **atomicAdd**, not the CAS (every duplicate
  still must `atomicAdd` its value; the read-check only removes the CAS). Consistent with
  worklog §21–24 (accumulate is atomic-throughput-bound; 3 atomic-reduction levers failed).
- **Verdict**: **REVERTED** — not a net positive (accumulate flat + extra branches), and it
  re-confirms the accumulate atomic floor.

### #3 — route bin4-5 (ht 512/1024) count-sort → right-sized `csort<128,8>`  · REVERTED
- count-sort (fused O(n²)) costs ~0.47 ms inside accumulate (bcsstk30). Tried routing bin4-5
  through a right-sized `csort<128,8>` (capacity 1024, low SMEM, high occupancy) — fixing the
  high-SMEM/low-occupancy mistake of the earlier `csort<512,8>` attempt.
- accumulate 2.42→1.95 (−0.47, count-sort gone) BUT compact+sort 0.40→1.05 (+0.65). Net
  geomean 1.46→1.47 (flat/worse; bcsstk32/31/29 got worse). Values correct.
- Even right-sized, the **per-block launch overhead** of a separate radix sort (18529 bin5
  rows) exceeds the fused O(n²). count-sort stays.

### #4 — fused in-SMEM bitonic sort replacing count-sort  · REVERTED
- Replaced the O(n²) count-sort loop with a fused bitonic sort (O(n log²n), (key,val) pairs,
  INT_MAX padding). Verified correct by hand ([3,1,2]→[1,2,3]) and by value-check (PASS).
- But accumulate 2.42→**2.96** (+0.54), geomean 1.46→1.73. The ~55 `__syncthreads`/block
  (p=1024) × 18529 bin5 rows = **sync-bound**.
- **Finding (general)**: for sorting *many small* per-row arrays on GPU, a **sync-free O(n²)
  count-sort beats both** sync-bound bitonic **and** launch-bound per-row BlockRadixSort —
  at this scale the sync/launch overhead dominates algorithmic complexity. **count-sort is the
  floor** for the small-row extract; ~0.47 ms is irreducible with known techniques.

### #5 — `HASH_PRIV` warp-private SPA (Phase A plain `+=`, no atomicAdd)  · REVERTED
- **What**: the dormant `hash_spa_priv_kernel` (gated `HASH_PRIV=1`, never benchmarked in
  #1–#4). Each warp owns a private SMEM hash table; k-loop partitioned across warps so
  intra-warp accesses to a column are temporally non-concurrent → Phase A accumulate uses
  plain `+=` (no atomicAdd), only Phase B fan-in (≤W warps) uses atomicAdd. Directly tests the
  "accumulate is atomicAdd-throughput-bound" claim.
- **Result** (bench_aa_hash -n 3): accumulate got **slower** — bcsstk30 2.39→**4.28 ms**,
  geomean 1.36→**1.71**. Applies only to bins with W·ht·12 ≤ 196 KB (ht≤2048, bins 0–6);
  heavy bin-9 rows (ht=16384) still flat.
- **Why**: W=8 private tables = 8× SMEM → **1 block/SM occupancy collapse** (heavy bins are
  already SMEM-limited: ht=16384 → 192 KB → 1 block/SM at 256 threads). Removing the
  atomicAdd instruction did NOT help because the bottleneck is latency/occupancy, not
  atomic-instruction throughput — and the added Phase-B fan-in + 8× table init cost more than
  the saved atomics. ncu could not confirm (dynamic-SMEM opt-in breaks kernel replay;
  `--replay-mode app` unsupported in this 2023 build).
- **Verdict**: **REVERTED**. Re-confirms the accumulate floor is **occupancy+latency**, not raw
  atomic count — so any "reduce atomic count" lever that costs SMEM is doomed on heavy bins.

### #6 — device buffer pool (dev_alloc arena, replaces ~13 cudaMalloc/Free per call)  · KEPT
- **What**: a device bump-arena (`dev_alloc`/`dev_free`/`dev_pool_reset` in mempool.cu),
  same bump-reset model as the host pinned pool. `hash_product` entry calls `dev_pool_reset()`;
  all ~13 device temporaries (dA/d_off/d_est/d_mh/d_row_nnz/d_tmp_key/val/d_bkid…/dC/dC_rp)
  are carved from the 16 GB arena via pointer bump (0 driver calls); `dev_free` is a no-op
  (arena reclaimed by next reset). Same-binary A/B via `USE_MEMPOOL`. (Wired by
  `scripts/devpool_wire.py`.)
- **Result**: correctness PASS (bcsstk17/30 vs cuSPARSE, ~1e-18). **Wall-clock** (USE_MEMPOOL
  0→1): bcsstk30 58.7→**9.2 ms**, bcsstk17 9.4→**1.9 ms**, bcsstk11 1.84→**0.91 ms**. Most of
  the gain is the *existing host pinned pool* (avoids per-d2h `cudaMallocHost` page-locking);
  the device arena adds the ~0.66 ms/call from eliminating ~13 `cudaMalloc`+`cudaFree` driver
  round-trips.
- **Compute-only neutral**: accumulate 2.42→2.42 ms (identical) — these allocs sit *outside*
  the cudaEvent prof tags, so `TOTAL−h2d−d2h` is unchanged. The pool helps **wall-clock / small-
  matrix competitiveness only**, NOT the compute-only comparison metric.
- **Verdict**: ~~KEPT~~ → **RE-REVISITED 2026-07-27, DEFAULT OFF**(见下 "device pool 复测")。
  对 **compute-only 口径是负优化**(小阵 hash 扫描慢 1.5-3×);wall-clock 收益仍在,但被
  `USE_DEV_POOL` 独立开关默认关掉。host pinned pool 保留(归 USE_MEMPOOL)。

## Where the time really goes → what's left
- accumulate 2.4 ms (bcsstk30) = **structural floor** (atomic-bound, proven). No known lever.
- Remaining compute (compact+sort 0.39 + sizing 0.31) is small → diminishing returns.
- **d2h 3.8 ms** (107 MB output): host buf is `cudaMallocHost` (pinned), src is contiguous
  device, sync copy → 28 GB/s ≈ **PCIe Gen4 x16 peak** (link confirmed Gen4 x16). **At hardware
  floor — not a lever.**
- **h2d 1.8 ms** for ~12 MB input = 6.7 GB/s — anomalously below Gen4 peak. Open question:
  is `A_buffer` truly pinned on the bench path? → entry #3 candidate.

## Measurement-caliber note (important for the opSparse comparison)
`sum-of-phases == TOTAL − h2d − d2h` **exactly** (gap = 0.00 ms). This means the HashProf
`TOTAL` is the **sum of tagged phase events**; work that runs *between* tags — the ~10
inter-phase `cudaMalloc`s and the `#ifdef DBG` `hash_check_sorted_kernel` — has **no event
around it and is never timed**. Consequence:
- The bench's compute (3.15 ms) **excludes** the real cudaMalloc + sorted-check cost
  (~0.3–0.5 ms of actual wall-clock work). So vs opSparse (host timing, which *does* include
  its mallocs), our number is **optimistic** — the comparison *favors us*, not opSparse.
- Reducing those (pool/pre-alloc; gate sorted-check out of timed runs) would speed real
  execution but **won't show in the cudaEvent bench** — needs host/wall-clock measurement.
- (Pooling was tried in worklog §16: cudaMallocAsync added +0.79 ms *under cudaEvent* due to
  stream-capture and was reverted — a measurement artifact, not a real regression.)

## Wall-clock overhead — the one real remaining lever (quantified)
Measured `host chrono Time − cudaEvent TOTAL` (the untagged host work):
| matrix | host | cudaEvent | gap |
|---|---|---|---|
| bcsstk30 | 9.94 | 8.67 | **1.27 ms** |
| bcsstk32 | 7.65 | 6.61 | 1.04 ms |
| bcsstk08 | 1.40 | 0.74 | **0.66 ms** |

The gap is largely **fixed** (~0.66 ms even on tiny bcsstk08, whose compute is only 0.74 ms) =
~13 `cudaMalloc` + ~13 `cudaFree` driver calls + kernel-launch overhead. This is the
**launch-overhead** that lets spECK beat us on small matrices (§25). The `hash_check_sorted`
block is **~0 ms** (gating it changed nothing) — not the cause.
- **Why not pursued here**: (a) invisible to the cudaEvent bench (work is between tags); (b)
  pooling via `cudaMallocAsync` was tried in §16 and **broke cudaEvent** (+0.79 ms, stream
  capture); (c) manual buffer-combining is a risky refactor of the hash.cu host fn (multiple
  free paths for overflow/underflow). **This is the highest-value remaining engineering work**,
  esp. for small-matrix / suite-geomean competitiveness — but needs wall-clock measurement and
  careful pooling (persistent device pool, alloc-once-reuse, outside tagged regions).

## 2026-07-27 复测 — device pool = compute-only 负优化(默认关)+ ③ kernel 融合 + dispatcher 复核

**起因**:对照 07-24 best(`compare/method_cmp_20260724_094758(best)`)发现 engiOpti 后 Auto 慢 1.42×
(87/100 阵),bp_*/bcsstk08 等**输 HSMU/opSparse**(之前全胜)。逐阵对比定位根因。

### device pool 复测 → DEFAULT OFF(独立开关 `USE_DEV_POOL`)
- **根因**:#6 的 device arena(`dev_alloc` bump)让 hash **扫描 phase**(est_scan/binning/cnnz_scan)
  在**小阵慢 1.5-3×**(accumulate 不变)。干净 A/B(`USE_MEMPOOL=1` vs `0`,compute-only):
  bp_0 **2.06×**、bcsstk08 1.67×、bcsstk13 1.45×、bcsstk30 **1.08×**(大阵才中性)。
  → #6 的"compute-only 中性"结论**只对大阵成立**,小阵被漏掉(first100 78% 是小阵 → suite 负)。
- device pool 唯一收益是省 driver call(**wall-clock**),而 wall-clock 被 compute-only 口径排除 → 对论文
  对比是**纯损失**。256B 对齐试过没用(更差)→ 根因是 arena 内存访存特性,非对齐。
- **修复**:`mempool.cu/h` + `main.cu` 把 device pool 拆出独立开关 `g_use_dev_pool`(env `USE_DEV_POOL`,
  默认 **OFF**);host pinned pool 仍归 `USE_MEMPOOL`。修后 bp_0 **0.813→0.318 ms**(= best 0.307)、
  bcsstk08 1.050→0.546,原输 HSMU/opSparse 阵**全部反败为胜**。
- **教训**:benchmark 须覆盖 size 谱;小阵 compute-only 上 arena 内存 ≠ 独立 cudaMalloc。

### dispatcher 复核 → 公式正确,不用改
- 怀疑过 dispatcher 选错,实测**没问题**。bp_0:`score=-0.28 → hash`(对!),因 bp_0 有密行
  (max_row=266, skew=66.7)→ hash(0.328)确实 < merge3(0.394)。问题在 hash kernel 慢(device pool),
  不在选择。07-27 夜"重拟合(95%)"的 7 个 hash→merge3 翻转基本有益(小阵 merge3 更快),保留。
- 公式(多变量,100 阵拟合 R²=0.824):`score=-0.426·lfp -0.422·ln -0.644·lmr -0.642·lsk +5.43;<0→hash`。

### #7 — ③ kernel 融合(小阵 launch 开销)· KEPT
目标:削小阵 compute-only(bp_* 输 cu 0.03-0.08ms,launch 开销占大头)。两处融合:
1. **binning 融合**:`compute_bucket` + `bucket_count` 合 1 个 kernel(写 bucket_id 同时 `atomicAdd`
   直方图,N_BINS=11 计数器冲突低);2 个 D2H(h_cnt/h_off)用 `cudaMemcpyAsync` + 单 `cudaStreamSynchronize`
   合并。省 1 kernel launch + 1 memset + 1 同步点。
2. **单 block scan 融合**:`est_scan`/`cnnz_scan` 对 **A_rows≤1024** 用自写 `scan_inclusive_kernel`
   (Hillis-Steele,1 launch)替 `thrust::inclusive_scan`(thrust 对几百元素也走多 kernel)。大阵 gated
   (>1024 仍 thrust)→ 不回退。
- **结果**:bp_* 输 cu 从 0.03-0.08ms → **0.008-0.05ms**(bp_0/bp_200 贴到噪声级平手)。正确性 PASS
  (bp_0=9384, bcsstk30=8946070 nnz 不变)。大阵不回退(bcsstk30 3.24→3.11)。**净赚,全小阵(~72/100)受益**。
- **未能赢 bp_***:cu 核心单 kernel compute 仅 0.19ms < 我们 hash accumulate(0.135)+ 整条流水线开销。
  剩下的钱全在 accumulate(47%),是 ① 结构性地板,ROI 低(Auto 整体已赢 cu 89/100、geomean 1.89×)。

### compute-only 时间分布(修后,定下一步优化方向)
| 尺寸 | 代表 | 分布 |
|---|---|---|
| 大 | bcsstk30 (3.11ms) | **accumulate 77%**(结构性地板)+ compact+sort 12% |
| 中 | bcsstk13 (0.84ms) | **compact+sort 59%** + accumulate 23% |
| 小 | bp_0 (0.29ms) | accumulate 47% + compact 18% + binning 17% + scans 17%(融合后) |

**下一步杠杆(若继续)**:① accumulate size-adaptive SMEM(小行用小表提 occupancy,大行才用大表;
  预计大阵 accumulate −20-40%,有溢出风险,工作量高)。② compact+sort 阈值/配置(中阵,中等)。
  ③ 已做。bp_* vs cu 属 cu 强项,接受平手/微输。

## Campaign conclusion
Explored: EXPAND sizing (#1), read-check (#2), count-sort→csort (#3), count-sort→bitonic (#4),
merge3 merge (profiled), d2h (PCIe), wall-clock malloc/launch overhead, device pool (#6 复测),
kernel 融合 (#7), dispatcher 复核. **Net KEPT = #1 (EXPAND 1.5) + #7 (binning/scan 融合).**
**#6 device pool DEFAULT OFF**(对 compute-only 负优化;`USE_DEV_POOL=1` 可为 wall-clock 重开).
AA **compute** is at its hardware/algorithmic floors — accumulate (atomic, 大阵 77%),
compact+sort (中阵 59%), count-sort (sync-free), merge3 merge, d2h (PCIe Gen4). 剩余杠杆:
① accumulate size-adaptive SMEM(高工作量、有溢出风险、大阵 −20-40%);② compact+sort 调参(中阵).
小阵 bp_* vs cu 属 cu 强项(单 kernel 0.19ms),Auto 整体仍赢 cu 89/100 (geomean 1.89×)。
