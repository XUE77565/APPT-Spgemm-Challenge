# docs/65 复验工作流(环境灾变中断项的系统清偿)

**日期**:2026-08-31 | **并入 loop 纪律**:**净窗优先排空复验队列,再开新实验**

## 1. 为什么需要

load 11-34 的环境灾变(gem5/spec-cpu)打断/污染了一批结论(docs/60 §5 是最新一例)。
GPU 事件计时在 load~11 时对中型阵尚可用(±0.7%),load 34 时全部作废。凡是在污染窗内
得到的"数值"都必须净窗重验;凡是被打断的实验都带着未定论状态。

## 2. 复验清单(scripts/reverify_queue.py 内置,带出处)

| 组 | 阵 | 未定论内容 |
|---|---|---|
| A 受害者 | c-big/Flan_1565/inline_1/F2/TSOPF_RS_b678_c2/rajat28/F1/dielFilter×2/bmwcra_1/nd12k/cage15/bone010(13) | v27 的救回值测于 load11(docs/60 §5);**F1 还有 ex1.4 误选待纠** |
| B 路由三赢 | c-64/c-64b/TSOPF_FS_b39_c7/3Dspectralwave/3Dspec2 | docs/63 §6 的赢面测于 load11-34 |
| C 翻赢复核 | pkustk01/olafu/brainpc2/c-53/bbmat/rajat25 | v26 窗口基本净,边界阵仍需交替定论 |
| D 回归守卫 | bcsstk30/pwtk/c-62/c-62ghs/Ga3As3H12/mult_dcop_03/Cube_Coup_dt0(7) | 每次净窗复验必带(保底) |
| E 高方差 | rajat16/18/20/a0nsdsil/fp | refresh 单值永不可信(±15-25%),只认交替中位 |

另:未跑过的 **v28 全量 refresh**(真终版基线)在复验队列排空后启动。

## 3. 机制

- 脚本自带**净窗门**(/proc/loadavg < 6,--force 才可跳过)
- 与生产路径同口径(CM.run_spgemm_method,含 harness 双 expand 取优)
- 交替×3 取中位 → 写回 v28 CSV(methods_cmp_v28_reverify.csv)
- **nnz 红旗**:复验 nnz ≠ CSV 记录立即标记(docs/59 纪律内化)
- loop 集成:每次触发先查负载;净窗 → 先跑复验(或其剩余),再开新实验(docs/64 的 B 机制)

## 4. 判定与行动

- 复验值与污染窗值差 <5% → 结论转正
- 差 5-15% → 用净窗值覆写,结论按新值重判(翻面/回归状态可能变化)
- 差 >15% → 该阵进入"环境敏感"名单,其结论永远只认交替中位 + 多窗均值
