# docs/58 PIPEFUSE step-1:device 门消 count_flop 后的 host 空档

**日期**:2026-08-30 | **状态**:代码就绪编译过,`PIPEFUSE=1` 开(默认关),**待 GPU 空闲后 A/B**

## 1. 动机(docs/57 的 DVFS toll 续)

管线同步点普查(前半段):①count_flop 的 thrust::reduce(**host 回传**)→ ②binning 三合一 sync →
③规模门 dense_sum D2H → ④count 后 est/nnz sums D2H → ⑤坍缩重扫+tmp_slots D2H → ⑥cnnz_scan。
每个 sync = GPU 流水排干 = 时钟下滑空档;mh_construct/merge 恰在空档①之后冷启(1.3-1.7ms toll
家族)。nodbg 实验证明 hash-prof 的 event sync 不是元凶 ⇒ 元凶是管线自身 sync(①还在)。

## 2. step-1 改动(最小侵入)

1. `flop_reduce_kernel`(block 归约 256T,atomicAdd 累加)+ `avg_gate_kernel`(单线程判
   avg_product≤AVG_FLOP_THR 写 device 门)—— total_flop 全程留 device。
2. `mh_construct_kernel`/`mh_merge_kernel` 增尾参 `avg_gate`:门命中时 construct 整体空转、
   merge 每 block 直填 `est=row_flop`(与 memcpy 路径逐位同值);门未命中 = 原 MinHash 路径。
3. host:`pfuse` 时 count_flop 后**无条件**背靠背 launch construct+merge(gate 命中的矩阵付
   两个空 kernel ~10μs);total_flop 推迟到 binning 已有 sync 批量取回(不新增同步点)。
4. 鲸鱼防护:`A_rows×MH_M×4B > 256MB` 自动回退老流程(d_mh 池浪费防护,333SP 类 gate 大阵)。
5. MHSAMP 与 pfuse 互斥(MHSAMP 的 host 决策破坏无同步性质)。
6. 顺手修:flop_reduce 首版 warp shfl off=128 越界(32 lane 上限 16)。

## 3. 待验证(下一 GPU 窗口)

- A/B:PIPEFUSE 0/1 交替 ×4 取中位(docs/56 方统计学),阵 = MinHash 集(TSOPF/brainpc2/c-64/
  Ga3/mult_dcop/bloweya)+ gate 集回归(pwtk/333SP = construct/merge 空转阵)。
- 观察:mh_merge/mh_construct 相位是否随空档消失而缩短(DVFS toll 理论的直接检验);
  cnnz 全对;gate 阵零回归。
- 若 toll 下降 → step-2:binning 折入 per-bin est/flop sums(消③④⑤,前半段 sync 收敛到 1 个);
  若不变 → toll 另有来源,回 nsys 时间线细查。

## 4. 风险

- merge 门早退分支写 est=flop:逐位等价 memcpy 路径 ✓;但 gate 边界(avg_product 恰在 64 附近)
  的浮点比较从 host(double 除法)变为 device 同式计算 —— 同表达式同结果 ✓。
- d_mh 无条件分配:256MB 上限内;pool 行为(no_collapse 教训:pool 扩容成本)—— 若 A/B 见
  分配回归,再收上限。
