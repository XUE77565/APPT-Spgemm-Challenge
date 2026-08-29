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

## 5. A/B 判决(2026-08-30,solo 交替 ×4):全中性 → 间隙理论死亡

TSOPF −0.5/brainpc2 −0.2/c-64 +0.3/Ga3 +0.4/mult_dcop −0.4/bloweya +2.3/pwtk −0.1/333SP +0.1;
mh_merge 相位全部不动(1.833→1.839 等);nnz 全对;gate 阵零回归。device 门基础设施正确
(total_flop 逐位一致)但消 count_flop 后的 host 空档对 toll 无效。

**toll 定论(合并全部证据)**:与 grid/工作量/字节/前序空档全部无关,只与运行历史负载形态
有关 —— h2d/d2h 在 copy engine 执行、SM 长闲 → 时钟 345MHz,front-half 小 kernel 全家冷跑
(construct 149μs vs 隔离 ~40μs = 3×;merge 1.7ms vs 40μs = 40×,延迟链对时钟最敏感)。
**代码侧无解;出路 = 管理员锁频(双卡 + Ocean 同条件重测 = 更公平的论文口径)**。
PIPEFUSE 默认关保留(device 门/规约基建可能复用);step-2(binning 折入)取消 —— 前提已死。
