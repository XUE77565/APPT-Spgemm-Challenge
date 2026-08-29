# docs/57 MHSAMP 与 mh_merge 的 DVFS toll 之谜

**日期**:2026-08-30 | **MHSAMP 状态**:代码在,默认关(`MHSAMP=1` 开)——正确但**不省时**,原因见下

## 1. MHSAMP 设计(正确性全部验证通过)

Ocean Ana2(3% 采样判 est-vs-precise)同哲学:dense_win 候选(14980<n≤200k, A_rows≥1000)先
stride 采样 merge(紧凑 grid,S=2048 行)→ 投影 Σest 过 DENSE_MIN_FRAC 门 → dense-注定:
- 免全量 merge;d_est=min(flop,n) 填充(dense_win+DIRECT5 下 per-row est 只喂门,d_off 随
  est 置零重扫,布局零消费——已 grep 验证全部消费者)
- total_est 用投影值覆写(uc 路由 dup 信号保持诚实)
- **投影精度:231,944,267 vs 全量 231,712,135 = +0.1% 误差**;TSOPF cnnz 精确一致

## 2. 但不省时:发现 mh_merge 的固定 toll

| 场景 | mh_merge 时间 |
|---|---|
| 管线内全量(28216 行) | 1.63-1.75ms(nsys 实锤,非仪器误差) |
| 管线内采样(2016 行,7% 工作) | **1.69-1.72ms(不变!)** |
| 隔离微基准(逐字同 kernel,同形数据,28216 行) | **0.040ms** |
| 隔离采样(2016 行) | 0.009ms |

采样数学证明工作真的只有 7%(s_sum=全量 7%)但时间 100% ⇒ **toll 与工作量无关,与管线上下文绑定**。

## 3. 根因:DVFS idle 时钟 + 冷上下文(证据链)

1. `nvidia-smi -lms 100` 轮询跑批中的 GPU1:**85% 样本 = 345MHz**(长 d2h 84ms 空档把 SM 频率
   打到 idle),只有 dense_direct 重 kernel 拉到 1755MHz。
2. 小 kernel 在 345MHz + 冷 TLB 下散射读 14.4MB sketch(d_mh,A_rows×128×4B)= 1.7ms;
   微基准背靠背 launch 维持 boost,同 kernel 40μs(≈5TB/s,L2 常驻)。
3. est_fill(触 113KB)同管线只 10μs ⇒ toll ∝ 触碰的冷数据量,非 launch 本身。

## 4. 含义(比 TSOPF 翻面更大的杠杆)

1. **每个 MinHash 阵(TSOPF/c-64/brainpc2/Ga3/mult_dcop…)都在付 1.3-1.7ms 的 DVFS toll**
   ——中近赢带 160 阵的"管理相位税"很可能大半是同类。
2. 攻法(未做,future):① 相位融合(mh_construct+merge+est_scan+binning 连成无同步链,消空档)
   ② PDL(docs/33:sm_90 相边界 ~2.1μs×launch)③ 锁频(`nvidia-smi -lgc` 需管理员;GPU1 曾有
   管理员 reset 通道)④ 减少管线 D2H 同步点(今天 DCFUSE 已砍 2 个)。
3. ⚠ 对照公平性:Ocean 的 estimation 占其论文 runtime ~4%(小 kernel 同样可能吃 DVFS toll);
   我们与 Ocean 都在同一流程下实测,toll 双方都计 —— 但我们管线同步点更多 = toll 更重,这是
   **真实的架构差距**,消 toll = 真实提速。

## 5. 资产

- `scripts/mh_microbench.cu`:隔离微基准(grid/stride/a_len/数据内容四变量扫描)
- `/tmp/mhs_nsys.nsys-rep`:nsys 现场(kernel 1.75ms × 6 instances 实锤)
- MHSAMP 代码(默认关):kernel `sample_stride` 参数 + host 采样分支 + est_fill_kernel +
  binning 后 total_est 覆写;`DCFUSE` 同批在 docs/56
