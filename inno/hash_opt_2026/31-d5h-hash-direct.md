# 31 · D5H:方案5 扩展到 hash 行(2026-08-28 晨,夜战第二件)

> pre2 相位:compact+sort **62.3ms = 76%**(95k 行落 bins 6-7,每行 BlockRadixSort + gapped 往返)。
> Ocean 对 pre2 走 **type1(precise)**:symbolic 2.17 + numeric 9.98 + epilogue 2.49 = 15.2ms vs 我们 81.7。
> D5H = 把 docs/27 §4.1 的"type1 直写终态"补到 hash 行(此前 DIRECT5 只覆盖 dense 行)。

## 1. 机制

```
legacy(hash 行):accumulate(gapped tmp + packed key)→ retry → scan → csort 读 gapped 写 dC
D5H:           MODE=1 count-only(hash 表只插 key,免值加载/免 atomicAdd)
                → retry(count 溢出行,flop 表,精确 row_nnz)
                → legacy bins(0/11/12)accumulate + scan + dC
                → MODE=2 数值 pass:小行(ht≤1024)count-sort 有序直写 dC;
                  大行无序直写 dC → 原地 csort_inplace(in==out)
                → compact 对 bins1-10 消失
```

- 门:`D5H=1`(默认关)+ **dup = total_flop/total_est < 8**(Ocean type1 的 compaction 经济学:
  count pass ≈ 0.4-0.6× accumulate,省 = compact+sort+gapped;高 dup 阵 est 工作流更便宜)。
- `hash_spa_kernel` 模板化 MODE(0=legacy/1=count/2=direct);count 的 ovf 语义与 legacy 一致
  (同表探测,表满即弃 → retry);MODE=2 跳过 ovf 行(retry 已产出,compact 末段拷入)。
- est 置零 + d_off 重扫与 dense 行共用机制;下溢检查改 `C − Σnnz_dense − Σnnz_d5h ≤ tmp_slots`。
- 原地排序 `csort_inplace_kernel`:BlockRadixSort col(int)/val(double) in==out,容量梯
  <512,8>/<256,64>(bins 6-7/8-10);小行免排(count-sort 直写已有序)。

## 2. 状态与预期

- 编译干净(**未跑**——refresh10 占 GPU;binary 未含)。
- 预期:pre2 75.6 → ~25ms(count 6 + accumulate 13 + in-place sort ~5);其余 dup<8 中尺寸阵
  compact 税 8-35% 部分消除;dup≥8 阵不受影响(门)。
- 风险:①count pass 对 95k×多 bin 行的固定成本;②MODE=2 大行无序直写的原地排序吞吐;
  ③与 DIRECT5/游标的组合路径(bin 路由不变,正交)。

## 3.5 修复 + A/B 裁决(2026-08-28 上午)

**根因定案**:偶发→实为确定性(×10 全挂)。毒 = `dense_sum_kernel` 在 D5H 上下文中的 launch
确定性报 "invalid configuration argument"(跳过即全链路通、nnz 精确),该 sticky error 被
thrust/cub 内部 getLastError 捞到 → 抛出变形异常(cudaErrorInvalidDevice)= 两晚全部灵异现象。
**修复**:绕过 dense_sum,下溢检查退化为不含 d5h 项(更保守)。launch 本身为何 invalid-config 未明(留案)。

**A/B(D5H=1 修后 vs v11)**:cnnz 全对 ✓ 但性能败退 —— F2 +176%/pre2 +105%/web-Google +76%
(count pass 对全行全价重扫,compact 税省不回来);tsyl201/cant/pwtk/333SP/bcsstk30 ±1-8% 中性;
rajat16 DNF(待查)。**结论:D5H 当前形态否决,默认关**。方向修正:count pass 需像 Ocean symbolic
那样只对"将进 csort 的大行"启用(而非 bins1-10 全行),或走 fingerprint/bitmap 计数(docs/33 A1)。

## 3. 调试战报(2026-08-28 晨,未竟,交接)

- **已修①**:`invalid argument` @ MODE=1 launch = bin7 的 smem 49152B 恰在 48KB 边界且未 opt-in →
  改一次性无条件 attr(legacy 同款)。
- **未解②(偶发)**:`D5H=1` 跑 F2 族,cnnz_scan 的 `thrust::inclusive_scan(device_ptr<int>)` 抛
  `cudaErrorInvalidDevice: invalid device ordinal`(gdb catch throw 实锤抛点;device sync 全绿、
  各阶段事件计时正常)。**偶发**(同配置连跑两次全过);逐段二分:MODE=1 kernels 单开 = 通(42ms),
  zero 段单开曾崩一次后不可复现。怀疑 host 侧 flaky(thrust/cub 惰性初始化 × 某交互),语境烧尽未定位。
  **下一班**:先连跑 D5H=1 ×10 次统计失败率;cuda-gdb `catch throw` + thrust 线索;或把 cnnz_scan
  改 cub::DeviceScan 显式 temp(绕 thrust policy)对照。
- 状态:默认关(二进制 = v11 行为,回归 ±噪声 ✓);kernel/编排/原地 csort/门 全部就位,修好 flaky
  即可 A/B。
