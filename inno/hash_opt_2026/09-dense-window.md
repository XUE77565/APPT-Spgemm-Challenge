# 09 · Dense 窗口路径(Step 3):TSOPF 类 3.63× → 1.72×

> 对标 Ocean `denseNumericIterKernel`:n 超 SMEM 预算的稠密输出矩阵,按列窗口迭代。

## 机制

```
门:n ≤ 200000 且 输出占比 ≥ 15%(TSOPF:28216²、29% 稠密 ✓;n ≤ 14980 仍走 dense v1)
每行一 CTA(512 线程):
  for 窗口 wi ∈ [0, ceil(n/14980)):
    [c0,c1) = 本窗口列域;SMEM 表 vals+flags+pref 复用 dense v1(190KB)
    每 k:dev_lower_bound×2 切出 B[k]∩[c0,c1)(免窗口外处理,读放大 ~0)
    累加:atomicAdd(dval[j]) + dflag 幂等写(同 v1)
    提取:窗口内 warp 前缀有序 → tmp[base + out + rank]
    out += 窗口计数
窗口升序 ⇒ 拼接天然全局有序;零跨 CTA 同步(07 撤回的 v2 缺陷就此绕开)
```

## 实测(TSOPF_FS_b39_c7)

| | 前 | 后 |
|---|---|---|
| accumulate | 57ms(中尺寸 SMEM bin × 76dup) | **31.3ms** |
| compute-only | 75.8 | **~36ms** |
| vs Ocean(20.9) | 3.63× | **1.72×** |

其余五阵零回归(exdata 走 v1/333SP 批量/bcsstk30/3Dspec2/pwtk 不变),C_nnz 全精确 + 0 乱序。

## 设计注记

- 每行 2 窗口(TSOPF)→ CTA 28216 个,1 CTA/SM × 190KB,占用率与 dense v1 相同受 SMEM 限,
  但行数多波次充分;31ms 对 2e9 积 = 64 G/s,与 dense v1 的 exdata 吞吐一致;
- 剩余 1.72×:d2h 112ms 在 compute 外;31ms 内 ~64 G/s 已接近 SMEM atomicAdd 直接寻址的
  实测上限;Ocean 18.8ms(106 G/s)的余量疑在其 denseNumericIter 的窗口宽度/线程形状调优。
