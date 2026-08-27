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

## 3. A/B 预案(refresh10 后)

pre2/web-Google/F2(csort 大户)+ 回归(pwtk/333SP/bcsstk30/Ga3/case39/bloweya)+ c-58;
`D5H=1` vs 0,3 连测中位数;cnnz 对表 + DBG sorted 校验。全量验证走 refresh11。
