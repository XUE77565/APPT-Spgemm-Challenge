# 08 · Heavy 混合路径(keys-in-SMEM):建对了,打错了(2026-08-26 Step 2)

> 结论:**hybrid kernel 正确落地但没带来收益** —— 因为 estimator 修复后 heavy 行已几乎消失,
> TSOPF 只剩 10 个 heavy 行。真正的 3.63× 差距在中尺寸 SMEM bin 的 76× dup 行。
> 又一次有价值的负结果:瓶颈定位必须跟着每次改动重新做。

## 做了什么

1. `hash_global_hybrid_kernel`:keys(int × ht)进 SMEM 动态共享(ht ≤ 32768 → ≤128KB),
   values 留全局 arena —— SMEM atomicCAS 比全局快 4-8×(对标 Ocean HYBRID_HASHMAP)。
2. 按 ht 拆批:heavy 行 copy_if 分 hybrid(≤32k)/big(>32k)两个行表分别 launch
   (统一 smem=max 的首版被单个 4M 巨型行顶爆 —— 混合规模必须拆批)。
3. 副产物:`heavy_gather_kernel`(子表 gather ht/tab_off 并原地把表内位置换成真实行号)。

## 为什么没用(瓶颈迁移)

```
estimator 修复前:TSOPF est 虚高 20× → 14030 行误入 heavy 全局表(82GB arena 熔断)
estimator 修复后:est 1.16× over → heavy 只剩 10 行(0.01GB)
现在的 57ms accumulate ∈ 中尺寸 SMEM bin(ht 8k-16k,1CTA/行,76× dup 行)
```

## 附加实验(均否决)

- 大表 G 组 32→8(TSOPF):286ms vs 190ms —— 更差;G 不是杠杆;
- 顺带发现:bins 8-9 的 96-192KB 表 → 1-2 CTA/SM,占用率才是嫌疑主犯。

## TSOPF 的正解(Step 3 设计,未实施)

TSOPF 输出 25% 稠密(C=199M / 28216²)—— **dense 累积器的理想候选,只差 n=28216 > SMEM 预算(14980)**。
两个方案:
- **A. 全局 dense 表**:vals[rows×n] doubles = 6.4GB(H100 可容),每 (行,片) CTA 直接寻址
  global atomicAdd(免 CAS 免探测);25% 稠密下 hash 的 75% 空槽探测全是浪费,dense 完胜;
- B. dense v2 行×列分片(07 文档撤回的两阶段版):片计数同步用两 kernel 解。
推荐 A(实现直、与 dense v1 共用 extract 骨架)。预期 TSOPF 3.63× → ~1.5×。
