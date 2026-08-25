# 01 · 安全根治:64 位化 + 硬上界 + 分配前 sanity

> commit `01be5b5`(merge3)+ est 64 位化(`7dad171` 内)。三次 GPU wedge(7-27、7-30、8-25)的教训代码化。

## 教训的因果链(每次 wedge 都同构)

```
病态输入(fp/Σflop/Σest 超 int)→ 计数变负 → (size_t)负数 = 天文数字
  → cudaMalloc 巨额分配 / kernel 超长循环 → 外层 420s 超时 SIGKILL
  → 杀在 CUDA context teardown → 驱动 D-state → GPU requires reset(无 sudo 不可逆)
```

## 三层防御(全部落地)

1. **计数 64 位**:merge3 的 `total_flop/d_row_off` 全程 long long(bucket_scan 模板化
   `<long long>`,kernel 索引 base/src ll);hash 的 `d_off/total_est` 同样(Ga3As3H12
   真实 Σflop=36.8 亿、TSOPF Σest 曾 147 亿——都超 int)。
2. **循环硬上界**:warp-merge 主循环 `for (guard < MRG3_LOOP_CAP)`(照抄 att_tiered 的
   ATT_LOOP_CAP 手法)——不变量破坏时输出错而不是挂死,kernel 必然终止。
3. **分配前 sanity**:`total_flop ≤ 0 || > 3e9`(MRG3_MAX_ENTRIES 可调)干净退出;
   heavy 表区 arena 超 24GB(GHT_MAX_BYTES)按 overflow 回退;所有 extract 写按
   est 槽位守卫(rank/pos ≥ cap → 记录 + 跳过,绝不越界砸下一行)。

## 纪律(写进 gpu_check.sh 与跑批流程)

- 任何跑批/计时前 `bash scripts/gpu_check.sh [id]`(nvidia-smi 状态 + 1-thread kernel+D2H 冒烟);
- 采集脚本熔断:fp>15 亿或 maxrow>10 万 SKIP;adaptive 行缺失不盲跑;
- 长跑 tmux + 增量 CSV;单阵超 2×timeout 人工介入,不等超时杀(SIGKILL 在 CUDA teardown
  上就是 wedge 机制)。
