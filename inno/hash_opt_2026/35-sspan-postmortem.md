# 35 · 小跨度 dense 首试尸检 + Ocean 侦察(2026-08-28 午,用户病休自主班)

## 1. Ocean 路由侦察(5 阵直测 stats.json,本档核心资产)

| 阵 | Ocean bins(hash1-6\|dense7-13) | 对我们的含义 |
|---|---|---|
| **nemeth18** | **全部 9506 行 → dense bin0(span≤236)** | 微跨度全族直接寻址,无 hash 无 MinHash 表 |
| cant | 60711 行 hash bin1-2 + 1737 行 dense bin0/2/4 | 大头仍 hash(与我们同构)|
| tsyl201 | 19869 hash + 816 dense bin2-3 | 同上 |
| oilpan | 72660 hash bin1 + 371 dense | 同上 |
| **TSOPF_FS_b300_c2** | 28.7k hash + **28082 行 dense bin6(iter)** | Ocean 确实用游标 iter 内核;我们输在**全局镜像的 O(a_len×窗数) 往返**,Ocean 的 start_map 在 **SMEM** |

## 2. 首试(单一大 SSPAN bin,1CTA/行,26KB 定长 SMEM)尸检

- 路由门:span≤2048 ∧ est≥span/2 → 全族进 bin。
- **败因①(性能)**:nemeth 族 +164~232% —— 9506 个 256 线程/26KB CTA 伺候 span≤236 的微行 =
  **1CTA/行 的第 N 次踩坑**(与 est=flop 门、Phase B v1 同族教训)。Ocean 的 dense bin0 是
  **64 线程小块 + 4KB SMEM + 32 块/SM** —— 密度是我们的 ~10×。
- **败因②(正确性)**:部分 launch 真失败("invalid argument" sticky → thrust 变形异常 → cnnz 错/tsyl201
  cnnz 7897689≠8321013)。查因未竟(26624B<48KB 本应合法;与 D5H 的 dense_sum 同款未解之谜,见 §4)。
- 已整体回退(git checkout),基线行为恢复 ✓。

## 3. 正确设计(下一班,照抄 Ocean 梯度)

1. **span 分桶子 bin**(非单一大 bin):DENSE_NUMERIC_BIN_SIZES 梯 {256,512,1024,2048} ×4 个子 bin;
   每子 bin **定长 SMEM = span×13B** + **小块(64/128 线程)** + 高占用(32+ 行/SM)。
2. est≥span/2 门保留(数组利用率);est<span/2 回 hash。
3. 修 launch 之谜:先用最小样例复现(可能需 cudaFuncSetAttribute 或 block/SMEM 校验)。
4. TSOPF 修法(正交):direct 内核的 start_map 搬进 SMEM(a_len≤~7000 时,与 dpref 区域分立),
   全局镜像只留大 a_len 回退 —— 消 O(a_len×窗数) 往返。

## 4. "invalid argument" sticky 之谜(两案未破,交接)

dense_sum_kernel(D5H 上下文)与 hash_sspan_kernel(本试)都在**特定管线位置**的 launch 报
invalid (configuration) argument,而参数表面合法;sticky error 被 thrust/cub 捞到后抛成
cudaErrorInvalidDevice = 灵异表象。**cnnz 前一次 cudaGetLastError() 清除可恢复运行**(本次实证),
但真失败的 launch = 那些行没跑 = 结果错。下一班:最小复现 + 逐 launch 二分(带 stream 维度,
两案都在 bin-stream 上或其后)。

## 5. 本班结论

- 近赢带(cant/tsyl201/oilpan)大头 = hash 路径(与我们同构)→ 差距在管理相位+探测,不在路由;
  **只有 nemeth 族(全微跨度)吃 span-dense 的肉**。
- 基线不变 1.8888×(v11);SSPAN 回退;侦察数据 + 正确设计已存本档。
