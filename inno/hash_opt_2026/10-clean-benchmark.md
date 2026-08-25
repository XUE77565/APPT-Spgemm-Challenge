# 10 · 干净基准全景 + Step4 巨型图修复(2026-08-26)

> 337 阵全量重跑(8 法,无人竞争时段)→ profiling → 发现并修复整个最大输阵类。
> 数据:`compare/ocean337/methods_cmp_clean.csv`(Ocean 列为干净直测,旧 CSV 作废)。

## 1. 干净全景(修复前)

```
Auto vs Ocean:赢 5 / 输 317,geomean 2.272×
分档:<1万 1.40× | 1-10万 2.35× | 10-100万 2.12× | >100万 2.72×
dispatcher 主动选 merge3:0/337(旧公式在新 kernel 上已完全失效——全押 hash)
Auto vs 其他:cuSPARSE 0.297×(230胜) | opSparse 0.730× | HSMU 1.007× | bhSparse 0.312×(309胜)
```

## 2. 输阵 Top20 的结构:巨型超稀疏图垄断

```
offshore 38.7× | germany_osm 13.1× | road_usa 12.8× | hugebubbles 9-10.7×
hugetrace 10.2-10.7× | road_central 10.5× | hugetric 9.6× | mult_dcop 9.2-9.4×
```
共同画像:**n ∈ 1.2万-2400万,平均度 2-6(超稀疏图),行数巨大**。
Ga/vsp/pkustk 等"目标类"反而 2.19× ≈ 平均水平——不是特殊问题。

## 3. profiling(germany_osm,n=1155万,avg_product=4.9)

| 相位 | ours | Ocean | 备注 |
|---|---|---|---|
| count+est | 2.6 | ~1.5 | flop 免 MinHash ✓ |
| accumulate | **4.5** | ~2.5 | **warp-per-row 批量已对路** |
| **compact+sort** | **62.0** | ~1.3 | **13× 差距全在这** |

## 4. 根因与修复(Step4)

**根因**:超稀疏图的行 99% 落在 ultra bin(est≤16)→ ultra 输出**无序** →
compact 给每行一个 `BlockRadixSort<64,1>` CTA —— **1150 万个 CTA 各排 ≤16 个元素**,
启动开销淹没一切。

**修复**(两行):
1. `hash_ultra_kernel` 尾部加**插入排序**(u_cnt≤32,单线程 ~512 次比较,可忽略)→ 输出有序;
2. compact 的 ultra 分支改 `hash_compact_copy_warp_kernel`(纯 copy,8 行/CTA)。

**实测**:

| 矩阵 | 前 TOTAL | 后 TOTAL | 前比 Ocean | 后比 Ocean(估) |
|---|---|---|---|---|
| germany_osm | 107ms | **49ms**(compact 62→4.9) | 13.1× | ~6× |
| road_central | 93* | 80ms | 10.5× | ~9× |
| hugetric-00010 | 44* | 47ms | 9.7× | ~10× |
| offshore | 74* | **20ms** | 38.7× | **~10×** |
| 333SP | 69 | 58ms | — | — |
| pwtk | 36 | 31ms | — | — |
| TSOPF | 192 | 140ms | — | — |

(*来自 CSV;后为直测。七阵+三图 C_nnz 全精确,0 乱序。)

**教训(补入小白讲解)**:profiling 的"相位级"分解至关重要——accumulate(算法核)已达标,
62ms 藏在"善后"相位里;逐 kernel 的 nsys 是定位这类问题的唯一可靠手段。

## 5. 修复后预计全景

巨型图类从 10-13× 收敛到 ~6-10×(accumulate 4.5 vs 2.5 还有 2× 空间,但已不是 13×)。
geomean 预计从 2.27× 降到 ~1.9-2.0×。待 REFRESH 出精确数字。

## 6. 下一步(按剩余差距排序)

1. **巨型图第二程**:accumulate 4.5→2.5(Ocean 的 spark kernel 每行只派 avg_product 档
   线程数,我们 warp=32 全额派出——改 sub-warp 档)
2. **merge3 测试**:在带状/offshore 类上实测 vs hash(带状阵合成数据上曾 35× 优;
   干净基准 0 阵被选——dispatcher 失效是前提问题)
3. **dispatcher 重建**(依赖 disptrain 数据采完 119/213 + 新特征"列跨度局部性")
