# pinned 与 arena 原理讲解(结合代码)

> 日期:2026-07-14
> 关联代码:`src/mempool.cu`、`include/mempool.h`、`src/matrix_utils.cu`(A 分配)、`src/main.cu`(开关/生灭)、5 个 d2h 调用点
> 一句话:**pinned 让 GPU DMA 直传(免 staging),arena 让锁页只付一次(免每次锁页)**。两者都把"每次都付的传输成本"变成"只付一次/完全不付",矩阵越大收益越大。

---

## 一、两个概念

### 1. pinned(锁页 / page-locked)

host 内存有两种:

| | 分配方式 | 特点 | GPU DMA 能直传吗 |
|---|---|---|---|
| **pageable** | `malloc` | OS 可把它**换出到磁盘**或**挪动物理页** | ❌ 物理地址会变,DMA 不敢直接访问 |
| **pinned** | `cudaMallocHost` | OS 把物理页**钉住**(不换出、地址固定) | ✅ DMA 直传 |

当 `cudaMemcpy(dA, A, H2D)` 而 A 是 **pageable** 时,GPU driver 不能直接搬,得**偷偷开一块 pinned 中转缓冲**:先把 A 拷进中转,再 DMA 上去。这叫 **staging**(中转),多一次拷贝 + 开销。

> 比方:pageable 像"快递柜"(东西可能被 OS 挪走,快递员得先找柜子确认);pinned 像"固定货架"(DMA 直接来搬)。

### 2. arena(内存池)

关键:`cudaMallocHost(N)` **不只是分配内存**,还得**锁页**——让 OS 把 N 字节钉在物理 RAM、建立 DMA 映射。**锁页有代价,且 ∝ N**(N/4KB 个物理页要一一钉住)。

原代码每次 d2h 都 `cudaMallocHost(C)` + `cudaFreeHost(C)`——**每次下载都重新锁一遍整个 C、用完解锁**。大矩阵 C=107MB 时,一锁一解就是几十 ms,**每次都付**。

arena 思路:**启动时一次性锁一大块**(256MB),之后所有 pinned 分配都从这块**切取**(bump 指针),**不再单独锁页**。**锁页只付一次。**

> 比方:每次 `cudaMallocHost` 像每次寄件都"现申请一个固定货架"(贵);arena 像一次性"租一整面货架墙",之后从上面取格子(几乎免费)。

---

## 二、代码里具体怎么做

开关在 `main.cu:37`:`g_use_mempool` 由 `USE_MEMPOOL`(`spgemm.h` 宏默认 1,环境变量可覆盖)决定。

### A 的输入上传(H2D)—— `matrix_utils.cu:126`

```c
void *buffer = nullptr;
if (g_use_mempool) {
    cudaMallocHost(&buffer, total_size);   // A 锁页 → H2D 直传 DMA
} else {
    buffer = malloc(total_size);           // A pageable → H2D 走 staging
}
```
A 在 `read_matrix_market` 里**只分配一次**(进程内复用)。`g_use_mempool=1` 时它就是 pinned。

### C 的结果下载(D2H)—— 5 个方法各一处,都调 `pinned_d2h_alloc`

`cusparse.cu:202`、`manual.cu:251/402`、`formulations.cu:176`(pack_and_download):
```c
CHECK_CUDA(pinned_d2h_alloc(&C_buffer, C_total_size));   // 拿 host 缓冲接 C
cudaMemcpy(C_buffer, dC_buffer, C_total_size, DeviceToHost);
```

`pinned_d2h_alloc`(`mempool.cu:60`)是分流核心:
```c
cudaError_t pinned_d2h_alloc(void** out, size_t bytes) {
    if (g_use_mempool) {
        g_off = 0;                  // ① reset:上一方法的 C 已被消费,arena 从头复用
        *out = arena_alloc(bytes);  // ② bump:返回 g_base+偏移,只算指针,不锁页
        return cudaSuccess;
    }
    return cudaMallocHost(out, bytes);  // 原路径:每次都锁页
}
```

`arena_alloc`(`mempool.cu:15`)是经典 bump 分配:16B 对齐 → 返回尾指针 → 游标前进。**没有任何锁页动作**(arena 启动时已锁好)。

### arena 生灭 —— `mempool.cu:28` / `main.cu:97,336`

```c
bool mempool_init(size_t cap_bytes) {
    ...
    cudaMallocHost((void**)&g_base, g_cap);   // 启动锁一次 256MB,整个进程持有
}
void mempool_destroy() { cudaFreeHost(g_base); ... }   // 退出解锁一次
```
**全程锁页只付这一次。**

### 释放对齐 —— `mempool.cu:73/77`

```c
void pinned_free(void* p) { if (!g_use_mempool) cudaFreeHost(p); }  // C:池模式 noop(reset 回收)
void host_free(void* p) {                                            // A:池模式 cudaFreeHost(A 是 pinned 分的),legacy 用 free
    if (g_use_mempool) cudaFreeHost(p); else free(p);
}
```
注意 C 和 A 释放语义**相反**:C 池模式是 arena 切的(不单独 free);A 池模式是 `cudaMallocHost` 单独分的(要 `cudaFreeHost`)。故两个 helper。

---

## 三、为什么这样能加速

### A 改 pinned → H2D 免 staging

原来 A pageable,每次 `cudaMemcpy(dA, A, H2D)` 走 driver staging(pageable→中转→DMA)。改 pinned 后 **DMA 直传**。
- 实测 h2d:0.21ms → 0.14ms。
- **顺带**治好 spgemm 上下文里"pageable A 拖慢其它内存操作"的毛病(pageable A 会让 `cudaMallocHost(C)` 都变慢,见 [[h2d_pinned_arena_tradeoff]])。

### C 改 arena → 每次 d2h 免锁页(大头)

原来每次 d2h:`cudaMallocHost(C)`(**每次都给整个 C 锁页**,∝ C 大小)→ D2H → `cudaFreeHost`。改 arena 后:启动锁一次,每次 d2h 只是 `g_off=0` + bump(µs),**不再锁页**。
- 实测 d2h:**2.0ms → 0.16ms(~10×)**。大矩阵(bcsstk30 C=107MB)33ms → ~5ms,省的是反复给 107MB 锁页。
- **矩阵越大、arena 收益越大**:锁页成本 ∝ C 大小。

### 对应到对比图

`compare/pinned-and-arena/` 的图里 10 根柱(5 法 × 有无 arena),**只有 d2h 段(深灰)在"有 arena"时大幅缩短**,其它段(计算/合并/…)几乎不动——因为 pin/arena 只动了**传输**,没碰计算。合计因此 −50~63%。

---

## 四、一句话总结

- **pinned(锁页)** 解决"GPU 能不能 DMA 直传":pin 住 → 免 staging → H2D/D2H 直传。**用在 A(输入)**。
- **arena(内存池)** 解决"锁页太贵能不能少做":启动锁一大块、之后切取复用 → 每次下载 C 不再单独锁页。**用在 C(输出)的 d2h**。
- 两者都把"每次都付的锁页/中转成本"变成"只付一次"或"完全不付"——锁页成本随数据量线性增长,故**矩阵越大、收益越大**。

---

## 附:运行时怎么切

```bash
USE_MEMPOOL=1 ./spgemm_test <mtx>     # pinned A + arena C(默认)
USE_MEMPOOL=0 ./spgemm_test <mtx>     # pageable A + 每次 cudaMallocHost 的 C(原路径)
MP_HOST_MB=512 ...                    # 覆盖 arena 大小(默认 256MB)
USE_MEMPOOL=1 bash scripts/run_aa.sh  # 整批 pool 版
bash scripts/ab_profile.sh            # 一键 A/B + 对比表/图
```

相关:[[h2d_pinned_arena_tradeoff]](arena 对 h2d 的副作用排查)、[[session_changelog_2026-07-13]]、[[profiling_analysis]]。
