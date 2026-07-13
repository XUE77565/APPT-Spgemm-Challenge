# h2d / d2h 传输优化方案

> 适用:`spgemm-challenge` 里 5 种 SpGEMM 公式(cuSPARSE / Gustavson / 外积 / 列向 / 内积)的结果回传与输入上传。
> 结论先行:**传输本身已到 PCIe 峰值(~24GB/s),优化空间在 `cudaMallocHost` 的锁页开销,以及"不需要时干脆不传"。**

---

## 1. 现状(profiling 实测)

分阶段打桩测得各方法在大矩阵上的 d2h(下载结果 C 到 host):

| 矩阵 | C.nnz | C 大小 | d2h 实测 |
|---|--:|--:|--:|
| vsp_bcsstk30 | 18.2M | ~146 MB | ~74 ms |
| email-Enron | 30.5M | ~244 MB | ~125 ms |
| **hangGlider_3** | **83.2M** | **~666 MB** | **~388 ms** |

- **h2d**(上传 A):A 很小(最大 vsp ~32MB),耗时 ~1–2ms,**不是问题**。
- **d2h**(下载 C):随 C 大小主导总耗时(大输出时占 80%+),**是头号瓶颈**。

---

## 2. 根因:慢在"锁页",不在"传输"

用 `/tmp/bw_test.cu` 实测本机(H100 PCIe)原始 D2H 带宽与 `cudaMallocHost` 开销:

| C 大小 | `cudaMallocHost`(锁页) | D2H 传输本身 | 传输带宽 |
|--:|--:|--:|--:|
| 10 MB | 5.9 ms | 0.9 ms | 11.7 GB/s |
| 100 MB | 43 ms | 4.5 ms | 23.6 GB/s |
| 333 MB | 158 ms | 14.4 ms | 24.2 GB/s |
| **666 MB** | **314 ms** | **30 ms** | **23.3 GB/s** |

**关键结论:**
- **D2H 传输本身只有 ~30ms(~24GB/s),已到 PCIe Gen4 ×16 峰值,无法再快**(除非换 Gen5/更宽槽位,那是硬件)。
- **`cudaMallocHost`(给 host 内存锁页)才是大头**:666MB 要 **314ms**,是传输的 **10 倍**,且**随大小线性增长(~0.47ms/MB)**。
- 所以 d2h ≈ `cudaMallocHost`(锁页) + `cudaMemcpy`(传输) + 打包;hangGlider 的 388ms ≈ 锁页 314 + 传输 30 + 打包。
- 现在每次调用都 `cudaMallocHost` + `cudaFreeHost`(反复锁页/解锁),**锁页成本被重复支付**。
- 另:`WRITE_MTX=0` 时 C 被下载后**立即 `cudaFreeHost` 释放、未使用** → 这 ~388ms **纯属白费**。

---

## 3. 优化方案

### 方案 A:pinned 内存池(锁页一次,复用)★★★ 最通用

**原理**:进程启动时锁一块足够大的 pinned arena,之后所有 d2h 从中切取/归还,**锁页只付一次**。

**适用**:需要 C 在 host(`WRITE_MTX=1`)的生产场景。

**收益**:d2h 从 ~388ms → **~30ms(纯传输)**。

**实现要点**(伪代码):
```cpp
// 全局池:启动时锁一大块(覆盖最大 C,比如 2GB)
static void *g_pin_pool = nullptr;
static size_t g_pin_used = 0, g_pin_cap = 0;
void pin_pool_init(size_t cap){ g_pin_cap = cap; cudaMallocHost(&g_pin_pool, cap); }
void* pin_pool_alloc(size_t bytes){ /* 简单 bump: 返回 g_pin_pool+used, used+=bytes; 满了就重置 */ }

// pack_and_download 里:
// 原:cudaMallocHost(&hb, total);   ← 每次 314ms (666MB)
// 改:hb = pin_pool_alloc(total);    ← 几乎 0ms (已锁页)
//   cudaMemcpy(hb, db, total, D2H); ← ~30ms
// 注意:不再 cudaFreeHost(hb),由池统一管理
```
- 或用 **CUDA 11.2+ 的 `cudaMallocAsync` + 内存池**(设 `cudaMemPoolSetAttribute(...,POOL_USE_CUDA_HOST_MEM)`),让驱动托管 pinned 池,更省心。
- 池大小要覆盖最大 C(hangGlider ~666MB,留余量 ~1–2GB)。
- cuSPARSE 的 host(`spgemm_self_product`)里那个 `cudaMallocHost(C_buffer)` 同样改成从池取。

> 顺带:`pack_and_download` 现在是 "3 次 D2D 拼块 + 1 次 D2H",可改成 **"3 次直接 D2H 到 host buffer 的对应偏移"**,省掉 3 次 D2D(pack 那 ~0.2ms 不大,但少一次 device 间搬运)。

---

### 方案 B:跳过 d2h(结果留 device)★★★ 见效最快

**原理**:加 `KEEP_ON_DEVICE` 开关,结果 C **不回传 host**,函数返回 device 指针(或只把 `nnz` 回传)。

**适用**:
- benchmark / 纯性能测量(`WRITE_MTX=0` 时 C 本来就下载完即丢);
- 后续计算还在 GPU 上(链式 `C = (A·A)·A`);
- 只需要验证 nnz(不验证数值)。

**收益**:d2h **直接归零**。hangGlider 总耗时从 ~490ms → **~100ms**。

**实现要点**:
```cpp
void spgemm_self_product_manual(..., void **C_buffer_out, ...){
    ...
    // 原:
    //   *C_buffer_out = pack_and_download(...);   // D2D+D2H+锁页
    // 改(#if KEEP_ON_DEVICE):
    //   *C_buffer_out = dC_buffer;   // 直接返回 device 指针, 调用者负责 cudaFree
    //   (跳过 pack_and_download 整段)
}
```
- 调用方(`main.cu`)相应改成:不 `cudaFreeHost` 而是 `cudaFree`;写盘时用 D2H 单独传(或干脆不写)。
- 注意 API 语义变化(返回 device 而非 host 指针),要在头文件/文档标明。

---

### 方案 C:异步传输 + 计算/传输重叠 ★ 进阶

**原理**:`cudaMemcpyAsync` + 独立 stream,让传输与计算重叠。

**适用**:
- **批量多矩阵**:矩阵 i 的 d2h 与矩阵 i+1 的计算重叠(流水线)。
- **单矩阵内分块**:把 C 分块,算完一块就 async 传一块(需 SpGEMM 支持分块输出)。

**限制**:
- 单矩阵、单次 SpGEMM 时,计算依赖输入、d2h 依赖计算完成,**重叠空间小**。
- 对本 benchmark(逐矩阵串行)主要价值在"矩阵间流水线"。

**实现要点**:
```cpp
cudaStream_t s_compute, s_xfer;
// 矩阵 i:在 s_compute 上算;算完 cudaMallocAsync + cudaMemcpyAsync(D2H, s_xfer)
// 矩阵 i+1 的 h2d 可与矩阵 i 的 d2h 重叠(两条 stream)
```
- 收益取决于矩阵间负载是否均衡;需测量是否真有重叠(用 nsys 看时间线)。

---

### 方案 D:h2d 侧(基本无需优化)

- A 很小(最大 vsp ~32MB),h2d ~1–2ms。
- A 已是 pinned(`read_matrix_market` 用 `cudaMallocHost`),H2D 已接近峰值。
- 若 `cudaMalloc(dA)` 设备分配有抖动(vsp h2d 偶见 17–30ms),可用 **device 内存池**(`cudaMallocAsync` 池)消除分配抖动。
- **优先级低**。

---

## 4. 传输本身(不可软件优化)

- D2H/H2D 已达 ~24GB/s(Gen4 ×16 峰值),**软件层面到顶**。
- 要更快只能靠硬件:PCIe Gen5(翻倍)、NVLink(若 GPU 间)、或减少需传输的数据量(稀疏压缩 / 只传需要部分)。

---

## 5. 实施优先级与预期收益(hangGlider,Gustavson)

| 方案 | d2h | 总耗时 | 难度 | 何时用 |
|---|--:|--:|:--:|---|
| 现状 | ~388ms | ~490ms | – | – |
| **B 跳过 d2h** | **0** | **~100ms** | 低 | benchmark / WRITE_MTX=0 |
| **A pinned 池** | **~30ms** | **~130ms** | 中 | 生产 / WRITE_MTX=1 |
| C 异步重叠 | 视情况 | 视情况 | 高 | 批量流水线 |
| D 传输本身 | – | – | 不可能 | 需硬件升级 |

**建议路线:**
1. **先做 B**(给 benchmark 加 `KEEP_ON_DEVICE` 开关)——几行代码,d2h 归零,立刻看到"纯 GPU 计算"的真实差距。
2. **再做 A**(pinned 池)——让生产路径(WRITE_MTX=1)也只付一次锁页。
3. C/D 按需。

---

## 6. 注意事项 / 坑

- **pinned 池大小**:必须 ≥ 最大 C(hangGlider 666MB),否则要么动态扩容(再锁页)要么溢出。建议启动时按"全集合最大 C"预锁。
- **`cudaMallocHost` vs `cudaHostRegister`**:`cudaMallocHost` 自己分配+锁;`cudaHostRegister` 锁已分配的普通内存。池两种都行,`cudaMallocHost` 一大块更简单。
- **WRITE_MTX=0 当前的浪费**:现状下 C 下载后即 `cudaFreeHost` 未用——这是"隐性 bug 级"的浪费,方案 B 直接消除。
- **多次测量取稳态**:`cudaMallocHost` 有抖动(见实测 314ms 是均值级),优化前后都用多次/中位数对比。
- **cuSPARSE 与手写法都要改**:d2h/锁页在 5 种方法的 host 里都有(cuSPARSE 的 `spgemm_self_product`、手写法的 `pack_and_download`),改池要统一覆盖。

---

## 附:实测脚本
带宽/锁页开销复现(`suitesparse_crawl/bw_test.cu`):
```bash
cd suitesparse_crawl
nvcc -O2 -o bw_test bw_test.cu
for mb in 10 100 333 666; do ./bw_test $mb; done
```
分阶段 d2h 数据:`suitesparse_crawl/profile_aa.csv`(列 `d2h`,按 `tag`/`name` 过滤)。
