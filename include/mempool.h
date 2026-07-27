#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <cstddef>
#include <cuda_runtime.h>

// ============================================================================
// Pinned host memory pool —— 给 d2h(下载 C)路径用的锁页内存池。
//
// 背景(见 suitesparse_crawl/profiling_analysis.md):d2h 的主成本不是 PCIe 传输
// (已到 Gen4×16 峰值 ~24GB/s),而是每次 cudaMallocHost 给 host 内存**锁页**——
// 大矩阵上锁页开销随 C 大小线性增长,且每次调用都重复支付。
//
// 方案:进程启动时一次性锁一大块 pinned arena,之后每次 d2h 从中 bump 切取,
//       锁页只付一次。
//
// 兼容/A-B:运行时开关 g_use_mempool(由环境变量 USE_MEMPOOL 控制)。
//   - 未设/0:走【原路径】cudaMallocHost(行为与引入本模块前完全一致)。
//   - 1:走【池子】。这样可同二进制 A/B 对比 d2h。
//
// 约定:每个 SpGEMM 方法对 host 只分配【一块】C_buffer(d2h 目标),且在调用下一
//       方法前已被 main 消费完。因此 pinned_d2h_alloc 在池模式下先 reset 再 bump,
//       arena 只需 ≥ 单个最大 C_buffer(默认 256MB,可 MP_HOST_MB 覆盖)。
// ============================================================================

// 运行时开关:main 启动时按 USE_MEMPOOL 环境变量置位。
extern bool g_use_mempool;

// 进程启动调一次。cap_bytes==0 → 默认 256MB,可被 MP_HOST_MB(单位 MB)覆盖。
// 池模式关闭时跳过分配(零浪费)。成功返回 true。
bool  mempool_init(size_t cap_bytes = 0);
void  mempool_destroy();

// d2h 专用分配(签名与 cudaMallocHost 一致,可直接塞进 CHECK_CUDA(...)):
//   池模式 :reset arena(上一方法的 C 已被调用者消费)→ bump 切一块 → 返回 cudaSuccess。
//   原模式 :退回 cudaMallocHost,行为不变。
// 池模式下 arena 溢出会打印并 exit(把 MP_HOST_MB 调大即可)。
cudaError_t pinned_d2h_alloc(void** out, size_t bytes);

// 对应释放:
//   池模式 :noop(arena 由下一次 pinned_d2h_alloc 的 reset 统一回收)。
//   原模式 :cudaFreeHost,行为不变。
void  pinned_free(void* p);

// 释放【按 g_use_mempool 方式分配的 host buffer】(A_buffer 用):
//   池模式(A 经 cudaMallocHost 锁页)→ cudaFreeHost;legacy 模式(A 经 malloc)→ free。
void  host_free(void* p);

// ============================================================================
// Device buffer pool —— 给 spgemm kernel 内部那 ~13 个 device 临时 buffer 复用。
//
// 背景(见 inno/engiOpti.md "wall-clock overhead"):每次 hash_product 调用做 ~13 次
// cudaMalloc + ~13 次 cudaFree,驱动往返 ~0.66ms 固定开销(小阵上甚至 > compute)。
// 这些 alloc 全是【调用局部】(一次 self-product 内分配、用完即弃),生命周期互不重叠
// 于下一次调用。故用与 host pinned arena 同构的 bump-reset 模型:进程启动锁一大块
// device arena,每次 hash_product 入口 reset(bump=0),内部 dev_alloc 仅做指针步进
// (0 driver call),调用结束所有 buffer 随下一次 reset 统一回收。
//
// 口径说明:这些 alloc 本就在 cudaEvent prof tag【之外】(prof 只裹 kernel/memset/scan),
// 故 compute-only 指标(TOTAL−h2d−d2h)本就不含它们 —— 池子【不改变 compute-only 数值】,
// 只削 wall-clock/启动开销(小阵竞争力、suite geomean 的真实耗时)。
//
// 同 g_use_mempool 门控(USE_MEMPOOL=1 同时开 host+device arena)。arena 上限默认 8GB,
// MP_DEV_MB 覆盖;溢出 → abort(把 MP_DEV_MB 调大)。
// ============================================================================
void  dev_pool_reset();                 // 新一次调用前 reset(bump=0)
void* dev_alloc(size_t bytes);          // 池:bump 指针步进;legacy:cudaMalloc
void  dev_free(void* p);                // 池:no-op(arena 由 reset 回收);legacy:cudaFree

// 调试/观察用。
size_t mempool_cap();
size_t mempool_used();
size_t dev_pool_used();

#endif
