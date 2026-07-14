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

// 调试/观察用。
size_t mempool_cap();
size_t mempool_used();

#endif
