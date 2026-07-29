#ifndef MEMPOOL_H
#define MEMPOOL_H

#include <cstddef>
#include <cuda_runtime.h>

// Pinned host memory pool:d2h 路径用的锁页内存池。启动时锁一大块 pinned arena,
// 之后每次 d2h 从中 bump 切取(锁页只付一次)。g_use_mempool 门控 A/B(USE_MEMPOOL)。

// 运行时开关:main 启动时按 USE_MEMPOOL 环境变量置位。
extern bool g_use_mempool;
// device arena 独立开关(USE_DEV_POOL,默认关):对 compute-only 是负优化(拖慢小阵 hash 扫描)。
extern bool g_use_dev_pool;

// 进程启动调一次。cap_bytes==0 → 默认 256MB(MP_HOST_MB 覆盖)。池关闭时跳过分配。成功返回 true。
bool  mempool_init(size_t cap_bytes = 0);
void  mempool_destroy();

// d2h 专用分配(签名同 cudaMallocHost):池模式 reset→bump;原模式退回 cudaMallocHost。溢出 exit。
cudaError_t pinned_d2h_alloc(void** out, size_t bytes);

// 对应释放:池模式 noop(arena 由 reset 回收);原模式 cudaFreeHost。
void  pinned_free(void* p);

// 释放按 g_use_mempool 方式分配的 host buffer(A_buffer 用):池→cudaFreeHost,legacy→free。
void  host_free(void* p);

// Device buffer pool:复用 kernel 内 ~13 个 device buffer(bump-reset),省 ~0.66ms driver 往返,
// 只削 wall-clock(不改 compute-only)。USE_MEMPOOL=1 门控,默认 8GB(MP_DEV_MB)。
void  dev_pool_reset();                 // 新一次调用前 reset(bump=0)
void* dev_alloc(size_t bytes);          // 池:bump 指针步进;legacy:cudaMalloc
void  dev_free(void* p);                // 池:no-op(arena 由 reset 回收);legacy:cudaFree

// 调试/观察用。
size_t mempool_cap();
size_t mempool_used();
size_t dev_pool_used();

#endif
