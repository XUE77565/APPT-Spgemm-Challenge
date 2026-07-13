#include "mempool.h"

#include <cstdio>
#include <cstdlib>

// 运行时开关,默认 false(原路径)。main 启动时按 USE_MEMPOOL 覆盖。
bool g_use_mempool = false;

// pinned arena 的全部状态。单线程串行使用(benchmark 一矩阵一进程),无需加锁。
static char*  g_base = nullptr;
static size_t g_cap  = 0;
static size_t g_off  = 0;

// ---- 内部:从 arena bump 切一块(16 字节对齐)。调用前需保证已 reset 到合适起点 ----
static void* arena_alloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    size_t aligned = (g_off + 15) & ~((size_t)15);
    if (aligned + bytes > g_cap) {
        fprintf(stderr,
                "[mempool] HOST arena overflow: need %zu B, cap %zu B, used %zu B "
                "(bump MP_HOST_MB)\n", bytes, g_cap, g_off);
        exit(EXIT_FAILURE);
    }
    g_off = aligned + bytes;
    return g_base + aligned;
}

bool mempool_init(size_t cap_bytes) {
    if (!g_use_mempool) {
        fprintf(stderr, "[mempool] OFF — legacy cudaMallocHost (USE_MEMPOOL unset/0)\n");
        return true;  // 原路径:不分配 arena,零开销
    }
    if (cap_bytes == 0) {
        const char* e = std::getenv("MP_HOST_MB");
        cap_bytes = e ? (size_t)std::atol(e) * 1024 * 1024
                      : (size_t)256 * 1024 * 1024;  // 默认 256MB
    }
    g_cap = cap_bytes;
    cudaError_t err = cudaMallocHost((void**)&g_base, g_cap);
    if (err != cudaSuccess || !g_base) {
        fprintf(stderr, "[mempool] cudaMallocHost(%zu MB) failed: %s\n",
                g_cap >> 20, cudaGetErrorString(err));
        g_base = nullptr;
        g_cap = 0;
        return false;
    }
    fprintf(stderr, "[mempool] host pinned arena ready: %zu MB (USE_MEMPOOL=ON)\n",
            g_cap >> 20);
    return true;
}

void mempool_destroy() {
    if (g_base) {
        cudaFreeHost(g_base);  // arena 本身的锁页内存用 cudaFreeHost 释放(合法)
        g_base = nullptr;
    }
    g_cap = g_off = 0;
}

cudaError_t pinned_d2h_alloc(void** out, size_t bytes) {
    if (g_use_mempool) {
        if (!g_base) {
            fprintf(stderr, "[mempool] pool used before mempool_init\n");
            return cudaErrorInitializationError;
        }
        g_off = 0;            // reset:上一方法的 C 已被调用者消费,arena 从头复用
        *out = arena_alloc(bytes);
        return cudaSuccess;
    }
    return cudaMallocHost(out, bytes);  // 原路径
}

void pinned_free(void* p) {
    if (!g_use_mempool) cudaFreeHost(p);  // 池模式:noop,arena 由 reset 回收
}

size_t mempool_cap() { return g_cap; }
size_t mempool_used() { return g_off; }
