#include "mempool.h"

#include <cstdio>
#include <cstdlib>

// 运行时开关,默认 false(原路径)。main 启动时按 USE_MEMPOOL 覆盖。
bool g_use_mempool = false;
// device arena 独立开关(USE_DEV_POOL,默认关)。device pool 省的是 driver 往返
// (wall-clock),但实测让 hash 扫描 phase 在小阵上慢 1.5-3×(compute-only 口径)→
// 对论文 compute-only 对比是负优化,故默认关;需 wall-clock 实验时 USE_DEV_POOL=1 开。
bool g_use_dev_pool = false;

// pinned arena 的全部状态。单线程串行使用(benchmark 一矩阵一进程),无需加锁。
static char*  g_base = nullptr;
static size_t g_cap  = 0;
static size_t g_off  = 0;

// device arena(给 spgemm kernel 的 device 临时 buffer 复用,见 mempool.h 注释)。
//   同 bump-reset 模型:进程启动 cudaMalloc 一大块,dev_pool_reset 把 bump 归零,
//   dev_alloc 仅步进指针(0 driver call)。所有 buffer 调用局部、随 reset 统一回收。
static char*  g_dev     = nullptr;
static size_t g_dev_cap = 0;
static size_t g_dev_off = 0;

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

    // device arena:给 spgemm kernel 的 device 临时 buffer 复用,省每调用 ~13 次
    // cudaMalloc/cudaFree 的 driver 往返(~0.66ms 固定开销)。默认 8GB,MP_DEV_MB 覆盖。
    // ⚠ 仅 USE_DEV_POOL=1 时分配(默认关:对 compute-only 是负优化,见 g_use_dev_pool 注释)。
    if (!g_use_dev_pool) {
        fprintf(stderr, "[mempool] device arena OFF (USE_DEV_POOL unset; compute-only 友好)\n");
        return true;
    }
    const char* ed = std::getenv("MP_DEV_MB");
    size_t dcap = ed ? (size_t)std::atol(ed) * 1024 * 1024
                     : (size_t)16384 * 1024 * 1024;
    cudaError_t errd = cudaMalloc((void**)&g_dev, dcap);
    if (errd != cudaSuccess || !g_dev) {
        fprintf(stderr, "[mempool] cudaMalloc dev arena (%zu MB) failed: %s — "
                "fallback 到 per-call cudaMalloc(性能不变,仅失去 pool 收益)\n",
                dcap >> 20, cudaGetErrorString(errd));
        g_dev = nullptr; g_dev_cap = g_dev_off = 0;
    } else {
        g_dev_cap = dcap; g_dev_off = 0;
        fprintf(stderr, "[mempool] device arena ready: %zu MB\n", g_dev_cap >> 20);
    }
    return true;
}

void mempool_destroy() {
    if (g_base) {
        cudaFreeHost(g_base);  // arena 本身的锁页内存用 cudaFreeHost 释放(合法)
        g_base = nullptr;
    }
    g_cap = g_off = 0;
    if (g_dev) {
        cudaFree(g_dev);
        g_dev = nullptr;
    }
    g_dev_cap = g_dev_off = 0;
}

// ---- device arena API(见 mempool.h)----
void dev_pool_reset() {
    if (g_use_dev_pool) g_dev_off = 0;
}

void* dev_alloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (!g_use_dev_pool) {                  // legacy:原路径 cudaMalloc(A/B 同二进制)
        void* p = nullptr;
        cudaError_t err = cudaMalloc(&p, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[mempool] cudaMalloc(%zu B) failed: %s\n",
                    bytes, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        return p;
    }
    size_t aligned = (g_dev_off + 15) & ~((size_t)15);   // 16B 对齐(double/uint4 友好)
    if (aligned + bytes > g_dev_cap) {
        fprintf(stderr,
                "[mempool] DEV arena overflow: need %zu B, cap %zu MB, used %zu MB "
                "(bump MP_DEV_MB)\n", bytes, g_dev_cap >> 20, g_dev_off >> 20);
        exit(EXIT_FAILURE);
    }
    g_dev_off = aligned + bytes;
    return g_dev + aligned;
}

void dev_free(void* p) {
    if (!g_use_dev_pool) cudaFree(p);       // 池模式:no-op,arena 由 dev_pool_reset 回收
}

size_t dev_pool_used() { return g_dev_off; }

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

void host_free(void* p) {
    if (!p) return;
    if (g_use_mempool) cudaFreeHost(p);   // A_buffer 经 cudaMallocHost 锁页
    else                free(p);           // A_buffer 经 malloc(pageable)
}

size_t mempool_cap() { return g_cap; }
size_t mempool_used() { return g_off; }
