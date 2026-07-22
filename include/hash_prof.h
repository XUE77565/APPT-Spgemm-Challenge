#pragma once
// cudaEvent phase profiler(抗 CPU 争用:elapsed 在 GPU stream 上量,不含 host 调度延迟)。
// RAII:函数返回(含 error 路径)时自动打印 TOTAL + 释放 event。DBG-gated;非 DBG 退化为直接执行。
// tag = 输出前缀(如 "hash-prof" / "mrg3-prof"),供 compare_methods 解析区分方法。
// 依赖 spgemm.h 的 dbg()(须在 spgemm.h 之后 include)。
#include <cuda_runtime.h>

#ifdef DBG
struct HashProf {
    const char *tag;
    cudaEvent_t s, e;
    double total = 0.0;
    HashProf(const char *tag_) : tag(tag_) { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~HashProf() {
        dbg("[%s] TOTAL(GPU)      %7.3f ms\n", tag, total);
        cudaEventDestroy(s); cudaEventDestroy(e);
    }
    template <class F> void operator()(const char *name, F &&fn) {
        cudaEventRecord(s);
        fn();
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, s, e);
        dbg("[%s] %-16s %7.3f ms\n", tag, name, (double)ms);
        total += ms;
    }
};
#else
struct HashProf {
    HashProf(const char *) {}
    template <class F> void operator()(const char *, F &&fn) { fn(); }
};
#endif
