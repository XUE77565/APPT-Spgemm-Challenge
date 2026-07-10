// 测本机 D2H 传输带宽 与 cudaMallocHost 锁页开销
// 编译: nvcc -O2 -o bw_test bw_test.cu
// 运行: ./bw_test 666   (参数 = C 的 MB 数)
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    size_t MB = argc > 1 ? atol(argv[1]) : 666;
    size_t N = MB * 1024 * 1024;
    void *d = nullptr, *h = nullptr, *h2 = nullptr;
    cudaMalloc(&d, N);
    cudaMallocHost(&h, N);
    cudaMemcpy(h, d, N, cudaMemcpyDeviceToHost);   // warmup
    cudaDeviceSynchronize();

    // cudaMallocHost(锁页) 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaMallocHost(&h2, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double mallochost_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // D2H 传输计时(取 5 次最优)
    double best = 1e9;
    for (int i = 0; i < 5; i++) {
        auto a = std::chrono::high_resolution_clock::now();
        cudaMemcpy(h, d, N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        auto b = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(b - a).count());
    }
    double gbs = (double)N / best / 1e6;   // bytes/ms -> GB/s
    printf("size=%4zuMB  cudaMallocHost=%6.1fms  D2H(best of 5)=%6.1fms  = %5.1f GB/s\n",
           MB, mallochost_ms, best, gbs);

    cudaFree(d); cudaFreeHost(h); cudaFreeHost(h2);
    return 0;
}
