// Trivial GPU health precheck (run AFTER nvidia-smi --gpu-reset).
// Exercises device enum, malloc, 1-thread kernel, D2H. Must finish <1s on a healthy GPU;
// if it hangs/errors the GPU is not healthy — do not launch real workloads.
#include <cstdio>
#include <cstdlib>

__global__ void k_one(int *p) { if (threadIdx.x == 0 && blockIdx.x == 0) *p = 0x12345678; }

int main() {
    int nd = -1;
    cudaError_t e = cudaGetDeviceCount(&nd);
    if (e != cudaSuccess || nd <= 0) {
        printf("PRECHECK FAIL: cudaGetDeviceCount err=%d nd=%d : %s\n", e, nd, cudaGetErrorString(e));
        return 2;
    }
    for (int i = 0; i < nd; ++i) {
        cudaSetDevice(i);
        int *d = nullptr;
        e = cudaMalloc(&d, sizeof(int));
        if (e != cudaSuccess) { printf("PRECHECK FAIL: dev %d malloc: %s\n", i, cudaGetErrorString(e)); return 3; }
        k_one<<<1,1>>>(d);
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("PRECHECK FAIL: dev %d sync: %s\n", i, cudaGetErrorString(e)); return 4; }
        int h = 0;
        e = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d);
        if (e != cudaSuccess) { printf("PRECHECK FAIL: dev %d d2h: %s\n", i, cudaGetErrorString(e)); return 5; }
        if (h != 0x12345678) { printf("PRECHECK FAIL: dev %d wrong value 0x%x\n", i, h); return 6; }
        printf("PRECHECK OK: dev %d (1-thread kernel + D2H verified)\n", i);
    }
    printf("PRECHECK ALL OK: %d device(s) healthy.\n", nd);
    return 0;
}
