// mh_merge_kernel 微基准:隔离 grid 尺寸 vs 时间,定位 1.6ms 固定开销来源
#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>


__global__ void mh_merge_kernel(
    const int *A_row_ptr, const int *A_col_ind, int A_rows,
    const unsigned int *b_mh, const int *row_flop, double expand,
    int *est_nnz, int sample_stride = 0);

// 用与原 kernel 相同的实现(直接 include 源文件太重;此处复制主体)
#define MH_M 128
#define MH_EMPTY 0xFFFFFFFFu
#define WARP_SIZE 32
#define EST_ULTRA_THR 16
#define GLOBAL_HT_MAX_SLOTS 131072
__global__ void mh_merge_bench(
    const int *A_row_ptr, const int *A_col_ind,
    int A_rows,
    const unsigned int *b_mh,           // [B_rows * MH_M] uint32 from Phase 1
    const int *row_flop,                // [A_rows] 每行精确乘积数(精确上界,封顶 MinHash 高估)
    double expand,                      // EST_EXPAND(运行时:小阵 1.4 免重试 / 大阵 1.15 省内存)
    int *est_nnz,                       // [A_rows] output
    int sample_stride = 0)              // MHSAMP:>1 时紧凑采样 grid(S 块,块→行 = bid×stride;实测
                                        // 28k 空块早退仍收全价 = dispatch/延迟限制,mod-skip 无效)
{
    int row = (sample_stride > 1) ? (int)blockIdx.x * sample_stride : (int)blockIdx.x;
    if (row >= A_rows) return;
    int tid = threadIdx.x;

    extern __shared__ int smem_merge_raw[];
    unsigned int *smem_merge = (unsigned int*)smem_merge_raw;   // [MH_M] uint32
    int base_i = tid * 4;                                       // 每线程 owning partitions [base_i..base_i+3]
    smem_merge[base_i + 0] = MH_EMPTY;
    smem_merge[base_i + 1] = MH_EMPTY;
    smem_merge[base_i + 2] = MH_EMPTY;
    smem_merge[base_i + 3] = MH_EMPTY;
    __syncthreads();

    int start_elem = A_row_ptr[row];
    int end_elem = A_row_ptr[row + 1];
    // 逐引用 B 行,vectorized uint4 min-merge(每线程固定 owning 4 个 partition → 无 race、无需内 sync)
    for (int e = start_elem; e < end_elem; e++) {
        int row_b = A_col_ind[e];
        const uint4 *bsk4 = (const uint4*)(b_mh + (size_t)row_b * MH_M);   // 行首 16B 对齐(512B/行)
        uint4 v = bsk4[tid];                                               // 16B coalesced
        unsigned int *sm = smem_merge + base_i;
        if (v.x < sm[0]) sm[0] = v.x;
        if (v.y < sm[1]) sm[1] = v.y;
        if (v.z < sm[2]) sm[2] = v.z;
        if (v.w < sm[3]) sm[3] = v.w;
    }
    __syncthreads();

    // 估计 reduce(单 warp):sum_min = Σ min_j(非空);V = 非空 partition 数
    double sum_min = 0.0;
    int V = 0;
    for (int k = 0; k < 4; k++) {
        unsigned int mv = smem_merge[base_i + k];
        if (mv != MH_EMPTY) { sum_min += (double)mv; V++; }
    }
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum_min += __shfl_down_sync(0xFFFFFFFF, sum_min, off);
        V += __shfl_down_sync(0xFFFFFFFF, V, off);
    }

    if (tid == 0) {
        double E;
        int v = V;
        if (v == 0) {
            E = 0.0;                                                      // 空行
        } else if (v < MH_M) {
            E = (double)MH_M * log((double)MH_M / (double)(MH_M - v));   // linear counting(小范围)
        } else {
            // min_j/2^32 ~ Exp(1/(D/m)) ⇒ E[min_j] = m·2^32/D ⇒ D ≈ m²·2^32/Σmin_j(算术均值,无偏)。
            // 旧式 2^32·Σ(1/min_j) 是调和均值,Jensen 高估 ~ln 倍(exdata_1 17.85×/TSOPF 20× 根因);
            // 首版修正漏了因子 m(低 EST 128 倍,exdata_1 est 254k vs C 11.3M)——都已修(2026-08-26)。
            E = (double)MH_M * MH_M * 0x100000000LL / sum_min;
        }
        int temp = (int)(E * expand);
        if (temp < 1) temp = 1;
        int est;
        if (temp <= EST_ULTRA_THR) {                                      // ultra:线性 kernel,est 保留紧 temp
            est = temp;
        } else {                                                          // snap 到 next_pow2 × 2 余量
            int ht = 32;                                                  // (2× 余量治小/中行低估表满;>HASH_CAP 交 heavy)
            while (ht < temp && ht < GLOBAL_HT_MAX_SLOTS) ht <<= 1;
            est = ht;
        }
        // flop = 每行乘积数 = distinct 的【精确上界】→ 封顶 MinHash 高估(TSOPF:Σest 从 147亿 → ≤Σflop,
        // 兼治 int 前缀和溢出)。min() 不引入新低估:flop ≥ distinct 恒真,低估风险仅剩原 2×snap 自身。
        int fl = row_flop[row];
        if (fl > 0 && est > fl) est = fl;
        est_nnz[row] = est;
    }
}

// MHSAMP:dense-注定阵的 est 门专用填充(min(flop,n) 上界;布局消费者不存在 —— d_off 随 est
// 置零重扫,tmp 只按坍缩后 hash 侧定容,total_est 由投影值在 binning 后覆写)
__global__ void est_fill_kernel(const int *row_flop, int n, int *est, int A_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < A_rows) { int f = row_flop[i]; est[i] = f < n ? f : n; }
}

// localLoadBalance 移植(docs/27 §4.2,Ocean AccumulatorCommon.cuh:67 同款语义):
// 按行 (a_len, flop, max_b_len) 动态选 2^log_nthr 线程/k —— 起步均值【排除最长 B 行】(它是
// straggler,由后续双向夹逼吸收),再按 max_sub_iter vs num_iters 的 2× 失衡双向调 G。
// 我们无 warp 内在依赖,G 上限 = HASH_BLOCK(整 block 伺候一个 k)。返回 log2(G)。

int main(int argc, char **argv) {
    int A_rows = 28216, a_len = 13;
    if (argc > 1) A_rows = atoi(argv[1]);
    if (argc > 2) a_len = atoi(argv[2]);
    // CSR: 每行 a_len 个随机列
    int nnzA = A_rows * a_len;
    int *h_rp = new int[A_rows + 1], *h_ci = new int[nnzA];
    for (int i = 0; i <= A_rows; i++) h_rp[i] = i * a_len;
    for (int i = 0; i < nnzA; i++) h_ci[i] = (i * 2654435761u) % A_rows;
    int *d_rp, *d_ci, *d_flop, *d_est;
    unsigned int *d_mh;
    cudaMalloc(&d_rp, (A_rows + 1) * 4); cudaMalloc(&d_ci, (size_t)nnzA * 4);
    cudaMalloc(&d_mh, (size_t)A_rows * MH_M * 4);
    cudaMalloc(&d_flop, A_rows * 4); cudaMalloc(&d_est, A_rows * 4);
    cudaMemcpy(d_rp, h_rp, (A_rows + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci, h_ci, (size_t)nnzA * 4, cudaMemcpyHostToDevice);
    {   // 随机 sketch 内容(真实哈希语义:min 更新会真实发生)
        unsigned int *h_mh = new unsigned int[(size_t)A_rows * MH_M];
        srand(42);
        for (size_t i = 0; i < (size_t)A_rows * MH_M; i++) h_mh[i] = ((unsigned)rand() << 16) ^ (unsigned)rand();
        cudaMemcpy(d_mh, h_mh, (size_t)A_rows * MH_M * 4, cudaMemcpyHostToDevice);
        delete[] h_mh;
    }
    cudaMemset(d_flop, 0x7f, A_rows * 4);
    int smem = MH_M * 4;
    for (int rep = 0; rep < 3; rep++) {
        for (int stride : {0, 1, 4, 14}) {
            int grid = stride > 1 ? (A_rows + stride - 1) / stride : A_rows;
            cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
            cudaEventRecord(s);
            mh_merge_bench<<<grid, 32, smem>>>(d_rp, d_ci, A_rows, d_mh, d_flop, 1.15, d_est, stride);
            cudaEventRecord(e); cudaEventSynchronize(e);
            float ms; cudaEventElapsedTime(&ms, s, e);
            printf("rep%d A_rows=%d a_len=%d stride=%2d grid=%6d : %8.3f ms\n", rep, A_rows, a_len, stride, grid, ms);
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
    }
    return 0;
}
