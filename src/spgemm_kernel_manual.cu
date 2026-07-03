#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ float row_dot_product(
    const int *col_idx, const float *val,
    int start_i, int end_i,
    int start_j, int end_j)
{
    float dot = 0.0f;
    int pi = start_i, pj = start_j;
    
    while (pi < end_i && pj < end_j) {
        int ci = col_idx[pi];
        int cj = col_idx[pj];
        
        if (ci == cj) {
            dot += val[pi] * val[pj];
            pi++;
            pj++;
        } else if (ci < cj) {
            pi++;
        } else {
            pj++;
        }
    }
    
    return dot;
}

// ========== A x A^T ==========

__global__ void count_full_nnz_kernel(
    const int *A_row_ptr, const int *A_col_idx, const float *A_val,
    int A_rows,
    int *row_nnz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    
    int A_i_start = A_row_ptr[i];
    int A_i_end = A_row_ptr[i + 1];
    
    int count = 0;
    for (int j = 0; j < A_rows; j++) {
        int A_j_start = A_row_ptr[j];
        int A_j_end = A_row_ptr[j + 1];
        
        float dot = row_dot_product(A_col_idx, A_val, 
                                   A_i_start, A_i_end,
                                   A_j_start, A_j_end);
        
        if (fabsf(dot) > 1e-12f) {
            count++;
        }
    }
    
    row_nnz[i] = count;
}

__global__ void fill_full_result_kernel(
    const int *A_row_ptr, const int *A_col_idx, const float *A_val,
    int A_rows,
    const int *C_row_ptr,
    int *C_col_idx, float *C_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) return;
    
    int A_i_start = A_row_ptr[i];
    int A_i_end = A_row_ptr[i + 1];
    
    int C_start = C_row_ptr[i];
    int write_pos = 0;
    
    for (int j = 0; j < A_rows; j++) {
        int A_j_start = A_row_ptr[j];
        int A_j_end = A_row_ptr[j + 1];
        
        float dot = row_dot_product(A_col_idx, A_val,
                                   A_i_start, A_i_end,
                                   A_j_start, A_j_end);
        
        if (fabsf(dot) > 1e-12f) {
            C_col_idx[C_start + write_pos] = j;
            C_val[C_start + write_pos] = dot;
            write_pos++;
        }
    }
}

// ========== A x A 优化版本 ==========

#define HASH_SIZE 4096
#define HASH_EMPTY -1

__device__ int hash_insert_or_add(
    int *hash_keys, float *hash_vals, 
    int key, float val)
{
    int slot = key & (HASH_SIZE - 1);  // 改用位运算，更快
    
    for (int attempt = 0; attempt < HASH_SIZE; attempt++) {
        //atomicCAS提取slot对应位置的key,如果是empty就写入key,返回值是老的key
        int old_key = atomicCAS(&hash_keys[slot], HASH_EMPTY, key);
        
        if (old_key == HASH_EMPTY || old_key == key) {
            atomicAdd(&hash_vals[slot], val);
            return slot;
        }
        
        slot = (slot + 1) & (HASH_SIZE - 1);
    }
    
    return -1;
}

// Bitonic sort in shared memory (for small arrays)
__device__ void bitonic_sort_shared(
    int *keys, float *vals, int n, int tid, int blockDim_x)
{
    // Bitonic sort 适合 power-of-2 大小，这里简化为冒泡排序
    // 对于 nnz < 100 的场景够用了
    
    for (int i = 0; i < n; i++) {
        for (int j = tid; j < n - 1; j += blockDim_x) {
            if (keys[j] > keys[j + 1]) {
                // Swap keys
                int tmp_key = keys[j];
                keys[j] = keys[j + 1];
                keys[j + 1] = tmp_key;
                
                // Swap vals
                float tmp_val = vals[j];
                vals[j] = vals[j + 1];
                vals[j + 1] = tmp_val;
            }
        }
        __syncthreads();
    }
}

__global__ void count_self_nnz_hash_kernel(
    const int *A_row_ptr, const int *A_col_idx, const float *A_val,
    int A_rows, int A_cols,
    int *row_nnz)
{
    //定义共享hash表和值
    __shared__ int shared_hash_keys[HASH_SIZE];
    __shared__ float shared_hash_vals[HASH_SIZE];
    
    int i = blockIdx.x;
    if (i >= A_rows) return;
    
    // 初始化, 一个线程处理一行
    for (int idx = threadIdx.x; idx < HASH_SIZE; idx += blockDim.x) {
        shared_hash_keys[idx] = HASH_EMPTY;
        shared_hash_vals[idx] = 0.0f;
    }
    __syncthreads();
    
    int A_i_start = A_row_ptr[i];
    int A_i_end = A_row_ptr[i + 1];
    
    // 累加到 hash table
    for (int p = A_i_start + threadIdx.x; p < A_i_end; p += blockDim.x) {
        int k = A_col_idx[p];
        float a_ik = A_val[p];
        
        int A_k_start = A_row_ptr[k];
        int A_k_end = A_row_ptr[k + 1];
        
        for (int q = A_k_start; q < A_k_end; q++) {
            int j = A_col_idx[q];
            float a_kj = A_val[q];
            
            hash_insert_or_add(shared_hash_keys, shared_hash_vals, j, a_ik * a_kj);
        }
    }
    __syncthreads();
    
    // 统计 nnz
    if (threadIdx.x == 0) {
        int count = 0;
        for (int idx = 0; idx < HASH_SIZE; idx++) {
            if (shared_hash_keys[idx] != HASH_EMPTY && 
                fabsf(shared_hash_vals[idx]) > 1e-12f) {
                count++;
            }
        }
        row_nnz[i] = count;
    }
}

__global__ void fill_self_result_hash_kernel(
    const int *A_row_ptr, const int *A_col_idx, const float *A_val,
    int A_rows, int A_cols,
    const int *C_row_ptr,
    int *C_col_idx, float *C_val)
{
    __shared__ int shared_hash_keys[HASH_SIZE];
    __shared__ float shared_hash_vals[HASH_SIZE];
    __shared__ int temp_cols[256];  // 临时存 (j, val)，假设每行 nnz < 256
    __shared__ float temp_vals[256];
    
    int i = blockIdx.x;
    if (i >= A_rows) return;
    
    // 初始化
    for (int idx = threadIdx.x; idx < HASH_SIZE; idx += blockDim.x) {
        shared_hash_keys[idx] = HASH_EMPTY;
        shared_hash_vals[idx] = 0.0f;
    }
    __syncthreads();
    
    int A_i_start = A_row_ptr[i];
    int A_i_end = A_row_ptr[i + 1];
    
    // 累加
    for (int p = A_i_start + threadIdx.x; p < A_i_end; p += blockDim.x) {
        int k = A_col_idx[p];
        float a_ik = A_val[p];
        
        int A_k_start = A_row_ptr[k];
        int A_k_end = A_row_ptr[k + 1];
        
        for (int q = A_k_start; q < A_k_end; q++) {
            int j = A_col_idx[q];
            float a_kj = A_val[q];
            
            hash_insert_or_add(shared_hash_keys, shared_hash_vals, j, a_ik * a_kj);
        }
    }
    __syncthreads();
    
    // Thread 0 收集到 temp 数组
    int nnz_count = 0;
    if (threadIdx.x == 0) {
        for (int idx = 0; idx < HASH_SIZE; idx++) {
            int j = shared_hash_keys[idx];
            if (j != HASH_EMPTY) {
                float val = shared_hash_vals[idx];
                if (fabsf(val) > 1e-12f) {
                    if (nnz_count < 256) {  // 防止越界
                        temp_cols[nnz_count] = j;
                        temp_vals[nnz_count] = val;
                        nnz_count++;
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // 在 shared memory 里排序（简化版冒泡）
    for (int pass = 0; pass < nnz_count; pass++) {
        for (int idx = threadIdx.x; idx < nnz_count - 1; idx += blockDim.x) {
            if (temp_cols[idx] > temp_cols[idx + 1]) {
                int tmp_col = temp_cols[idx];
                temp_cols[idx] = temp_cols[idx + 1];
                temp_cols[idx + 1] = tmp_col;
                
                float tmp_val = temp_vals[idx];
                temp_vals[idx] = temp_vals[idx + 1];
                temp_vals[idx + 1] = tmp_val;
            }
        }
        __syncthreads();
    }
    
    // 写回 global memory
    int C_start = C_row_ptr[i];
    for (int idx = threadIdx.x; idx < nnz_count; idx += blockDim.x) {
        C_col_idx[C_start + idx] = temp_cols[idx];
        C_val[C_start + idx] = temp_vals[idx];
    }
}

// ========== A x A^T Host ==========

void spgemm_transpose_product_manual(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(float);
    size_t A_total_size = A_row_ptr_size + A_col_idx_size + A_val_size;

    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice));

    char *dA_base = (char*)dA_buffer;
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    float *dA_val = (float*)(dA_base + A_row_ptr_size + A_col_idx_size);

    int *dC_row_nnz;
    CHECK_CUDA(cudaMalloc(&dC_row_nnz, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_nnz, 0, A_rows * sizeof(int)));

    int block_size = 256;
    int grid_size = (A_rows + block_size - 1) / block_size;
    
    count_full_nnz_kernel<<<grid_size, block_size>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows, dC_row_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());

    int *dC_row_ptr;
    CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));

    thrust::exclusive_scan(
        thrust::device_ptr<int>(dC_row_nnz),
        thrust::device_ptr<int>(dC_row_nnz + A_rows),
        thrust::device_ptr<int>(dC_row_ptr + 1));

    int C_nnz_result;
    CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_row_ptr + A_rows, 
                         sizeof(int), cudaMemcpyDeviceToHost));

    int *dC_col_idx;
    float *dC_val;
    CHECK_CUDA(cudaMalloc(&dC_col_idx, C_nnz_result * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dC_val, C_nnz_result * sizeof(float)));

    fill_full_result_kernel<<<grid_size, block_size>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows,
        dC_row_ptr, dC_col_idx, dC_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 去掉逐行排序，改成批量排序
    // 注意：这里结果已经是按列号递增的（因为 j 递增遍历）
    // 所以不需要排序！

    size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t C_col_idx_size = C_nnz_result * sizeof(int);
    size_t C_val_size = C_nnz_result * sizeof(float);
    size_t C_row_ptr_aligned = (C_row_ptr_size + 3) & ~3;
    size_t C_col_idx_aligned = (C_col_idx_size + 3) & ~3;
    size_t C_total_size = C_row_ptr_aligned + C_col_idx_aligned + C_val_size;

    void *dC_buffer;
    CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));

    char *dC_base = (char*)dC_buffer;
    CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size, 
                         cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned, dC_col_idx, 
                         C_col_idx_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned + C_col_idx_aligned, 
                         dC_val, C_val_size, cudaMemcpyDeviceToDevice));

    void *C_buffer = nullptr;
    CHECK_CUDA(cudaMallocHost(&C_buffer, C_total_size));
                         
    CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, 
                         cudaMemcpyDeviceToHost));

    *C_buffer_out = C_buffer;
    *C_rows = A_rows;
    *C_cols = A_rows;
    *C_nnz = C_nnz_result;

    cudaFree(dA_buffer);
    cudaFree(dC_row_nnz);
    cudaFree(dC_row_ptr);
    cudaFree(dC_col_idx);
    cudaFree(dC_val);
    cudaFree(dC_buffer);
}

// ========== A x A Host ==========

void spgemm_self_product_manual(
    void *A_buffer, int A_rows, int A_cols, int A_nnz,
    void **C_buffer_out, int *C_rows, int *C_cols, int *C_nnz)
{
    size_t A_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t A_col_idx_size = A_nnz * sizeof(int);
    size_t A_val_size = A_nnz * sizeof(float);
    size_t A_total_size = A_row_ptr_size + A_col_idx_size + A_val_size;

    void *dA_buffer;
    CHECK_CUDA(cudaMalloc(&dA_buffer, A_total_size));
    CHECK_CUDA(cudaMemcpy(dA_buffer, A_buffer, A_total_size, cudaMemcpyHostToDevice));

    char *dA_base = (char*)dA_buffer;
    int *dA_row_ptr = (int*)dA_base;
    int *dA_col_idx = (int*)(dA_base + A_row_ptr_size);
    float *dA_val = (float*)(dA_base + A_row_ptr_size + A_col_idx_size);

    int *dC_row_nnz;
    CHECK_CUDA(cudaMalloc(&dC_row_nnz, A_rows * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_nnz, 0, A_rows * sizeof(int)));

    int block_size = 256;
    int grid_size = A_rows;
    
    count_self_nnz_hash_kernel<<<grid_size, block_size>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows, A_cols,
        dC_row_nnz);
    CHECK_CUDA(cudaDeviceSynchronize());

    int *dC_row_ptr;
    CHECK_CUDA(cudaMalloc(&dC_row_ptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(dC_row_ptr, 0, sizeof(int)));

    thrust::exclusive_scan(
        thrust::device_ptr<int>(dC_row_nnz),
        thrust::device_ptr<int>(dC_row_nnz + A_rows),
        thrust::device_ptr<int>(dC_row_ptr + 1));

    int C_nnz_result;
    CHECK_CUDA(cudaMemcpy(&C_nnz_result, dC_row_ptr + A_rows,
                          sizeof(int), cudaMemcpyDeviceToHost));

    int *dC_col_idx;
    float *dC_val;
    CHECK_CUDA(cudaMalloc(&dC_col_idx, C_nnz_result * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dC_val, C_nnz_result * sizeof(float)));

    fill_self_result_hash_kernel<<<grid_size, block_size>>>(
        dA_row_ptr, dA_col_idx, dA_val, A_rows, A_cols,
        dC_row_ptr, dC_col_idx, dC_val);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 不需要排序了，已经在 kernel 里排好了

    size_t C_row_ptr_size = (A_rows + 1) * sizeof(int);
    size_t C_col_idx_size = C_nnz_result * sizeof(int);
    size_t C_val_size = C_nnz_result * sizeof(float);
    size_t C_row_ptr_aligned = (C_row_ptr_size + 3) & ~3;
    size_t C_col_idx_aligned = (C_col_idx_size + 3) & ~3;
    size_t C_total_size = C_row_ptr_aligned + C_col_idx_aligned + C_val_size;

    void *dC_buffer;
    CHECK_CUDA(cudaMalloc(&dC_buffer, C_total_size));

    char *dC_base = (char*)dC_buffer;
    CHECK_CUDA(cudaMemcpy(dC_base, dC_row_ptr, C_row_ptr_size,
                          cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned, dC_col_idx,
                          C_col_idx_size, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dC_base + C_row_ptr_aligned + C_col_idx_aligned,
                          dC_val, C_val_size, cudaMemcpyDeviceToDevice));

    void *C_buffer = nullptr;
    CHECK_CUDA(cudaMallocHost(&C_buffer, C_total_size));
    CHECK_CUDA(cudaMemcpy(C_buffer, dC_buffer, C_total_size, cudaMemcpyDeviceToHost));

    *C_buffer_out = C_buffer;
    *C_rows = A_rows;
    *C_cols = A_cols;
    *C_nnz = C_nnz_result;

    CHECK_CUDA(cudaFree(dA_buffer));
    CHECK_CUDA(cudaFree(dC_row_nnz));
    CHECK_CUDA(cudaFree(dC_row_ptr));
    CHECK_CUDA(cudaFree(dC_col_idx));
    CHECK_CUDA(cudaFree(dC_val));
    CHECK_CUDA(cudaFree(dC_buffer));
}