#include "spgemm.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cstdlib>   // malloc/free

struct MatrixMarketHeader {
    bool is_pattern;  // 是否为 pattern 格式
    bool is_symmetric;
    int rows, cols, nnz;
};

// 解析 Matrix Market 头部
bool parse_mm_header(std::ifstream &file, MatrixMarketHeader &header) {
    std::string line;
    
    // 读取第一行 banner
    std::getline(file, line);
    if (line.find("%%MatrixMarket") == std::string::npos) {
        std::cerr << "Not a valid Matrix Market file\n";
        return false;
    }
    
    // 检查是否为 pattern 格式
    header.is_pattern = (line.find("pattern") != std::string::npos);
    header.is_symmetric = (line.find("symmetric") != std::string::npos);
    
    // 跳过注释行
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }
    
    // 读取矩阵维度
    std::istringstream iss(line);
    iss >> header.rows >> header.cols >> header.nnz;
    
    return true;
}

bool read_matrix_market(const char *filename, void **buffer_out, 
    int **row_ptr_out, int **col_idx_out, float **val_out,
    int *rows, int *cols, int *nnz) 
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return false;
    }

    // 跳过注释行
    char line[1024];
    do {
        if (!fgets(line, sizeof(line), fp)) {
            fclose(fp);
            return false;
        }
    } while (line[0] == '%');

    // 读取维度
    int M, N, nz;
    sscanf(line, "%d %d %d", &M, &N, &nz);

    // 计算总内存大小
    size_t row_ptr_size = (M + 1) * sizeof(int);
    size_t col_idx_size = nz * sizeof(int);
    size_t val_size = nz * sizeof(float);
    size_t total_size = row_ptr_size + col_idx_size + val_size;

    // 分配单块连续普通 host memory
    void *buffer = malloc(total_size);
    if (!buffer) {
        fprintf(stderr, "malloc failed, size = %zu bytes\n", total_size);
        fclose(fp);
        return false;
    }

    // 设置三个指针偏移
    char *base = (char*)buffer;
    int *row_ptr = (int*)base;
    int *col_idx = (int*)(base + row_ptr_size);
    float *val = (float*)(base + row_ptr_size + col_idx_size);

    // 读取三元组并转换为 CSR
    std::vector<std::vector<std::pair<int, float>>> rows_data(M);
    for (int i = 0; i < nz; ++i) {
        int r, c;
        float v;
        if (fscanf(fp, "%d %d %f", &r, &c, &v) != 3) {
            free(buffer);
            fclose(fp);
            return false;
        }
        rows_data[r - 1].push_back({c - 1, v});
    }
    fclose(fp);

    // 构建 CSR
    row_ptr[0] = 0;
    int idx = 0;
    for (int i = 0; i < M; ++i) {
        auto &row = rows_data[i];
        std::sort(row.begin(), row.end());
        for (auto &p : row) {
            col_idx[idx] = p.first;
            val[idx] = p.second;
            ++idx;
        }
        row_ptr[i + 1] = idx;
    }

    *buffer_out = buffer;
    *row_ptr_out = row_ptr;
    *col_idx_out = col_idx;
    *val_out = val;
    *rows = M;
    *cols = N;
    *nnz = nz;

    return true;
}

bool write_matrix_market(const char *filename,
                        const int *row_ptr, const int *col_idx, const float *val,
                        int rows, int cols, int nnz) 
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to write " << filename << std::endl;
        return false;
    }

    file << "%%MatrixMarket matrix coordinate real general\n";
    file << rows << " " << cols << " " << nnz << "\n";

    for (int i = 0; i < rows; i++) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            file << (i + 1) << " " << (col_idx[j] + 1) << " " 
                 << val[j] << "\n";
        }
    }

    file.close();
    return true;
}

void print_matrix_info(const char *name, int rows, int cols, int nnz) {
    double sparsity = (rows * cols > 0) ? 
        (1.0 - (double)nnz / ((double)rows * cols)) * 100 : 0;
    std::cout << name << ": " << rows << " × " << cols 
              << ", nnz = " << nnz 
              << ", sparsity = " << sparsity << "%\n";
}

bool verify_csr_matrices(const int *row_ptr1, const int *col_idx1, const float *val1,
                        const int *row_ptr2, const int *col_idx2, const float *val2,
                        int rows, int cols, int nnz1, int nnz2, float tolerance) 
{
    // 检查非零元素数量
    if (nnz1 != nnz2) {
        fprintf(stderr, "Verification failed: nnz mismatch (%d vs %d)\n", nnz1, nnz2);
        return false;
    }

    // 检查每一行
    for (int i = 0; i < rows; ++i) {
        int start1 = row_ptr1[i];
        int end1 = row_ptr1[i + 1];
        int start2 = row_ptr2[i];
        int end2 = row_ptr2[i + 1];

        if (end1 - start1 != end2 - start2) {
            fprintf(stderr, "Verification failed: row %d has different nnz (%d vs %d)\n",
                    i, end1 - start1, end2 - start2);
            return false;
        }

        // 检查该行的列索引和值
        for (int j = start1, k = start2; j < end1; ++j, ++k) {
            if (col_idx1[j] != col_idx2[k]) {
                fprintf(stderr, "Verification failed: row %d, col index mismatch (%d vs %d)\n",
                        i, col_idx1[j], col_idx2[k]);
                return false;
            }

            float diff = std::abs(val1[j] - val2[k]);
            float max_val = std::max(std::abs(val1[j]), std::abs(val2[k]));
            float relative_error = (max_val > 0) ? diff / max_val : diff;

            if (relative_error > tolerance) {
                fprintf(stderr, "Verification failed: row %d, col %d, value mismatch (%.6e vs %.6e, rel_err=%.6e)\n",
                        i, col_idx1[j], val1[j], val2[k], relative_error);
                return false;
            }
        }
    }

    return true;
}
