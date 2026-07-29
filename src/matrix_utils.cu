#include "spgemm.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <cstdlib>   // malloc/free
#include <cctype>    // tolower

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
    int **row_ptr_out, int **col_idx_out, double **val_out,
    int *rows, int *cols, int *nnz) 
{
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return false;
    }

    // 读第一行 banner，检测 pattern / symmetric（大小写不敏感）
    char banner[1024];
    if (!fgets(banner, sizeof(banner), fp)) {
        fclose(fp);
        return false;
    }
    std::string bs(banner);
    for (char &ch : bs) ch = (char)tolower((unsigned char)ch);
    bool is_pattern = (bs.find("pattern") != std::string::npos);
    bool is_symmetric = (bs.find("symmetric") != std::string::npos);
    bool is_complex = (bs.find("complex") != std::string::npos);

    // 跳过剩余注释行
    char line[1024];
    do {
        if (!fgets(line, sizeof(line), fp)) {
            fclose(fp);
            return false;
        }
    } while (line[0] == '%');

    // 读取维度（symmetric 时 nz 只是下三角条目数，展开后会变大）
    int M, N, nz;
    sscanf(line, "%d %d %d", &M, &N, &nz);

    // 读取三元组：pattern 无 value（隐含 1.0），symmetric 需镜像补全上三角
    std::vector<std::vector<std::pair<int, double>>> rows_data(M);
    for (int i = 0; i < nz; ++i) {
        int r, c;
        double v;
        if (is_pattern) {
            if (fscanf(fp, "%d %d", &r, &c) != 2) {
                fclose(fp);
                return false;
            }
            v = 1.0f;
        } else if (is_complex) {
            // complex 数据行是 "row col real imag"，这里只保留实部
            double re, im;
            if (fscanf(fp, "%d %d %lf %lf", &r, &c, &re, &im) != 4) {
                fclose(fp);
                return false;
            }
            v = re;
        } else {
            if (fscanf(fp, "%d %d %lf", &r, &c, &v) != 3) {
                fclose(fp);
                return false;
            }
        }
        r -= 1;
        c -= 1;
        rows_data[r].push_back({c, v});
        // symmetric 只存下三角，r != c 时补镜像项 (c, r)；对角线 r==c 只存一次
        if (is_symmetric && r != c) {
            rows_data[c].push_back({r, v});
        }
    }
    fclose(fp);

    // 展开后的实际 nnz
    int actual_nnz = 0;
    for (int i = 0; i < M; ++i) actual_nnz += (int)rows_data[i].size();

    // 计算总内存大小（按展开后的 nnz）
    size_t row_ptr_size = (M + 1) * sizeof(int);
    size_t col_idx_size = actual_nnz * sizeof(int);
    size_t val_size = actual_nnz * sizeof(double);
    size_t total_size = ALIGN8(row_ptr_size + col_idx_size) + val_size;

    // 分配单块连续 host memory:USE_MEMPOOL=1 → pinned(DMA 直传);0 → pageable(driver staging)。便于 A/B。
    void *buffer = nullptr;
    if (g_use_mempool) {
        if (cudaMallocHost(&buffer, total_size) != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed, size = %zu bytes\n", total_size);
            return false;
        }
    } else {
        buffer = malloc(total_size);
        if (!buffer) {
            fprintf(stderr, "malloc failed, size = %zu bytes\n", total_size);
            return false;
        }
    }

    // 设置三个指针偏移
    char *base = (char*)buffer;
    int *row_ptr = (int*)base;
    int *col_idx = (int*)(base + row_ptr_size);
    double *val = (double*)(base + ALIGN8(row_ptr_size + col_idx_size));

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
    *nnz = actual_nnz;

    return true;
}

bool write_matrix_market(const char *filename,
                        const int *row_ptr, const int *col_idx, const double *val,
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

bool verify_csr_matrices(const int *row_ptr1, const int *col_idx1, const double *val1,
                        const int *row_ptr2, const int *col_idx2, const double *val2,
                        int rows, int cols, int nnz1, int nnz2, double tolerance) 
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

            double diff = std::abs(val1[j] - val2[k]);
            double max_val = std::max(std::abs(val1[j]), std::abs(val2[k]));
            double relative_error = (max_val > 0) ? diff / max_val : diff;

            if (relative_error > tolerance) {
                fprintf(stderr, "Verification failed: row %d, col %d, value mismatch (%.6e vs %.6e, rel_err=%.6e)\n",
                        i, col_idx1[j], val1[j], val2[k], relative_error);
                return false;
            }
        }
    }

    return true;
}
