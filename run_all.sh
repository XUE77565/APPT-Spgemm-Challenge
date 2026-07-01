#!/bin/bash

DATA_DIR="./data"
RESULTS_DIR="./results"
mkdir -p $RESULTS_DIR

# 清理旧结果
rm -f $RESULTS_DIR/*.csv

echo "matrix,rows,cols,nnz,self_time_ms,self_nnz,transpose_time_ms,transpose_nnz" > $RESULTS_DIR/summary.csv

for matrix_dir in $DATA_DIR/*/; do
    matrix_name=$(basename "$matrix_dir")
    mtx_file="$matrix_dir/${matrix_name}.mtx"
    
    if [ ! -f "$mtx_file" ]; then
        echo "Skipping $matrix_name (no .mtx file)"
        continue
    fi
    
    echo "Processing $matrix_name..."
    ./spgemm_test "$mtx_file" 2>&1 | tee "$RESULTS_DIR/${matrix_name}.log"
    
    # 从输出中提取结果追加到 CSV
    # 这里需要根据你的实际输出格式调整解析逻辑
done

echo "All tests completed. Results in $RESULTS_DIR/"