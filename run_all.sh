#!/bin/bash
# 跑全部矩阵的完整 SpGEMM benchmark，并统计成功/失败数量
# 判定依据：spgemm_test 的退出码
#   0        -> 成功（读入 + 计算全部完成）
#   非零      -> 失败（读入失败 / CUDA 错误 / 崩溃 segv/abort 等）
# 关键：不能用 "| tee" 后查 $?（那会是 tee 的退出码，恒 0），必须用 pipefail 或直接重定向

set -o pipefail   # 让管道返回 spgemm_test 的真实退出码而不是 tee 的

DATA_DIR="./data"
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/*.csv

echo "matrix,status,rows,cols,nnz" > "$RESULTS_DIR/summary.csv"

ok=0
fail=0
total=0
failed=""

for matrix_dir in "$DATA_DIR"/*/; do
    name=$(basename "$matrix_dir")
    mtx="${matrix_dir}${name}.mtx"

    if [ ! -f "$mtx" ]; then
        echo "Skipping $name (no .mtx file)"
        continue
    fi

    total=$((total + 1))
    echo "Processing $name ..."

    log="$RESULTS_DIR/${name}.log"
    # 直接重定向拿真实退出码；同时 tee 到终端保持实时输出
    ./spgemm_test "$mtx" 2>&1 | tee "$log"
    rc=$?

    # 从输出中解析 "Input A: <rows> x <cols>, nnz = <nnz>"
    read rows cols nnz <<< "$(grep -m1 'Input A:' "$log" | awk -F'[^0-9]+' '{print $2, $3, $4}')"

    if [ "$rc" -eq 0 ]; then
        ok=$((ok + 1))
        status="OK"
    else
        fail=$((fail + 1))
        status="FAIL"
        failed="${failed}\n    ${name} (rc=${rc})"
    fi

    echo "${name},${status},${rows},${cols},${nnz}" >> "$RESULTS_DIR/summary.csv"
done

echo ""
echo "============================="
echo "Total: $total"
echo "OK:    $ok"
echo "FAIL:  $fail"
echo "============================="
if [ -n "$failed" ]; then
    echo -e "Failed:${failed}"
fi
echo ""
echo "Summary: $RESULTS_DIR/summary.csv"
echo "Per-matrix logs: $RESULTS_DIR/<name>.log"
