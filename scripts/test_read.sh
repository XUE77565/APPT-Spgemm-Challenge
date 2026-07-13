#!/bin/bash
# 专门测试 read_matrix_market 能否正确读入所有矩阵
# 判定依据：spgemm_test 的退出码
#   TEST_READ 模式下：读成功 return 0；读失败 return EXIT_FAILURE；崩溃(abort/segv) 非零
# 关键：不能用 "| tee" 后查 $?（那会是 tee 的退出码，恒 0），必须直接重定向

cd "$(dirname "$(readlink -f "$0")")/.."   # 脚本在 scripts/ 下,cd 回仓库根
DATA_DIR="./data"
OUT_DIR="./read_test"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.csv "$OUT_DIR"/*.log

echo "matrix,status,rows,cols,nnz" > "$OUT_DIR/read_summary.csv"

ok=0
fail=0
total=0
failed=""

for matrix_dir in "$DATA_DIR"/*/; do
    name=$(basename "$matrix_dir")
    mtx="${matrix_dir}${name}.mtx"

    if [ ! -f "$mtx" ]; then
        echo "Skipping $name (no .mtx)"
        continue
    fi

    total=$((total + 1))
    printf "  %-22s ... " "$name"

    log="$OUT_DIR/${name}.log"
    ./spgemm_test "$mtx" > "$log" 2>&1
    rc=$?

    # 读成功时 main 会打印 "Input A: <rows> x <cols>, nnz = <nnz>"，解析出来
    read rows cols nnz <<< "$(grep -m1 'Input A:' "$log" | awk -F'[^0-9]+' '{print $2, $3, $4}')"

    if [ "$rc" -eq 0 ]; then
        ok=$((ok + 1))
        status="OK"
    else
        fail=$((fail + 1))
        status="FAIL"
        failed="${failed}\n    ${name} (rc=${rc})"
    fi
    echo "$status"

    echo "${name},${status},${rows},${cols},${nnz}" >> "$OUT_DIR/read_summary.csv"
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
echo "Detail: $OUT_DIR/read_summary.csv"
echo "Per-matrix logs: $OUT_DIR/<name>.log (看具体崩溃信息)"
