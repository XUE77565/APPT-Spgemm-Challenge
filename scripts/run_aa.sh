#!/bin/bash
# 跑 data/rep/ 下的【代表矩阵】(扁平 .mtx)做 SpGEMM benchmark,统计成功/失败/超时
# 与 run_all.sh 的区别:data/rep/ 是扁平结构(<name>.mtx),这里直接遍历 .mtx 文件
# 判定依据:spgemm_test 的退出码
#   0        -> 成功(读入 + 计算全部完成)
#   非零      -> 失败(读入失败 / CUDA 错误 / 崩溃 segv/abort 等)
#   124/137   -> 超时(超过 TIMEOUT 秒被 kill,单独计数,跳过该矩阵继续下一个)
# 关键:不能用 "| tee" 后查 $? (那会是 tee 的退出码,恒 0),必须用 pipefail 拿真实退出码

set -o pipefail   # 让管道返回 spgemm_test 的真实退出码而不是 tee 的
cd "$(dirname "$(readlink -f "$0")")/.."   # 脚本在 scripts/ 下,cd 回仓库根,使 ./spgemm_test ./data ./results 等相对路径生效


# 单个矩阵最大允许耗时（秒）。可用环境变量覆盖，如：TIMEOUT=60 ./run_all.sh
TIMEOUT=${TIMEOUT:-600}
DATA_DIR="./data/first100"
# 可经 AA_RESULTS_DIR 覆盖(供 A/B:legacy / pool 各跑一个目录,互不覆盖)
RESULTS_DIR="${AA_RESULTS_DIR:-./results/aa/first100_aa}"
LOG_DIR="$RESULTS_DIR/log"
MATRIX_DIR="$RESULTS_DIR/matrices"
SUMMARY="$RESULTS_DIR/summary.csv"

# 清理上次的结果
rm -rf "$RESULTS_DIR"/*

mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$MATRIX_DIR"
rm -f "$SUMMARY"
echo "matrix,status,rows,cols,nnz" > "$SUMMARY"


ok=0
fail=0
timeout_cnt=0
total=0
failed=""
timedout=""

  # data/rep/ 是扁平结构:data/rep/<name>.mtx(不是 <name>/<name>.mtx)
  # 直接遍历 .mtx 文件,而不是子目录
  for mtx in "$DATA_DIR"/*.mtx; do
      [ -e "$mtx" ] || continue        # 目录为空时 glob 不展开,跳过
      name=$(basename "$mtx" .mtx)

      total=$((total + 1))
      echo "Processing $name ... (max ${TIMEOUT}s)"

      log="$LOG_DIR/${name}.log"
      timeout -k 10 "$TIMEOUT" ./spgemm_test "$mtx" 2>&1 | tee "$log"
      rc=$?

      # 从输出中解析 "Input A: <rows> x <cols>, nnz = <nnz>"
      read rows cols nnz <<< "$(grep -m1 'Input A:' "$log" | awk -F'[^0-9]+' '{print $2, $3, $4}')"

      if [ "$rc" -eq 0 ]; then
          ok=$((ok + 1))
          status="OK"
      elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
          timeout_cnt=$((timeout_cnt + 1))
          status="TIMEOUT"
          timedout="${timedout}\n    ${name} (>${TIMEOUT}s)"
          echo "  -> TIMEOUT (>${TIMEOUT}s), skipped"
      else
          fail=$((fail + 1))
          status="FAIL"
          failed="${failed}\n    ${name} (rc=${rc})"
      fi

      echo "${name},${status},${rows},${cols},${nnz}" >> "$SUMMARY"
  done

  echo ""
  echo "============================="
  echo "Total:    $total"
  echo "OK:       $ok"
  echo "FAIL:     $fail"
  echo "TIMEOUT:  $timeout_cnt  (>${TIMEOUT}s 自动跳过)"
  echo "============================="
  if [ -n "$failed" ]; then
      echo -e "Failed:${failed}"
  fi
  if [ -n "$timedout" ]; then
      echo -e "Timed out:${timedout}"
  fi
  echo ""
  echo "Summary: $SUMMARY"
  echo "Per-matrix logs: $LOG_DIR/<name>.log"
  echo "Matrix outputs:  $MATRIX_DIR/<name>/"