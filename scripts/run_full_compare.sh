#!/bin/bash
# ===========================================================================
#  终极 6 法对比脚本(first100)
#  跑 6 个方法,生成比较日志 + 6 方法 compute-only 柱状图,打包到 compare/full_compare_<ts>/
#
#    C 法(spgemm_test,USE_MEMPOOL=1 arena):
#      cuSPARSE / gust(ESC) / merge(ser) / merge(par)   ← 4 法,日志→ results/aa/.../log
#    Python baseline:
#      Ocean   = baseline_ocean.py(GPU 阶段求和,compute-only)
#      cuBLAS  = baseline_cublas.py(sgemm kernel,compute-only)
#    profile_aa.py:读以上数据 → 比较日志 + profile_methods_bar.png(6 法柱状图)
#
#  用法:
#    bash scripts/run_full_compare.sh                 # 全跑(~15-20min,大矩阵稠密 GEMM 慢)
#    TIMEOUT=300 bash scripts/run_full_compare.sh     # 缩短 spgemm_test 每矩阵上限
#    PY=python SKIP_C=1 bash scripts/run_full_compare.sh   # 跳过 [2/5](复用已有 C 日志)
# ===========================================================================
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # 脚本在 scripts/ 下,cd 回仓库根

TIMEOUT=${TIMEOUT:-600}
PY=${PY:-.venv/bin/python}
SKIP_C=${SKIP_C:-0}
TS=$(date +%Y%m%d_%H%M%S)
OUT="compare/full_compare_${TS}"
DATA=./data/first100
mkdir -p "$OUT"

echo "############################################################"
echo "#  终极 6 法对比  →  $OUT/"
echo "#  (cuSPARSE / gust(ESC) / merge(ser) / merge(par) / Ocean / cuBLAS)"
echo "############################################################"

echo "### [1/5] 编译 spgemm_test ###"
make >/dev/null 2>&1 || { echo "  ✗ make 失败"; exit 1; }
echo "  ✓ done"

if [ "$SKIP_C" -eq 1 ]; then
  echo "### [2/5] 跳过 spgemm_test(SKIP_C=1,复用 results/aa/first100_aa/log/)###"
else
  echo "### [2/5] spgemm_test 4 法(USE_MEMPOOL=1,$(ls $DATA/*.mtx 2>/dev/null | wc -l) 矩阵)###"
  LOGDIR="results/aa/first100_aa/log"
  rm -rf results/aa/first100_aa/*; mkdir -p "$LOGDIR"
  total=$(ls $DATA/*.mtx 2>/dev/null | wc -l); i=0
  for mtx in $DATA/*.mtx; do
    [ -e "$mtx" ] || continue
    name=$(basename "$mtx" .mtx); i=$((i+1))
    printf "  spgemm_test [%3d/%d] %s\n" "$i" "$total" "$name"
    USE_MEMPOOL=1 timeout "$TIMEOUT" ./spgemm_test "$mtx" > "$LOGDIR/${name}.log" 2>&1
  done
  echo "  ✓ C 日志:$(ls $LOGDIR/*.log 2>/dev/null | wc -l) 个"
fi

echo "### [3/5] cuBLAS baseline(稠密 sgemm kernel,compute-only;大矩阵 O(n³) 慢)###"
$PY suitesparse_crawl/baseline_cublas.py 2>/dev/null | tee "$OUT/baseline_cublas.log"
cp suitesparse_crawl/baseline_cublas.csv "$OUT/" 2>/dev/null \
  || echo "  ✗ baseline_cublas 失败(见 $OUT/baseline_cublas.log)"

echo "### [4/5] Ocean baseline(GPU 阶段求和,compute-only)###"
$PY suitesparse_crawl/baseline_ocean.py 2>/dev/null | tee "$OUT/baseline_ocean.log"
cp suitesparse_crawl/baseline_ocean.csv "$OUT/" 2>/dev/null \
  || echo "  ✗ baseline_ocean 失败(见 $OUT/baseline_ocean.log)"

echo "### [5/5] profile_aa.py 生成比较日志 + 6 方法柱状图 ###"
$PY suitesparse_crawl/profile_aa.py > "$OUT/profile_aa_log.txt" 2>&1 \
  || echo "  ✗ profile_aa 失败(见 $OUT/profile_aa_log.txt)"
cp suitesparse_crawl/charts/profile_methods_bar.png "$OUT/" 2>/dev/null
cp suitesparse_crawl/profile_aa_summary.csv "$OUT/" 2>/dev/null
echo "  ✓ 日志 profile_aa_log.txt + 柱状图 profile_methods_bar.png"

echo
echo "############################################################"
echo "#  完成!产物在 $OUT/"
echo "############################################################"
ls -1 "$OUT"
echo
echo "================ 类别聚合(compute-only,ms)================"
sed -n '/按类别聚合/,/跳过/p' "$OUT/profile_aa_log.txt" 2>/dev/null | head -9
echo
echo "柱状图:$OUT/profile_methods_bar.png"
