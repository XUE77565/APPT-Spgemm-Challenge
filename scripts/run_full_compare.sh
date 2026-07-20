#!/bin/bash
# ===========================================================================
#  终极对比脚本(first100)
#  跑 spgemm_test(5 C 法)+ Ocean baseline,生成比较日志 + 柱状图
#
#    C 法(spgemm_test,USE_MEMPOOL=1 arena):
#      cuSPARSE / gust(ESC) / merge(ser) / merge2(par) / merge3(bucket)
#    Python baseline:
#      Ocean   = baseline_ocean.py(GPU 阶段求和,compute-only)
#    profile_aa.py:读以上数据 → 比较日志 + profile_methods_bar.png
#
#  用法:
#    bash scripts/run_full_compare.sh                 # 全跑
#    TIMEOUT=300 bash scripts/run_full_compare.sh     # 缩短 spgemm_test 每矩阵上限
#    PY=python SKIP_C=1 bash scripts/run_full_compare.sh   # 跳过 spgemm_test(复用已有日志)
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
echo "#  SpGEMM 对比  →  $OUT/"
echo "#  (cuSPARSE / gust(ESC) / merge(ser) / merge2(par) / merge3(bucket) / Ocean)"
echo "############################################################"

echo "### [1/4] 编译 spgemm_test ###"
make >/dev/null 2>&1 || { echo "  ✗ make 失败"; exit 1; }
echo "  ✓ done"

if [ "$SKIP_C" -eq 1 ]; then
  echo "### [2/4] 跳过 spgemm_test(SKIP_C=1,复用 results/aa/first100_aa/log/)###"
else
  echo "### [2/4] spgemm_test 5 法(USE_MEMPOOL=1,$(ls $DATA/*.mtx 2>/dev/null | wc -l) 矩阵)###"
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

echo "### [3/4] Ocean baseline(GPU 阶段求和,compute-only)###"
$PY suitesparse_crawl/baseline_ocean.py 2>/dev/null | tee "$OUT/baseline_ocean.log"
cp suitesparse_crawl/baseline_ocean.csv "$OUT/" 2>/dev/null \
  || echo "  ✗ baseline_ocean 失败(见 $OUT/baseline_ocean.log)"

echo "### [4/4] profile_aa.py 生成比较日志 + 柱状图 ###"
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
