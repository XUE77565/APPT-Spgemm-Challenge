#!/bin/bash
# ===========================================================================
#  5-method comparison driver (first100):
#    Auto (ours, adaptive hash/merge3) / cuSPARSE / opSparse / HSMU / dense
#
#  Engines:
#    scripts/compare_paper.py  — runs the 5 methods → compare/paper_cmp.csv
#                                 (incremental resume: keeps existing timings,
#                                  i.e. "same results as before"; fills gaps)
#    scripts/paper_figures.py  — figures + LaTeX table from the CSV
#
#  Each method's timing口径 (compute-only, exclude h2d/d2h where possible):
#    Auto      = METHOD=adaptive, cudaEvent TOTAL − h2d − d2h
#    cuSPARSE  = METHOD=cu, [dbg ms][cu] phase intervals (excl h2d/d2h)
#    opSparse  = external OpSparse binary, "total" ms
#    HSMU      = external HSMU test binary, NHC CSV col6
#    dense     = spgemm_dense, naive scalar GEMM (densify+dgemm+sparsify)
#
#  用法:
#    bash scripts/run_full_compare.sh                 # resume: show existing
#                                                      # results + regen figures
#    FRESH=1 bash scripts/run_full_compare.sh          # delete CSV, re-run all 5
#    DENSE_ONLY=1 bash scripts/run_full_compare.sh     # refresh dense column only
#    LIMIT=10 bash scripts/run_full_compare.sh         # smoke test (first 10)
#    TIMEOUT=200 DENSE_TIMEOUT=600 bash scripts/run_full_compare.sh
# ===========================================================================
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # 脚本在 scripts/ 下,cd 回仓库根

TIMEOUT=${TIMEOUT:-200}        # per-call timeout for Auto/cuSPARSE/opSparse/HSMU
DENSE_TIMEOUT=${DENSE_TIMEOUT:-600}  # naive dense is slow on large matrices
PY=${PY:-.venv/bin/python}
FRESH=${FRESH:-0}
DENSE_ONLY=${DENSE_ONLY:-0}
LIMIT=${LIMIT:-0}
DATA=${DATA:-./data/first100}
CSV=compare/paper_cmp.csv
export TIMEOUT DENSE_TIMEOUT

echo "############################################################"
echo "#  SpGEMM 5-method compare"
echo "#  Auto(ours) / cuSPARSE / opSparse / HSMU / dense"
echo "#  matrix dir: $DATA  $([ "$LIMIT" -gt 0 ] && echo "(前 $LIMIT 阵)" || echo "(全部)")"
[ "$FRESH"      -eq 1 ] && echo "#  FRESH=1      → 清空 CSV,全 5 法重跑"
[ "$DENSE_ONLY" -eq 1 ] && echo "#  DENSE_ONLY=1 → 只刷新 dense 列(其余 4 法保持原值)"
echo "############################################################"

echo "### [1/3] 编译 spgemm_test(DBG=1 cudaEvent 计时) + spgemm_dense ###"
make DBG=1 >/dev/null 2>&1 || { echo "  ✗ make spgemm_test 失败"; exit 1; }
make dense  >/dev/null 2>&1 || { echo "  ✗ make dense 失败"; exit 1; }
echo "  ✓ done"

# ---- decide run mode ----
if [ "$FRESH" -eq 1 ]; then
  rm -f "$CSV"
fi

echo "### [2/3] 跑 5 法(compare_paper.py)→ $CSV ###"
ARGS="--dir $DATA"
[ "$LIMIT" -gt 0 ] && ARGS="$ARGS --limit $LIMIT"
if [ "$DENSE_ONLY" -eq 1 ]; then
  ARGS="$ARGS --dense-only"
elif [ "$FRESH" -ne 1 ] && [ -s "$CSV" ]; then
  echo "  (resume:已有 $(($(wc -l < "$CSV")-1)) 阵,保持原结果,只补缺;FRESH=1 可全重跑)"
fi
$PY -u scripts/compare_paper.py $ARGS 2>&1 | tee compare/_last_run.log
echo "  ✓ CSV: $CSV ($(($(wc -l < "$CSV")-1)) 阵)"

echo "### [3/3] 出图 + LaTeX 表(paper_figures.py)###"
$PY scripts/paper_figures.py "$CSV" 2>&1 | sed 's/^/  /'
$PY scripts/paper_figures.py "$CSV" --slide >/dev/null 2>&1 \
  && echo "  ✓ slide 变体也生成"

echo
echo "############################################################"
echo "#  完成!产物:"
echo "#    $CSV                       (5 法 timing)"
echo "#    compare/paper_speedup_table.tex   (LaTeX,可直接 \\input)"
echo "#    fig/paper_speedup_overall{,_slide}.png"
echo "#    fig/paper_speedup_by_density{,_slide}.png"
echo "#    fig/paper_speedup_scatter{,_slide}.png"
echo "############################################################"
echo
echo "================ 结果汇总(同口径,compute-only ms)================"
$PY scripts/compare_paper.py --report 2>&1 | sed -n '/Auto speedup by density/,$p'
