#!/bin/bash
# ===========================================================================
#  方法对比脚本(first100)— 6 法  【跑一次,以后只读】
#    cuSPARSE / Auto(ours,自适应) / Ocean / opSparse / HSMU / dense
#
#  数据存【稳定 CSV】compare/methods_cmp.csv(增量 resume:已有矩阵跳过,不重跑)。
#  报告/图也稳定:compare/methods_cmp_{report.txt,bar.png}。
#
#  模式(三选一):
#    bash scripts/run_method_cmp.sh                 # 默认:resume(已有跳过)+ 重出报告/图
#                                                     → 首次慢(全跑),之后秒级(全已存)
#    REPORT_ONLY=1 bash scripts/run_method_cmp.sh   # 不跑,只从 CSV 重出报告+图(最快)
#    REFRESH=dense bash scripts/run_method_cmp.sh   # 只重跑 dense 列(改了 spgemm_dense 后)
#    REFRESH=Auto bash scripts/run_method_cmp.sh    # 只重跑 Auto 列(改了 hash/dispatcher 后)
#    FRESH=1 bash scripts/run_method_cmp.sh         # 备份 CSV 后清空,全 6 法重跑
#
#  跳过某法:NO_OCEAN=1 / NO_HSMU=1 / NO_OPSPARSE=1 / NO_DENSE=1
#  冒烟:LIMIT=10 ...   超时:TIMEOUT=300 DENSE_TIMEOUT=600 ...
#
#  每矩阵口径:
#    cu/Auto → METHOD=<m> spgemm_test,compute-only(去 h2d/d2h)
#    Ocean   → ocean/convert+spgemm stats.json 各 phase 求和
#    opSparse→ 外部 OpSparse,total ms
#    HSMU    → 外部 HSMU test,NHC CSV col6
#    dense   → spgemm_dense(tiled FP64 GEMM),Kernel time ms
#  注意:HSMU 会写 /tmp/*.csv(figure 要读)→ 跑前备份、跑后(EXIT trap)恢复。
# ===========================================================================
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # 脚本在 scripts/ 下,cd 回仓库根

TIMEOUT=${TIMEOUT:-200}
DENSE_TIMEOUT=${DENSE_TIMEOUT:-600}
export TIMEOUT DENSE_TIMEOUT
PY=${PY:-.venv/bin/python}
MATRIX_DIR=${MATRIX_DIR:-./data/first100}
LIMIT=${LIMIT:-0}
NO_HSMU=${NO_HSMU:-0}
NO_OPSPARSE=${NO_OPSPARSE:-0}
NO_DENSE=${NO_DENSE:-0}
NO_CUBLAS=${NO_CUBLAS:-0}
NO_OCEAN=${NO_OCEAN:-0}
FRESH=${FRESH:-0}
REPORT_ONLY=${REPORT_ONLY:-0}
REFRESH=${REFRESH:-}        # cu/Auto/Ocean/opSparse/HSMU/dense

# ---- 稳定产物路径 ----
CSV=compare/methods_cmp.csv
REPORT=compare/methods_cmp_report.txt
BAR=compare/methods_cmp_bar.png
LOG=compare/methods_cmp_run.log

# ---- HSMU /tmp CSV 备份/恢复(只在真跑 HSMU 时有意义)----
TS=$(date +%Y%m%d_%H%M%S)
HSMU_CSVS=(NHC_4080S_result small_step_runtime new_compressed_step_runtime small_time_conversion)
BACKUP="/tmp/hsmu_backup_${TS}"
mkdir -p "$BACKUP"
restore_hsmu() {
  for f in "${HSMU_CSVS[@]}"; do [ -f "$BACKUP/$f.csv" ] && cp "$BACKUP/$f.csv" "/tmp/$f.csv"; done
}
trap restore_hsmu EXIT INT TERM

echo "############################################################"
echo "#  SpGEMM 6 法对比  →  $CSV(稳定,resume)"
echo "#  cuSPARSE / Auto(ours) / Ocean / opSparse / HSMU / dense"
[ "$REPORT_ONLY" -eq 1 ] && echo "#  模式:REPORT_ONLY(只读 CSV,不跑)"
[ -n "$REFRESH" ]        && echo "#  模式:REFRESH=$REFRESH(只重跑该列)"
[ "$FRESH" -eq 1 ]       && echo "#  模式:FRESH(备份+清空 CSV,全重跑)"
echo "############################################################"

# ---- REPORT_ONLY:不编译不跑,直接出报告+图 ----
if [ "$REPORT_ONLY" -eq 1 ]; then
  [ ! -s "$CSV" ] && { echo "✗ $CSV 不存在,先跑一次(去掉 REPORT_ONLY)"; exit 1; }
  echo "### 只读 $CSV($(($(wc -l < "$CSV")-1)) 阵)→ 报告 + 图 ###"
  $PY scripts/report_methods_cmp.py "$CSV" "$REPORT" >/dev/null 2>&1 && echo "  ✓ $REPORT"
  $PY scripts/plot_methods_cmp.py   "$CSV" "$BAR"    >/dev/null 2>&1 && echo "  ✓ $BAR"
  sed -n '/Auto vs cuSPARSE/,$p' "$REPORT" 2>/dev/null
  exit 0
fi

echo "### [1/3] 编译 spgemm_test(DBG=1)$([ "$NO_DENSE" -eq 0 ] && echo ' + spgemm_dense')$([ "$NO_CUBLAS" -eq 0 ] && echo ' + cublas') ###"
make DBG=1 >/dev/null 2>&1 || { echo "  ✗ make spgemm_test 失败"; exit 1; }
if [ "$NO_DENSE" -eq 0 ]; then
  make dense >/dev/null 2>&1 || { echo "  ✗ make dense 失败"; exit 1; }
fi
if [ "$NO_CUBLAS" -eq 0 ]; then
  make cublas >/dev/null 2>&1 || { echo "  ✗ make cublas 失败"; exit 1; }
fi
echo "  ✓ done"

echo "### [1b/3] 备份 HSMU /tmp CSV → $BACKUP ###"
for f in "${HSMU_CSVS[@]}"; do [ -f "/tmp/$f.csv" ] && cp "/tmp/$f.csv" "$BACKUP/"; done
echo "  ✓ 备份 $(ls "$BACKUP" 2>/dev/null | wc -l) 个"

# ---- 决定 run 模式 ----
if [ "$FRESH" -eq 1 ]; then
  [ -f "$CSV" ] && cp "$CSV" "${CSV}.bak_${TS}"
  rm -f "$CSV"
fi
ARGS="--out $CSV --dir $MATRIX_DIR"
[ "$LIMIT" -gt 0 ]       && ARGS="$ARGS --limit $LIMIT"
[ "$NO_OCEAN" -eq 1 ]    && ARGS="$ARGS --no-ocean"
[ "$NO_HSMU" -eq 1 ]     && ARGS="$ARGS --no-hsmu"
[ "$NO_OPSPARSE" -eq 1 ] && ARGS="$ARGS --no-opsparse"
[ "$NO_DENSE" -eq 1 ]    && ARGS="$ARGS --no-dense"
[ "$NO_CUBLAS" -eq 1 ]   && ARGS="$ARGS --no-cublas"
[ -n "$REFRESH" ]        && ARGS="$ARGS --refresh-col $REFRESH"

echo "### [2/3] 跑对比(compare_methods.py $ARGS)###"
$PY -u scripts/compare_methods.py $ARGS 2>&1 | tee "$LOG" \
  || { echo "  ✗ compare_methods.py 失败(见 $LOG)"; exit 1; }
echo "  ✓ CSV: $CSV ($(($(wc -l < "$CSV")-1)) 阵)"

echo "### [3/3] 报告 + 柱状图 ###"
$PY scripts/report_methods_cmp.py "$CSV" "$REPORT" >/dev/null 2>&1 \
  && echo "  ✓ $REPORT" || echo "  ✗ 报告失败"
$PY scripts/plot_methods_cmp.py "$CSV" "$BAR" >/dev/null 2>&1 \
  && [ -f "$BAR" ] && echo "  ✓ $BAR" || echo "  ✗ 画图失败"

echo
echo "############################################################"
echo "#  完成!稳定产物(HSMU /tmp CSV 由 EXIT trap 已恢复):"
echo "#    $CSV          (6 法 timing,resume)"
echo "#    $REPORT       (Auto vs 各基线 输赢)"
echo "#    $BAR          (按密度类别柱状图)"
echo "############################################################"
echo
echo "================ Auto vs 各基线 输赢 ===================="
sed -n '/Auto vs cuSPARSE/,$p' "$REPORT" 2>/dev/null
