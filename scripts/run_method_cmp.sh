#!/bin/bash
# ===========================================================================
#  方法对比脚本(first100)— 5 法
#    cuSPARSE / Auto(ours,自适应) / opSparse / HSMU / dense
#  (无 Ocean、无 merge serial/merge3)
#
#  每矩阵:
#    cu / Auto  → METHOD=<m> 跑 spgemm_test,compute-only(去 h2d/d2h)
#    opSparse   → 外部 OpSparse binary,total ms
#    HSMU       → 外部 HSMU test binary,NHC CSV col6
#    dense      → spgemm_dense(naive scalar GEMM),Kernel time ms
#  → 增量 CSV + 报告(Auto 对各法赢/输,仿原 Ocean 对比)+ 柱状图
#
#  注意:HSMU 的 test binary 会追加写 /tmp/*.csv(figure 的 gen_sota_clean.py 要读),
#  本脚本跑前备份、跑后(含中断)恢复,确保不影响 figure。
#
#  用法:
#    bash scripts/run_method_cmp.sh                        # 全跑 first100
#    LIMIT=10 bash scripts/run_method_cmp.sh               # 只前 10 阵(冒烟)
#    TIMEOUT=300 DENSE_TIMEOUT=600 bash scripts/run_method_cmp.sh
#    NO_HSMU=1 bash scripts/run_method_cmp.sh              # 跳过 HSMU
#    NO_OPSPARSE=1 bash scripts/run_method_cmp.sh          # 跳过 opSparse
#    NO_DENSE=1 bash scripts/run_method_cmp.sh             # 跳过 dense
#    FRESH=1 bash scripts/run_method_cmp.sh                # 清空 CSV 全重跑
# ===========================================================================
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # 脚本在 scripts/ 下,cd 回仓库根

TIMEOUT=${TIMEOUT:-200}
DENSE_TIMEOUT=${DENSE_TIMEOUT:-600}
export TIMEOUT DENSE_TIMEOUT   # 透传给 compare_methods.py 的 per-call 超时
PY=${PY:-.venv/bin/python}
MATRIX_DIR=${MATRIX_DIR:-./data/first100}
LIMIT=${LIMIT:-0}
NO_HSMU=${NO_HSMU:-0}
NO_OPSPARSE=${NO_OPSPARSE:-0}
NO_DENSE=${NO_DENSE:-0}
FRESH=${FRESH:-0}
TS=$(date +%Y%m%d_%H%M%S)
OUT="compare/method_cmp_${TS}"
mkdir -p "$OUT"

# ---- HSMU /tmp CSV 备份/恢复(figure 要读,不能被污染)----
HSMU_CSVS=(NHC_4080S_result small_step_runtime new_compressed_step_runtime small_time_conversion)
BACKUP="/tmp/hsmu_backup_${TS}"
mkdir -p "$BACKUP"
restore_hsmu() {
  for f in "${HSMU_CSVS[@]}"; do [ -f "$BACKUP/$f.csv" ] && cp "$BACKUP/$f.csv" "/tmp/$f.csv"; done
}
trap restore_hsmu EXIT INT TERM

echo "############################################################"
echo "#  SpGEMM 5 法对比  →  $OUT/"
echo "#  cuSPARSE / Auto(ours) / opSparse / HSMU / dense"
echo "#  矩阵: $MATRIX_DIR  $([ $LIMIT -gt 0 ] && echo "(前 $LIMIT 阵)" || echo "(全部)")"
echo "############################################################"

echo "### [1/3] 编译 spgemm_test(DBG=1 cudaEvent 计时)$([ "$NO_DENSE" -eq 0 ] && echo ' + spgemm_dense') ###"
make DBG=1 >/dev/null 2>&1 || { echo "  ✗ make spgemm_test 失败"; exit 1; }
if [ "$NO_DENSE" -eq 0 ]; then
  make dense >/dev/null 2>&1 || { echo "  ✗ make dense 失败"; exit 1; }
fi
echo "  ✓ done"

echo "### [1b/3] 备份 HSMU /tmp CSV → $BACKUP ###"
for f in "${HSMU_CSVS[@]}"; do [ -f "/tmp/$f.csv" ] && cp "/tmp/$f.csv" "$BACKUP/"; done
echo "  ✓ 备份 $(ls "$BACKUP" 2>/dev/null | wc -l) 个"

echo "### [2/3] 对比(cu+Auto+opSparse+HSMU+dense,每法/矩阵超时 ${TIMEOUT}s/dense ${DENSE_TIMEOUT}s)###"
CSV="$OUT/methods_cmp.csv"
[ "$FRESH" -eq 1 ] && rm -f compare/methods_cmp.csv
ARGS="--out $CSV --dir $MATRIX_DIR"
[ "$LIMIT" -gt 0 ]      && ARGS="$ARGS --limit $LIMIT"
[ "$NO_HSMU" -eq 1 ]    && ARGS="$ARGS --no-hsmu"
[ "$NO_OPSPARSE" -eq 1 ] && ARGS="$ARGS --no-opsparse"
[ "$NO_DENSE" -eq 1 ]   && ARGS="$ARGS --no-dense"
$PY -u scripts/compare_methods.py $ARGS 2>&1 | tee "$OUT/run.log" \
  || { echo "  ✗ compare_methods.py 失败(见 $OUT/run.log)"; exit 1; }
echo "  ✓ 对比表: $CSV ($(($(wc -l < "$CSV")-1)) 阵)"

echo "### [3/3] 报告 + 柱状图 ###"
$PY scripts/report_methods_cmp.py "$CSV" "$OUT/methods_cmp_report.txt" >/dev/null 2>&1 \
  && echo "  ✓ 报告: $OUT/methods_cmp_report.txt" || echo "  ✗ 报告失败"
$PY scripts/plot_methods_cmp.py "$CSV" "$OUT/methods_cmp_bar.png" >/dev/null 2>&1 \
  && [ -f "$OUT/methods_cmp_bar.png" ] && echo "  ✓ 图: $OUT/methods_cmp_bar.png" || echo "  ✗ 画图失败"

echo
echo "############################################################"
echo "#  完成!产物在 $OUT/  (HSMU /tmp CSV 由 EXIT trap 已恢复)"
echo "############################################################"
ls -1 "$OUT"
echo
echo "================ Auto vs 各基线 输赢(从报告)================"
sed -n '/Auto vs cuSPARSE/,$p' "$OUT/methods_cmp_report.txt" 2>/dev/null
echo
echo "(完整每阵表 + 类别聚合: $OUT/methods_cmp_report.txt)"
echo "对比表: $CSV | 柱状图: $OUT/methods_cmp_bar.png"
