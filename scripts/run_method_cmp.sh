#!/bin/bash
# ===========================================================================
#  方法对比脚本(仿 run_full_compare.sh)
#  对比:Ocean / HSMU / dense-baseline(-O0) / merge(serial) / merge3 / Auto
#    Auto = src/spgemm_adaptive.cu 的自适应(flop_proxy=A_nnz²/n >1e8 → hash,否则 merge3)
#  每矩阵:METHOD=<m> 跑 spgemm_test 取该方法时间 + ocean/convert+spgemm + HSMU test
#  → 增量 CSV + 几何均值。
#
#  注意:HSMU 的 test binary 会追加写 /tmp/*.csv(figure 的 gen_sota_clean.py 要读),
#  本脚本跑前备份、跑后(含中断)恢复,确保不影响 figure。
#
#  用法:
#    bash scripts/run_method_cmp.sh                        # 全跑 first100
#    LIMIT=10 bash scripts/run_method_cmp.sh               # 只前 10 阵(冒烟)
#    TIMEOUT=300 bash scripts/run_method_cmp.sh            # 缩短每方法/矩阵超时
#    MATRIX_DIR=./data/sota_27_final bash scripts/run_method_cmp.sh
#    NO_OCEAN=1 bash scripts/run_method_cmp.sh             # 跳过 Ocean
#    NO_HSMU=1  bash scripts/run_method_cmp.sh             # 跳过 HSMU
# ===========================================================================
set -o pipefail
cd "$(dirname "$(readlink -f "$0")")/.."   # 脚本在 scripts/ 下,cd 回仓库根

TIMEOUT=${TIMEOUT:-200}
export TIMEOUT   # 透传给 compare_methods.py 的 per-call 超时
PY=${PY:-.venv/bin/python}
MATRIX_DIR=${MATRIX_DIR:-./data/first100}
LIMIT=${LIMIT:-0}
NO_OCEAN=${NO_OCEAN:-0}
NO_HSMU=${NO_HSMU:-0}
NO_DENSE=${NO_DENSE:-0}
TS=$(date +%Y%m%d_%H%M%S)
OUT="compare/method_cmp_${TS}"
mkdir -p "$OUT"

# ---- HSMU /tmp CSV 备份/恢复(figure 要读,不能被污染)----
HSMU_CSVS=(NHC_4080S_result small_step_runtime new_compressed_step_runtime small_time_conversion)
BACKUP="/tmp/hsmu_backup_${TS}"
mkdir -p "$BACKUP"
restore_hsmu() {
  echo "### 恢复 HSMU /tmp CSV(从 $BACKUP)###"
  for f in "${HSMU_CSVS[@]}"; do [ -f "$BACKUP/$f.csv" ] && cp "$BACKUP/$f.csv" "/tmp/$f.csv"; done
}
trap restore_hsmu EXIT INT TERM

echo "############################################################"
echo "#  SpGEMM 方法对比  →  $OUT/"
echo "#  Ocean / HSMU / dense-baseline(-O0) / merge(serial) / merge3 / Auto"
echo "#  矩阵: $MATRIX_DIR  $([ $LIMIT -gt 0 ] && echo "(前 $LIMIT 阵)" || echo "(全部)")"
echo "############################################################"

echo "### [1/4] 编译 spgemm_test(DBG=1,cudaEvent phase 计时,含 hash + Auto) + spgemm_dense(-O0 baseline)###"
make DBG=1 dense >/dev/null 2>&1 || { echo "  ✗ make 失败"; exit 1; }
echo "  ✓ done(spgemm 法 = cudaEvent compute-only[同 Ocean 口径];dense baseline = -O0 cudaEvent kernel)"

echo "### [1b/4] dense baseline cache 检查(缺失则生成,一次性 ~10min)###"
if [ -f compare/dense_baseline.csv ] && [ "$(($(wc -l < compare/dense_baseline.csv)-1))" -gt 0 ]; then
  echo "  ✓ cache 已存在: compare/dense_baseline.csv ($(($(wc -l < compare/dense_baseline.csv)-1)) 阵)"
else
  echo "  cache 缺失 → 生成 dense baseline(-O0,first100 全阵)..."
  $PY -u scripts/run_dense_baseline.py 2>&1 | tail -3
fi

echo "### [2/4] 备份 HSMU /tmp CSV → $BACKUP ###"
for f in "${HSMU_CSVS[@]}"; do [ -f "/tmp/$f.csv" ] && cp "/tmp/$f.csv" "$BACKUP/"; done
echo "  ✓ 备份 $(ls "$BACKUP" 2>/dev/null | wc -l) 个"

echo "### [3/4] 对比(Ocean+HSMU+dense[cache]+serial+merge3+Auto,每方法/矩阵超时 ${TIMEOUT}s)###"
ARGS="--out $OUT/methods_cmp.csv --dir $MATRIX_DIR"
[ "$LIMIT" -gt 0 ]      && ARGS="$ARGS --limit $LIMIT"
[ "$NO_OCEAN" -eq 1 ]   && ARGS="$ARGS --no-ocean"
[ "$NO_HSMU" -eq 1 ]    && ARGS="$ARGS --no-hsmu"
[ "$NO_DENSE" -eq 1 ]   && ARGS="$ARGS --no-dense"
$PY -u scripts/compare_methods.py $ARGS 2>&1 | tee "$OUT/run.log" \
  || { echo "  ✗ compare_methods.py 失败(见 $OUT/run.log)"; exit 1; }
echo "  ✓ 对比表: $OUT/methods_cmp.csv ($(($(wc -l < "$OUT/methods_cmp.csv")-1)) 阵)"

echo "### [4/4] 报告 + 柱状图(仿 full_compare_K=5.txt + profile_methods_bar.png)###"
# 报告:每阵表 + 类别聚合 + Auto/m3 输给 Ocean(主产物)
$PY scripts/report_methods_cmp.py "$OUT/methods_cmp.csv" "$OUT/methods_cmp_report.txt" >/dev/null 2>&1 \
  && echo "  ✓ 报告: $OUT/methods_cmp_report.txt" || echo "  ✗ 报告失败"
# 柱状图
$PY scripts/plot_methods_cmp.py "$OUT/methods_cmp.csv" "$OUT/methods_cmp_bar.png" >/dev/null 2>&1 \
  && [ -f "$OUT/methods_cmp_bar.png" ] && echo "  ✓ 图: $OUT/methods_cmp_bar.png" || echo "  ✗ 画图失败"

echo
echo "############################################################"
echo "#  完成!产物在 $OUT/  (HSMU /tmp CSV 由 EXIT trap 已恢复)"
echo "############################################################"
ls -1 "$OUT"
echo
echo "================ 报告(每阵表 + 类别聚合 + 输给 Ocean)================"
sed -n '/compute-only/,/另有/p' "$OUT/methods_cmp_report.txt" 2>/dev/null | head -40
echo
echo "(完整报告: $OUT/methods_cmp_report.txt)"
echo "对比表: $OUT/methods_cmp.csv | 柱状图: $OUT/methods_cmp_bar.png"