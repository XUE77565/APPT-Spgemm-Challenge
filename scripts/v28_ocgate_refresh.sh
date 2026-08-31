#!/bin/bash
# v28-OCCGATE 刷新(净窗用;在 reverify 排空后跑):Auto 列全 337,OCCGATE=1,
# 基底 = v27 拷贝,Ocean 列保持 v9 干净口径。nnz 红旗由 compare_methods 内建。
# 用法:bash scripts/v28_ocgate_refresh.sh [--force](跳过负载门)
set -e
cd "$(dirname "$0")/.."
la=$(awk '{print $1}' /proc/loadavg)
if [ "$1" != "--force" ] && awk -v l="$la" 'BEGIN{exit !(l>=6)}'; then
  echo "✗ load=$la ≥6(非净窗)——v28 刷新只在净窗跑;--force 强制作业"; exit 2
fi
echo "✓ 净窗(load=$la),v28-OCCGATE Auto 列刷新开始 $(date +%T)"
cp -n compare/ocean337/methods_cmp_v27_harness.csv compare/ocean337/methods_cmp_v28_ocgate.csv 2>/dev/null || true
REFRESH=Auto OUT_CSV=compare/ocean337/methods_cmp_v28_ocgate OCCGATE=1 MP_HOST_MB=8192 \
  bash scripts/run_method_cmp.sh
echo "完成 $(date +%T);报告:.venv/bin/python scripts/report_v2X_clean.py compare/ocean337/methods_cmp_v28_ocgate.csv 2>/dev/null || true"
