#!/bin/bash
# ============================================================================
# 一键重生成"有无 pinned+arena"对比图:
#   build → 跑 USE_MEMPOOL=0(无 arena) → profile 出 log
#         → 跑 USE_MEMPOOL=1(有 arena) → profile 出 log
#         → compare_pinned.py 合成 per_method_compare.png
#
# 用法:bash scripts/regen_pinned_compare.sh
#       TIMEOUT=120 bash scripts/regen_pinned_compare.sh   # 覆盖每矩阵超时
# ============================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
CRAWL=suitesparse_crawl
OUT=compare/pinned-and-arena
NO_DIR=results/aa/no_pinned        # USE_MEMPOOL=0 的日志
YES_DIR=results/aa/pinned          # USE_MEMPOOL=1 的日志

echo "### build ###"
make 2>&1 | grep -vE "warning #177-D" > /tmp/regen_build.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "BUILD FAILED"; cat /tmp/regen_build.log; exit 1
fi
mkdir -p "$OUT"

echo "### [1/2] no-pinned+arena  (USE_MEMPOOL=0) ###"
USE_MEMPOOL=0 AA_RESULTS_DIR="$NO_DIR" bash scripts/run_aa.sh > /tmp/run_nopin.out 2>&1
echo "    logs: $(ls $NO_DIR/log/*.log | wc -l)"
$PY "$CRAWL/profile_aa.py" "$NO_DIR/log" > "$OUT/no-pinned+arena.log" 2>&1
echo "    -> $OUT/no-pinned+arena.log"

echo "### [2/2] pinned+arena  (USE_MEMPOOL=1) ###"
USE_MEMPOOL=1 AA_RESULTS_DIR="$YES_DIR" bash scripts/run_aa.sh > /tmp/run_pin.out 2>&1
echo "    logs: $(ls $YES_DIR/log/*.log | wc -l)"
$PY "$CRAWL/profile_aa.py" "$YES_DIR/log" > "$OUT/pinned+arena.log" 2>&1
echo "    -> $OUT/pinned+arena.log"

echo "### 合成对比图 ###"
$PY "$CRAWL/compare_pinned.py" \
    "$OUT/no-pinned+arena.log" "$OUT/pinned+arena.log" "$OUT/per_method_compare.png"

echo "### done ###"
echo "对比图:$OUT/per_method_compare.png"
