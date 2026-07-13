#!/bin/bash
# ============================================================================
# A/B profiling(交错版):每个矩阵 legacy / pool 【背靠背】各跑一次。
#
# 为什么交错:旧版"先跑完 100 个 legacy、再跑完 100 个 pool",两趟 sweep 相隔
# 几分钟,GPU 状态(温度/boost clock/OS)漂移会污染亚毫秒级的小测量(h2d 等),
# 让 pool "显得"更慢——其实是后跑的那趟吃了漂移的亏。交错后每个矩阵的 legacy/
# pool 只相隔一次运行的时间,环境配对,漂移抵消 → 小阶段数据也可信。
#
# 同一二进制、同一批矩阵,只差 USE_MEMPOOL 这一个开关。
# 产物(分目录、分文件名,互不覆盖):
#   results/aa/first100_aa_legacy/  + profile_aa_legacy.csv
#   results/aa/first100_aa_pool/    + profile_aa_pool.csv
# 末尾打印 d2h/h2d/合计 对比表(+ 写日志 + 出图)。
#
# 用法:bash scripts/ab_profile.sh
#       TIMEOUT=120 bash scripts/ab_profile.sh
# ============================================================================
set -uo pipefail                      # 注意:不开 -e,单个矩阵失败不中断
cd "$(dirname "$(readlink -f "$0")")/.."

PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
CRAWL=suitesparse_crawl
DATA=./data/first100
LEG_DIR=./results/aa/first100_aa_legacy
POOL_DIR=./results/aa/first100_aa_pool
TIMEOUT=${TIMEOUT:-600}

echo "### build ###"
make 2>&1 | grep -vE "warning #177-D" > /tmp/ab_build.log
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "### BUILD FAILED ###"; cat /tmp/ab_build.log; exit 1
fi
echo "    (build ok / up-to-date)"

echo "### 交错 A/B:每矩阵 legacy→pool 背靠背,data=$DATA ###"
rm -rf "$LEG_DIR" "$POOL_DIR"
mkdir -p "$LEG_DIR/log" "$POOL_DIR/log"
total_all=$(ls "$DATA"/*.mtx 2>/dev/null | wc -l)

ok0=0; ok1=0; fail=0; total=0
for mtx in "$DATA"/*.mtx; do
    [ -e "$mtx" ] || continue
    name=$(basename "$mtx" .mtx)
    total=$((total+1))
    # 背靠背:先 legacy,紧跟 pool(同一 GPU 状态下配对)
    USE_MEMPOOL=0 timeout -k 10 "$TIMEOUT" ./spgemm_test "$mtx" > "$LEG_DIR/log/$name.log" 2>&1; rc0=$?
    USE_MEMPOOL=1 timeout -k 10 "$TIMEOUT" ./spgemm_test "$mtx" > "$POOL_DIR/log/$name.log" 2>&1; rc1=$?
    s0=$([ $rc0 -eq 0 ] && echo OK || echo "rc$rc0")
    s1=$([ $rc1 -eq 0 ] && echo OK || echo "rc$rc1")
    [ $rc0 -eq 0 ] && ok0=$((ok0+1))
    [ $rc1 -eq 0 ] && ok1=$((ok1+1))
    [ $rc0 -ne 0 -o $rc1 -ne 0 ] && fail=$((fail+1))
    printf "  [%3d/%d] %-22s legacy:%-5s pool:%-5s\n" "$total" "$total_all" "$name" "$s0" "$s1"
done
echo "### done: legacy OK=$ok0  pool OK=$ok1  (failures=$fail) ###"

# profile 两路 → 各自 CSV
for tag in legacy pool; do
    dir=$([ "$tag" = legacy ] && echo "$LEG_DIR" || echo "$POOL_DIR")
    $PY "$CRAWL/profile_aa.py" "$dir/log" > /tmp/profile_$tag.out 2>&1
    mv "$CRAWL/profile_aa.csv"         "$CRAWL/profile_aa_$tag.csv"
    mv "$CRAWL/profile_aa_summary.csv" "$CRAWL/profile_aa_summary_$tag.csv"
done

echo "### 对比 ###"
$PY "$CRAWL/compare_ab.py" "$CRAWL/profile_aa_legacy.csv" "$CRAWL/profile_aa_pool.csv"
