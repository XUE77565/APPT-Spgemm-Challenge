#!/bin/bash
# ============================================================================
# 一键对比:你的 Gustavson(ESC sort)vs cuSPARSE vs Ocean(hash)—— 纯计算(不含 IO)
# 跑全 100 矩阵,按 C_nnz 分桶,生成阶段拆解堆叠柱状图(对数轴,每块标 µs+%)
#
# 用法:
#   bash scripts/run_compare_ocean.sh
#   TIMEOUT=120 bash scripts/run_compare_ocean.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
CRAWL=suitesparse_crawl
OCEAN=ocean
OUT=compare/three_method_profiling
TIMEOUT=${TIMEOUT:-600}

mkdir -p "$OUT" /tmp/ocean_csr

echo "### [1/4] Build ###"
make -j8 2>&1 | grep -iE "error" && exit 1 || true
cd "$OCEAN" && make -j8 2>&1 | grep -iE "error" && exit 1 || true
cd ..

echo "### [2/4] 跑 spgemm_test(USE_MEMPOOL=1,100 矩阵)###"
CMP_DIR=results/aa/cmp
rm -rf "$CMP_DIR"; mkdir -p "$CMP_DIR/log"
total_mats=$(ls data/first100/*.mtx | wc -l)
i2=0
for mtx in data/first100/*.mtx; do
  name=$(basename "$mtx" .mtx); i2=$((i2+1))
  printf "  [spgemm_test %3d/%d] %-22s" "$i2" "$total_mats" "$name"
  USE_MEMPOOL=1 timeout "$TIMEOUT" ./spgemm_test "$mtx" > "$CMP_DIR/log/${name}.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then echo " OK"; else echo " FAIL($rc)"; fi
done
$PY "$CRAWL/profile_aa.py" "$CMP_DIR/log" > /tmp/cmp_profile.out 2>&1
echo "    profile 完成($i2 矩阵)"

echo "### [3/4] 跑 Ocean bench_detail(全 100 矩阵,收阶段拆解)###"
echo "name,cnnz,analysis,estimation,numeric,epilogue,total" > /tmp/ocean_phases_all.csv
i=0; total=$(ls data/first100/*.mtx | wc -l)
for mtx in data/first100/*.mtx; do
  name=$(basename "$mtx" .mtx); i=$((i+1))
  csr="/tmp/ocean_csr/$name.csr"
  [ -f "$csr" ] || "$OCEAN/convert" "$mtx" "$csr" >/dev/null 2>&1
  timeout "$TIMEOUT" "$OCEAN/spgemm" "$csr" "$OCEAN/config/bench_detail.json" >/dev/null 2>&1
  $PY -c "
import json
try:
    s=json.load(open('ocean/stats.json'))
    t=s['timing']
    an=t['analysis']['product_calc']+t['analysis']['reduce']+t['analysis']['mem_cpy']
    est=t['estimation']['hll_construct']+t['estimation']['hll_merge']+t['estimation']['malloc']+t['estimation']['sampling']
    num=sum(t['numeric'].values())
    epi=t['epilogue']['sort']+t['epilogue']['copy']+t['epilogue']['scan']
    tot=an+est+num+epi+t['prologue']
    cnnz=s.get('C',{}).get('nnz',0)
    print(f'$name,{cnnz},{an*1000:.1f},{est*1000:.1f},{num*1000:.1f},{epi*1000:.1f},{tot*1000:.1f}')
except Exception as e:
    print(f'$name,0,0,0,0,0,0')
" >> /tmp/ocean_phases_all.csv 2>/dev/null
  printf "  [%3d/%d] %s\n" "$i" "$total" "$name"
done

echo "### [4/4] 生成对比图(全 100 分桶,对数堆叠柱)###"
$PY - <<'PYEOF'
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK): fm.fontManager.addfont(_CJK)
plt.rcParams.update({
    "figure.facecolor":"#fcfcfb","axes.facecolor":"#fcfcfb","savefig.facecolor":"#fcfcfb",
    "text.color":"#0b0b0b","axes.labelcolor":"#52514e","axes.titlecolor":"#0b0b0b",
    "axes.edgecolor":"#c3c2b7","xtick.color":"#898781","ytick.color":"#898781",
    "axes.grid":True,"grid.color":"#e1e0d9","grid.linewidth":0.8,
    "axes.linewidth":0.8,"axes.spines.top":False,"axes.spines.right":False,
    "font.family":["Noto Sans CJK SC","DejaVu Sans"],"axes.unicode_minus":False,
    "font.size":9,"figure.dpi":140,
})

ROOT = "."
OUT = f"{ROOT}/compare/three_method_profiling"
os.makedirs(OUT, exist_ok=True)

# === 加载数据 ===
ph = pd.read_csv(f"{ROOT}/suitesparse_crawl/profile_aa.csv", na_values=[""])
sm = pd.read_csv(f"{ROOT}/suitesparse_crawl/profile_aa_summary.csv", na_values=[""])
ocean_raw = pd.read_csv("/tmp/ocean_phases_all.csv")

# C_nnz 统一用 cu_cnnz(从 summary)
cnnz_map = sm[["name","cu_cnnz"]].dropna()

# 给三方的数据都标上 cnnz,按 log10(cnnz) 分桶
bucket_labels = ["小\n(C<1K)", "中\n(1K-10K)", "大\n(10K-100K)", "巨大\n(C>100K)"]
bucket_bins = [0, 3, 4, 5, 99]

# cu/gust:加 cnnz 列
ph_c = ph.merge(cnnz_map, on="name", how="inner")
ph_c["bucket"] = pd.cut(np.log10(ph_c["cu_cnnz"].clip(lower=1)),
                         bins=bucket_bins, labels=bucket_labels)
# ocean:加 cnnz 列
ocean_raw["bucket"] = pd.cut(np.log10(ocean_raw["cnnz"].clip(lower=1)),
                               bins=bucket_bins, labels=bucket_labels)

def mean_by_bucket(df, bucket_col, val_col):
    return [df[df[bucket_col]==b][val_col].fillna(0).mean() for b in bucket_labels]

# === 定义三法阶段 ===
methods = [
    ("cuSPARSE\n(不含IO)", [
        ("workest", lambda t: mean_by_bucket(ph_c[ph_c.tag=="cu"], "bucket", "workest"), "#1baf7a"),
        ("compute", lambda t: mean_by_bucket(ph_c[ph_c.tag=="cu"], "bucket", "compute"), "#2a78d6"),
        ("copy",    lambda t: mean_by_bucket(ph_c[ph_c.tag=="cu"], "bucket", "copy"), "#eda100"),
        ("pack",    lambda t: mean_by_bucket(ph_c[ph_c.tag=="cu"], "bucket", "pack"), "#7f5f99"),
    ], "#2a78d6"),
    ("Gustavson\n(ESC sort)", [
        ("count+scan", lambda t: [a+b for a,b in zip(
            mean_by_bucket(ph_c[ph_c.tag=="gust"], "bucket", "count"),
            mean_by_bucket(ph_c[ph_c.tag=="gust"], "bucket", "scan"))], "#1baf7a"),
        ("expand", lambda t: mean_by_bucket(ph_c[ph_c.tag=="gust"], "bucket", "expand"), "#2a78d6"),
        ("sort",   lambda t: mean_by_bucket(ph_c[ph_c.tag=="gust"], "bucket", "sort"), "#e34948"),
        ("reduce", lambda t: mean_by_bucket(ph_c[ph_c.tag=="gust"], "bucket", "reduce"), "#eda100"),
        ("final",  lambda t: mean_by_bucket(ph_c[ph_c.tag=="gust"], "bucket", "final"), "#a86620"),
    ], "#eb6834"),
    ("Ocean\n(hash)", [
        ("analysis", lambda t: mean_by_bucket(ocean_raw, "bucket", "analysis"), "#1baf7a"),
        ("HLL",      lambda t: mean_by_bucket(ocean_raw, "bucket", "estimation"), "#7f5f99"),
        ("numeric",  lambda t: mean_by_bucket(ocean_raw, "bucket", "numeric"), "#2a78d6"),
        ("epilogue", lambda t: mean_by_bucket(ocean_raw, "bucket", "epilogue"), "#a86620"),
    ], "#1baf7a"),
]

n_buckets = len(bucket_labels)
n_methods = len(methods)
bar_w = 0.25
group_gap = 0.15

fig, ax = plt.subplots(figsize=(16, 6.5))
x_centers = np.arange(n_buckets) * (n_methods * bar_w + group_gap)

for mi, (method_name, phases, method_color) in enumerate(methods):
    # 先算出所有阶段值
    phase_data = []
    for label, func, color in phases:
        vals = np.array(func(None)) * 1000  # 转 µs
        phase_data.append((label, vals, color))

    for bi in range(n_buckets):
        x_pos = x_centers[bi] + (mi - 1) * bar_w
        bottom = 0
        total = sum(pd[1][bi] for pd in phase_data)

        for label, vals, color in phase_data:
            val = vals[bi]
            pct = val / total * 100 if total > 0 else 0
            ax.bar(x_pos, val, bar_w, bottom=bottom, color=color, edgecolor="white",
                   linewidth=0.5, zorder=3)
            if val > total * 0.08 and val > 5:
                ax.text(x_pos, bottom + val / 2,
                        f"{val:.0f}\n{pct:.0f}%",
                        ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
            bottom += val

        ax.text(x_pos, total * 1.03, f"{total:.0f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=method_color)

# 每桶矩阵数
counts = []
for b in bucket_labels:
    c = len(ph_c[(ph_c.tag=="cu") & (ph_c.bucket==b)])
    counts.append(c)

ax.set_xticks(x_centers)
ax.set_xticklabels([f"{b}\n(#{n})" for b, n in zip(bucket_labels, counts)], fontsize=10)
ax.set_yscale("log")
ax.set_ylabel("计算耗时 (µs, 对数轴)")
ax.set_title("三种 SpGEMM 阶段拆解对比(全 100 矩阵分桶均值,纯计算:cuSPARSE 不含 H2D/D2H)",
             fontweight="bold", fontsize=11)
ax.grid(axis="x", visible=False)
ax.set_ylim(10, 50000)

# 方法名标在 x 轴下方
for bi in range(n_buckets):
    for mi, (method_name, _, method_color) in enumerate(methods):
        x_pos = x_centers[bi] + (mi - 1) * bar_w
        short = method_name.split("\n")[0]
        ax.text(x_pos, 8, short, ha="center", va="top", fontsize=7, color=method_color, fontweight="bold")

# 图例
all_handles = []
for method_name, phases, _ in methods:
    for label, _, color in phases:
        all_handles.append(Patch(facecolor=color, edgecolor="white",
                                  label=f"{method_name.split(chr(10))[0]}: {label}"))
ax.legend(handles=all_handles, frameon=False, fontsize=6.5, loc="upper left", ncol=1)

fig.tight_layout()
fig.savefig(f"{OUT}/three_method_phases.png", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT}/three_method_phases.png")

# 保存数据
ocean_raw.to_csv(f"{OUT}/ocean_phases.csv", index=False)
ph_c[ph_c.tag.isin(["cu","gust"])].to_csv(f"{OUT}/profile_aa_phases.csv", index=False)
print(f"saved: {OUT}/ocean_phases.csv + profile_aa_phases.csv")

# 打印摘要
print("\n=== 摘要(各桶计算均值 µs)===")
for bi, b in enumerate(bucket_labels):
    cu_total = sum(pd[1][bi] for pd in methods[0][1]) * 1000  # wrong, already in µs
    # 直接从 phase_data 重算
    row = []
    for mi, (method_name, phases, _) in enumerate(methods):
        phase_data = []
        for label, func, color in phases:
            vals = np.array(func(None)) * 1000
            phase_data.append(vals[bi])
        tot = sum(phase_data)
        row.append(f"{method_name.split(chr(10))[0]}={tot:.0f}µs")
    print(f"  {b.replace(chr(10),' ')} (#{counts[bi]}): " + "  ".join(row))

print("done")
PYEOF

echo "### 全部完成 ###"
echo "图表在: $OUT/"
ls -la "$OUT/"
