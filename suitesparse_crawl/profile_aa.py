#!/usr/bin/env python3
"""
Profiling run_rep.sh 的结果:解析每个日志里的 dbg 阶段时间戳,
拆解 cuSPARSE 与 manual 自乘的各阶段耗时,按稀疏类别(D/MS/HS/ES)聚合,找瓶颈。

阶段(cuSPARSE T1): workEstimation / compute / copy / D2H / 写盘
阶段(manual  T4): count kernel / fill kernel / 写盘
"""

import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK).exists():
    fm.fontManager.addfont(_CJK)
SURFACE, INK_PRI, INK_SEC, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE = "#e1e0d9", "#c3c2b7"
C_CU, C_MAN = "#2a78d6", "#eb6834"   # cuSPARSE 蓝 / manual 橙
C_KERNEL, C_D2H, C_WRITE, C_OTHER = "#2a78d6", "#1baf7a", "#e34948", "#898781"
CLASS_COLOR = {"Dense": "#2a78d6", "Mildly sparse": "#eb6834",
               "Highly sparse": "#008300", "Extremely sparse": "#4a3aa7"}
CLASS_ORDER = ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK_PRI, "axes.labelcolor": INK_SEC, "axes.titlecolor": INK_PRI,
    "axes.edgecolor": BASELINE, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.8,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 130,
})

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
# scripts/run_aa.sh 把日志写到 results/aa/first100_aa/log;可用 argv[1] 或 AA_LOG_DIR 覆盖
LOG_DIR = Path(sys.argv[1] if len(sys.argv) > 1
              else os.environ.get("AA_LOG_DIR",
                                  str(REPO / "results" / "aa" / "first100_aa" / "log")))
OUT_CSV = HERE / "profile_aa_summary.csv"   # 每矩阵汇总(time/nnz/class)
PHASES_CSV = HERE / "profile_aa.csv"        # 每矩阵×方法 分阶段明细
CHART_DIR = HERE / "charts"
CHART_DIR.mkdir(exist_ok=True)


# 稀疏度分类(与 classify_by_sparsity.py 阈值一致);直接从日志解析的 n/A_nnz 算,
# 覆盖全部 100 个矩阵 —— 不再依赖只含 32 个代表的 representatives.csv。
# 注:read_matrix_market 不展开对称,故日志里的 A_nnz 与 SuiteSparse 元数据一致。
def classify_density(density_pct):
    if pd.isna(density_pct):
        return "?"
    if density_pct >= 10.0:
        return "Dense"
    if density_pct >= 1.0:
        return "Mildly sparse"
    if density_pct >= 0.1:
        return "Highly sparse"
    return "Extremely sparse"


def parse_log(path):
    ev = {}      # event -> ms
    buf2 = None
    a_nnz = a_rows = None
    times = []        # stdout "Time:"(cuSPARSE, manual)
    result_cnnz = []  # stdout "Result C: nnz="(cuSPARSE, manual)
    phases = {}       # (tag, phase) -> ms(取最后一次 = 计时那次,跳过 warmup)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[dbg\s+([\d.]+)\s+ms\]\s+(.*)", line)
        if m:
            ms, msg = float(m.group(1)), m.group(2).strip()
            tag = None
            if msg.startswith("read done"):
                mm = re.search(r"(\d+)\s*x\s*\d+,\s*nnz=(\d+)", msg)
                if mm:
                    a_rows, a_nnz = int(mm.group(1)), int(mm.group(2))
            elif "T1 cuSPARSE self_product: start" in msg: tag = "t1_start"
            elif "cusparse: workEstimation begin" in msg: tag = "we_b"
            elif "cusparse: workEstimation done" in msg:
                tag = "we_d"; mm = re.search(r"buf1=(\d+)", msg)
            elif "cusparse: compute begin" in msg: tag = "co_b"
            elif "cusparse: compute done" in msg:
                tag = "co_d"; mm = re.search(r"buf2=(\d+)", msg)
                if mm: buf2 = int(mm.group(1))
            elif "cusparse: copy begin" in msg: tag = "cp_b"
            elif "cusparse: copy done" in msg: tag = "cp_d"
            elif "C D2H begin" in msg: tag = "d2_b"
            elif "C D2H done" in msg: tag = "d2_d"
            elif "T1 cuSPARSE self_product: done" in msg:
                tag = "t1_d"; mm = re.search(r"C_nnz=(\d+)", msg)
                if mm: ev["cu_cnnz"] = int(mm.group(1))
            elif "T1 writing" in msg: tag = "t1w_b"
            elif "T1 write done" in msg: tag = "t1w_d"
            elif "T4 manual self_product: start" in msg: tag = "t4_start"
            elif "manual: count kernel begin" in msg: tag = "cnt_b"
            elif "manual: count kernel done" in msg: tag = "cnt_d"
            elif "manual: fill kernel begin" in msg: tag = "fil_b"
            elif "manual: fill kernel done" in msg: tag = "fil_d"
            elif "T4 manual self_product: done" in msg:
                tag = "t4_d"; mm = re.search(r"C_nnz=(\d+)", msg)
                if mm: ev["man_cnnz"] = int(mm.group(1))
            elif "T4 writing" in msg: tag = "t4w_b"
            elif "T4 write done" in msg: tag = "t4w_d"
            if tag:
                ev[tag] = ms
            mph = re.match(r"\[(cu|gust|outer|colw|inner|merge|mrg2|mrg3)\]\s+(\w+)", msg)
            if mph:
                phases[(mph.group(1), mph.group(2))] = ms
        else:
            mt = re.search(r"^Time:\s+([\d.]+)\s+ms", line)
            if mt:
                times.append(float(mt.group(1)))
            mi = re.search(r"Input A:\s*(\d+)\s*x\s*\d+,\s*nnz\s*=\s*([\d,]+)", line)
            if mi:
                a_rows, a_nnz = int(mi.group(1)), int(mi.group(2).replace(",", ""))
            mc = re.search(r"Result C:.*nnz\s*=\s*([\d,]+)", line)
            if mc:
                result_cnnz.append(int(mc.group(1).replace(",", "")))

    def d(a, b):
        return (ev[b] - ev[a]) if (a in ev and b in ev) else np.nan
    return {
        "n": a_rows, "A_nnz": a_nnz,
        "cu_we": d("we_b", "we_d"), "cu_compute": d("co_b", "co_d"),
        "cu_copy": d("cp_b", "cp_d"), "cu_d2h": d("d2_b", "d2_d"),
        "cu_total": d("t1_start", "t1_d"),
        "cu_time": times[0] if len(times) >= 1 else np.nan,
        "cu_write": d("t1w_b", "t1w_d"),
        "man_count": d("cnt_b", "cnt_d"), "man_fill": d("fil_b", "fil_d"),
        "man_total": d("t4_start", "t4_d"),
        "man_time": times[1] if len(times) >= 2 else np.nan,
        "man_write": d("t4w_b", "t4w_d"),
        # T4b:串行 k-way merge(times[2]=merge 总耗时,result_cnnz[2]=merge 的 nnz)
        "merge_time": times[2] if len(times) >= 3 else np.nan,
        # T4c:并行 k-way merge v2(times[3],result_cnnz[3])
        "merge2_time": times[3] if len(times) >= 4 else np.nan,
        # T4d: merge3(均匀分桶)
        "merge3_time": times[4] if len(times) >= 5 else np.nan,
        "cu_cnnz": result_cnnz[0] if len(result_cnnz) >= 1 else ev.get("cu_cnnz", np.nan),
        "man_cnnz": result_cnnz[1] if len(result_cnnz) >= 2 else ev.get("man_cnnz", np.nan),
        "merge_cnnz": result_cnnz[2] if len(result_cnnz) >= 3 else np.nan,
        "merge2_cnnz": result_cnnz[3] if len(result_cnnz) >= 4 else np.nan,
        "merge3_cnnz": result_cnnz[4] if len(result_cnnz) >= 5 else np.nan,
        "buf2": buf2,
        "phases": phases,
    }


PHASE_ORDER = ["h2d", "csc", "count", "scan", "expand", "merge", "compact",
               "sort", "reduce", "final", "numeric", "pack", "d2h"]
CU_PHASES = ["h2d", "workest", "compute", "copy", "pack", "d2h"]
ALL_PHASES = ["h2d", "csc", "count", "scan", "expand", "merge", "compact",
              "workest", "compute", "copy", "sort", "reduce", "final", "numeric", "pack", "d2h"]
METHOD_NAME = {"cu": "cuSPARSE", "gust": "Gustavson", "merge": "Merge(ser)", "mrg2": "Merge(par)", "mrg3": "Merge3"}
# 阶段分组(用于堆叠图与汇总表);cuSPARSE 的 workest/compute/copy 归入"计算"
# 传输拆成 h2d(上传 A)与 d2h(下载 C)两列,便于分别看
PHASE_GROUP = {
    "h2d": "h2d", "d2h": "d2h",
    "csc": "符号", "count": "符号", "scan": "符号",
    "expand": "计算", "workest": "计算", "compute": "计算", "copy": "计算",
    "sort": "排序", "reduce": "去重", "final": "收尾",
    # merge 法:用 "merge" 阶段替代 sort+reduce(归并去重);映射到「排序」列以便与 gust.sort 直接对照
    "merge": "排序", "compact": "收尾",
    "numeric": "数值归并", "pack": "打包",
}
GROUP_ORDER = ["h2d", "d2h", "符号", "计算", "排序", "去重", "收尾", "数值归并", "打包"]
GROUP_COLOR = {"h2d": "#b8b8b0", "d2h": "#6f6f68",   # 上传浅灰 / 下载深灰(同属"传输")
               "符号": "#1baf7a", "计算": "#2a78d6",
               "排序": "#eda100", "去重": "#c98a1e", "收尾": "#a86620",   # 合并三段:黄→深黄→棕黄
               "数值归并": "#e34948", "打包": "#4a3aa7"}


def phase_order_for(tag):
    return CU_PHASES if tag == "cu" else PHASE_ORDER


def method_durations(phases, tag):
    """从 [tag] phase 时间戳算该方法的各阶段耗时(ms)。
    按真实时间戳排序(而非 PHASE_ORDER 标签顺序),这样各法执行顺序不同
    (如 merge 的 final(scan)在 compact 之前,ESC 的 final 在 reduce 之后)也能正确归位。"""
    order = ["start"] + phase_order_for(tag)
    present = [p for p in order if (tag, p) in phases]
    if len(present) < 2:
        return {}
    # 按时间戳升序排,保证相邻相减得正、归属正确
    present = sorted(present, key=lambda p: phases[(tag, p)])
    ts = [phases[(tag, p)] for p in present]
    return {present[i]: ts[i] - ts[i - 1] for i in range(1, len(present))}


def phase_breakdown(df):
    TAGS = ["cu", "gust", "merge", "mrg2", "mrg3"]
    rows = []
    for _, r in df.iterrows():
        ph = r.get("phases", {}) or {}
        for tag in TAGS:
            d = method_durations(ph, tag)
            if not d:
                continue
            row = {"name": r["name"], "class": r["class"], "tag": tag,
                   "n": r["n"], "A_nnz": r["A_nnz"]}
            for p in ALL_PHASES:
                row[p] = d.get(p, np.nan)
            rows.append(row)
    if not rows:
        print("\n(无 [tag] phase 数据:确认 DBG=1 且日志带分阶段桩)")
        return
    pdf = pd.DataFrame(rows)
    pdf.to_csv(PHASES_CSV, index=False)

    # ---- 按方法聚合:分组成 h2d/d2h/符号/计算/合并/数值归并/打包(均值 ms)----
    print("\n" + "=" * 112)
    print("分阶段构成(各矩阵均值,ms)— 按阶段分组,六法可比")
    print("-" * 112)
    print(f"{'方法':<12}" + "".join(f"{g:>10}" for g in GROUP_ORDER) + f"{'合计':>10}")
    agg = {}
    for tag in TAGS:
        sub = pdf[pdf["tag"] == tag]
        if len(sub) == 0:
            continue
        gvals = {}
        for g in GROUP_ORDER:
            cols = [p for p in ALL_PHASES if PHASE_GROUP.get(p) == g]
            gvals[g] = float(sub[cols].sum(axis=1).mean()) if cols else 0.0
        agg[tag] = gvals
        tot = sum(gvals.values())
        parts = [("    --   " if (pd.isna(gvals[g]) or gvals[g] == 0) else f"{gvals[g]:>10.2f}")
                 for g in GROUP_ORDER]
        print(f"{METHOD_NAME[tag]:<12}" + "".join(parts) + f"{tot:>10.2f}")

    # ---- 堆叠图:每方法的阶段构成 ----
    present_tags = [t for t in TAGS if t in agg]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(present_tags))
    bottom = np.zeros(len(present_tags))
    for g in GROUP_ORDER:
        vals = np.array([agg[t].get(g, 0.0) or 0.0 for t in present_tags])
        if np.nanmax(vals) <= 0:
            continue
        ax.barh(y, vals, left=bottom, color=GROUP_COLOR[g], height=0.6,
                label=g, zorder=3)
        bottom += np.nan_to_num(vals)
    ax.set_yticks(y); ax.set_yticklabels([METHOD_NAME[t] for t in present_tags])
    ax.invert_yaxis()
    ax.set_xlabel("耗时 (ms, 各矩阵均值)")
    ax.set_title("各 SpGEMM 公式的阶段构成(分阶段打桩,含 cuSPARSE)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "profile_aa.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n分阶段明细: profile_aa.csv ; 图: charts/profile_aa.png")


def chart_methods_bar(df):
    """7 方法(cuBLAS/Ocean/cuSPARSE/ESC/merge-ser/merge-par/merge3)compute-only 均值,按稀疏类别分组柱(log y)。"""
    methods = [("cuBLAS", "cublas_ms", "#2a78d6"),
               ("Ocean", "ocean_ms", "#4a3aa7"),
               ("cuSPARSE", "cu_compute_t", "#898781"),
               ("gust(ESC)", "man_compute_t", "#eb6834"),
               ("merge(ser)", "merge_compute_t", "#c98a1e"),
               ("merge(par)", "merge2_compute_t", "#1baf7a"),
               ("merge3", "merge3_compute_t", "#e0533d")]
    classes = [c for c in CLASS_ORDER if len(df[df["class"] == c])]
    if not classes:
        return
    means = {}
    for lab, col, _ in methods:
        means[lab] = []
        for c in classes:
            sub = df[df["class"] == c]
            vals = sub[col].dropna() if col in sub.columns else pd.Series(dtype=float)
            means[lab].append(float(vals.mean()) if len(vals) else np.nan)
    x = np.arange(len(classes))
    nm = len(methods)
    w = 0.13
    fig, ax = plt.subplots(figsize=(11, 5.0))
    for i, (lab, _, color) in enumerate(methods):
        vals = np.array([v if not np.isnan(v) else 1e-3 for v in means[lab]])
        ax.bar(x + (i - (nm - 1) / 2) * w, vals, width=w, color=color, label=lab, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({len(df[df['class'] == c])})" for c in classes])
    ax.set_ylabel("耗时 (ms, 对数, compute-only)")
    ax.set_title("7 方法 compute-only 对照(按稀疏类别):cuBLAS / Ocean / cuSPARSE / ESC / merge(ser) / merge(par) / merge3")
    ax.legend(frameon=False, fontsize=8.5, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, 1.00))  # 图例放图顶,避免挡柱
    ax.grid(axis="y", which="both", color="#e1e0d9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "profile_methods_bar.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  图: charts/profile_methods_bar.png  (7 方法 compute-only 柱状图;具体数值见上面的类别聚合表)")


def print_lose_ocean(df):
    """merge(merge2/merge3)哪些 case 输给 Ocean,按比值降序。"""
    print("\n" + "=" * 100)
    print("merge vs Ocean:输给 Ocean 的 case(ratio > 1.0,越小越好)")
    print("=" * 100)
    for label, col, ocean_col in [("merge2(par)", "merge2_compute_t", "ocean_ms"),
                                  ("merge3(bucket)", "merge3_compute_t", "ocean_ms")]:
        d = df.dropna(subset=[col, ocean_col]).copy()
        d["ratio"] = d[col] / d[ocean_col]
        losers = d[d["ratio"] > 1.0].sort_values("ratio", ascending=False)
        winners = d[d["ratio"] < 1.0]
        geo = np.exp(np.log(d["ratio"]).mean())
        print(f"\n--- {label}:赢 {len(winners)} / 输 {len(losers)},几何均值 {geo:.3f}× ---")
        if len(losers) == 0:
            print("  (全部赢 Ocean)")
        else:
            print(f"{'name':<14}{'n':>7}{'C_nnz':>10}{'Ocean':>9}{label:>14}{'ratio':>8}")
            print("-" * 70)
            for _, r in losers.head(20).iterrows():
                print(f"{r['name']:<14}{int(r['n']):>7}{int(r['cu_cnnz']):>10}"
                      f"{r[ocean_col]:>9.3f}{r[col]:>14.3f}{r['ratio']:>7.2f}×")
            if len(losers) > 20:
                print(f"  ... 另有 {len(losers)-20} 个")


def main():
    rows = []
    for log in sorted(LOG_DIR.glob("*.log")):
        p = parse_log(log)
        p["name"] = log.stem
        rows.append(p)
    df = pd.DataFrame(rows)
    # 稀疏度分类:直接从日志的 n/A_nnz 算,不再 join representatives.csv
    df["density_pct"] = df["A_nnz"] / (df["n"].astype(float) ** 2) * 100.0
    df["class"] = df["density_pct"].apply(classify_density)

    # cuSPARSE 各阶段:aa 日志只打 begin 不打 done,parse_log 里 cu_we/cu_compute/cu_copy/
    # cu_d2h(begin/done 配对)解析为 NaN。这里改从 [cu] phase 时间戳取(method_durations),
    # 与 profile_aa.csv 分阶段明细一致,使 cu_kernel 与 cuSPARSE 拆解图可用。
    def _cu_phase(phases, ph):
        return method_durations(phases or {}, "cu").get(ph, np.nan)
    ph = df["phases"]
    df["cu_we"] = df["cu_we"].fillna(ph.apply(lambda p: _cu_phase(p, "workest")))
    df["cu_compute"] = df["cu_compute"].fillna(ph.apply(lambda p: _cu_phase(p, "compute")))
    df["cu_copy"] = df["cu_copy"].fillna(ph.apply(lambda p: _cu_phase(p, "copy")))
    df["cu_d2h"] = df["cu_d2h"].fillna(ph.apply(lambda p: _cu_phase(p, "d2h")))

    # 正确性:gust / merge / merge2 vs cuSPARSE 的 nnz 偏差
    df["gust_gap"] = (df["cu_cnnz"] - df["man_cnnz"]).abs()
    df["merge_gap"] = (df["cu_cnnz"] - df["merge_cnnz"]).abs()
    df["merge2_gap"] = (df["cu_cnnz"] - df["merge2_cnnz"]).abs()
    df["cnnz_gap_pct"] = df["gust_gap"] / df["cu_cnnz"] * 100
    df["cu_kernel"] = df[["cu_we", "cu_compute", "cu_copy"]].sum(axis=1, min_count=1)
    df["man_kernel"] = df[["man_count", "man_fill"]].sum(axis=1, min_count=1)

    # compute-only 时间(各 dbg 阶段求和,去掉 h2d/d2h 传输):与 cuBLAS(kernel-only)公平对比
    def _compute_only(phases, tag):
        dd = method_durations(phases or {}, tag)
        return float(sum(v for p, v in dd.items() if p not in ("h2d", "d2h")))
    df["cu_compute_t"]    = df["phases"].apply(lambda p: _compute_only(p, "cu"))
    df["man_compute_t"]   = df["phases"].apply(lambda p: _compute_only(p, "gust"))
    df["merge_compute_t"] = df["phases"].apply(lambda p: _compute_only(p, "merge"))
    df["merge2_compute_t"]= df["phases"].apply(lambda p: _compute_only(p, "mrg2"))
    df["merge3_compute_t"]= df["phases"].apply(lambda p: _compute_only(p, "mrg3"))

    cols = ["class", "name", "n", "A_nnz", "density_pct",
            "cu_cnnz", "man_cnnz", "merge_cnnz", "merge2_cnnz",
            "merge3_cnnz", "cnnz_gap_pct", "buf2",
            "cu_we", "cu_compute", "cu_copy", "cu_d2h", "cu_kernel", "cu_time",
            "cu_write", "man_count", "man_fill", "man_kernel", "man_time",
            "merge_time", "merge2_time", "merge3_time", "man_write",
            "cu_compute_t", "man_compute_t", "merge_compute_t", "merge2_compute_t",
            "merge3_compute_t",
            "phases"]
    df = df[[c for c in cols if c in df.columns]]
    # cuBLAS 稠密 GEMM baseline(Python ctypes 直调,新主 baseline;cuSPARSE 降为旧 baseline)
    cub_path = HERE / "baseline_cublas.csv"
    if cub_path.exists():
        df = df.merge(pd.read_csv(cub_path)[["name", "cublas_ms"]], on="name", how="left")
    else:
        df["cublas_ms"] = np.nan
    # Ocean SpGEMM baseline(compute-only = GPU 阶段求和)
    oce_path = HERE / "baseline_ocean.csv"
    if oce_path.exists():
        df = df.merge(pd.read_csv(oce_path)[["name", "ocean_ms"]], on="name", how="left")
    else:
        df["ocean_ms"] = np.nan
    df = df.sort_values(["density_pct", "name"], ascending=[False, True])
    df.to_csv(OUT_CSV, index=False)
    print(f"写出 {OUT_CSV}\n")

    # ---- 每矩阵明细(compute-only):cuSPARSE / Ocean / gust(ESC) / mrgP(merge2) / m3(merge3) + 比值 ----
    print("=" * 116)
    print("[compute-only:稀疏法=各阶段求和去 h2d/d2h;Ocean=GPU 阶段求和]  ms")
    print(f"{'class':<4} {'name':<22}{'n':>7}"
          f"{'cuSP':>8}{'Ocean':>8}{'gust':>7}{'mrgP':>7}{'m3':>7}{'mP/Oce':>8}{'m3/cu':>7}{'m3/Oce':>8}")
    print("-" * 116)
    for _, r in df.iterrows():
        def f(v, w=8, p=False):
            if pd.isna(v): return "  --".rjust(w)
            return f"{v:{'.2f' if p else ',.0f'}}".rjust(w)
        def rr(a, b):
            return (a / b) if (pd.notna(a) and pd.notna(b) and b > 0) else np.nan
        mp = r['merge2_compute_t']
        m3 = r['merge3_compute_t']
        cu = r['cu_compute_t']
        r_oce_p = rr(mp, r['ocean_ms'])
        r_cu_3   = rr(m3, cu)
        r_oce_3 = rr(m3, r['ocean_ms'])
        print(f"{str(r['class'])[:4]:<4} {r['name']:<22}{f(r['n'],7)}"
              f"{f(cu,8,'t')}{f(r['ocean_ms'],8,'t')}{f(r['man_compute_t'],7,'t')}{f(mp,7,'t')}{f(m3,7,'t')}"
              f"{f(r_oce_p,8,'t')}{f(r_cu_3,7,'t')}{f(r_oce_3,8,'t')}")

    # ---- 按类别聚合(compute-only)----
    print("\n" + "=" * 116)
    print("按类别聚合(compute-only 均值,毫秒)")
    print("-" * 116)
    print(f"{'class':<18}{'#':>3}{'cuBLAS':>10}{'Ocean':>8}{'cuSPARSE':>10}{'gust(ESC)':>10}{'mrg(ser)':>9}{'mrgP':>8}{'mrg3':>8}"
          f"{'par/Oce':>8}{'m3/cu':>7}{'m3/Oce':>8}")
    def g(s, w, dec=1):
        v = s.mean()
        return "--".rjust(w) if np.isnan(v) else f"{v:.{dec}f}".rjust(w)
    # 比值用几何均值(算术均值会被 bp_* 等离群点带偏,得出相反结论)
    def gm(series):
        v = series.dropna()
        return np.exp(np.log(v).mean()) if len(v) else np.nan
    for c in CLASS_ORDER:
        s = df[df["class"] == c]
        if len(s) == 0:
            continue
        mp = s['merge2_compute_t']
        m3 = s['merge3_compute_t']
        cu = s['cu_compute_t']
        r_oce_p = gm(mp / s['ocean_ms'])
        r_cu_3   = gm(m3 / cu)
        r_oce_3 = gm(m3 / s['ocean_ms'])
        print(f"{c:<18}{len(s):>3}"
              f"{g(s['cublas_ms'],10)}{g(s['ocean_ms'],8)}{g(s['cu_compute_t'],10)}{g(s['man_compute_t'],10)}"
              f"{g(s['merge_compute_t'],9)}{g(s['merge2_compute_t'],8)}{g(s['merge3_compute_t'],8)}"
              f"{g(pd.Series([r_oce_p]),8,2)}{g(pd.Series([r_cu_3]),7,2)}{g(pd.Series([r_oce_3]),8,2)}")

    charts(df)
    phase_breakdown(df)
    print()  # 7 方法 compute-only 柱状图
    chart_methods_bar(df)

    # ---- merge vs Ocean 专项:哪些 case 输给 Ocean ----
    print_lose_ocean(df)

    print(f"\n图表在 {CHART_DIR}")


def charts(df):
    d = df.dropna(subset=["cu_time"]).copy()
    d = d.sort_values(["density_pct", "name"], ascending=[False, True])
    labels = [f"{r['name']}\n[{r['class'][:2]}]" for _, r in d.iterrows()]

    # 图1: 两种方法总GPU耗时对照(cuSPARSE / Gustavson)
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(d) + 1.5))
    y = np.arange(len(d))
    for shift, col, lab, c in [(-0.12, "cu_time", "cuSPARSE", C_CU),
                               (0.12, "man_time", "Gustavson", C_MAN)]:
        ax.barh(y + shift, d[col], height=0.25, color=c, label=lab, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("GPU 耗时 (ms, 对数)")
    ax.set_xscale("log"); ax.set_title("cuSPARSE vs Gustavson 自乘总耗时对照(不含写盘)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(CHART_DIR / "profile_totals.png", bbox_inches="tight"); plt.close(fig)

    # 图2 仅在有 dbg 阶段数据时绘制(DBG=0 的日志没有)
    if not (d["cu_kernel"].notna().any() and d["cu_write"].notna().any()):
        print("  (跳过 profile_cu_breakdown.png:日志无 dbg 阶段时间戳,需 DBG=1 重跑)")
        return

    # 图2: cuSPARSE 全墙钟拆解(计算 kernel / D2H / 写盘 / 其它)
    d2 = d.copy()
    other = (d2["cu_time"] - d2["cu_kernel"] - d2["cu_d2h"]).clip(lower=0)
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(d2) + 1.5))
    y = np.arange(len(d2))
    left = np.zeros(len(d2))
    for vals, color, lab in [(d2["cu_kernel"], C_KERNEL, "SpGEMM kernel(we+compute+copy)"),
                             (d2["cu_d2h"], C_D2H, "D2H 回传"),
                             (other, C_OTHER, "其余(H2D/分配)"),
                             (d2["cu_write"], C_WRITE, "写盘 .mtx")]:
        vals = vals.fillna(0)
        ax.barh(y, vals, left=left, color=color, height=0.7, label=lab, zorder=3)
        left += vals
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("墙钟耗时 (ms, 对数)")
    ax.set_title("cuSPARSE 全流程墙钟拆解(kernel / D2H / 其余 / 写盘)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(CHART_DIR / "profile_cu_breakdown.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()