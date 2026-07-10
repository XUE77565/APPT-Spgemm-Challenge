#!/usr/bin/env python3
"""
A·Aᵀ 上三角 profiling:解析 `./spgemm_test <mtx> att` 日志(results/rep_att/log/)。
5 法:cuSPARSE 全量 + 外积/Gustavson/列向/内积(上三角)。
正确性:4 个上三角法 nnz 应一致,且 2×upper−diag == cuSPARSE 全量(diag 由关系式推)。
"""
import re
import os
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
SURF, P1, P2, MUTE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASE = "#c3c2b7"
C_CU, C_OUT, C_GUST, C_COL, C_INN = "#2a78d6", "#eda100", "#eb6834", "#008300", "#4a3aa7"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": P1, "axes.labelcolor": P2, "axes.titlecolor": P1, "axes.edgecolor": BASE,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": .8, "axes.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold", "figure.dpi": 130,
})

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.environ.get("ATT_LOG_DIR", str(REPO / "results" / "rep_att" / "log")))
REPS_CSV = Path(__file__).resolve().parent / "representatives.csv"
OUT_CSV = Path(__file__).resolve().parent / "profile_att.csv"
CHART_DIR = Path(__file__).resolve().parent / "charts"
CHART_DIR.mkdir(exist_ok=True)
# att 方法顺序(与 main.cu att 一致):0=cuSPARSE全量, 1=outer, 2=gust, 3=colw, 4=inner
TAGS = ["cu", "outer", "gust", "colw", "inner"]
NAME = {"cu": "cuSPARSE(全量)", "outer": "外积(上三角)", "gust": "Gustavson(上三角)",
        "colw": "列向(上三角)", "inner": "内积(上三角)"}
# att 阶段 tag(日志里的 [xxx])
PTAG = {"cu": "cut", "outer": "atto", "gust": "attg", "colw": "attc", "inner": "atti"}
ALL_PHASES = ["h2d", "transpose", "csc", "count", "scan", "expand", "compact",
              "workest", "compute", "copy", "sort", "reduce", "final", "numeric", "pack", "d2h"]
PORDER = {"cu": ["h2d", "transpose", "workest", "compute", "copy", "pack", "d2h"],
          "outer": ["h2d", "csc", "count", "scan", "expand", "sort", "reduce", "final", "pack", "d2h"],
          "gust": ["h2d", "csc", "count", "scan", "expand", "compact", "sort", "reduce", "final", "pack", "d2h"],
          "colw": ["h2d", "csc", "count", "scan", "expand", "compact", "sort", "reduce", "final", "pack", "d2h"],
          "inner": ["h2d", "csc", "count", "scan", "expand", "compact", "sort", "reduce", "final", "numeric", "pack", "d2h"]}
PGROUP = {"h2d": "传输", "d2h": "传输", "transpose": "符号", "csc": "符号", "count": "符号", "scan": "符号",
          "expand": "计算", "workest": "计算", "compute": "计算", "copy": "计算",
          "sort": "合并", "reduce": "合并", "final": "合并", "compact": "合并",
          "numeric": "数值归并", "pack": "打包"}
GORDER = ["传输", "符号", "计算", "合并", "数值归并", "打包"]
GCOLOR = {"传输": "#898781", "符号": "#1baf7a", "计算": "#2a78d6", "合并": "#eda100",
          "数值归并": "#e34948", "打包": "#4a3aa7"}


def parse_log(path):
    times, cnnz, phases = [], [], {}
    a_rows = a_nnz = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[dbg\s+([\d.]+)\s+ms\]\s+(.*)", line)
        if m:
            ms, msg = float(m.group(1)), m.group(2).strip()
            mp = re.match(r"\[(cut|atto|attg|attc|atti)\]\s+(\w+)", msg)
            if mp:
                phases[(mp.group(1), mp.group(2))] = ms
            mr = re.search(r"(\d+)\s*x\s*\d+,\s*nnz=(\d+)", msg)  # read done
            if mr:
                a_rows, a_nnz = int(mr.group(1)), int(mr.group(2))
        else:
            mt = re.search(r"Time:\s+([\d.]+)\s+ms", line)
            if mt:
                times.append(float(mt.group(1)))
            mi = re.search(r"Input A:\s*(\d+)\s*x\s*\d+,\s*nnz\s*=\s*([\d,]+)", line)
            if mi:
                a_rows, a_nnz = int(mi.group(1)), int(mi.group(2).replace(",", ""))
            mc = re.search(r"Result C.*nnz\s*=\s*([\d,]+)", line)
            if mc:
                cnnz.append(int(mc.group(1).replace(",", "")))
    def t(i): return times[i] if len(times) > i else np.nan
    def c(i): return cnnz[i] if len(cnnz) > i else np.nan
    return {"n": a_rows, "A_nnz": a_nnz,
            "cu_time": t(0), "outer_time": t(1), "gust_time": t(2), "colw_time": t(3), "inner_time": t(4),
            "full_nnz": c(0), "upper_outer": c(1), "upper_gust": c(2), "upper_colw": c(3), "upper_inner": c(4),
            "phases": phases}


def method_durations(phases, m):
    pt = PTAG[m]
    order = ["start"] + PORDER[m]
    present = [p for p in order if (pt, p) in phases]
    if len(present) < 2:
        return {}
    ts = [phases[(pt, p)] for p in present]
    return {present[i]: ts[i] - ts[i - 1] for i in range(1, len(present))}


def main():
    reps = {r["name"]: r for r in pd.read_csv(REPS_CSV).to_dict("records")}
    rows = []
    for log in sorted(LOG_DIR.glob("*.log")):
        p = parse_log(log)
        p["name"] = log.stem
        r = reps.get(log.stem, {})
        p["class"] = r.get("class", "?")
        p["density_pct"] = r.get("density_pct", np.nan)
        rows.append(p)
    df = pd.DataFrame(rows)
    # 上三角法一致性 + 推 diag
    ucols = ["upper_outer", "upper_gust", "upper_colw", "upper_inner"]
    df["upper_agree"] = df[ucols].apply(lambda r: int(len(set(r.dropna())) <= 1), axis=1)
    df["upper_nnz"] = df["upper_outer"]
    df["implied_diag"] = 2 * df["upper_nnz"] - df["full_nnz"]
    df.to_csv(OUT_CSV, index=False)
    order = {"Dense": 0, "Mildly sparse": 1, "Highly sparse": 2, "Extremely sparse": 3}
    df["_o"] = df["class"].map(order).fillna(9)
    df = df.sort_values(["_o", "cu_time"]).drop(columns="_o")

    print("=" * 116)
    print(f"{'class':<4} {'name':<26}{'n':>9}{'cuSPARSE':>10}{'外积':>9}{'Gust':>9}{'列向':>9}{'内积':>9}"
          f"{'全量nnz':>12}{'上三角nnz':>12}{'一致':>5}")
    print("-" * 116)
    def fm(v, w=9):
        return "  --" if pd.isna(v) else f"{v:.2f}".rjust(w)
    for _, r in df.iterrows():
        print(f"{str(r['class'])[:4]:<4} {r['name']:<26}{(str(int(r['n'])) if pd.notna(r['n']) else '--'):>9}"
              f"{fm(r['cu_time'],10)}{fm(r['outer_time'])}{fm(r['gust_time'])}{fm(r['colw_time'])}{fm(r['inner_time'])}"
              f"{(f'{int(r['full_nnz']):,}' if pd.notna(r['full_nnz']) else '--'):>12}"
              f"{(f'{int(r['upper_nnz']):,}' if pd.notna(r['upper_nnz']) else '--'):>12}"
              f"{'✓' if r['upper_agree'] else '✗':>5}")

    print("\n" + "=" * 70)
    print("按类别聚合(均值,ms)")
    print("-" * 70)
    print(f"{'class':<18}{'#':>3}{'cuSPARSE':>10}{'外积':>9}{'Gust':>9}{'列向':>9}{'内积':>9}")
    for c in ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]:
        s = df[df["class"] == c]
        if not len(s):
            continue
        g = lambda col: "--" if pd.isna(s[col].mean()) else f"{s[col].mean():.1f}".rjust(9)
        print(f"{c:<18}{len(s):>3}{g('cu_time').rjust(10) if pd.notna(s['cu_time'].mean()) else '--':>10}"
              f"{g('outer_time')}{g('gust_time')}{g('colw_time')}{g('inner_time')}")

    # 阶段构成
    print("\n" + "=" * 78)
    print("分阶段构成(各矩阵均值,ms)")
    print("-" * 78)
    print(f"{'方法':<16}" + "".join(f"{g:>10}" for g in GORDER))
    agg = {}
    for m in TAGS:
        gvals = {}
        for g in GORDER:
            cols = [p for p in ALL_PHASES if PGROUP.get(p) == g]
            tot = []
            for _, r in df.iterrows():
                d = method_durations(r.get("phases", {}) or {}, m)
                if d:
                    tot.append(sum(d.get(p, 0) for p in cols))
            gvals[g] = np.mean(tot) if tot else 0.0
        agg[m] = gvals
        parts = "".join(("    --   " if (pd.isna(v) or v == 0) else f"{v:>10.2f}") for v in (gvals[g] for g in GORDER))
        print(f"{NAME[m]:<16}{parts}")

    # 堆叠图
    present = [m for m in TAGS if m in agg]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(present))
    bot = np.zeros(len(present))
    for g in GORDER:
        vals = np.array([agg[m].get(g, 0) or 0 for m in present])
        if np.nanmax(vals) <= 0:
            continue
        ax.barh(y, vals, left=bot, color=GCOLOR[g], height=0.6, label=g, zorder=3)
        bot += np.nan_to_num(vals)
    ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in present], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("耗时 (ms, 各矩阵均值)")
    ax.set_title("A·Aᵀ 上三角 各公式阶段构成(cuSPARSE 全量 vs 上三角四法)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(CHART_DIR / "profile_att.png", bbox_inches="tight"); plt.close(fig)
    print(f"\n写出 {OUT_CSV} ; 图 charts/profile_att.png")


if __name__ == "__main__":
    main()
