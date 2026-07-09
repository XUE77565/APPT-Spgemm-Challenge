#!/usr/bin/env python3
"""
Profiling run_rep.sh 的结果:解析每个日志里的 dbg 阶段时间戳,
拆解 cuSPARSE 与 manual 自乘的各阶段耗时,按稀疏类别(D/MS/HS/ES)聚合,找瓶颈。

阶段(cuSPARSE T1): workEstimation / compute / copy / D2H / 写盘
阶段(manual  T4): count kernel / fill kernel / 写盘
"""

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
LOG_DIR = REPO / "results" / "rep" / "log"
REPS_CSV = Path(__file__).resolve().parent / "representatives.csv"
OUT_CSV = Path(__file__).resolve().parent / "profile_rep.csv"
CHART_DIR = Path(__file__).resolve().parent / "charts"
CHART_DIR.mkdir(exist_ok=True)


def parse_log(path):
    ev = {}      # event -> ms
    buf2 = None
    a_nnz = a_rows = None
    times = []        # stdout "Time:"(cuSPARSE, manual)
    result_cnnz = []  # stdout "Result C: nnz="(cuSPARSE, manual)
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
        "cu_cnnz": result_cnnz[0] if len(result_cnnz) >= 1 else ev.get("cu_cnnz", np.nan),
        "man_cnnz": result_cnnz[1] if len(result_cnnz) >= 2 else ev.get("man_cnnz", np.nan),
        "buf2": buf2,
    }


def main():
    reps = {r["name"]: r for r in pd.read_csv(REPS_CSV).to_dict("records")}
    rows = []
    for log in sorted(LOG_DIR.glob("*.log")):
        name = log.stem
        p = parse_log(log)
        p["name"] = name
        r = reps.get(name, {})
        p["class"] = r.get("class", "?")
        p["density_pct"] = r.get("density_pct", np.nan)
        rows.append(p)
    df = pd.DataFrame(rows)

    # 正确性:manual 丢的非零(cuSPARSE 为基准)
    df["cnnz_gap"] = df["cu_cnnz"] - df["man_cnnz"]
    df["cnnz_gap_pct"] = df["cnnz_gap"] / df["cu_cnnz"] * 100
    df["cu_kernel"] = df[["cu_we", "cu_compute", "cu_copy"]].sum(axis=1, min_count=1)
    df["man_kernel"] = df[["man_count", "man_fill"]].sum(axis=1, min_count=1)

    cols = ["class", "name", "n", "A_nnz", "density_pct", "cu_cnnz", "man_cnnz",
            "cnnz_gap", "cnnz_gap_pct", "buf2",
            "cu_we", "cu_compute", "cu_copy", "cu_d2h", "cu_kernel", "cu_time",
            "cu_write", "man_count", "man_fill", "man_kernel", "man_time", "man_write"]
    df = df[[c for c in cols if c in df.columns]]
    # 按稀疏度排列:密度从高到低(稠密→稀疏,即 D→MS→HS→ES)
    df = df.sort_values(["density_pct", "name"], ascending=[False, True])
    df.to_csv(OUT_CSV, index=False)
    print(f"写出 {OUT_CSV}\n")

    # ---- 每矩阵明细(关键列) ----
    print("=" * 100)
    print(f"{'class':<4} {'name':<30}{'n':>9}{'A_nnz':>10}{'cu_time':>10}"
          f"{'man_time':>10}{'cu_comp':>9}{'cu_d2h':>8}{'cu_write':>10}{'gap%':>8}")
    print("-" * 100)
    for _, r in df.iterrows():
        def f(v, w=9, p=False):
            if pd.isna(v): return "  --".rjust(w)
            return f"{v:{'.2f' if p else ',.0f'}}".rjust(w)
        print(f"{str(r['class'])[:4]:<4} {r['name']:<30}{f(r['n'],9)}"
              f"{f(r['A_nnz'],10)}{f(r['cu_time'],10,'t')}{f(r['man_time'],10,'t')}"
              f"{f(r['cu_compute'],9,'t')}{f(r['cu_d2h'],8,'t')}{f(r['cu_write'],10,'t')}"
              f"{f(r['cnnz_gap_pct'],8,'t')}")

    # ---- 按类别聚合 ----
    print("\n" + "=" * 70)
    print("按类别聚合(均值,毫秒)")
    print("-" * 70)
    print(f"{'class':<18}{'#':>3}{'cu_time':>10}{'man_time':>11}{'cu_compute':>12}"
          f"{'cu_write':>11}{'gap%':>9}")
    def g(s, w, dec=1):
        v = s.mean()
        return "--".rjust(w) if np.isnan(v) else f"{v:.{dec}f}".rjust(w)
    for c in CLASS_ORDER:
        s = df[df["class"] == c]
        if len(s) == 0:
            continue
        print(f"{c:<18}{len(s):>3}"
              f"{g(s['cu_time'],10)}{g(s['man_time'],11)}"
              f"{g(s['cu_compute'],12)}{g(s['cu_write'],11)}"
              f"{g(s['cnnz_gap_pct'],9,2)}")

    charts(df)
    print(f"\n图表在 {CHART_DIR}")


def charts(df):
    d = df.dropna(subset=["cu_time"]).copy()
    d = d.sort_values(["density_pct", "name"], ascending=[False, True])
    labels = [f"{r['name']}\n[{r['class'][:2]}]" for _, r in d.iterrows()]

    # 图1: cuSPARSE vs manual 总GPU耗时
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["cu_time"], height=0.4, color=C_CU, label="cuSPARSE", zorder=3)
    ax.barh(y + 0.2, d["man_time"], height=0.4, color=C_MAN, label="manual", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("GPU 耗时 (ms, 对数)")
    ax.set_xscale("log"); ax.set_title("cuSPARSE vs manual 自乘总耗时(不含写盘)")
    ax.legend(frameon=False, fontsize=9)
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
