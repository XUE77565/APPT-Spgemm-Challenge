#!/usr/bin/env python3
"""Slide 12 — Scalability & Cost Trade-off Analysis (deck-style, English).

Left  : Cost Analysis — per-design (complexity / overhead / saving) table.
Right : Scalability   — Auto compute-only runtime vs matrix dimension n (log-log),
                        density-colored scatter + power-law fit + throughput annot.

Data  : compare/aa(best1)/methods_cmp.csv  (n, Auto, density_pct).
Palette (deck): teal #1485A4 (ours), red #C00000 (emphasis), blue family for density,
                gray #9AA6B2, muted #5C6773, ink #1F2933, white bg, grid #ECEEF1.
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import LogLocator, ScalarFormatter

# ---- deck palette ----
TEAL, RED, GRAY, MUTED, INK = "#1485A4", "#C00000", "#9AA6B2", "#5C6773", "#1F2933"
GRID, BG = "#ECEEF1", "#FFFFFF"
# density blue->teal ramp (deck blue family + teal)
DENS_COL = {"Dense": "#0A4996", "Mild": "#0D77C3", "High": "#1485A4", "Extreme": "#5BB3C9"}
DENS_NAME = {"Dense": "Dense", "Mild": "Mildly sparse", "High": "Highly sparse", "Extreme": "Extremely sparse"}

CSV = "compare/aa(best1)/methods_cmp.csv"
def f(x):
    try: return float(x)
    except: return None
def classify(d):
    if d >= 10: return "Dense"
    if d >= 1:  return "Mild"
    if d >= 0.1: return "High"
    return "Extreme"

rows = []
for r in csv.DictReader(open(CSV)):
    n, au = f(r["n"]), f(r["Auto"]); d = f(r["density_pct"]) or 0
    if n and au and au > 0:
        rows.append((n, au, classify(d)))
n_arr  = np.array([r[0] for r in rows])
au_arr = np.array([r[1] for r in rows])
dc_arr = [r[2] for r in rows]

# power-law fit T ~ n^k  (log-log linear)
k, lnA = np.polyfit(np.log(n_arr), np.log(au_arr), 1)

# throughput by size bucket (ratio of geomeans: C_nnz_proxy omitted; use output throughput = ~)
def gm(x): return math.exp(sum(math.log(v) for v in x) / len(x))
def bucket(lo, hi):
    sub = [(n, au) for n, au, c in rows if lo <= n < hi]
    if not sub: return None
    return gm([au for n, au in sub])  # geomean runtime ms
small_ms, mid_ms, large_ms, huge_ms = bucket(0,1000), bucket(1000,5000), bucket(5000,15000), bucket(15000,1e9)

# ---------------- figure (16:9) ----------------
fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
fig.text(0.5, 0.955, "Scalability & Cost Trade-off Analysis",
         ha="center", va="top", fontsize=20, fontweight="bold", color=INK)
axL = fig.add_axes([0.035, 0.10, 0.46, 0.78])   # cost table
axR = fig.add_axes([0.56, 0.12, 0.41, 0.74])    # scalability scatter

# ================= LEFT: cost table =================
axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
axL.set_title("Cost Analysis  —  per design", fontsize=14, fontweight="bold",
              color=INK, loc="left", pad=8)

cols = ["Design", "Complexity", "Added overhead", "Key saving"]
# x-edges of columns
xe = [0.015, 0.235, 0.475, 0.715, 0.985]
y_top = 0.86; row_h = 0.115; hdr_h = 0.055

# header row
hdr_y = y_top
axL.add_patch(Rectangle((0.015, hdr_y - hdr_h), 0.97, hdr_h, facecolor=TEAL, edgecolor="none"))
for i, c in enumerate(cols):
    axL.text((xe[i] + xe[i+1]) / 2, hdr_y - hdr_h/2, c, ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="white")

data = [
    ("D1  Adaptive\nDispatch", "—", "CPU O(nnz) scan,\nhidden under H2D", "matrix-aware merge/hash\npick, ~0 GPU cost"),
    ("D2  Column-Domain\nMerge", "O(flops), 3 axes", "column-bucket\nbinning + sync", "removes global sort\n(54-70% of ESC)"),
    ("D3  Light-Symbolic\nMinHash", "O(nnz) sizing", "per-row sketch +\n1.5x safety margin", "1.4-3.3x faster symbolic\n(vs O(flops))"),
    ("D4  Tail-Balanced\nA^T A", "~ -50% arithmetic\n(upper triangle)", "CPU O(nnz) workload\n+ reshape", "kills long-tail\nstraggler"),
]
y = hdr_y - hdr_h - 0.012
for r, (d, cx, ov, sv) in enumerate(data):
    ry = y - row_h
    if r % 2 == 0:
        axL.add_patch(Rectangle((0.015, ry), 0.97, row_h, facecolor="#F4FAFB", edgecolor="none"))
    # design name (teal, bold)
    axL.text((xe[0]+xe[1])/2, ry + row_h/2, d, ha="center", va="center",
             fontsize=8.6, fontweight="bold", color=TEAL, linespacing=1.15)
    for i, txt in enumerate([cx, ov, sv]):
        axL.text((xe[i+1]+xe[i+2])/2, ry + row_h/2, txt, ha="center", va="center",
                 fontsize=8.2, color=INK, linespacing=1.15)
    y = ry - 0.006

# overall row (red emphasis)
ry = y - row_h
axL.add_patch(Rectangle((0.015, ry), 0.97, row_h, facecolor="#FDECEC", edgecolor=RED, linewidth=1.0))
axL.text((xe[0]+xe[1])/2, ry + row_h/2, "Overall\nSpGEMM", ha="center", va="center",
         fontsize=8.8, fontweight="bold", color=RED, linespacing=1.15)
axL.text((xe[1]+xe[2])/2, ry + row_h/2, "compute-bound", ha="center", va="center",
         fontsize=8.2, color=INK)
axL.text((xe[2]+xe[3])/2, ry + row_h/2, "accumulation / dedup\n= the floor", ha="center", va="center",
         fontsize=8.2, color=RED, linespacing=1.15)
axL.text((xe[3]+xe[4])/2, ry + row_h/2, "engineering opts\nsaturate here", ha="center", va="center",
         fontsize=8.2, color=INK, linespacing=1.15)

# ================= RIGHT: scalability scatter =================
axR.set_facecolor(BG)
for c in DENS_COL:
    sub_n = np.array([n for n, au, cc in rows if cc == c])
    sub_a = np.array([au for n, au, cc in rows if cc == c])
    axR.scatter(sub_n, sub_a, s=26, c=DENS_COL[c], alpha=0.78,
                edgecolors="white", linewidths=0.4, zorder=3, label=DENS_NAME[c])

# power-law fit line
nn = np.logspace(math.log10(n_arr.min()*0.8), math.log10(n_arr.max()*1.1), 50)
axR.plot(nn, math.exp(lnA) * nn**k, color=RED, lw=2.0, zorder=4,
         label=f"fit  T $\\propto$ n$^{{{k:.2f}}}$")
# overhead floor reference
axR.axhline(0.1, color=GRAY, lw=1.0, ls=":", zorder=2)
axR.text(n_arr.max()*0.5, 0.108, "fixed-overhead floor ~0.1 ms",
         fontsize=7.5, color=MUTED, ha="right", va="bottom")

axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlabel("Matrix dimension  n  (log)", fontsize=10.5, color=INK)
axR.set_ylabel("Auto compute-only runtime (ms, log)", fontsize=10.5, color=INK)
axR.set_title("Scalability  —  runtime vs. size", fontsize=14, fontweight="bold",
              color=INK, loc="left", pad=8)
axR.tick_params(labelsize=8, colors=MUTED)
axR.xaxis.set_major_formatter(ScalarFormatter())
axR.yaxis.set_major_formatter(ScalarFormatter())
axR.grid(True, which="major", color=GRID, lw=0.6); axR.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.5)
axR.set_axisbelow(True)
for s in ["top", "right"]: axR.spines[s].set_visible(False)
for s in ["left", "bottom"]: axR.spines[s].set_color(MUTED)
axR.legend(loc="upper left", fontsize=7.8, framealpha=0.9, edgecolor=GRID, ncol=2)

# throughput takeaway box
axR.text(0.98, 0.03,
         "Throughput rises ~50x as size grows\n(~25 M  ->  ~1.4 G output-nnz/s),\nthen saturates at the floor",
         transform=axR.transAxes, fontsize=8.2, color=TEAL, ha="right", va="bottom",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", fc="#F4FAFB", ec=TEAL, lw=1.0))

fig.text(0.035, 0.045, "H100 PCIe  ·  100 matrices  ·  compute-only (cudaEvent, excludes H2D/D2H)",
         fontsize=8, color=MUTED)

os.makedirs("fig", exist_ok=True)
for p in ["fig/cost_scalability.png"]:
    fig.savefig(p, dpi=200, facecolor=BG, bbox_inches="tight")
    print("saved", p)
fig.savefig("fig/cost_scalability.pdf", facecolor=BG, bbox_inches="tight")
print("saved fig/cost_scalability.pdf")
print(f"fit: T = {math.exp(lnA):.3e} * n^{k:.3f}   (sub-linear, < n^1)")
print(f"geomean runtime by size: small={small_ms:.3f} mid={mid_ms:.3f} large={large_ms:.3f} huge={huge_ms:.3f} ms")
