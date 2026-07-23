#!/usr/bin/env python3
"""Generate publication-quality chart + performance model for the SpGEMM adaptive dispatcher paper.

Chart: scatter plot (Auto vs Ocean, log-log, colored by hash/merge3 choice).
Model: roofline-style regression on H100 PCIe characteristics.

Usage: .venv/bin/python scripts/gen_paper_chart.py compare/method_cmp_20260723_123627/methods_cmp.csv
"""
import os, sys, csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

# ---- Palette (dataviz skill, light mode) ----
C_HASH   = '#2a78d6'   # categorical slot 1 (blue) — Auto picked hash
C_MERGE  = '#eb6834'   # categorical slot 2 (orange) — Auto picked merge3
C_DIAG   = '#898781'   # muted — diagonal y=x
C_GRID   = '#e1e0d9'   # gridline
C_TEXT   = '#0b0b0b'
C_TEXT2  = '#52514e'
C_SURFACE = '#fcfcfb'
C_WIN    = '#e8f0fb'   # light blue tint for "Auto wins" region
C_LOSE   = '#fdf0eb'   # light orange tint for "Auto loses" region

# ---- H100 PCIe specs ----
H100_FP32_TFLOPS = 51.0      # FP32 peak (PCIe)
H100_BW_TBPS      = 2.0       # HBM3 bandwidth (PCIe)
H100_SMS          = 114       # SMs (PCIe)

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "compare/methods_cmp.csv"
    out_dir  = os.path.dirname(csv_path)

    rows = []
    for r in csv.DictReader(open(csv_path)):
        try:
            au = float(r["Auto"]); oc = float(r["Ocean"])
            n  = int(r["n"]); cnnz = int(r["cnnz"])
            dens = float(r["density_pct"])
            choice = r.get("Auto_choice", "")
        except (ValueError, KeyError):
            continue
        rows.append({"matrix": r["matrix"], "n": n, "cnnz": cnnz,
                     "dens": dens, "au": au, "oc": oc, "choice": choice,
                     "m3": float(r["merge3"]) if r.get("merge3","") else None})

    hash_rows  = [r for r in rows if r["choice"].startswith("hash")]
    merge_rows = [r for r in rows if r["choice"].startswith("merge3")]

    # ================================================================
    # Chart 1: Scatter — Auto vs Ocean (log-log, colored by dispatch)
    # ================================================================
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    # Win/lose regions (subtle background)
    lim_lo = 0.01; lim_hi = 30
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], [lim_hi, lim_hi],
                    color=C_WIN, alpha=0.3, zorder=0)   # below diagonal = Auto wins
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_lo], [lim_lo, lim_hi],
                    color=C_LOSE, alpha=0.3, zorder=0)  # above = Auto loses

    # Diagonal y=x
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=C_DIAG, lw=1.2, ls='--', zorder=1, label='Break-even (Auto = Ocean)')

    # Scatter: merge3-chosen (orange) + hash-chosen (blue)
    if merge_rows:
        ax.scatter([r["oc"] for r in merge_rows], [r["au"] for r in merge_rows],
                   c=C_MERGE, s=30, alpha=0.7, edgecolors='white', linewidths=0.4,
                   zorder=3, label=f'Auto → merge3 (n={len(merge_rows)})')
    if hash_rows:
        ax.scatter([r["oc"] for r in hash_rows], [r["au"] for r in hash_rows],
                   c=C_HASH, s=36, alpha=0.8, edgecolors='white', linewidths=0.4,
                   marker='^', zorder=4, label=f'Auto → hash (n={len(hash_rows)})')

    # Annotate notable matrices
    for name in ["bcsstk30", "bcsstk32", "bcsstk33", "bp_0", "bcsstk16", "bcsstm21"]:
        r = next((r for r in rows if r["matrix"] == name), None)
        if r:
            offset = (6, 6) if r["au"] < r["oc"] else (6, -10)
            ax.annotate(name, (r["oc"], r["au"]), fontsize=6.5, color=C_TEXT2,
                        xytext=offset, textcoords='offset points')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.xaxis.set_major_locator(LogLocator(numticks=8))
    ax.yaxis.set_major_locator(LogLocator(numticks=8))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(labelsize=8, colors=C_TEXT2)
    ax.set_xlabel('Ocean compute-only (ms)', fontsize=10, color=C_TEXT)
    ax.set_ylabel('Auto (adaptive) compute-only (ms)', fontsize=10, color=C_TEXT)
    ax.set_title('Adaptive SpGEMM Dispatcher vs Ocean on NVIDIA H100 PCIe\n(99 matrices, C = A·A self-product, cudaEvent same methodology)',
                 fontsize=10, color=C_TEXT, pad=10)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9, edgecolor=C_GRID)
    ax.grid(True, which='major', color=C_GRID, lw=0.5)
    ax.grid(True, which='minor', color=C_GRID, lw=0.3, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)

    # Win/lose labels
    ax.text(0.03, 0.97, f'Auto wins: {sum(1 for r in rows if r["au"] <= r["oc"])}/{len(rows)}',
            transform=ax.transAxes, fontsize=8, color=C_HASH, fontweight='bold', va='top')
    ax.text(0.97, 0.03, f'Auto loses: {sum(1 for r in rows if r["au"] > r["oc"])}/{len(rows)}',
            transform=ax.transAxes, fontsize=8, color=C_MERGE, fontweight='bold', ha='right', va='bottom')

    fig.tight_layout()
    chart_path = os.path.join(out_dir, "auto_vs_ocean_scatter.png")
    fig.savefig(chart_path, dpi=200, facecolor=C_SURFACE)
    print(f"Chart saved: {chart_path}")
    plt.close(fig)

    # ================================================================
    # Chart 2: Grouped bar by density class (geomean by class)
    # ================================================================
    CLASS_ORDER = ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]
    CLASS_TAG = {"Dense": "Dense", "Mildly sparse": "Mild", "Highly sparse": "High", "Extremely sparse": "Extr"}

    def classify(dens):
        if dens >= 10: return "Dense"
        if dens >= 1:  return "Mildly sparse"
        if dens >= 0.1: return "Highly sparse"
        return "Extremely sparse"

    def gmean(xs):
        xs = [x for x in xs if x and x > 0]
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')

    classes_data = {}
    for c in CLASS_ORDER:
        sub = [r for r in rows if classify(r["dens"]) == c]
        if not sub: continue
        classes_data[c] = {
            "Ocean": gmean([r["oc"] for r in sub]),
            "m3":    gmean([r["m3"] for r in sub if r["m3"]]),
            "Auto":  gmean([r["au"] for r in sub]),
            "count": len(sub),
        }

    fig2, ax2 = plt.subplots(figsize=(8, 4.5), facecolor=C_SURFACE)
    ax2.set_facecolor(C_SURFACE)
    x = np.arange(len(CLASS_ORDER))
    w = 0.22
    bars_oc = ax2.bar(x - w, [classes_data.get(c,{}).get("Ocean", 0) for c in CLASS_ORDER], w,
                       color='#4a3aa7', label='Ocean', edgecolor='white', linewidth=0.5)
    bars_m3 = ax2.bar(x,     [classes_data.get(c,{}).get("m3", 0) for c in CLASS_ORDER], w,
                       color=C_MERGE, label='merge3', edgecolor='white', linewidth=0.5)
    bars_au = ax2.bar(x + w, [classes_data.get(c,{}).get("Auto", 0) for c in CLASS_ORDER], w,
                       color=C_HASH, label='Auto (adaptive)', edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{CLASS_TAG[c]}\n(n={classes_data.get(c,{}).get("count",0)})' for c in CLASS_ORDER],
                        fontsize=9, color=C_TEXT)
    ax2.set_ylabel('Geometric mean compute-only (ms)', fontsize=10, color=C_TEXT)
    ax2.set_title('Performance by Matrix Density Class (geometric mean, lower is better)',
                   fontsize=10, color=C_TEXT, pad=8)
    ax2.legend(fontsize=8, framealpha=0.9, edgecolor=C_GRID)
    ax2.grid(axis='y', color=C_GRID, lw=0.5)
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=8, colors=C_TEXT2)
    for spine in ax2.spines.values():
        spine.set_color(C_GRID)
    # Value labels on bars
    for bars in [bars_oc, bars_m3, bars_au]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                         ha='center', va='bottom', fontsize=6.5, color=C_TEXT2)
    fig2.tight_layout()
    chart2_path = os.path.join(out_dir, "method_comparison_by_class.png")
    fig2.savefig(chart2_path, dpi=200, facecolor=C_SURFACE)
    print(f"Chart saved: {chart2_path}")
    plt.close(fig2)

    # ================================================================
    # Performance Model (roofline-style regression)
    # ================================================================
    print("\n" + "="*70)
    print("PERFORMANCE MODEL (roofline-style, H100 PCIe)")
    print("="*70)

    # Build feature matrix
    # Features: C_nnz (output work), A_nnz^2/n (flop proxy), 1 (overhead)
    Y = []
    X = []
    for r in rows:
        A_nnz_est = r["dens"] / 100.0 * r["n"]**2  # estimated A_nnz from density
        if A_nnz_est < 1: A_nnz_est = 1
        flop_proxy = A_nnz_est**2 / r["n"]  # = A_nnz^2/n
        C_nnz = r["cnnz"]
        Y.append(r["au"])
        X.append([C_nnz, flop_proxy, 1.0])
    Y = np.array(Y)
    X = np.array(X)

    # Log-space linear regression: log(T) = a*log(C_nnz) + b*log(flop) + c
    # (more robust for power-law scaling)
    Yl = np.log(Y)
    Xl = np.column_stack([np.log(X[:, 0]), np.log(X[:, 1]), np.ones(len(Y))])

    coeffs, residuals, rank, sv = np.linalg.lstsq(Xl, Yl, rcond=None)
    a_c, a_f, a_0 = coeffs
    Yl_pred = Xl @ coeffs
    ss_res = np.sum((Yl - Yl_pred)**2)
    ss_tot = np.sum((Yl - Yl_pred.mean())**2)
    r2_log = 1 - ss_res / ss_tot

    # Also linear regression (T = a*C_nnz + b*flop + c) for comparison
    coeffs_lin, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    a_lin, b_lin, c_lin = coeffs_lin
    Y_pred_lin = X @ coeffs_lin
    ss_res_lin = np.sum((Y - Y_pred_lin)**2)
    ss_tot_lin = np.sum((Y - Y.mean())**2)
    r2_lin = 1 - ss_res_lin / ss_tot_lin

    # H100-derived constants
    bw_Bps   = H100_BW_TBPS * 1e12     # bytes/s
    flops    = H100_FP32_TFLOPS * 1e12  # FLOP/s
    bytes_per_C_item = 12   # key(8) + val(4)
    bytes_per_A_item = 12

    print(f"\nH100 PCIe specs:")
    print(f"  FP32 peak   : {H100_FP32_TFLOPS:.0f} TFLOPS")
    print(f"  HBM3 BW     : {H100_BW_TBPS:.1f} TB/s")
    print(f"  SMs         : {H100_SMS}")

    print(f"\n--- Power-law model (log-space, R² = {r2_log:.4f}) ---")
    print(f"  T(ms) = {math.exp(a_0):.2e} × C_nnz^{a_c:.3f} × flop_proxy^{a_f:.3f}")
    print(f"  where flop_proxy = A_nnz²/n (intermediate-product proxy)")
    print(f"  Coefficients: a(C_nnz)={a_c:.4f}, b(flop)={a_f:.4f}, c(const)={a_0:.4f}")

    print(f"\n--- Linear model (R² = {r2_lin:.4f}) ---")
    print(f"  T(ms) = {a_lin:.2e} × C_nnz + {b_lin:.2e} × flop_proxy + {c_lin:.4f}")
    print(f"  Slope α (per C_nnz item): {a_lin:.2e} ms/item")
    print(f"    → implied throughput: {1/a_lin/1e6:.1f} M items/s")
    print(f"    → vs H100 BW ({bw_Bps/1e12:.0f} TB/s): {bytes_per_C_item/(a_lin*1e-3*bw_Bps):.1f}% utilization")

    print(f"\n--- Auto vs Ocean summary ---")
    wins = sum(1 for r in rows if r["au"] <= r["oc"])
    losses = len(rows) - wins
    geomean_ratio = gmean([r["au"]/r["oc"] for r in rows])
    print(f"  Auto wins {wins}/{len(rows)} matrices")
    print(f"  Geomean(Auto/Ocean) = {geomean_ratio:.3f}× (Auto is {1/geomean_ratio:.2f}× faster)")
    hash_wins = sum(1 for r in hash_rows if r["au"] <= r["oc"])
    merge_wins = sum(1 for r in merge_rows if r["au"] <= r["oc"])
    print(f"  hash path: {hash_wins}/{len(hash_rows)} win; merge3 path: {merge_wins}/{len(merge_rows)} win")

    # Save model to text
    model_path = os.path.join(out_dir, "performance_model.txt")
    with open(model_path, "w") as f:
        f.write(f"Performance Model — Adaptive SpGEMM on NVIDIA H100 PCIe\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Power-law model (R² = {r2_log:.4f}):\n")
        f.write(f"  T(ms) = {math.exp(a_0):.2e} × C_nnz^{a_c:.3f} × flop_proxy^{a_f:.3f}\n")
        f.write(f"  flop_proxy = A_nnz²/n\n\n")
        f.write(f"Linear model (R² = {r2_lin:.4f}):\n")
        f.write(f"  T(ms) = {a_lin:.2e} × C_nnz + {b_lin:.2e} × flop_proxy + {c_lin:.4f}\n\n")
        f.write(f"H100 PCIe constants:\n")
        f.write(f"  FP32 peak = {H100_FP32_TFLOPS:.0f} TFLOPS\n")
        f.write(f"  HBM3 BW = {H100_BW_TBPS:.1f} TB/s\n")
        f.write(f"  SMs = {H100_SMS}\n\n")
        f.write(f"Auto vs Ocean: {wins}/{len(rows)} win, geomean {geomean_ratio:.3f}×\n")
    print(f"\nModel saved: {model_path}")

if __name__ == "__main__":
    main()
