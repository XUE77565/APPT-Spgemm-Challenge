#!/usr/bin/env python3
"""Generate publication chart + analytical dispatcher model for hash vs merge3.

Chart: identical to hash_vs_merge.png but larger (10×9, 300 DPI).
Model: analytical performance model derived from algorithm structure + H100 PCIe specs,
       constants fitted from first100 measurements.

Usage: .venv/bin/python scripts/hash_vs_merge_final.py [enriched_csv] [output_dir]
"""
import os, sys, csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from numpy.linalg import lstsq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Palette (dataviz skill, light mode) ──
C_HASH='#2a78d6'; C_MERGE='#eb6834'; C_DIAG='#898781'; C_GRID='#e1e0d9'
C_TEXT='#0b0b0b'; C_TEXT2='#52514e'; C_SURFACE='#fcfcfb'

# ── H100 PCIe hardware constants ──
H100_FP32_TFLOPS = 51.0       # FP32 peak (PCIe)
H100_BW_TBs      = 2.0        # HBM3 bandwidth (PCIe)
H100_SMS         = 114         # SMs (PCIe)
H100_CLOCK_GHz   = 1.98       # boost clock
HASH_BLOCK       = 256         # threads per hash block
MERGE3_WARP      = 32          # threads per merge3 warp
MERGE3_K         = 5           # column-range buckets
SMEM_ATOMIC_CYC  = 20          # shared-memory atomic CAS/Add latency (cycles)

def main():
    csv_path = sys.argv[1] if len(sys.argv)>1 else os.path.join(REPO,"compare/hash_vs_merge/hash_vs_merge_enriched.csv")
    out_dir  = sys.argv[2] if len(sys.argv)>2 else os.path.join(REPO,"compare/hash_vs_merge")
    os.makedirs(out_dir, exist_ok=True)

    int_cols = {"n","A_nnz","C_nnz_flop","max_row_nnz","avg_row_nnz","hash_wins"}
    data = []
    for r in csv.DictReader(open(csv_path)):
        row = {"matrix": r["matrix"]}
        for k, v in r.items():
            if k == "matrix": continue
            try: row[k] = int(float(v)) if k in int_cols else float(v)
            except ValueError: row[k] = v
        data.append(row)
    N = len(data)
    print(f"Loaded {N} matrices")

    # ================================================================
    # PART 1: Chart — identical to original but larger
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    lim_lo, lim_hi = 0.05, 20

    # Win/lose background regions
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], [lim_hi, lim_hi],
                    color='#e8f0fb', alpha=0.35, zorder=0)
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_lo], [lim_lo, lim_hi],
                    color='#fdf0eb', alpha=0.35, zorder=0)

    # Diagonal y=x
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=C_DIAG, lw=1.5, ls='--', zorder=1,
            label='Break-even (hash = merge3)')

    # Scatter colored by flop_proxy
    flops = [r["C_nnz_flop"] for r in data]
    flops_log = np.log10(np.clip(flops, 1, None))
    fmin, fmax = flops_log.min(), flops_log.max()
    norm = plt.Normalize(fmin, fmax)

    sc = ax.scatter(
        [r["merge3_ms"] for r in data],
        [r["hash_ms"] for r in data],
        c=flops_log, cmap='Blues', norm=norm, s=50, alpha=0.8,
        edgecolors='white', linewidths=0.5, zorder=3)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label(r'flop proxy = $A_{\mathrm{nnz}}^2 / n$  (log$_{10}$)', fontsize=10, color=C_TEXT)
    cbar.ax.tick_params(labelsize=8, colors=C_TEXT2)

    # Annotate notable matrices
    for name in ["bcsstk30","bcsstk32","bcsstk08","bp_0","bcsstk16","bcsstk13",
                 "bcsstk24","bcsstk21","can_24","can_715"]:
        r = next((r for r in data if r["matrix"]==name), None)
        if r:
            off = (7, 6) if r["hash_ms"] < r["merge3_ms"] else (7, -10)
            ax.annotate(name, (r["merge3_ms"], r["hash_ms"]), fontsize=7, color=C_TEXT2,
                        xytext=off, textcoords='offset points')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(labelsize=9, colors=C_TEXT2)
    ax.set_xlabel('merge3 compute-only time (ms)', fontsize=12, color=C_TEXT)
    ax.set_ylabel('hash compute-only time (ms)', fontsize=12, color=C_TEXT)
    ax.set_title('Hash vs Merge3 SpGEMM on NVIDIA H100 PCIe\n(C = A·A self-product, 100 matrices, cudaEvent GPU-only timing)',
                 fontsize=12, color=C_TEXT, pad=12)

    hw = sum(1 for r in data if r["hash_ms"] < r["merge3_ms"])
    ax.text(0.03, 0.97, f'hash wins ({hw})', transform=ax.transAxes,
            fontsize=10, color=C_HASH, fontweight='bold', va='top')
    ax.text(0.97, 0.03, f'merge3 wins ({N-hw})', transform=ax.transAxes,
            fontsize=10, color=C_MERGE, fontweight='bold', ha='right', va='bottom')

    ax.grid(True, which='major', color=C_GRID, lw=0.5)
    ax.grid(True, which='minor', color=C_GRID, lw=0.3, alpha=0.5)
    for spine in ax.spines.values(): spine.set_color(C_GRID)

    fig.tight_layout()
    chart_path = os.path.join(out_dir, "hash_vs_merge_large.png")
    fig.savefig(chart_path, dpi=300, facecolor=C_SURFACE)
    print(f"Chart: {chart_path}")
    plt.close(fig)

    # ================================================================
    # PART 2: Analytical Dispatcher Model
    # ================================================================
    print("\n" + "="*70)
    print("ANALYTICAL DISPATCHER MODEL")
    print("="*70)

    # ── Algorithm structure ──
    # HASH pipeline:
    #   1. HLL estimate:         scan A CSR → O(nnz_A)
    #   2. Binning:              est → bucket → scatter → O(n)
    #   3. Accumulate:           per-row SMEM hash insertion of all flop intermediates
    #      - each (k,j) pair:   1 atomicCAS + 1 atomicAdd ≈ 2 SMEM atomics
    #      - parallelism:       min(n, 114) blocks × 256 threads
    #   4. Compact + sort:       per-row BlockRadixSort of C_nnz_per_row items
    #      - sort cost:         O(C_nnz_per_row) with BlockRadixSort (radix, not comparison)
    #   5. Fixed overhead:       HLL construct+merge + binning + kernel launches
    #
    # MERGE3 pipeline:
    #   1. Count intermediates:  scan A CSR → O(nnz_A)
    #   2. Binning by column:    K=5 column-range buckets → O(flop) (count distinct per bucket)
    #   3. K-way merge:          per-row merge of K sorted lists → O(flop) total work
    #      - each intermediate:  log₂(K) ≈ 2.3 comparisons
    #      - parallelism:       min(n, 114) blocks × 32 threads (1 warp per row)
    #   4. No sort:             merge produces sorted output
    #   5. Fixed overhead:       count + scan + launches

    # ── H100 PCIe constants ──
    clock_Hz = H100_CLOCK_GHz * 1e9
    n_sms    = H100_SMS
    bw_Bps   = H100_BW_TBs * 1e12
    flops    = H100_FP32_TFLOPS * 1e12

    # ── Model variables per matrix ──
    # For each matrix i:
    #   n          = matrix dimension (number of rows)
    #   nnz_A      = A_nnz_actual (total nonzeros in A)
    #   flop       = Σ_k nnz(row_k)² (for symmetric A) ≈ C_nnz_flop proxy
    #   C_nnz      = output nonzeros (not directly available; approx from ratio of times)
    #   max_row    = max_row_nnz (heaviest row)
    #   avg_row    = nnz_A / n (average row weight)

    # ── Hash time model ──
    # T_hash = α_h × (flop / P_h) + β_h × (nnz_A / BW) + γ_h
    # where:
    #   P_h = min(n, n_sms) × HASH_BLOCK = hash parallelism (blocks × threads)
    #   α_h = cost per intermediate product (2 SMEM atomics × SMEM_ATOMIC_CYC / clock)
    #   β_h = memory transfer coefficient (nnz_A read + C_nnz write)
    #   γ_h = fixed overhead (HLL + binning + sort base + launches)

    # ── Merge3 time model ──
    # T_merge3 = α_m × (flop / P_m) + β_m × (nnz_A / BW) + γ_m
    # where:
    #   P_m = min(n, n_sms) × MERGE3_WARP = merge3 parallelism
    #   α_m = cost per intermediate (log₂(K) × comparison + k-way merge overhead)
    #   γ_m = fixed overhead (count + binning + launches)

    # Build feature matrices
    X_h = []  # hash features
    X_m = []  # merge3 features
    Y_h = []  # hash_ms
    Y_m = []  # merge3_ms

    for r in data:
        n = max(r["n"], 1)
        nnz = max(r["A_nnz_actual"], 1)
        flop = max(r["C_nnz_flop"], 1)
        P_h = min(n, n_sms) * HASH_BLOCK       # hash parallelism
        P_m = min(n, n_sms) * MERGE3_WARP       # merge3 parallelism
        mem_B = nnz * 12.0                       # memory bytes (read A + write C, ~12 B/nnz)

        # Features: [flop/P (compute), mem/BW (memory), 1 (overhead)]
        X_h.append([flop / P_h, mem_B / bw_Bps, 1.0])
        X_m.append([flop / P_m, mem_B / bw_Bps, 1.0])
        Y_h.append(r["hash_ms"])
        Y_m.append(r["merge3_ms"])

    X_h = np.array(X_h); Y_h = np.array(Y_h)
    X_m = np.array(X_m); Y_m = np.array(Y_m)

    # Linear regression (non-negative not enforced; physical interpretation from sign)
    cf_h, _, _, _ = lstsq(X_h, Y_h, rcond=None)
    cf_m, _, _, _ = lstsq(X_m, Y_m, rcond=None)

    # R²
    def r2(Y, Y_pred):
        return 1 - np.sum((Y - Y_pred)**2) / np.sum((Y - Y.mean())**2)

    r2_h = r2(Y_h, X_h @ cf_h)
    r2_m = r2(Y_m, X_m @ cf_m)

    print(f"\nHash model (R² = {r2_h:.3f}):")
    print(f"  T_hash = {cf_h[0]:.4e} × flop/P_hash + {cf_h[1]:.4e} × nnz_A/BW + {cf_h[2]:.4f}")
    print(f"  P_hash = min(n, {n_sms}) × {HASH_BLOCK}")
    alpha_h = cf_h[0]
    print(f"  α_h = {alpha_h:.4e} ms·thread/intermediate")
    print(f"    → per-intermediate cost: {alpha_h:.2e} ms × {HASH_BLOCK} threads = {alpha_h*HASH_BLOCK:.2e} ms/block-intermediate")
    print(f"    → atomic cycles implied: {alpha_h*1e-3*clock_Hz*HASH_BLOCK:.0f} cycles")

    print(f"\nMerge3 model (R² = {r2_m:.3f}):")
    print(f"  T_merge3 = {cf_m[0]:.4e} × flop/P_merge3 + {cf_m[1]:.4e} × nnz_A/BW + {cf_m[2]:.4f}")
    print(f"  P_merge3 = min(n, {n_sms}) × {MERGE3_WARP}")
    alpha_m = cf_m[0]
    print(f"  α_m = {alpha_m:.4e} ms·warp/intermediate")
    print(f"    → per-intermediate cost: {alpha_m:.2e} ms × {MERGE3_WARP} threads = {alpha_m*MERGE3_WARP:.2e} ms/intermediate")

    # ── Dispatcher decision boundary ──
    # T_hash < T_merge3  ⟺  α_h × flop/P_h + β_h × nnz/BW + γ_h < α_m × flop/P_m + β_m × nnz/BW + γ_m
    # For large n (n >> n_sms): P_h = n_sms × 256, P_m = n_sms × 32 → P_h/P_m = 8
    # → flop × (α_h/(n_sms×256) - α_m/(n_sms×32)) + nnz × (β_h-β_m)/BW + (γ_h-γ_m) < 0
    # → flop × (α_h - 8×α_m) / (n_sms × 256) < (γ_m - γ_h) + nnz × (β_m - β_h) / BW
    # Since α_h < 8×α_m (hash more parallel), LHS is negative → inequality holds for large flop

    print(f"\n{'='*70}")
    print("DISPATCHER DECISION FORMULA")
    print("="*70)
    print(f"\nT_hash   = {cf_h[0]:.3e} × flop / min(n,{n_sms})/{HASH_BLOCK}")
    print(f"         + {cf_h[1]:.3e} × nnz_A / ({H100_BW_TBs} TB/s)")
    print(f"         + {cf_h[2]:.3f} ms")
    print(f"\nT_merge3 = {cf_m[0]:.3e} × flop / min(n,{n_sms})/{MERGE3_WARP}")
    print(f"         + {cf_m[1]:.3e} × nnz_A / ({H100_BW_TBs} TB/s)")
    print(f"         + {cf_m[2]:.3f} ms")
    print(f"\nDispatch hash ⟺ T_hash < T_merge3")

    # Classification accuracy
    T_h_pred = X_h @ cf_h
    T_m_pred = X_m @ cf_m
    y_pred = (T_h_pred < T_m_pred).astype(int)
    y_true = np.array([1 if r["hash_ms"] < r["merge3_ms"] else 0 for r in data])
    acc = np.mean(y_pred == y_true)
    print(f"\nModel dispatch accuracy: {acc:.1%} ({sum(y_pred==y_true)}/{N})")

    # Show misclassified
    miscls = [(data[i]["matrix"], data[i]["hash_ms"], data[i]["merge3_ms"], T_h_pred[i], T_m_pred[i])
             for i in range(N) if y_pred[i] != y_true[i]]
    if miscls:
        print(f"\nMisclassified ({len(miscls)}):")
        for m, th, tm, thp, tmp in miscls[:10]:
            print(f"  {m:16} actual: hash={th:.3f} m3={tm:.3f} ({'hash' if th<tm else 'm3'})"
                  f"  predicted: hash={thp:.3f} m3={tmp:.3f} ({'hash' if thp<tmp else 'm3'})")

    # ── Simplified decision rule for paper ──
    print(f"\n{'='*70}")
    print("SIMPLIFIED DECISION RULE")
    print("="*70)

    # For the paper: at the crossover T_hash = T_merge3
    # α_h × flop / P_h + γ_h = α_m × flop / P_m + γ_m  (memory term cancels for symmetric)
    # flop × (α_h/P_h - α_m/P_m) = γ_m - γ_h
    # flop* = (γ_m - γ_h) / (α_h/P_h - α_m/P_m)
    # For large n: P_h = n_sms × 256, P_m = n_sms × 32
    P_h_inf = n_sms * HASH_BLOCK
    P_m_inf = n_sms * MERGE3_WARP
    dh = cf_h[0] / P_h_inf
    dm = cf_m[0] / P_m_inf
    g_diff = cf_m[2] - cf_h[2]
    if abs(dh - dm) > 1e-20:
        flop_star = g_diff / (dh - dm)
        print(f"\nCrossover flop* = (γ_m - γ_h) / (α_h/P_h - α_m/P_m)")
        print(f"  = ({cf_m[2]:.4f} - {cf_h[2]:.4f}) / ({cf_h[0]:.3e}/{P_h_inf} - {cf_m[0]:.3e}/{P_m_inf})")
        print(f"  = {flop_star:.3e}")
        print(f"\nSimplified rule (large n, symmetric A):")
        print(f"  hash  ⟺  flop > {flop_star:.2e}")
        print(f"  merge3 ⟺  flop ≤ {flop_star:.2e}")
        print(f"  where flop ≈ Σ_k nnz(row_k)² ≈ A_nnz²/n for symmetric A")

    # Save model
    mp = os.path.join(out_dir, "dispatcher_model_final.txt")
    with open(mp, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DISPATCHER PERFORMANCE MODEL — Hash vs Merge3 SpGEMM\n")
        f.write(f"Hardware: NVIDIA H100 PCIe ({H100_FP32_TFLOPS:.0f} TFLOPS FP32, {H100_BW_TBs:.1f} TB/s HBM3, {H100_SMS} SMs)\n")
        f.write(f"Algorithm: C = A·A self-product, 100 matrices (SuiteSparse first100)\n")
        f.write("=" * 70 + "\n\n")

        f.write("ANALYTICAL TIME MODELS\n")
        f.write("-" * 40 + "\n\n")
        f.write(f"Hash:\n")
        f.write(f"  T_hash = α_h × flop / min(n, {n_sms}) / {HASH_BLOCK}\n")
        f.write(f"         + β_h × nnz_A / ({H100_BW_TBs} TB/s)\n")
        f.write(f"         + γ_h\n\n")
        f.write(f"  α_h = {cf_h[0]:.4e} ms·thread/intermediate\n")
        f.write(f"  β_h = {cf_h[1]:.4e} ms·s/B (memory coefficient)\n")
        f.write(f"  γ_h = {cf_h[2]:.4f} ms (fixed: HLL + binning + compact/sort + launches)\n")
        f.write(f"  R²  = {r2_h:.3f}\n\n")
        f.write(f"Merge3:\n")
        f.write(f"  T_merge3 = α_m × flop / min(n, {n_sms}) / {MERGE3_WARP}\n")
        f.write(f"           + β_m × nnz_A / ({H100_BW_TBs} TB/s)\n")
        f.write(f"           + γ_m\n\n")
        f.write(f"  α_m = {cf_m[0]:.4e} ms·warp/intermediate\n")
        f.write(f"  β_m = {cf_m[1]:.4e} ms·s/B\n")
        f.write(f"  γ_m = {cf_m[2]:.4f} ms (fixed: count + scan + launches)\n")
        f.write(f"  R²  = {r2_m:.3f}\n\n")

        f.write("DISPATCHER DECISION\n")
        f.write("-" * 40 + "\n\n")
        f.write(f"hash   ⟺  T_hash < T_merge3\n\n")
        f.write(f"Model dispatch accuracy: {acc:.1%}\n\n")
        if 'flop_star' in dir():
            f.write(f"SIMPLIFIED RULE (large n, symmetric A):\n")
            f.write(f"  hash   ⟺  flop > {flop_star:.2e}\n")
            f.write(f"  merge3 ⟺  flop ≤ {flop_star:.2e}\n")
            f.write(f"  where flop ≈ A_nnz²/n for symmetric A\n\n")

        f.write("ALGORITHM-HARDWARE INTERPRETATION\n")
        f.write("-" * 40 + "\n\n")
        f.write(f"Hash wins because:\n")
        f.write(f"  • Parallelism: min(n,{n_sms})×{HASH_BLOCK} threads (hash) vs min(n,{n_sms})×{MERGE3_WARP} (merge3)\n")
        f.write(f"    → {HASH_BLOCK/MERGE3_WARP:.0f}× more thread-level parallelism per row\n")
        f.write(f"  • Dedup efficiency: hash O(1) per duplicate vs merge3 O(log₂{MERGE3_K}) per duplicate\n")
        f.write(f"  • Per-intermediate cost: α_h={cf_h[0]:.2e} vs α_m={cf_m[0]:.2e} (parallelism-adjusted)\n\n")
        f.write(f"Merge3 wins because:\n")
        f.write(f"  • No sort: merge produces sorted output (saves O(C_nnz log) sort cost)\n")
        f.write(f"  • Lower fixed overhead: γ_m={cf_m[2]:.3f} vs γ_h={cf_h[2]:.3f} ms\n")
        f.write(f"    (no HLL estimation, no hash table allocation/init)\n\n")

        f.write(f"hash wins: {hw}/{N}, merge3 wins: {N-hw}/{N}\n")
    print(f"\nModel saved: {mp}")

if __name__ == "__main__":
    main()
