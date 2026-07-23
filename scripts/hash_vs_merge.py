#!/usr/bin/env python3
"""Run hash on all first100, combine with existing merge3 times, generate
crossover chart + dispatcher decision model.

Usage: .venv/bin/python scripts/hash_vs_merge.py <merge3_csv> <output_dir>
  merge3_csv = existing method_cmp CSV (has merge3 column)
  output_dir = where to save chart + model

Outputs:
  hash_vs_merge.csv     — per-matrix: n, density, merge3_ms, hash_ms, ratio, flop_proxy
  hash_vs_merge.png     — crossover scatter (merge3 vs hash, log-log)
  dispatcher_model.txt  — fitted decision formula
"""
import os, sys, csv, re, math, subprocess, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import LogLocator, ScalarFormatter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
DATA = os.path.join(REPO, "data/first100")

# Palette
C_HASH  = '#2a78d6'
C_MERGE = '#eb6834'
C_DIAG  = '#898781'
C_GRID  = '#e1e0d9'
C_TEXT  = '#0b0b0b'
C_TEXT2 = '#52514e'
C_SURFACE = '#fcfcfb'

def compute_only_hashprof(stderr):
    """Parse [hash-prof] cudaEvent phases → compute-only = TOTAL - h2d - d2h."""
    rx = re.compile(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s+ms")
    phases = {}
    for line in stderr.splitlines():
        m = rx.search(line)
        if m:
            phases[m.group(1)] = float(m.group(2))
    if not phases:
        return None
    total = phases.get("TOTAL(GPU)")
    if total is not None:
        return total - phases.get("h2d", 0.0) - phases.get("d2h", 0.0)
    return sum(v for k, v in phases.items() if k not in ("h2d", "d2h"))

def mtx_info(path):
    """Return (n, density_pct, A_nnz_est)."""
    n = 0; nnz_stored = 0
    with open(path) as f:
        first = f.readline(); sym = "symmetric" in first
        for l in f:
            if l.startswith("%"): continue
            p = l.split()
            if not p: continue
            if n == 0:
                n = int(p[0]); continue
            nnz_stored += 1
    a_nnz = nnz_stored * (2 if sym else 1)
    dens = (a_nnz / (n * n) * 100.0) if n > 0 else 0.0
    return n, dens, a_nnz

def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')

def main():
    merge3_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "compare/method_cmp_20260723_123627/methods_cmp.csv")
    out_dir    = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "compare/hash_vs_merge")

    os.makedirs(out_dir, exist_ok=True)

    # Read existing merge3 times
    m3_times = {}
    for r in csv.DictReader(open(merge3_csv)):
        try:
            m3 = float(r["merge3"])
            m3_times[r["matrix"]] = m3
        except (ValueError, KeyError):
            pass

    # Run hash on all first100
    mtxs = sorted(f[:-4] for f in os.listdir(DATA) if f.endswith(".mtx"))
    combined = []
    print(f"Running hash on {len(mtxs)} matrices...", flush=True)
    for i, name in enumerate(mtxs):
        mtx_path = os.path.join(DATA, name + ".mtx")
        n, dens, a_nnz = mtx_info(mtx_path)
        flop_proxy = a_nnz * a_nnz / max(n, 1)
        m3 = m3_times.get(name)
        # Run hash
        try:
            env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash")
            r = subprocess.run([BIN, mtx_path], capture_output=True, text=True, env=env, timeout=120)
            h = compute_only_hashprof(r.stderr)
        except Exception:
            h = None
        if h is not None and m3 is not None:
            ratio = h / m3
            combined.append({"matrix": name, "n": n, "density_pct": round(dens, 4),
                             "A_nnz": a_nnz, "C_nnz_flop": flop_proxy,
                             "merge3_ms": m3, "hash_ms": h, "ratio": ratio})
            winner = "hash" if h < m3 else "merge3"
            print(f"[{i+1}/{len(mtxs)}] {name:16} n={n:<6} m3={m3:.3f} hash={h:.3f} → {winner}", flush=True)
        else:
            print(f"[{i+1}/{len(mtxs)}] {name:16} SKIP (m3={m3}, hash={h})", flush=True)

    # Save combined CSV
    csv_path = os.path.join(out_dir, "hash_vs_merge.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["matrix", "n", "density_pct", "A_nnz", "C_nnz_flop",
                                          "merge3_ms", "hash_ms", "ratio"])
        w.writeheader()
        w.writerows(combined)
    print(f"\nSaved: {csv_path}")

    hash_wins = sum(1 for r in combined if r["ratio"] < 1.0)
    merge_wins = len(combined) - hash_wins
    print(f"hash wins: {hash_wins}/{len(combined)}, merge3 wins: {merge_wins}/{len(combined)}")

    # ================================================================
    # Chart 1: Crossover scatter (merge3 vs hash, log-log)
    # ================================================================
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    lim_lo = 0.005; lim_hi = 50

    # Color by flop_proxy (sequential blue ramp)
    flops = [r["C_nnz_flop"] for r in combined]
    flops_log = np.log10(np.clip(flops, 1, None))
    fmin, fmax = flops_log.min(), flops_log.max()
    norm = plt.Normalize(fmin, fmax)
    cmap = plt.cm.Blues

    sc = ax.scatter(
        [r["merge3_ms"] for r in combined],
        [r["hash_ms"] for r in combined],
        c=flops_log, cmap=cmap, norm=norm, s=35, alpha=0.8,
        edgecolors='white', linewidths=0.4, zorder=3)

    # Diagonal y=x
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=C_DIAG, lw=1.2, ls='--', zorder=1,
            label='Break-even (hash = merge3)')

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('flop proxy = $A_{nnz}^2 / n$ (log$_{10}$)', fontsize=9, color=C_TEXT)
    cbar.ax.tick_params(labelsize=7, colors=C_TEXT2)

    # Annotate notable matrices
    for name in ["bcsstk30", "bcsstk32", "bcsstk08", "can_24", "bp_0", "bcsstk16", "bcsstk13"]:
        r = next((r for r in combined if r["matrix"] == name), None)
        if r:
            offset = (5, 5) if r["hash_ms"] < r["merge3_ms"] else (5, -8)
            ax.annotate(name, (r["merge3_ms"], r["hash_ms"]), fontsize=6, color=C_TEXT2,
                        xytext=offset, textcoords='offset points')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.xaxis.set_major_locator(LogLocator(numticks=8))
    ax.yaxis.set_major_locator(LogLocator(numticks=8))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(labelsize=8, colors=C_TEXT2)
    ax.set_xlabel('merge3 compute-only (ms)', fontsize=10, color=C_TEXT)
    ax.set_ylabel('hash compute-only (ms)', fontsize=10, color=C_TEXT)
    ax.set_title('Hash vs Merge3 on NVIDIA H100 PCIe\n(Points below diagonal: hash wins; color = flop proxy)',
                 fontsize=10, color=C_TEXT, pad=10)

    # Region labels
    ax.text(0.03, 0.97, f'hash wins ({hash_wins})', transform=ax.transAxes,
            fontsize=8, color=C_HASH, fontweight='bold', va='top')
    ax.text(0.97, 0.03, f'merge3 wins ({merge_wins})', transform=ax.transAxes,
            fontsize=8, color=C_MERGE, fontweight='bold', ha='right', va='bottom')

    ax.grid(True, which='major', color=C_GRID, lw=0.5)
    ax.grid(True, which='minor', color=C_GRID, lw=0.3, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)

    fig.tight_layout()
    chart_path = os.path.join(out_dir, "hash_vs_merge.png")
    fig.savefig(chart_path, dpi=200, facecolor=C_SURFACE)
    print(f"Chart saved: {chart_path}")
    plt.close(fig)

    # ================================================================
    # Chart 2: Ratio vs flop_proxy (dispatcher decision threshold)
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(7.5, 5), facecolor=C_SURFACE)
    ax2.set_facecolor(C_SURFACE)

    ratios = [r["ratio"] for r in combined]
    fps = [r["C_nnz_flop"] for r in combined]
    ns = [r["n"] for r in combined]

    # Color by n (sequential)
    sc2 = ax2.scatter(fps, ratios, c=ns, cmap='viridis', s=30, alpha=0.7,
                      edgecolors='white', linewidths=0.4, zorder=3, norm=LogNorm())
    ax2.axhline(y=1.0, color=C_DIAG, lw=1.2, ls='--', zorder=1, label='Break-even (ratio = 1)')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('flop proxy = $A_{nnz}^2 / n$', fontsize=10, color=C_TEXT)
    ax2.set_ylabel('hash / merge3 time ratio', fontsize=10, color=C_TEXT)
    ax2.set_title('Dispatcher Decision: Hash vs Merge3 by Flop Proxy\n(below 1 = hash faster; color = matrix dimension $n$)',
                  fontsize=10, color=C_TEXT, pad=8)
    cbar2 = fig2.colorbar(sc2, ax=ax2, shrink=0.75, pad=0.02)
    cbar2.set_label('matrix dimension $n$', fontsize=9, color=C_TEXT)
    cbar2.ax.tick_params(labelsize=7, colors=C_TEXT2)
    ax2.tick_params(labelsize=8, colors=C_TEXT2)
    ax2.grid(True, which='major', color=C_GRID, lw=0.5)
    ax2.grid(True, which='minor', color=C_GRID, lw=0.3, alpha=0.5)
    for spine in ax2.spines.values():
        spine.set_color(C_GRID)

    # Fit crossover threshold
    # log(ratio) = a * log(flop_proxy) + b → ratio=1 at flop_proxy = exp(-b/a)
    valid = [(math.log(r["C_nnz_flop"]), math.log(r["ratio"])) for r in combined
             if r["C_nnz_flop"] > 0 and r["ratio"] > 0]
    if len(valid) > 3:
        Xf = np.array([[v[0], 1] for v in valid])
        Yf = np.array([v[1] for v in valid])
        cf, _, _, _ = np.linalg.lstsq(Xf, Yf, rcond=None)
        slope, intercept = cf
        crossover = math.exp(-intercept / slope) if abs(slope) > 1e-6 else float('inf')
        Yf_pred = Xf @ cf
        ss_res = np.sum((Yf - Yf_pred)**2)
        ss_tot = np.sum((Yf - Yf.mean())**2)
        r2 = 1 - ss_res / ss_tot

        # Plot fit line
        x_fit = np.logspace(np.log10(min(fps)), np.log10(max(fps)), 100)
        y_fit = [math.exp(slope * math.log(x) + intercept) for x in x_fit]
        ax2.plot(x_fit, y_fit, color=C_HASH, lw=1.5, alpha=0.7, zorder=2,
                 label=f'Fit: ratio = {math.exp(intercept):.2e} × flop$^{{{slope:.3f}}}$ ($R^2$={r2:.3f})')
        if 0 < crossover < 1e15:
            ax2.axvline(x=crossover, color=C_MERGE, lw=1, ls=':', alpha=0.6, zorder=1,
                        label=f'Crossover: flop_proxy ≈ {crossover:.2e}')
        ax2.legend(fontsize=7, loc='lower left', framealpha=0.9, edgecolor=C_GRID)

        print(f"\nDispatcher model:")
        print(f"  log(hash/merge3) = {slope:.4f} × log(flop_proxy) + {intercept:.4f}")
        print(f"  Crossover (ratio=1): flop_proxy ≈ {crossover:.2e}")
        print(f"  R² = {r2:.4f}")

    fig2.tight_layout()
    chart2_path = os.path.join(out_dir, "dispatcher_decision.png")
    fig2.savefig(chart2_path, dpi=200, facecolor=C_SURFACE)
    print(f"Chart saved: {chart2_path}")
    plt.close(fig2)

    # ================================================================
    # Separate performance models for hash and merge3
    # ================================================================
    print("\n" + "="*60)
    print("PERFORMANCE MODELS (H100 PCIe)")
    print("="*60)

    # log(T) = a*log(C_nnz_flop) + b*log(n) + c
    for method, col in [("hash", "hash_ms"), ("merge3", "merge3_ms")]:
        valid = [(r["C_nnz_flop"], r["n"], r[col]) for r in combined if r[col] > 0]
        if len(valid) < 5: continue
        Xl = np.column_stack([np.log(np.array([v[0] for v in valid])),
                              np.log(np.array([v[1] for v in valid])),
                              np.ones(len(valid))])
        Yl = np.log(np.array([v[2] for v in valid]))
        cf, _, _, _ = np.linalg.lstsq(Xl, Yl, rcond=None)
        Yl_pred = Xl @ cf
        r2 = 1 - np.sum((Yl - Yl_pred)**2) / np.sum((Yl - Yl.mean())**2)

        print(f"\n{method}: T(ms) = {math.exp(cf[2]):.2e} × flop_proxy^{cf[0]:.3f} × n^{cf[1]:.3f}")
        print(f"  R² = {r2:.4f}")

    # Save model
    model_path = os.path.join(out_dir, "dispatcher_model.txt")
    with open(model_path, "w") as f:
        f.write("Dispatcher Decision Model — Hash vs Merge3 on NVIDIA H100 PCIe\n")
        f.write("="*60 + "\n\n")
        f.write(f"Decision: hash when flop_proxy > threshold; merge3 otherwise\n")
        f.write(f"  flop_proxy = A_nnz² / n (intermediate-product proxy)\n\n")
        if 'crossover' in dir():
            f.write(f"Crossover threshold: flop_proxy ≈ {crossover:.2e}\n")
            f.write(f"  (current dispatcher uses 1e8 → validates threshold)\n\n")
        f.write(f"Ratio model: hash/merge3 = {math.exp(intercept):.2e} × flop_proxy^{slope:.3f}\n")
        f.write(f"  R² = {r2:.3f}\n\n")
        f.write(f"Summary: hash wins {hash_wins}/{len(combined)}, merge3 wins {merge_wins}/{len(combined)}\n")
    print(f"\nModel saved: {model_path}")

if __name__ == "__main__":
    main()
