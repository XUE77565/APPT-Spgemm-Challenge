#!/usr/bin/env python3
"""Hash vs Merge3 multi-variable analysis (numpy-only, no sklearn).
Collect max_row_nnz from mtx, fit multi-variable model, generate charts.

Usage: .venv/bin/python scripts/hash_vs_merge_v2.py [hash_vs_merge.csv] [output_dir]
"""
import os, sys, csv, math, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
from numpy.linalg import lstsq

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data/first100")

C_HASH='#2a78d6'; C_MERGE='#eb6834'; C_DIAG='#898781'; C_GRID='#e1e0d9'
C_TEXT='#0b0b0b'; C_TEXT2='#52514e'; C_SURFACE='#fcfcfb'

def read_mtx_row_stats(path):
    n=0; sym=False; rc=collections.Counter()
    with open(path) as f:
        first=f.readline(); sym="symmetric" in first
        for l in f:
            if l.startswith("%"): continue
            p=l.split()
            if not p: continue
            if n==0: n=int(p[0]); continue
            r=int(p[0]); rc[r]+=1
            if sym and int(p[0])!=int(p[1]): rc[int(p[1])]+=1
    nnz=sum(rc.values()); mr=max(rc.values()) if rc else 0
    return n, sym, nnz, mr, (nnz/n if n else 0)

def main():
    csv_path = sys.argv[1] if len(sys.argv)>1 else os.path.join(REPO,"compare/hash_vs_merge/hash_vs_merge.csv")
    out_dir  = sys.argv[2] if len(sys.argv)>2 else os.path.join(REPO,"compare/hash_vs_merge")

    data=[]
    for r in csv.DictReader(open(csv_path)): data.append(dict(r))
    print(f"Loaded {len(data)} matrices")

    for r in data:
        mtx=os.path.join(DATA, r["matrix"]+".mtx")
        if os.path.exists(mtx):
            n,sym,nnz,mr,avg=read_mtx_row_stats(mtx)
            r["max_row_nnz"]=mr; r["avg_row_nnz"]=avg; r["skew"]=mr/max(avg,1e-9); r["A_nnz_actual"]=nnz
        else:
            r["max_row_nnz"]=0; r["avg_row_nnz"]=0; r["skew"]=0; r["A_nnz_actual"]=0
    for r in data:
        for k in ("n","A_nnz","C_nnz_flop","max_row_nnz","avg_row_nnz"): r[k]=int(float(r[k]))
        for k in ("merge3_ms","hash_ms","ratio","density_pct","skew","A_nnz_actual"): r[k]=float(r[k])
        r["hash_wins"]=1 if r["ratio"]<1.0 else 0

    hw=sum(r["hash_wins"] for r in data)
    print(f"hash wins: {hw}/{len(data)}")

    # Features in log10 space
    feats=["flop_proxy","n","max_row_nnz"]
    X=np.array([[math.log10(max(r["C_nnz_flop"],1)), math.log10(max(r["n"],1)),
                 math.log10(max(r["max_row_nnz"],1))] for r in data])
    Y_ratio=np.array([math.log10(max(r["ratio"],1e-3)) for r in data])
    y_label=np.array([r["hash_wins"] for r in data])

    # Multi-variable linear model: log10(ratio) = β0 + β1*log10(fp) + β2*log10(n) + β3*log10(mr)
    X_aug=np.column_stack([X, np.ones(len(Y_ratio))])
    cf,_,_,_=lstsq(X_aug, Y_ratio, rcond=None)
    Y_pred=X_aug@cf
    r2=1-np.sum((Y_ratio-Y_pred)**2)/np.sum((Y_ratio-Y_ratio.mean())**2)
    print(f"\nMulti-variable model (R²={r2:.3f}):")
    names=feats+["const"]
    for n2,c in zip(names,cf): print(f"  {n2}: {c:.4f}")

    # Classification accuracy (ratio=1 boundary)
    y_pred_cls=(Y_pred < 0).astype(int)
    acc=np.mean(y_pred_cls==y_label)
    print(f"Classification accuracy (ratio=1 boundary): {acc:.1%}")

    # Single-variable for comparison
    X1=np.column_stack([X[:,0], np.ones(len(Y_ratio))])
    cf1,_,_,_=lstsq(X1, Y_ratio, rcond=None)
    Y_pred1=X1@cf1
    r2_1=1-np.sum((Y_ratio-Y_pred1)**2)/np.sum((Y_ratio-Y_ratio.mean())**2)
    print(f"Single-variable (flop_proxy only) R²={r2_1:.3f}")

    # ================================================================
    # Chart 1: 2-panel decision space
    # ================================================================
    fig,axes=plt.subplots(1,2,figsize=(14,6),facecolor=C_SURFACE)
    hd=[r for r in data if r["hash_wins"]]
    md=[r for r in data if not r["hash_wins"]]

    def plot_panel(ax, xlabel, ylabel, xdata_key, ydata_key, title, med_idx):
        ax.set_facecolor(C_SURFACE)
        ax.scatter([r[xdata_key] for r in md],[r[ydata_key] for r in md],
                   c=C_MERGE,s=42,alpha=0.7,edgecolors='white',linewidths=0.4,marker='o',zorder=3,
                   label=f'merge3 wins ({len(md)})')
        ax.scatter([r[xdata_key] for r in hd],[r[ydata_key] for r in hd],
                   c=C_HASH,s=48,alpha=0.8,edgecolors='white',linewidths=0.4,marker='^',zorder=4,
                   label=f'hash wins ({len(hw)})' if False else f'hash wins ({len(hd)})')

        # Decision boundary from multi-var model
        # log10(ratio)=0 → β1*log10(x) + β2*log10(y) + β3*log10(med) + β0 = 0
        # → log10(y) = -(β1*log10(x) + β3*log10(med) + β0) / β2
        b0,b_fp,b_n,b_mr = cf[3],cf[0],cf[1],cf[2]
        med_val=np.median([math.log10(max(r[med_idx],1)) for r in data])
        x_fit=np.logspace(1,11,200)
        if med_idx=="max_row_nnz":
            # Panel A: x=flop_proxy, y=n, fix max_row_nnz at median
            log_y=-(b_fp*np.log10(x_fit)+b_mr*med_val+b0)/b_n
        else:
            # Panel B: x=flop_proxy, y=max_row_nnz, fix n at median
            log_y=-(b_fp*np.log10(x_fit)+b_n*med_val+b0)/b_mr
        y_fit=10**log_y
        valid=(y_fit>0.5)&(y_fit<1e6)
        ax.plot(x_fit[valid],y_fit[valid],color=C_DIAG,lw=2,ls='--',zorder=2,
                label=f'Decision boundary (multi-var)')

        for name in ["bcsstk30","bcsstk32","bcsstk08","bp_0","bcsstk16","bcsstk21","bcsstk24","bcsstk14","can_715"]:
            r=next((r for r in data if r["matrix"]==name),None)
            if r: ax.annotate(name,(r[xdata_key],r[ydata_key]),fontsize=6,color=C_TEXT2,xytext=(4,3),textcoords='offset points')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(xlabel,fontsize=10,color=C_TEXT)
        ax.set_ylabel(ylabel,fontsize=10,color=C_TEXT)
        ax.set_title(title,fontsize=9,color=C_TEXT,pad=6)
        ax.legend(fontsize=7.5,loc='upper left',framealpha=0.9,edgecolor=C_GRID)
        ax.grid(True,which='major',color=C_GRID,lw=0.5)
        ax.tick_params(labelsize=8,colors=C_TEXT2)
        for sp in ax.spines.values(): sp.set_color(C_GRID)

    plot_panel(axes[0], 'flop proxy = $A_{nnz}^2 / n$', 'matrix dimension $n$',
               'C_nnz_flop','n','(a) flop proxy vs $n$ (fixed median max-row)', 'max_row_nnz')
    plot_panel(axes[1], 'flop proxy = $A_{nnz}^2 / n$', 'max row nnz',
               'C_nnz_flop','max_row_nnz','(b) flop proxy vs max-row-nnz (fixed median $n$)', 'n')

    fig.suptitle('Hash vs Merge3 Dispatcher Decision Space (H100 PCIe, 100 matrices)',
                 fontsize=11,color=C_TEXT,y=1.02)
    fig.tight_layout()
    p1=os.path.join(out_dir,"dispatcher_decision_2d.png")
    fig.savefig(p1,dpi=200,facecolor=C_SURFACE,bbox_inches='tight')
    print(f"\nChart: {p1}")
    plt.close(fig)

    # ================================================================
    # Chart 2: Enlarged ratio vs flop_proxy (colored by max_row_nnz)
    # ================================================================
    fig2,ax2=plt.subplots(figsize=(9,6.5),facecolor=C_SURFACE)
    ax2.set_facecolor(C_SURFACE)
    mrs=np.log10(np.clip([r["max_row_nnz"] for r in data],1,None))
    norm=plt.Normalize(mrs.min(),mrs.max())
    sc=ax2.scatter([r["C_nnz_flop"] for r in data],[r["ratio"] for r in data],
                   c=mrs,cmap='plasma',norm=norm,s=55,alpha=0.8,edgecolors='white',linewidths=0.5,zorder=3)
    ax2.axhline(y=1.0,color=C_DIAG,lw=1.5,ls='--',zorder=1,label='Break-even (ratio = 1)')
    # Multi-var prediction (fix n and max_row at median)
    med_n=np.median([math.log10(max(r["n"],1)) for r in data])
    med_mr=np.median([math.log10(max(r["max_row_nnz"],1)) for r in data])
    x_fit2=np.logspace(1,11,200)
    log_ratio_pred=cf[0]*np.log10(x_fit2)+cf[1]*med_n+cf[2]*med_mr+cf[3]
    y_fit2=10**log_ratio_pred
    ax2.plot(x_fit2,y_fit2,color=C_HASH,lw=2,alpha=0.6,zorder=2,
             label=f'Multi-var fit (R²={r2:.3f})')

    cbar=fig2.colorbar(sc,ax=ax2,shrink=0.75,pad=0.02)
    cbar.set_label('max row nnz (log$_{10}$)',fontsize=9,color=C_TEXT)
    cbar.ax.tick_params(labelsize=7,colors=C_TEXT2)
    for name in ["bcsstk30","bcsstk32","bcsstk08","bp_0","bcsstk16","bcsstk21","bcsstk24","bcsstk14","can_715"]:
        r=next((r for r in data if r["matrix"]==name),None)
        if r:
            off=(5,5) if r["ratio"]<1.5 else (5,-8)
            ax2.annotate(name,(r["C_nnz_flop"],r["ratio"]),fontsize=6.5,color=C_TEXT2,xytext=off,textcoords='offset points')
    ax2.set_xscale('log');ax2.set_yscale('log')
    ax2.set_xlabel('flop proxy = $A_{nnz}^2 / n$',fontsize=11,color=C_TEXT)
    ax2.set_ylabel('hash / merge3 time ratio',fontsize=11,color=C_TEXT)
    ax2.set_title('Hash vs Merge3 Ratio by Flop Proxy\n(color = max row nnz; blue line = multi-var model at median $n$ & max-row)',
                  fontsize=10,color=C_TEXT,pad=8)
    ax2.legend(fontsize=8,loc='upper right',framealpha=0.9,edgecolor=C_GRID)
    ax2.grid(True,which='major',color=C_GRID,lw=0.5)
    ax2.grid(True,which='minor',color=C_GRID,lw=0.3,alpha=0.5)
    ax2.tick_params(labelsize=9,colors=C_TEXT2)
    for sp in ax2.spines.values(): sp.set_color(C_GRID)
    ax2.text(0.03,0.97,f'hash wins {hw}/{len(data)}',transform=ax2.transAxes,fontsize=9,color=C_HASH,fontweight='bold',va='top')
    ax2.text(0.97,0.03,f'merge3 wins {len(data)-hw}/{len(data)}',transform=ax2.transAxes,fontsize=9,color=C_MERGE,fontweight='bold',ha='right',va='bottom')
    fig2.tight_layout()
    p2=os.path.join(out_dir,"ratio_vs_flopproxy_multivar.png")
    fig2.savefig(p2,dpi=200,facecolor=C_SURFACE,bbox_inches='tight')
    print(f"Chart: {p2}")
    plt.close(fig2)

    # ================================================================
    # Save enriched CSV + model
    # ================================================================
    ep=os.path.join(out_dir,"hash_vs_merge_enriched.csv")
    fields=["matrix","n","density_pct","A_nnz","A_nnz_actual","max_row_nnz","avg_row_nnz","skew","C_nnz_flop","merge3_ms","hash_ms","ratio","hash_wins"]
    with open(ep,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in data: w.writerow({k:r.get(k,"") for k in fields})
    print(f"Enriched CSV: {ep}")

    mp=os.path.join(out_dir,"dispatcher_model_v2.txt")
    with open(mp,"w") as f:
        f.write("Multi-Variable Dispatcher Model (H100 PCIe, 100 matrices)\n")
        f.write("="*60+"\n\n")
        f.write("log10(hash/merge3 ratio) = ")
        f.write(" + ".join(f"{cf[i]:.4f}×log10({feats[i]})" for i in range(3)))
        f.write(f" + {cf[3]:.4f}\n")
        f.write(f"  R² = {r2:.3f}\n")
        f.write(f"  Classification accuracy (ratio=1 boundary): {acc:.1%}\n\n")
        f.write(f"Decision: hash when log10(ratio) < 0\n")
        f.write(f"  → {cf[0]:.4f}×log10(flop_proxy) + {cf[1]:.4f}×log10(n) + {cf[2]:.4f}×log10(max_row_nnz) < {-cf[3]:.4f}\n\n")
        f.write(f"Single-variable (flop_proxy only) R² = {r2_1:.3f} → multi-var adds {(r2-r2_1)*100:.1f}pp\n\n")
        f.write(f"hash wins: {hw}/{len(data)}, merge3 wins: {len(data)-hw}/{len(data)}\n")
    print(f"Model: {mp}")

if __name__=="__main__":
    main()
