#!/usr/bin/env python3
"""Run-all-methods comparison: Auto (ours) vs cuSPARSE / dense / opSparse / HSMU
on first100. NO Ocean, NO merge3/serial. All timed compute-only (exclude h2d/d2h)
where possible. Incremental (resume). Writes CSV + speedup summary.

This is the canonical "run all methods" driver. Five methods per matrix:
    Auto      = adaptive (hash/merge3)  cudaEvent compute-only (TOTAL - h2d - d2h)
    cuSPARSE  = METHOD=cu  [dbg ms][cu] phase intervals (excl h2d/d2h)
    dense     = spgemm_dense  naive scalar GEMM kernel (densify+dgemm+sparsify)
    opSparse  = external OpSparse binary, "total" ms
    HSMU      = external HSMU test binary, NHC CSV col6

Usage:
  .venv/bin/python scripts/compare_paper.py                 # full run (all 5), resume
  .venv/bin/python scripts/compare_paper.py --dense-only    # regenerate ONLY the dense
                                                            # column (keep the other 4 as-is)
  TIMEOUT=200 DENSE_TIMEOUT=600 .venv/bin/python scripts/compare_paper.py --dense-only
"""
import os, sys, re, csv, subprocess, argparse, time, math
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
DENSE_BIN = os.path.join(REPO, "spgemm_dense")
HSMU_BIN = os.path.join(REPO, "external_sota/HSMU-SpGEMM/evaluation/script/test")
HSMU_CSV = "/tmp/NHC_4080S_result.csv"
OPSPARSE_RUN = os.path.join(REPO, "external_sota/HSMU-SpGEMM/other_spgemm_code/OpSparse/opsparse")
CALL_TIMEOUT = int(os.environ.get("TIMEOUT", "200"))
# naive dense is slow on large matrices; give it its own (larger) budget.
DENSE_TIMEOUT = int(os.environ.get("DENSE_TIMEOUT", "600"))

def find_mtx(name):
    for d in ("data/first100", "data/sota_27_final", "data/sota_27"):
        p = os.path.join(REPO, d, name + ".mtx")
        if os.path.exists(p): return p
    return None

def mtx_header(path):
    n=0; sym=False; nnz=0; sized=False
    with open(path) as f:
        first=f.readline(); sym="symmetric" in first
        for l in f:
            if l.startswith("%"): continue
            p=l.split()
            if not p: continue
            if not sized: n=int(p[0]); sized=True; continue
            nnz+=1
    return n, sym, nnz, (nnz/(n*n)*100.0 if n>0 else 0.0)

# ---- Auto (adaptive): cudaEvent compute-only (TOTAL - h2d - d2h) ----
def run_auto(mtx):
    env=dict(os.environ, USE_MEMPOOL="1", METHOD="adaptive")
    try: r=subprocess.run([BIN,mtx],capture_output=True,text=True,env=env,timeout=CALL_TIMEOUT)
    except subprocess.TimeoutExpired: return None
    err=r.stdout+r.stderr
    choice="hash" if "→ hash" in err else ("merge3" if "→ merge3" in err else "")
    tag="hash-prof" if choice=="hash" else "mrg3-prof"
    ph={}
    for ln in err.splitlines():
        m=re.search(r"\["+re.escape(tag)+r"\]\s+(\S+)\s+([0-9.]+)\s+ms",ln)
        if m: ph[m.group(1)]=float(m.group(2))
    tot=ph.get("TOTAL(GPU)")
    nz=re.search(r"Result C:.*?nnz\s*=\s*(\d+)", r.stdout)
    comp=(tot-ph.get("h2d",0)-ph.get("d2h",0)) if tot else None
    return (comp, choice, int(nz.group(1)) if nz else -1)

# ---- cuSPARSE: compute-only from [dbg ms][cu] phase intervals (exclude h2d/d2h) ----
def run_cusparse(mtx):
    env=dict(os.environ, USE_MEMPOOL="1", METHOD="cu")
    try: r=subprocess.run([BIN,mtx],capture_output=True,text=True,env=env,timeout=CALL_TIMEOUT)
    except subprocess.TimeoutExpired: return None
    phases={}
    for ln in (r.stdout+r.stderr).splitlines():
        m=re.search(r"\[dbg\s+([\d.]+)\s+ms\]\s*\[cu\]\s*(\w+)",ln)
        if m: phases[m.group(2)]=float(m.group(1))
    if len(phases)>=2:
        items=sorted(phases.items(),key=lambda kv:kv[1])
        total=0.0
        for i in range(1,len(items)):
            if items[i][0] in ("h2d","d2h","h2dmalloc"): continue
            total+=items[i][1]-items[i-1][1]
        return total
    m=re.search(r"Time:\s*([0-9.]+)\s*ms",r.stdout)
    return float(m.group(1)) if m else None

# ---- dense: naive scalar GEMM cudaEvent kernel time (densify+dgemm+sparsify) ----
def run_dense(mtx):
    try: r=subprocess.run([DENSE_BIN,mtx,"/tmp/dense_cmp.mtx"],capture_output=True,text=True,timeout=DENSE_TIMEOUT)
    except subprocess.TimeoutExpired: return "timeout"
    m=re.search(r"Kernel time:\s*([0-9.]+)\s*ms",r.stdout)
    return float(m.group(1)) if m else "fail"

def run_opsparse(mtx):
    try: r=subprocess.run([OPSPARSE_RUN,os.path.abspath(mtx)],capture_output=True,text=True,timeout=CALL_TIMEOUT)
    except Exception: return None
    out=r.stdout+r.stderr
    t=re.search(r"(?m)^\s+total\s+([0-9.]+)ms",out)
    return float(t.group(1)) if t else None

def run_hsmu(mtx, name):
    try: subprocess.run([HSMU_BIN,mtx],capture_output=True,timeout=CALL_TIMEOUT)
    except Exception: return None
    last=None
    if os.path.exists(HSMU_CSV):
        for l in open(HSMU_CSV):
            p=l.strip().split(",")
            if len(p)>=7 and p[0]==name:
                try: last=float(p[6])
                except: pass
    return last

def geomean(xs): return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float('nan')

def fnum(x):
    try: return float(x)
    except: return None

# ---- speedup summary: baseline/Auto (>1 = Auto faster); dense timeout/fail = Auto win ----
def summary(rows):
    print("\n=== Auto speedup (baseline_time / Auto_time, geomean; >1 = we win) ===")
    for col in ["cuSPARSE","dense","opSparse","HSMU"]:
        ratios=[]; win=0; lose=0; dnf=0
        for r in rows:
            ta=fnum(r.get("Auto")); tb=fnum(r.get(col))
            if not (ta and ta>0): continue
            if tb is None or (isinstance(r.get(col),str) and r[col] in ("timeout","fail")):
                dnf+=1; win+=1; continue      # baseline DNF => Auto wins
            if tb>0:
                ratios.append(tb/ta)
                if tb>=ta: win+=1
                else: lose+=1
        extra=f" (+{dnf} DNF)" if dnf else ""
        if ratios:
            print(f"  vs {col:9}: win {win:3}/lose {lose:3}{extra} | geomean speedup {geomean(ratios):.2f}x")

def dens_class(d):
    if d >= 10.0:  return "Dense"
    if d >= 1.0:   return "Mildly sparse"
    if d >= 0.1:   return "Highly sparse"
    return "Extremely sparse"

def _gm_speedup(rows, col):
    rs = []
    for r in rows:
        a, b = fnum(r.get("Auto")), fnum(r.get(col))
        if a and b and a > 0 and b > 0:
            rs.append(b / a)
    return geomean(rs) if rs else float('nan')

def report(rows):
    """Full report: per-matrix timings + per-density speedup + overall summary."""
    cols = ["cuSPARSE", "opSparse", "HSMU", "dense"]
    # ---- per-matrix table ----
    print("\n=== Per-matrix timings (ms, compute-only) ===")
    hdr = f"{'matrix':16} {'n':>6} {'dens%':>6}  {'Auto':>8}  " + "  ".join(f"{c:>9}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: int(x["n"]) if str(x.get("n")).isdigit() else 0):
        def c(k):
            v = r.get(k, "")
            try: return f"{float(v):9.3f}"
            except: return f"{v:>9}"
        dens = fnum(r.get("density_pct"))
        print(f"{r['matrix']:16} {r.get('n',''):>6} {(f'{dens:.2f}' if dens else ''):>6}  "
              f"{c('Auto')}  " + "  ".join(c(k) for k in cols))
    # ---- per-density class geomean speedup ----
    print("\n=== Auto speedup by density class (geomean ×, >1 = Auto faster) ===")
    for g in ["Dense", "Mildly sparse", "Highly sparse", "Extremely sparse"]:
        sub = [r for r in rows if dens_class(fnum(r.get("density_pct")) or 0) == g]
        if not sub:
            continue
        cells = "  ".join(f"{_gm_speedup(sub, c):>8.2f}×" if _gm_speedup(sub, c)==_gm_speedup(sub, c) else f"{'—':>9}"
                          for c in cols)
        print(f"  {g:18} ({len(sub):2}):  " + cells)
    print("  " + "  ".join(f"{c:>9}" for c in cols))
    summary(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",default=os.path.join(REPO,"data/first100"))
    ap.add_argument("--out",default=os.path.join(REPO,"compare/paper_cmp.csv"))
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--dense-only",action="store_true",
                    help="regenerate only the dense column; keep Auto/cuSPARSE/opSparse/HSMU")
    ap.add_argument("--report",action="store_true",
                    help="just print the full report from the existing CSV (no runs)")
    args=ap.parse_args()

    if args.report:
        if not os.path.exists(args.out):
            sys.exit(f"--report needs existing {args.out}; run a full pass first.")
        report(list(csv.DictReader(open(args.out))))
        return

    names=sorted(os.path.basename(p)[:-4] for p in os.listdir(args.dir) if p.endswith(".mtx"))
    if args.limit: names=names[:args.limit]
    os.makedirs(os.path.dirname(args.out),exist_ok=True)
    fieldnames=["matrix","n","sym","density_pct","Auto","Auto_choice","cuSPARSE","dense","opSparse","HSMU","cnnz"]

    # ---- dense-only: load existing rows, refresh dense column, rewrite ----
    if args.dense_only:
        if not os.path.exists(args.out):
            sys.exit(f"--dense-only needs existing {args.out}; run a full pass first.")
        rows=list(csv.DictReader(open(args.out)))
        print(f"Dense-only refresh: {len(rows)} matrices (DENSE_TIMEOUT={DENSE_TIMEOUT}s) → {args.out}\n",flush=True)
        for i,r in enumerate(rows):
            p=find_mtx(r["matrix"])
            if not p: print(f"[{i+1}/{len(rows)}] {r['matrix']}: not found",flush=True); continue
            t0=time.time(); d=run_dense(p); r["dense"]=(round(d,3) if isinstance(d,float) else d)
            print(f"[{i+1}/{len(rows)}] {r['matrix']:14} n={r['n']:<6} dense={r['dense']:<10} ({time.time()-t0:.1f}s)",flush=True)
            # write back incrementally so progress is saved
            with open(args.out,"w",newline="") as fp:
                w=csv.DictWriter(fp,fieldnames=fieldnames); w.writeheader()
                for rr in rows: w.writerow({k:rr.get(k,"") for k in fieldnames})
        report(rows)
        return

    # ---- full pass (all 5 methods), incremental resume ----
    done=set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)): done.add(r["matrix"])
    fout=open(args.out,"a",newline=""); w=csv.DictWriter(fout,fieldnames=fieldnames)
    if not os.path.exists(args.out) or os.path.getsize(args.out)==0: w.writeheader()
    N=len(names)
    print(f"Paper cmp (all 5 methods): {N} matrices (resume: {len(done)} done) → {args.out}\n",flush=True)
    for i,name in enumerate(names):
        if name in done: continue
        p=find_mtx(name)
        if not p: print(f"[{i+1}/{N}] {name}: not found",flush=True); continue
        n,sym,annz,dens=mtx_header(p)
        print(f"[{i+1}/{N}] {name:14} n={n:<6}",end="   ",flush=True)
        t0=time.time(); row={"matrix":name,"n":n,"sym":"Y" if sym else "N","density_pct":round(dens,4)}
        a=run_auto(p)
        if a: row["Auto"]=round(a[0],3); row["Auto_choice"]=a[1]; row["cnnz"]=a[2]
        cu=run_cusparse(p); row["cuSPARSE"]=round(cu,3) if cu is not None else ""
        d=run_dense(p);     row["dense"]=round(d,3) if isinstance(d,float) else d
        op=run_opsparse(p); row["opSparse"]=round(op,3) if op is not None else ""
        hs=run_hsmu(p,name);row["HSMU"]=round(hs,3) if hs is not None else ""
        w.writerow(row); fout.flush(); dt=time.time()-t0
        print(f"Auto={row['Auto']}({row['Auto_choice']}) cu={row['cuSPARSE']} dense={row['dense']} "
              f"opSp={row['opSparse']} HSMU={row['HSMU']} ({dt:.1f}s)",flush=True)
    fout.close()
    rows=list(csv.DictReader(open(args.out)))
    report(rows)

if __name__=="__main__": main()
