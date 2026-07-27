#!/usr/bin/env python3
"""对比 cuSPARSE / opSparse / HSMU / dense / Auto(自适应) 的 SpGEMM 自乘时间。
Auto = src 里的 spgemm_self_product_adaptive(flop>thr→hash 否则 merge3)。
每矩阵:
  cu / Auto  → METHOD=<m> 跑 spgemm_test,compute-only(排除 h2d/d2h)
  opSparse   → 外部 OpSparse binary,"total" ms
  HSMU       → 外部 HSMU test binary,NHC CSV col6
  dense      → spgemm_dense(naive scalar GEMM),"Kernel time" ms
增量写 CSV(可断点续跑),末尾汇总 + Auto vs 各法赢/输(仿 report_methods_cmp)。
用法:compare_methods.py [--dir data/first100] [--out cmp.csv] [--no-hsmu] [--no-opsparse] [--no-dense] [--limit N]
"""
import os, sys, re, csv, subprocess, argparse, time, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
DENSE_BIN = os.path.join(REPO, "spgemm_dense")
HSMU_BIN = os.path.join(REPO, "external_sota/HSMU-SpGEMM/evaluation/script/test")
HSMU_CSV = "/tmp/NHC_4080S_result.csv"
OPSPARSE_RUN = os.path.join(REPO, "external_sota/HSMU-SpGEMM/other_spgemm_code/OpSparse/opsparse")

CALL_TIMEOUT = int(os.environ.get("TIMEOUT", "200"))
DENSE_TIMEOUT = int(os.environ.get("DENSE_TIMEOUT", "600"))   # naive dense 大阵慢

# spgemm_test 方法 → cudaEvent prof tag(adaptive 由 choice 决定)
METHOD_TAG = {"serial": "merge-prof", "merge3": "mrg3-prof", "hash": "hash-prof", "adaptive": None}
# 无 HashProf 的方法(如 cuSPARSE)用 host [dbg][tag] phase 退回
DBG_TAG = {"serial": "merge", "merge3": "mrg3", "cu": "cu"}

def compute_only_from_prof(stderr, tag):
    """[tag-prof] cudaEvent phase → TOTAL(GPU) − h2d − d2h。取最后一次(timed run)。"""
    if not tag:
        return None
    rx = re.compile(r"\[" + re.escape(tag) + r"\]\s+(\S+)\s+([0-9.]+)\s+ms")
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

def compute_only_from_dbg(stderr, tag):
    """[dbg ms][tag] host 时间戳 → 区间和(排除 h2d/d2h)。cuSPARSE 用此口径。"""
    if not tag:
        return None
    phases = {}
    rx = re.compile(r"\[dbg\s+([\d.]+)\s+ms\]\s*\[" + re.escape(tag) + r"\]\s*(\w+)")
    for line in stderr.splitlines():
        m = rx.search(line)
        if m:
            phases[m.group(2)] = float(m.group(1))
    if len(phases) < 2:
        return None
    items = sorted(phases.items(), key=lambda kv: kv[1])
    total = 0.0
    for i in range(1, len(items)):
        if items[i][0] in ("h2d", "d2h", "h2dmalloc"):
            continue
        total += items[i][1] - items[i - 1][1]
    return total

# 跑的 spgemm_test 方法:cuSPARSE + Auto
SPGEMM_METHODS = [("cu", "cu"), ("Auto", "adaptive")]

def find_mtx(name):
    for d in ("data/first100", "data/sota_27_final", "data/sota_27"):
        p = os.path.join(REPO, d, name + ".mtx") if not name.endswith(".mtx") else os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p
    return None

def mtx_header(path):
    n = 0; sym = False; nnz = 0; sized = False
    with open(path) as f:
        first = f.readline(); sym = "symmetric" in first
        for l in f:
            if l.startswith("%"):
                continue
            p = l.split()
            if not p:
                continue
            if not sized:
                n = int(p[0]); sized = True; continue
            nnz += 1
    dens = (nnz / (n * n) * 100.0) if n > 0 else 0.0
    return n, sym, nnz, dens

def run_spgemm_method(mtx, method_key, timeout=CALL_TIMEOUT):
    """METHOD=method_key 跑 spgemm_test → (compute_only_ms, wall_ms, cnnz, choice) 或 None。"""
    env = dict(os.environ, USE_MEMPOOL="1", METHOD=method_key)
    try:
        r = subprocess.run([BIN, mtx], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout
    wall = re.search(r"Time:\s*([0-9.]+)\s*ms", out)
    nz = re.search(r"Result C:.*?nnz\s*=\s*(\d+)", out)
    adapt = " ".join(l for l in out.splitlines() if "[adapt]" in l)
    if "overflow" in adapt:
        choice = "merge3(回退)"
    else:
        mm = re.findall(r"→\s*(hash|merge3)", adapt)
        choice = mm[-1] if mm else ""
    tag = METHOD_TAG.get(method_key)
    if method_key == "adaptive":
        tag = "hash-prof" if choice.startswith("hash") else "mrg3-prof"
    comp = compute_only_from_prof(r.stderr, tag)
    if comp is None:
        comp = compute_only_from_dbg(r.stderr, DBG_TAG.get(method_key))
    if wall:
        w = float(wall.group(1))
        return (comp if comp is not None else w, w, int(nz.group(1)) if nz else -1, choice)
    return None

def run_hsmu(mtx, name, timeout=CALL_TIMEOUT):
    try:
        subprocess.run([HSMU_BIN, mtx], capture_output=True, timeout=timeout)
    except Exception:
        return None
    last = None
    if os.path.exists(HSMU_CSV):
        for l in open(HSMU_CSV):
            p = l.strip().split(",")
            if len(p) >= 7 and p[0] == name:
                try: last = float(p[6])
                except: pass
    return last

def run_opsparse(mtx, timeout=CALL_TIMEOUT):
    try:
        r = subprocess.run([OPSPARSE_RUN, os.path.abspath(mtx)],
                           capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    out = r.stdout + r.stderr
    t = re.search(r"(?m)^\s+total\s+([0-9.]+)ms", out)
    nz = re.search(r"C\.nnz=(\d+)", out)
    if t:
        return (float(t.group(1)), int(nz.group(1)) if nz else None)
    return None

def run_dense(mtx, timeout=DENSE_TIMEOUT):
    """spgemm_dense naive scalar GEMM → 'Kernel time' ms,或 'timeout'/'fail'。"""
    try:
        r = subprocess.run([DENSE_BIN, mtx, "/tmp/dense_cmp.mtx"],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", r.stdout)
    return float(m.group(1)) if m else "fail"

def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="*", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "compare/methods_cmp.csv"))
    ap.add_argument("--no-hsmu", action="store_true")
    ap.add_argument("--no-opsparse", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-ocean", action="store_true", help="(兼容占位,Ocean 已移除)")
    args = ap.parse_args()

    if args.matrices:
        names = args.matrices
    else:
        d = args.dir or os.path.join(REPO, "data/first100")
        names = sorted(os.path.basename(p)[:-4] for p in os.listdir(d) if p.endswith(".mtx"))
    if args.limit:
        names = names[:args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            done.add(r["matrix"])
    fieldnames = ["matrix", "n", "sym", "density_pct", "cu", "Auto",
                  "Auto_choice", "opSparse", "HSMU", "dense", "cnnz"]
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fieldnames)
    if not os.path.exists(args.out) or os.path.getsize(args.out) == 0:
        w.writeheader()

    N = len(names)
    methods_desc = [m[0] for m in SPGEMM_METHODS] + ["opSparse", "HSMU", "dense"]
    print(f"对比 {N} 阵 × {methods_desc} → {args.out}\n", flush=True)
    for i, name in enumerate(names):
        if name in done:
            continue
        p = find_mtx(name)
        if not p:
            print(f"[{i+1}/{N}] {name}: 未找到,跳过", flush=True); continue
        n, sym, annz, dens = mtx_header(p)
        print(f"[{i+1}/{N}] {name:14} n={n:<6}", end="   ", flush=True)
        row = {"matrix": name, "n": n, "sym": "Y" if sym else "N", "density_pct": round(dens, 4)}
        t0 = time.time()
        for label, key in SPGEMM_METHODS:
            r = run_spgemm_method(p, key)
            row[label] = round(r[0], 3) if r else ""
            if label == "Auto" and r:
                row["cnnz"] = r[2]
                row["Auto_choice"] = r[3]
        op = run_opsparse(p) if not args.no_opsparse else None
        row["opSparse"] = round(op[0], 3) if op else ("" if args.no_opsparse else "DNF")
        if op and op[1] and row.get("cnnz") and str(op[1]) != str(row["cnnz"]):
            print(f"  ⚠ opSparse C.nnz={op[1]} ≠ Auto cnnz={row['cnnz']}", flush=True)
        row["HSMU"] = round(run_hsmu(p, name), 3) if not args.no_hsmu else ""
        if not args.no_dense:
            d = run_dense(p)
            row["dense"] = round(d, 3) if isinstance(d, float) else d
        w.writerow(row); fout.flush()
        dt = time.time() - t0
        dense_str = f" dense={row.get('dense','-')!s:>8}" if not args.no_dense else ""
        print(f"cu={row['cu']!s:>7} Auto={row['Auto']!s:>7}({row.get('Auto_choice','?'):<6}) "
              f"opSp={row['opSparse']!s:>7} HSMU={row['HSMU']!s:>7}{dense_str} ({dt:.1f}s)", flush=True)
    fout.close()
    print(f"\n完成 → {args.out}")

    rows = list(csv.DictReader(open(args.out)))
    print("\n=== 几何均值(ms) ===")
    for col in ["cu", "Auto", "opSparse", "HSMU", "dense"]:
        xs = []
        for r in rows:
            try: xs.append(float(r.get(col)))
            except (TypeError, ValueError): pass
        if xs:
            print(f"  {col:9} {geomean(xs):8.3f}")
    n_hash = sum(1 for r in rows if r.get("Auto_choice") == "hash")
    n_m3 = sum(1 for r in rows if r.get("Auto_choice", "").startswith("merge3"))
    print(f"\n=== Auto 选择(共 {n_hash + n_m3} 阵)===  hash {n_hash} / merge3 {n_m3}")

    # ---- Auto vs 各基线 赢/输(仿 report_methods_cmp vs_section)----
    def vs_section(ref_col, ref_name):
        pairs = []
        for r in rows:
            try: t = float(r.get("Auto")); ref = float(r.get(ref_col))
            except (TypeError, ValueError): continue
            if t > 0 and ref > 0: pairs.append((r, t / ref))
        if not pairs:
            print(f"\n=== Auto vs {ref_name} === (无数据)"); return
        win = [p for p in pairs if p[1] <= 1.0]
        lose = sorted([p for p in pairs if p[1] > 1.0], key=lambda p: -p[1])
        gm = math.exp(sum(math.log(p[1]) for p in pairs) / len(pairs))
        print(f"\n=== Auto vs {ref_name}:赢 {len(win)} / 输 {len(lose)},几何均值(Auto/{ref_name}) {gm:.3f}× ===")
        print(f"  {'name':<16}{'n':>8}{'C_nnz':>12}{ref_name:>10}{'Auto':>10}{'ratio':>9}")
        for r, ratio in lose[:15]:
            try: rt = float(r.get("Auto")); rf = float(r.get(ref_col))
            except: rt = rf = 0.0
            print(f"  {r['matrix']:<16}{r.get('n',''):>8}{r.get('cnnz',''):>12}{rf:>10.3f}{rt:>10.3f}{ratio:>8.2f}×")
        if len(lose) > 15:
            print(f"  ... 另有 {len(lose)-15} 个")
    vs_section("cu", "cuSPARSE")
    if not args.no_opsparse: vs_section("opSparse", "opSparse")
    if not args.no_hsmu:    vs_section("HSMU", "HSMU")
    if not args.no_dense:   vs_section("dense", "dense")

if __name__ == "__main__":
    main()
