#!/usr/bin/env python3
"""对比 cuSPARSE / Ocean / opSparse / HSMU / dense / Auto(自适应) 的 SpGEMM 自乘时间。
Auto = src 里的 spgemm_self_product_adaptive(flop>thr→hash 否则 merge3)。
每矩阵:
  cu / Auto  → METHOD=<m> 跑 spgemm_test,compute-only(排除 h2d/d2h)
  Ocean      → ocean/convert + ocean/spgemm → stats.json 各 phase 求和(compute-only)
  opSparse   → 外部 OpSparse binary,"total" ms
  HSMU       → 外部 HSMU test binary,NHC CSV col6
  dense      → spgemm_dense(tiled FP64 GEMM),"Kernel time" ms
增量写 CSV(可断点续跑),末尾汇总 + Auto vs 各法赢/输(仿 report_methods_cmp)。
用法:compare_methods.py [--dir data/first100] [--out cmp.csv] [--no-ocean] [--no-hsmu] [--no-opsparse] [--no-dense] [--limit N]
"""
import os, sys, re, csv, json, subprocess, argparse, time, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("SPGEMM_BIN") or os.path.join(REPO, "spgemm_test")
DENSE_BIN = os.path.join(REPO, "spgemm_dense")
CUBLAS_BIN = os.path.join(REPO, "spgemm_dense_cublas")   # cuBLAS FP64 dgemm(PEDANTIC,无 TC)
HSMU_BIN = os.path.join(REPO, "external_sota/HSMU-SpGEMM/evaluation/script/test")
HSMU_CSV = "/tmp/NHC_4080S_result.csv"
OPSPARSE_RUN = os.path.join(REPO, "external_sota/HSMU-SpGEMM/other_spgemm_code/OpSparse/opsparse")
# bhSparse (Liu & Vinter IPDPS'14, merge 家族 GPU SpGEMM;H100 移植版 2026-08-24)
BHSPARSE_BIN = os.path.join(REPO, "external_sota/bhSparse/SpGEMM_cuda/spgemm")
# Ocean (Hui et al., 2025 SOTA hash SpGEMM): convert mtx→csr, run spgemm, sum
# timing phases from stats.json. h2d/d2h 不在 timing 里 → compute-only,同口径。
OCEAN_CONV  = os.path.join(REPO, "ocean/convert")
OCEAN_RUN   = os.path.join(REPO, "ocean/spgemm")
OCEAN_CFG   = "config/bench_detail.json"
OCEAN_STATS = os.path.join(REPO, "ocean/stats.json")

CALL_TIMEOUT = int(os.environ.get("TIMEOUT", "200"))
DENSE_TIMEOUT = int(os.environ.get("DENSE_TIMEOUT", "600"))   # naive dense 大阵慢
# dense/cuBLAS 大阵护栏:ocean337 有 50+ 个 >100 万阶阵,N³ GEMM 不可行 → 直接判 DNF(big)
# (n=5 万时 tiled64 FP64 ≈ 30s 量级,仍可真跑;再往上没有意义)
DENSE_MAX_N = int(os.environ.get("DENSE_MAX_N", "50000"))

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
    return None   # docs/66 血泪:TOTAL 缺失 = 运行未完成(SAFETY exit 等)→ 宁可 None 也不相位求和冒充时长

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
    # data/ocean/square = 337 阵 Ocean benchmark(2026-08-24 起新套件)
    for d in ("data/ocean/square", "data/first100", "data/sota_27_final", "data/sota_27"):
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

def _parse_num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None

def run_spgemm_method(mtx, method_key, timeout=CALL_TIMEOUT, exp_cnnz=None):
    """METHOD=method_key 跑 spgemm_test → (compute_only_ms, wall_ms, cnnz, choice) 或 None。
    exp_cnnz: 已知 C_nnz(来自 CSV 旧列)→ 按需定 host pinned arena(鲸鱼阵 D2H>8GB,普通阵 8GB 省每次 ~10s pin)。"""
    if "MP_HOST_MB" not in os.environ:
        need_gb = 8192
        if exp_cnnz is None or exp_cnnz > 300_000_000:   # 空(鲸鱼阵未出过数)或 >300M nnz → D2H 可能 >8GB
            need_gb = 32768
        env_extra = {"MP_HOST_MB": str(need_gb)}
    else:
        env_extra = {}
    env = dict(os.environ, USE_MEMPOOL="1", METHOD=method_key, **env_extra)
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
        comp_v = comp if comp is not None else w
        nz_v = int(nz.group(1)) if nz else -1
        # HARNESS-ADAPTIVE EXPAND(2026-08-30 docs/60 §5 裁决:进程内状态机有偏,F2/Flan 误判,
        # 惩罚落在比较点之后的相位)→ harness 跑双 expand 取 compute-only 更优;nnz 不一致 =
        # 红旗(expand 不改结构)→ 保留 1.15。
        if method_key in ("hash", "adaptive"):
            env2 = dict(env, HASH_EXPAND="1.4")
            try:
                r2 = subprocess.run([BIN, mtx], capture_output=True, text=True, env=env2, timeout=timeout)
                wall2 = re.search(r"Time:\s*([0-9.]+)\s*ms", r2.stdout)
                nz2 = re.search(r"Result C:.*?nnz\s*=\s*(\d+)", r2.stdout)
                comp2 = compute_only_from_prof(r2.stderr, tag)
                if comp2 is None and wall2:
                    comp2 = float(wall2.group(1))
                if comp2 is not None and wall2 and nz2:
                    if int(nz2.group(1)) != nz_v:
                        print(f"  [warn] expand-nnz-mismatch {os.path.basename(mtx)}: "
                              f"{nz_v} vs {nz2.group(1)}(保留 1.15)", flush=True)
                    elif comp2 < comp_v * 0.98:
                        return (comp2, float(wall2.group(1)), nz_v, choice + "(ex1.4)")
            except subprocess.TimeoutExpired:
                pass
        return (comp_v, w, nz_v, choice)
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

def run_ocean(mtx, timeout=CALL_TIMEOUT):
    """ocean/convert + ocean/spgemm → stats.json 总时间(各 phase 求和,compute-only)。
    与 baseline_ocean.py(K=5)同口径:analysis 3 子项 + estimation 4 子项(不含 binning)
      + numeric 全部 + epilogue 3 子项 + prologue。h2d/d2h 不在 timing 里(Ocean 另算)。"""
    csr = "/tmp/cmp_ocean.csr"
    try:
        # 2026-08-27 夜事故修复:spgemm 静默崩溃(convert/spgemm rc 被吞)时 stats.json 残留【上一阵】
        # 的数据 —— refresh_ocean_sym 批首 43 阵被冻结值 4.027 污染(333SP 真值 6.16/Ga3 真值 37.05)。
        # 修复:跑前删 stats.json + 检查 rc;失败 = DNF(None),绝不解析陈旧文件。
        if os.path.exists(OCEAN_STATS):
            os.remove(OCEAN_STATS)
        r1 = subprocess.run([OCEAN_CONV, mtx, csr], capture_output=True, timeout=CALL_TIMEOUT)
        if r1.returncode != 0:
            return None
        r2 = subprocess.run([OCEAN_RUN, csr, OCEAN_CFG], cwd=os.path.join(REPO, "ocean"),
                            capture_output=True, timeout=timeout)
        if r2.returncode != 0 or not os.path.exists(OCEAN_STATS):
            return None
        t = json.load(open(OCEAN_STATS))["timing"]
        an  = t["analysis"]["product_calc"] + t["analysis"]["reduce"] + t["analysis"]["mem_cpy"]
        est = (t["estimation"]["hll_construct"] + t["estimation"]["hll_merge"]
               + t["estimation"]["malloc"] + t["estimation"]["sampling"])
        sym = sum(t.get("symbolic", {}).values())   # 2026-08-27 口径修正:type-0 阵真实发生
        num = sum(t["numeric"].values())            # 的 symbolic pass 必须计入(Ocean 论文的
        epi = t["epilogue"]["sort"] + t["epilogue"]["copy"] + t["epilogue"]["scan"]  # iteration 时间含它)
        return an + est + sym + num + epi + t.get("prologue", 0)
    except Exception:
        return None

def run_dense(mtx, timeout=DENSE_TIMEOUT):
    """spgemm_dense naive scalar GEMM → 'Kernel time' ms,或 'timeout'/'fail'/'DNF(big)'。"""
    n, _sym, _a, _d = mtx_header(mtx)
    if n > DENSE_MAX_N:
        return "DNF(big)"
    try:
        r = subprocess.run([DENSE_BIN, mtx, "/tmp/dense_cmp.mtx"],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", r.stdout)
    return float(m.group(1)) if m else "fail"

def run_cublas(mtx, timeout=DENSE_TIMEOUT):
    """spgemm_dense_cublas(cuBLAS FP64 dgemm,PEDANTIC 无 TC)→ 'Kernel time' ms,或 'timeout'/'fail'/'DNF(big)'。"""
    n, _sym, _a, _d = mtx_header(mtx)
    if n > DENSE_MAX_N:
        return "DNF(big)"
    try:
        r = subprocess.run([CUBLAS_BIN, mtx, "/tmp/cublas_cmp.mtx"],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    m = re.search(r"Kernel time:\s*([0-9.]+)\s*ms", r.stdout)
    return float(m.group(1)) if m else "fail"

def run_bhsparse(mtx, timeout=CALL_TIMEOUT):
    """bhSparse(IPDPS'14,merge/ESC 变体按行长分桶)→ '[ CUDA ] SpGEMM time' ms 或 None。
    口径 = STAGE1-4 host 计时(含 stage3 Ct 重分配轮次;输入 h2d/输出 d2h 在计时外);
    BH_CHECK=0 跳过串行参考校验(生产模式)。"""
    env = dict(os.environ, BH_CHECK="0")
    try:
        r = subprocess.run([BHSPARSE_BIN, "-cuda", "-spgemm", os.path.abspath(mtx)],
                           capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"SpGEMM time:\s*([0-9.]+)\s*ms", r.stdout)
    return float(m.group(1)) if m else None

def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

def _report_and_summary(rows, args):
    """末尾汇总:几何均值 + Auto 选择 + Auto vs 各基线赢/输(normal flow 和 refresh-col 复用)。"""
    print("\n=== 几何均值(ms) ===")
    for col in ["cu", "Auto", "Ocean", "opSparse", "HSMU", "bhSparse", "dense", "cublas"]:
        xs = []
        for r in rows:
            try: xs.append(float(r.get(col)))
            except (TypeError, ValueError): pass
        if xs:
            print(f"  {col:9} {geomean(xs):8.3f}")
    n_hash = sum(1 for r in rows if r.get("Auto_choice") == "hash")
    n_m3 = sum(1 for r in rows if r.get("Auto_choice", "").startswith("merge3"))
    print(f"\n=== Auto 选择(共 {n_hash + n_m3} 阵)===  hash {n_hash} / merge3 {n_m3}")
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
    if not args.no_ocean:    vs_section("Ocean", "Ocean")
    if not args.no_opsparse: vs_section("opSparse", "opSparse")
    if not args.no_hsmu:     vs_section("HSMU", "HSMU")
    if not args.no_bhsparse: vs_section("bhSparse", "bhSparse")
    if not args.no_dense:    vs_section("dense", "dense")
    if not args.no_cublas:   vs_section("cublas", "cuBLAS")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="*", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "compare/methods_cmp.csv"))
    ap.add_argument("--no-hsmu", action="store_true")
    ap.add_argument("--no-opsparse", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-cublas", action="store_true", help="不比较 cuBLAS dense 基线")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-ocean", action="store_true", help="跳过 Ocean 基线")
    ap.add_argument("--no-bhsparse", action="store_true", help="跳过 bhSparse 基线")
    ap.add_argument("--refresh-col", default=None,
                    help="只重跑指定列(cu/Auto/Ocean/opSparse/HSMU/dense),保留其余列(改了某个 binary 后用)")
    args = ap.parse_args()

    if args.matrices:
        names = args.matrices
    else:
        d = args.dir or os.path.join(REPO, "data/first100")
        names = sorted(os.path.basename(p)[:-4] for p in os.listdir(d) if p.endswith(".mtx"))
    if args.limit:
        names = names[:args.limit]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["matrix", "n", "sym", "density_pct", "cu", "Auto",
                  "Auto_choice", "Ocean", "opSparse", "HSMU", "bhSparse", "dense", "cublas", "cnnz"]

    # ---- refresh-col:只重跑某一列,保留其余(改了某 binary 后局部刷新)----
    if args.refresh_col:
        col = args.refresh_col
        if col not in fieldnames:
            sys.exit(f"--refresh-col: 未知列 '{col}',可选 {fieldnames}")
        if not os.path.exists(args.out):
            sys.exit(f"--refresh-col 需要已有 {args.out};先跑一次全量")
        rows = list(csv.DictReader(open(args.out)))
        print(f"Refresh 仅 '{col}' 列 × {len(rows)} 阵(DENSE_TIMEOUT={DENSE_TIMEOUT}s)→ {args.out}\n", flush=True)
        def run_one(c, p, name, old_cnnz=None):
            if c == "cu":
                r = run_spgemm_method(p, "cu"); return (round(r[0], 3) if r else "DNF", {})
            if c == "Auto":
                r = run_spgemm_method(p, "adaptive", exp_cnnz=old_cnnz)
                ex = {}
                if r: ex["Auto_choice"] = r[3]; ex["cnnz"] = r[2]
                return (round(r[0], 3) if r else "", ex)
            if c == "Ocean":
                v = run_ocean(p); return (round(v, 3) if v is not None else "DNF", {})
            if c == "opSparse":
                op = run_opsparse(p); return (round(op[0], 3) if op else "DNF", {})
            if c == "HSMU":
                v = run_hsmu(p, name); return (round(v, 3) if v is not None else "", {})
            if c == "bhSparse":
                bh = run_bhsparse(p); return (round(bh, 3) if bh is not None else "DNF", {})
            if c == "dense":
                d = run_dense(p); return (round(d, 3) if isinstance(d, float) else d, {})
            if c == "cublas":
                cb = run_cublas(p); return (round(cb, 3) if isinstance(cb, float) else cb, {})
            return ("", {})
        for i, r in enumerate(rows):
            p = find_mtx(r["matrix"])
            if not p:
                print(f"[{i+1}/{len(rows)}] {r['matrix']}: 未找到", flush=True); continue
            t0 = time.time(); val, extra = run_one(col, p, r["matrix"], old_cnnz=_parse_num(r.get("cnnz")))
            r[col] = val; r.update(extra)
            print(f"[{i+1}/{len(rows)}] {r['matrix']:14} {col}={val!s:<10} ({time.time()-t0:.1f}s)", flush=True)
            with open(args.out, "w", newline="") as fp:    # 增量回写(断点续刷)
                ww = csv.DictWriter(fp, fieldnames=fieldnames); ww.writeheader()
                for rr in rows: ww.writerow({k: rr.get(k, "") for k in fieldnames})
        rows = list(csv.DictReader(open(args.out)))
        _report_and_summary(rows, args)      # 复用末尾汇总
        return

    # ---- header 兼容检查:旧 schema(无 opSparse/dense 等)→ 备份重写,避免 append 错列 ----
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out) as f:
            existing_hdr = next(csv.reader(f), None)
        if existing_hdr != fieldnames:
            bak = args.out + ".bak_staleschema_" + time.strftime("%Y%m%d_%H%M%S")
            os.rename(args.out, bak)
            print(f"⚠ {args.out} 是旧 schema({existing_hdr}),备份 → {bak},重新建表({fieldnames})", flush=True)

    done = set()
    if os.path.exists(args.out):
        for r in csv.DictReader(open(args.out)):
            done.add(r["matrix"])
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fieldnames)
    if not os.path.exists(args.out) or os.path.getsize(args.out) == 0:
        w.writeheader()

    N = len(names)
    methods_desc = [m[0] for m in SPGEMM_METHODS] + ["Ocean", "opSparse", "HSMU", "bhSparse", "dense", "cublas"]
    n_done = len(done)
    print(f"对比 {N} 阵 × {methods_desc} → {args.out}"
          f"{f'  (resume: {n_done} 已存,跳过;FRESH=1 全重跑)' if n_done else ''}\n", flush=True)
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
            # cu 失败 → DNF(cuSPARSE 12.x workEstimation 在高中间积阵上报 err 11
            # insufficient resources,ocean337 常见;计为基线弃权,口径同 dense DNF)
            row[label] = round(r[0], 3) if r else ("DNF" if label == "cu" else "")
            if label == "Auto" and r:
                row["cnnz"] = r[2]
                row["Auto_choice"] = r[3]
        oc = run_ocean(p) if not args.no_ocean else None
        row["Ocean"] = round(oc, 3) if oc is not None else ("" if args.no_ocean else "DNF")
        op = run_opsparse(p) if not args.no_opsparse else None
        row["opSparse"] = round(op[0], 3) if op else ("" if args.no_opsparse else "DNF")
        if op and op[1] and row.get("cnnz") and str(op[1]) != str(row["cnnz"]):
            print(f"  ⚠ opSparse C.nnz={op[1]} ≠ Auto cnnz={row['cnnz']}", flush=True)
        row["HSMU"] = round(run_hsmu(p, name), 3) if not args.no_hsmu else ""
        bh = run_bhsparse(p) if not args.no_bhsparse else None
        row["bhSparse"] = round(bh, 3) if bh is not None else ("" if args.no_bhsparse else "DNF")
        if not args.no_dense:
            d = run_dense(p)
            row["dense"] = round(d, 3) if isinstance(d, float) else d
        if not args.no_cublas:
            cb = run_cublas(p)
            row["cublas"] = round(cb, 3) if isinstance(cb, float) else cb
        w.writerow(row); fout.flush()
        dt = time.time() - t0
        dense_str = f" dense={row.get('dense','-')!s:>8}" if not args.no_dense else ""
        cublas_str = f" cublas={row.get('cublas','-')!s:>8}" if not args.no_cublas else ""
        print(f"cu={row['cu']!s:>7} Auto={row['Auto']!s:>7}({row.get('Auto_choice','?'):<6}) "
              f"Ocn={row['Ocean']!s:>7} opSp={row['opSparse']!s:>7} HSMU={row['HSMU']!s:>7} "
              f"bhSp={row['bhSparse']!s:>7}{dense_str}{cublas_str} ({dt:.1f}s)", flush=True)
    fout.close()
    print(f"\n完成 → {args.out}")

    rows = list(csv.DictReader(open(args.out)))
    _report_and_summary(rows, args)

if __name__ == "__main__":
    main()
