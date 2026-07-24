#!/usr/bin/env python3
"""对比 Ocean / HSMU / cuSPARSE / merge(serial) / merge3 / Auto(自适应) 的 SpGEMM 自乘时间。
Auto = 集成在 src 里的 spgemm_self_product_adaptive(flop>thr→hash 否则 merge3),非外部脚本。
每矩阵:METHOD=<m> 跑 spgemm_test 取该方法时间;ocean 跑 ocean/spgemu;hsmu 跑 test binary。
增量写 CSV(可断点续跑),末尾汇总。
用法:compare_methods.py [--matrices a b c | --dir data/first100] [--out cmp.csv] [--ocean/--no-ocean] [--hsmu/--no-hsmu]
"""
import os, sys, re, json, csv, subprocess, argparse, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "spgemm_test")
OCEAN_CONV = os.path.join(REPO, "ocean/convert")
OCEAN_RUN = os.path.join(REPO, "ocean/spgemm")
OCEAN_CFG = "config/bench_detail.json"
OCEAN_STATS = os.path.join(REPO, "ocean/stats.json")
HSMU_BIN = os.path.join(REPO, "external_sota/HSMU-SpGEMM/evaluation/script/test")
HSMU_CSV = "/tmp/NHC_4080S_result.csv"
# spECK (Parger PPoPP'20, hash+dense hybrid SOTA, cuda11 分支) — 自乘 C=A·A,double(与 Ocean/HSMU 同精度;spgemm_test 已统一 double)
SPECK_RUN = os.path.join(REPO, "spECK/build/runspECK")
SPECK_CFG = os.path.join(REPO, "spECK/config.ini")
# dense baseline(-O0 朴素 dense matmul,cudaEvent kernel time,同口径)缓存:由 scripts/run_dense_baseline.py 生成
DENSE_CACHE = os.path.join(REPO, "compare/dense_baseline.csv")

def load_dense_cache():
    """读 compare/dense_baseline.csv → {matrix: dense_ms(float) 或 'timeout'/'fail'}。"""
    d = {}
    if os.path.exists(DENSE_CACHE):
        for r in csv.DictReader(open(DENSE_CACHE)):
            v = r.get("dense_ms", "")
            try: v = float(v)
            except: pass
            d[r["matrix"]] = v
    return d

# 每次 spgemm_test/ocean/hsmu 调用的超时(秒),可被 env TIMEOUT 覆盖
CALL_TIMEOUT = int(os.environ.get("TIMEOUT", "200"))

# METHOD → cudaEvent prof tag(HashProf 输出 [tag-prof];adaptive 由 choice 决定 hash/mrg3)
METHOD_TAG = {"serial": "merge-prof", "merge3": "mrg3-prof", "hash": "hash-prof", "adaptive": None}

def compute_only_from_prof(stderr, tag):
    """从 [tag-prof] cudaEvent phase 求 compute-only = TOTAL(GPU) − h2d − d2h。
    纯 GPU 时间(抗 CPU 争用),内部控制流 D2H 已在 phase 块内 → 与 Ocean 同口径。取最后一次(timed run)。"""
    if not tag:
        return None
    rx = re.compile(r"\[" + re.escape(tag) + r"\]\s+(\S+)\s+([0-9.]+)\s+ms")
    phases = {}
    for line in stderr.splitlines():
        m = rx.search(line)
        if m:
            phases[m.group(1)] = float(m.group(2))    # 覆盖 → 保留最后一次
    if not phases:
        return None
    total = phases.get("TOTAL(GPU)")
    if total is not None:
        return total - phases.get("h2d", 0.0) - phases.get("d2h", 0.0)
    return sum(v for k, v in phases.items() if k not in ("h2d", "d2h"))

# fallback:host 时间戳(供无 HashProf 的方法,如 serial)。被 CPU 争用放大,仅次选。
DBG_TAG = {"serial": "merge", "merge3": "mrg3"}
def compute_only_from_dbg(stderr, tag):
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
        if items[i][0] in ("h2d", "d2h"):
            continue
        total += items[i][1] - items[i - 1][1]
    return total

# spgemm_test 方法名 → METHOD 关键字(cuSPARSE→baseline dense;serial 已移除)
SPGEMM_METHODS = [("merge3", "merge3"), ("Auto", "adaptive")]

def find_mtx(name):
    for d in ("data/first100", "data/sota_27_final", "data/sota_27"):
        p = os.path.join(REPO, d, name + ".mtx") if not name.endswith(".mtx") else os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p
    return None

def mtx_header(path):
    """返回 (n, sym, A_nnz_stored, density_pct)。density = stored_nnz/n²×100(与 profile_aa 同口径)。"""
    n = 0; sym = False; nnz = 0; sized = False
    with open(path) as f:
        first = f.readline(); sym = "symmetric" in first
        for l in f:
            if l.startswith("%"):
                continue
            p = l.split()
            if not p:
                continue
            if not sized:                  # 首个非注释行 = size 行
                n = int(p[0]); sized = True; continue
            nnz += 1                       # entry(pattern 2列 / real 3列)
    dens = (nnz / (n * n) * 100.0) if n > 0 else 0.0
    return n, sym, nnz, dens

def run_spgemm_method(mtx, method_key, timeout=CALL_TIMEOUT):
    """METHOD=method_key 跑 spgemm_test。返回 (compute_only_ms, wall_ms, cnnz, choice) 或 None。
    compute_only 由 stderr 的 [dbg][tag] phase 解析(排除 h2d/d2h,对标 K=5);解析失败退回 wall。"""
    env = dict(os.environ, USE_MEMPOOL="1", METHOD=method_key)
    try:
        r = subprocess.run([BIN, mtx], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout
    wall = re.search(r"Time:\s*([0-9.]+)\s*ms", out)
    nz = re.search(r"Result C:.*?nnz\s*=\s*(\d+)", out)
    # Auto 的方法选择(取 timed run 那次;溢出回退单独标)
    adapt = " ".join(l for l in out.splitlines() if "[adapt]" in l)
    if "overflow" in adapt:
        choice = "merge3(回退)"
    else:
        mm = re.findall(r"→\s*(hash|merge3)", adapt)
        choice = mm[-1] if mm else ""
    # cudaEvent prof tag:adaptive 用 choice 决定 hash-prof/mrg3-prof
    tag = METHOD_TAG.get(method_key)
    if method_key == "adaptive":
        tag = "hash-prof" if choice.startswith("hash") else "mrg3-prof"
    comp = compute_only_from_prof(r.stderr, tag)
    if comp is None:   # fallback:host 时间戳(serial 等无 HashProf 的方法,被争用放大,次选)
        comp = compute_only_from_dbg(r.stderr, DBG_TAG.get(method_key))
    if wall:
        w = float(wall.group(1))
        return (comp if comp is not None else w, w, int(nz.group(1)) if nz else -1, choice)
    return None

def run_ocean(mtx, timeout=CALL_TIMEOUT):
    """ocean/convert + ocean/spgemm → stats.json 总时间(各 phase 求和)。"""
    csr = "/tmp/cmp_ocean.csr"
    try:
        subprocess.run([OCEAN_CONV, mtx, csr], capture_output=True, timeout=CALL_TIMEOUT)
        subprocess.run([OCEAN_RUN, csr, OCEAN_CFG], cwd=os.path.join(REPO, "ocean"),
                       capture_output=True, timeout=timeout)
        s = json.load(open(OCEAN_STATS))
        t = s["timing"]
        # 与 baseline_ocean.py(K=5)完全同口径:analysis 3 子项 + estimation 4 子项(不含 binning)
        #   + numeric 全部 + epilogue 3 子项 + prologue。h2d/d2h 不在 timing 里(Ocean 另算),故为 compute-only。
        an  = t["analysis"]["product_calc"] + t["analysis"]["reduce"] + t["analysis"]["mem_cpy"]
        est = (t["estimation"]["hll_construct"] + t["estimation"]["hll_merge"]
               + t["estimation"]["malloc"] + t["estimation"]["sampling"])
        num = sum(t["numeric"].values())
        epi = t["epilogue"]["sort"] + t["epilogue"]["copy"] + t["epilogue"]["scan"]
        return an + est + num + epi + t.get("prologue", 0)
    except Exception:
        return None

def run_hsmu(mtx, name, timeout=CALL_TIMEOUT):
    """HSMU test binary → NHC CSV 里该矩阵最后出现的 total(col6)。"""
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

def run_speck(mtx, timeout=CALL_TIMEOUT):
    """spECK(cuda11,hash+dense hybrid,2020)自乘 C=A·A。返回 (compute_ms, C_nnz) 或 None。
    解析 'var-SpGEMM SpGEMM: X ms' 与 'var-SpGEMM -> NNZ: Y'。compute-only(均值,排除加载)。"""
    try:
        r = subprocess.run([SPECK_RUN, os.path.abspath(mtx), SPECK_CFG],
                           capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    out = r.stdout + r.stderr
    t = re.search(r"SpGEMM:\s*([0-9.]+)\s*ms", out)     # "var-SpGEMM SpGEMM: 2.95661 ms"
    nz = re.search(r"->\s*NNZ:\s*(\d+)", out)            # "var-SpGEMM -> NNZ: 8946070"
    if t:
        return (float(t.group(1)), int(nz.group(1)) if nz else None)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="*", default=None)
    ap.add_argument("--dir", default=None, help="矩阵目录(取目录下所有 .mtx)")
    ap.add_argument("--out", default=os.path.join(REPO, "compare/methods_cmp.csv"))
    ap.add_argument("--no-ocean", action="store_true")
    ap.add_argument("--no-hsmu", action="store_true")
    ap.add_argument("--no-speck", action="store_true", help="不比较 spECK(hash+dense hybrid SOTA)")
    ap.add_argument("--no-dense", action="store_true", help="不比较 dense baseline(dense 未跑完时用)")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个(调试)")
    args = ap.parse_args()

    # 矩阵列表
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
    fieldnames = ["matrix", "n", "sym", "density_pct", "merge3", "Auto",
                  "Auto_choice", "Ocean", "HSMU", "spECK", "cnnz"]
    if not args.no_dense:
        fieldnames.insert(4, "baseline")   # dense baseline 列(在 density_pct 后)
    fout = open(args.out, "a", newline="")
    w = csv.DictWriter(fout, fieldnames=fieldnames)
    if not os.path.exists(args.out) or os.path.getsize(args.out) == 0:
        w.writeheader()

    N = len(names)
    dense_cache = load_dense_cache() if not args.no_dense else {}
    print(f"对比 {N} 阵 × {[m[0] for m in SPGEMM_METHODS]} + Ocean + HSMU"
          + ("" if args.no_speck else " + spECK")
          + ("" if args.no_dense else " + dense-baseline(cache)") + f" → {args.out}\n", flush=True)
    for i, name in enumerate(names):
        if name in done:
            continue
        p = find_mtx(name)
        if not p:
            print(f"[{i+1}/{N}] {name}: 未找到,跳过", flush=True); continue
        n, sym, annz, dens = mtx_header(p)
        # 进度头:立即 flush(实时 1-100),后面接结果
        print(f"[{i+1}/{N}] {name:14} n={n:<6}", end="   ", flush=True)
        row = {"matrix": name, "n": n, "sym": "Y" if sym else "N", "density_pct": round(dens, 4)}
        t0 = time.time()
        # spgemm 方法
        for label, key in SPGEMM_METHODS:
            r = run_spgemm_method(p, key)
            row[label] = round(r[0], 3) if r else ""    # compute-only(排除 h2d/d2h)
            if label == "Auto" and r:
                row["cnnz"] = r[2]
                row["Auto_choice"] = r[3]          # hash / merge3 / merge3(回退)
        # Ocean
        row["Ocean"] = round(run_ocean(p), 3) if not args.no_ocean else ""
        # HSMU
        row["HSMU"] = round(run_hsmu(p, name), 3) if not args.no_hsmu else ""
        # spECK(hash+dense hybrid SOTA)
        sk = run_speck(p) if not args.no_speck else None
        row["spECK"] = round(sk[0], 3) if sk else ""
        if sk and sk[1] and row.get("cnnz") and str(sk[1]) != str(row["cnnz"]):
            print(f"  ⚠ spECK C_nnz={sk[1]} ≠ Auto cnnz={row['cnnz']}", flush=True)
        # dense baseline(从 cache 读,不重跑;--no-dense 时跳过)
        if not args.no_dense:
            bv = dense_cache.get(name, "")
            row["baseline"] = round(bv, 3) if isinstance(bv, float) else bv
        w.writerow(row); fout.flush()
        dt = time.time() - t0
        # 结果行
        dense_str = f" dense={row['baseline']!s:>8}" if not args.no_dense else ""
        print(f"Auto→{row.get('Auto_choice','?'):<6}{dense_str} "
              f"m3={row['merge3']!s:>7} Auto={row['Auto']!s:>7} Ocean={row['Ocean']!s:>7} "
              f"HSMU={row['HSMU']!s:>7} spECK={row['spECK']!s:>7} ({dt:.1f}s)", flush=True)
    fout.close()
    print(f"\n完成 → {args.out}")
    # 汇总:各方法几何均值(相对 Auto)
    import math
    rows = list(csv.DictReader(open(args.out)))
    def geomean(col):
        xs = []
        for r in rows:
            v = r.get(col)
            if v in (None, "", "None"): continue
            try: xs.append(float(v))
            except ValueError: continue    # 跳过 'timeout' 等非数字
        if not xs: return float("nan")
        return math.exp(sum(math.log(x) for x in xs) / len(xs))
    print("\n=== 几何均值(ms) ===")
    cols = ([["baseline"]] if not args.no_dense else []) + [m[0] for m in SPGEMM_METHODS] + ["Ocean", "HSMU"]
    if not args.no_speck:
        cols += ["spECK"]
    for col in cols:
        gm = geomean(col)
        if gm == gm:
            print(f"  {col:8} {gm:8.3f}")
    # Auto 方法选择统计
    n_hash = sum(1 for r in rows if r.get("Auto_choice") == "hash")
    n_m3   = sum(1 for r in rows if r.get("Auto_choice", "").startswith("merge3"))
    print(f"\n=== Auto 选择(共 {n_hash + n_m3} 阵)===  hash {n_hash} / merge3 {n_m3}")
    if n_hash:
        print("  选 hash 的:" + ", ".join(r["matrix"] for r in rows if r.get("Auto_choice") == "hash"))

if __name__ == "__main__":
    main()