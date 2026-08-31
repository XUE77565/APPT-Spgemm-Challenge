#!/usr/bin/env python3
"""per-row 采数(docs/66):同阵分别跑 hash(HASH_ROWTIME)与 merge3(MRG3_ROWTIME),
逐行执行周期 + 特征(kcnt/flop/est/span)落盘 → merge3 路由判据画像。

输出:<outdir>/<mat>.hash.tsv(行 kcnt flop est span ovf cycles)
      <outdir>/<mat>.mrg3.tsv(行 桶 cycles;t_mrg=各桶 max=块墙钟)
摘要:总量对比 + 按 flop 十分位的 t_mrg/t_hash 中位 + 简单门搜索(net win 最大的特征门)。
用法:.venv/bin/python scripts/collect_rowtime.py <mtx>... [--outdir compare/rowtime]
"""
import argparse, os, subprocess, sys, statistics, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_methods as CM

def run(mtx, method, rt_env, rt_path):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD=method, MP_HOST_MB="8192", rt_env: rt_path)
    try:
        r = subprocess.run([CM.BIN, mtx], capture_output=True, text=True, env=env, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    return os.path.exists(rt_path) and os.path.getsize(rt_path) > 0

def load_hash(p):
    rows = {}
    for ln in open(p):
        r, kc, fl, es, sp, ov, cy = ln.split()
        rows[int(r)] = dict(kc=int(kc), flop=int(fl), est=int(es), span=int(sp), ovf=int(ov), cyc=int(cy))
    return rows

def load_mrg(p):
    rows = collections.defaultdict(dict)
    for ln in open(p):
        r, b, cy = ln.split()
        rows[int(r)][int(b)] = int(cy)
    return {r: max(b.values()) for r, b in rows.items()}   # 块墙钟 = max over 桶(桶间并行)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mtx", nargs="+")
    ap.add_argument("--outdir", default="compare/rowtime")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for mtx in a.mtx:
        mat = os.path.splitext(os.path.basename(mtx))[0]
        hp, mp = os.path.join(a.outdir, mat + ".hash.tsv"), os.path.join(a.outdir, mat + ".mrg3.tsv")
        if not run(mtx, "hash", "HASH_ROWTIME", hp): print(f"{mat}: hash 采数失败"); continue
        if not run(mtx, "merge3", "MRG3_ROWTIME", mp): print(f"{mat}: merge3 采数失败"); continue
        H, M = load_hash(hp), load_mrg(mp)
        common = [r for r in H if r in M and H[r]["ovf"] == 0 and H[r]["cyc"] > 0]
        if not common: print(f"{mat}: 无可比对行"); continue
        th, tm = sum(H[r]["cyc"] for r in common), sum(M[r] for r in common)
        print(f"\n== {mat}({len(common)} 行可比,ovf 行已剔)==")
        print(f"  Σt_hash={th:.3g} Σt_mrg={tm:.3g} → 整阵 merge3 {'优' if tm<th else '劣'} {th/tm:.2f}x")
        # flop 十分位画像
        rows = sorted(common, key=lambda r: H[r]["flop"])
        dec = max(1, len(rows)//10)
        print(f"  {'flop段':>16} {'行数':>6} {'中位dup':>8} {'中位span/n':>10} {'t_mrg/t_hash':>12}")
        for i in range(10):
            seg = rows[i*dec:(i+1)*dec] if i < 9 else rows[9*dec:]
            if not seg: continue
            dups = [H[r]["flop"]/max(1,H[r]["est"]) for r in seg]
            dens = [H[r]["span"] for r in seg]
            ratios = [M[r]/H[r]["cyc"] for r in seg]
            print(f"  flop≥{H[seg[0]]['flop']:>8} {len(seg):>6} {statistics.median(dups):>8.1f} "
                  f"{statistics.median(dens):>10} {statistics.median(ratios):>11.2f}")
        # 门搜索:特征 × 阈值 → 门内 Σ(t_hash−t_mrg) 最大
        best = None
        feats = ["flop", "kc", "dup", "span"]
        for f in feats:
            vals = sorted({ (H[r]["flop"] if f=="flop" else H[r]["kc"] if f=="kc"
                            else H[r]["flop"]/max(1,H[r]["est"]) if f=="dup" else H[r]["span"]) for r in common })
            for q in (0.3, 0.5, 0.7, 0.9):
                thr = vals[min(len(vals)-1, int(q*len(vals)))]
                def fv(r):
                    h = H[r]
                    return h["flop"] if f=="flop" else h["kc"] if f=="kc" else h["flop"]/max(1,h["est"]) if f=="dup" else h["span"]
                gate = [r for r in common if fv(r) >= thr]
                if len(gate) < len(common)*0.05: continue
                net = sum(H[r]["cyc"]-M[r] for r in gate)
                if best is None or net > best[0]: best = (net, f, thr, len(gate))
        if best and best[0] > 0:
            net, f, thr, n = best
            print(f"  ★ 最优门:{f} ≥ {thr:.4g}(捕 {n} 行,{n/len(common)*100:.0f}%)net win = {net:.3g} 周期({net/th*100:+.1f}% of Σt_hash)")
        else:
            print("  (无正收益门:该阵 merge3 无行级生态位)")

if __name__ == "__main__":
    main()
