#!/usr/bin/env python3
"""dense_direct 逐行 cursor/search 判据采集(docs/66 §8,任务=docs/63 §6 遗留)。
每阵三跑:PB2_CURSOR=1(路由:avgB≥64→cursor)/ =0(全 search)/ HASH_ROWTIME(特征)。
join 逐行:特征(kcnt/flop/est/span/avgB) × (t_cursor, t_search) → 判据画像 + 门搜索。
用法:.venv/bin/python scripts/collect_dd_rowtime.py [--outdir compare/dd_rowtime] [mat ...]
"""
import argparse, os, subprocess, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BIN = os.path.join(os.path.dirname(HERE), "spgemm_test")
DIR = "data/ocean/square"
DEFAULT = ["c-64", "TSOPF_FS_b39_c7", "3Dspectralwave2", "Cube_Coup_dt0", "c-58", "brainpc2", "mult_dcop_03"]

def run(mtx, env_extra, timeout=900):
    env = dict(os.environ, USE_MEMPOOL="1", METHOD="hash", MP_HOST_MB="8192", **env_extra)
    try:
        subprocess.run([BIN, os.path.join(DIR, mtx + ".mtx")], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True

def load_dd(p):
    rows = {}
    if not os.path.exists(p): return rows
    for ln in open(p):
        r, cy, md = ln.split()
        rows[int(r)] = (int(cy), int(md))
    return rows

def load_feat(p):
    rows = {}
    for ln in open(p):
        r, kc, fl, es, sp, ov, cy = ln.split()
        h = dict(kc=int(kc), flop=int(fl), est=int(es), span=int(sp))
        rows[int(r)] = h
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="*", default=DEFAULT)
    ap.add_argument("--outdir", default="compare/dd_rowtime")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for m in a.mat:
        fc = os.path.join(a.outdir, m + ".cur.tsv")
        fs = os.path.join(a.outdir, m + ".srh.tsv")
        ff = os.path.join(a.outdir, m + ".feat.tsv")
        if not (os.path.exists(fc) and os.path.exists(fs)):
            run(m, {"PB2_CURSOR": "1", "DD_ROWTIME": fc}) and run(m, {"PB2_CURSOR": "0", "DD_ROWTIME": fs})
        if not os.path.exists(ff):
            run(m, {"HASH_ROWTIME": ff})
        C, S, F = load_dd(fc), load_dd(fs), load_feat(ff)
        common = [r for r in C if r in S and C[r][1] == 1 and S[r][1] == 0 and S[r][0] > 0 and r in F]
        print(f"\n== {m}: cursor 模式行 {sum(1 for _,md in C.values() if md==1)},search 模式行 {sum(1 for _,md in C.values() if md==0)},可比 {len(common)}")
        if len(common) < 30:
            print("   (可比行不足,cursor 未触或 dense 行少)"); continue
        tc, ts = sum(C[r][0] for r in common), sum(S[r][0] for r in common)
        print(f"   Σt_cursor={tc:.3g} Σt_search={ts:.3g} → {'cursor 优' if tc<ts else 'search 优'} {max(tc,ts)/min(tc,ts):.2f}x")
        ratios = sorted((C[r][0]/max(1,S[r][0]), r) for r in common)
        for q, lab in [(0.1, "cursor 大胜"), (0.5, "中位"), (0.9, "cursor 大败")]:
            rr, r = ratios[min(len(ratios)-1, int(q*len(ratios)))]
            f = F[r]
            print(f"   {lab}: t_cur/t_srh={rr:6.2f} @kcnt={f['kc']} flop={f['flop']} est={f['est']} span={f['span']} avgB={f['flop']//max(1,f['kc'])}")
        # 门搜索:特征 × 阈值,net = Σ(search−cursor) 只在被选行
        best = None
        for feat, key in [("kcnt", lambda f: f["kc"]), ("flop", lambda f: f["flop"]),
                          ("span", lambda f: f["span"]), ("avgB", lambda f: f["flop"]/max(1,f["kc"])),
                          ("nwin(span/4k)", lambda f: f["span"]/4096+1)]:
            vals = sorted(key(F[r]) for r in common)
            for q in (0.1, 0.25, 0.5, 0.75, 0.9):
                thr = vals[min(len(vals)-1, int(q*len(vals)))]
                gate = [r for r in common if key(F[r]) <= thr]
                if len(gate) < len(common)*0.05: continue
                net = sum(S[r][0]-C[r][0] for r in gate)
                if best is None or net > best[0]: best = (net, feat, thr, len(gate), sum(S[r][0] for r in gate))
        if best:
            net, feat, thr, n, ts_g = best
            print(f"   ★ 门 {feat}≤{thr:.4g}(选 {n} 行/{n/len(common)*100:.0f}%):Σt_srh={ts_g:.3g} → Σt_cur={ts_g-net:.3g}(net {'+' if net>0 else ''}{net/ts_g*100:.1f}%)")
        else:
            print("   (无正收益门)")

if __name__ == "__main__":
    main()
