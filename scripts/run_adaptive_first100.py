#!/usr/bin/env python3
"""Run the DEPLOYED dispatcher (METHOD=adaptive, new 95% formula) on first100 and
compare against the old in-tree formula, on the same per-matrix hash/merge3
timings collected in compare/dispatcher_fit_*.csv (apples-to-apples, no noise).

Reports per-matrix choice + correctness, and totals:
  new (actual adaptive run) vs new (re-eval on collected) vs old (re-eval) vs oracle.
"""
import os, sys, math, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_methods as CM

REPO = CM.REPO
NEW_W = (-0.4259, -0.4215, -0.6441, -0.6420); NEW_B = 5.4297   # deployed
OLD_W = (-1.3085, 1.2131, 1.9815, -2.4331); OLD_B = 1.2943    # previous in-tree

# load collected timings + features
fit = {}
for r in csv.DictReader(open(sorted([os.path.join(REPO, "compare", f) for f in os.listdir(os.path.join(REPO, "compare")) if f.startswith("dispatcher_fit_")])[-1])):
    try:
        ht = float(r["hash_ms"]); mt = float(r["merge3_ms"])
    except Exception:
        continue
    n = int(r["n"]); maxrow = int(r["maxrow"]); skew = float(r["skew"]); flop = float(r["flop"])
    x = (math.log10(max(flop, 1)), math.log10(n), math.log10(max(maxrow, 1)), math.log10(max(skew, 1)))
    hw = 0 if not math.isfinite(ht) else (1 if ht < mt else 0)
    fit[r["matrix"]] = dict(ht=ht, mt=mt, hw=hw, x=x)

def score(x, W, b):
    return sum(w * xi for w, xi in zip(W, x)) + b

names = sorted(p[:-4] for p in os.listdir(os.path.join(REPO, "data/first100")) if p.endswith(".mtx"))
print(f"running METHOD=adaptive (new formula) on {len(names)} matrices ...\n", flush=True)

new_total_act = 0.0; new_correct = 0; new_hash = 0; ran = 0
old_total = 0.0; old_correct = 0; oracle = 0.0
mismatches = []
for i, name in enumerate(names):
    p = CM.find_mtx(name)
    if not p or name not in fit:
        continue
    ran += 1
    f = fit[name]
    # actual deployed run
    r = CM.run_spgemm_method(p, "adaptive")
    if r is None or r[0] is None:
        print(f"[{i+1}] {name}: adaptive run failed", flush=True); continue
    new_choice = "hash" if (r[3].startswith("hash")) else "merge3"
    new_auto = r[0]
    new_total_act += new_auto
    if new_choice == "hash": new_hash += 1
    true_w = "hash" if f["hw"] else "merge3"
    if new_choice == true_w: new_correct += 1
    # re-eval old + new + oracle on collected timings
    old_choice = "hash" if score(f["x"], OLD_W, OLD_B) < 0 else "merge3"
    new_choice_eval = "hash" if score(f["x"], NEW_W, NEW_B) < 0 else "merge3"
    old_total += f["ht"] if old_choice == "hash" and math.isfinite(f["ht"]) else (f["mt"] if old_choice == "hash" else f["mt"])
    if old_choice == true_w: old_correct += 1
    better = f["ht"] if (f["hw"] and math.isfinite(f["ht"])) else f["mt"]
    oracle += better
    mark = " " if new_choice == true_w else "X"
    if new_choice != new_choice_eval:
        mismatches.append(name)
    print(f"[{ran:3}/{len(names)}] {name:14} new→{new_choice:6} (true {true_w:6}) {mark}  "
          f"auto={new_auto:8.3f}ms", flush=True)

print("\n================ DISPATCHER first100 COMPARISON ================", flush=True)
print(f"matrices run              : {ran}")
print(f"new formula (deployed)    : picks hash {new_hash}/{ran}   total(actual) {new_total_act:.2f} ms")
print(f"  actual-run choices vs re-eval mismatches: {mismatches}  (should be empty)")
print(f"accuracy (vs true faster) : NEW {new_correct}/{ran} = {100*new_correct/ran:.1f}%   | "
      f"OLD {old_correct}/{ran} = {100*old_correct/ran:.1f}%")
print(f"total compute (collected) : NEW {sum(f['ht'] if (score(f['x'],NEW_W,NEW_B)<0 and math.isfinite(f['ht'])) else f['mt'] for f in fit.values()):.2f} | "
      f"OLD {old_total:.2f} | oracle {oracle:.2f} ms")
print(f"  → NEW saves {old_total - sum(f['ht'] if (score(f['x'],NEW_W,NEW_B)<0 and math.isfinite(f['ht'])) else f['mt'] for f in fit.values()):.2f} ms vs OLD "
      f"(gap-to-oracle NEW {sum(f['ht'] if (score(f['x'],NEW_W,NEW_B)<0 and math.isfinite(f['ht'])) else f['mt'] for f in fit.values())-oracle:+.2f} / OLD {old_total-oracle:+.2f})", flush=True)
