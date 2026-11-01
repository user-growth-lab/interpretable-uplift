"""Honest rebuild of Table 4: SCRL vs Causal Tree (top-K leaves) vs CURLS, 10 seeds.

Protocol (documented for the paper):
  - Semi-synthetic IHDP (5 modifiers, signal 2.0, noise 0.3), seeds 42-51.
  - Metric: Definition 5 / 5b coverage fidelity (oos_coverage_fidelity).
  - SCRL: v2 defaults (n_rules=5, n_bootstrap=10, n_candidates=20).
  - Causal Tree: leaf->rule conversion restricted to the top-5 leaves by
    predicted leaf CATE (the union of ALL leaves covers the population by
    construction, which degenerates Definition 5 to exactly 0; matching K=5
    to SCRL's rule count gives a fair, non-degenerate comparison).
  - CURLS: 5 rules, its own selection.
SCRL results for seeds 42-46 are reused from audit_scrl_reverify.json.
"""
import json
import numpy as np
from scipy import stats
from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL
from src.models.baselines import CausalTreeBaseline, CURLSBaseline
from src.metrics.fidelity import oos_coverage_fidelity

SEEDS = list(range(42, 52))
prev = json.load(open("results/paper_tables/audit_scrl_reverify.json"))
res = {"scrl": {"in": list(prev["scrl"]["in"]), "oos": list(prev["scrl"]["oos"]), "mods": list(prev["scrl"]["mods"])},
       "ct": {"in": [], "oos": [], "mods": []},
       "curls": {"in": [], "oos": [], "mods": []},
       "per_rule_cov_test": list(prev["per_rule_cov_test"]),
       "union_cov_test": list(prev["union_cov_test"])}

for seed in SEEDS:
    train, test = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed)
    mods = set(int(i) for i in train.effect_modifier_indices)
    m = TLearner(); m.fit(train.X, train.T, train.Y)
    tau_tr = m.predict_cate(train.X); tau_te = m.predict_cate(test.X)

    # --- SCRL (only new seeds 47-51) ---
    if seed >= 47:
        s = SCRL(cate_model=m, n_rules=5, n_bootstrap=10, n_candidates=20, seed=seed)
        s.fit(train.X, train.T, train.Y)
        mk_tr = [r.covers(train.X) for r in s.rules]
        mk_te = [r.covers(test.X) for r in s.rules]
        res["scrl"]["in"].append(oos_coverage_fidelity(train.X, tau_tr, train.X, tau_tr, mk_tr))
        res["scrl"]["oos"].append(oos_coverage_fidelity(train.X, tau_tr, test.X, tau_te, mk_te))
        res["scrl"]["mods"].append(len({int(c[0]) for r in s.rules for c in r.conditions} & mods))
        res["per_rule_cov_test"] += [float(mk.mean()) for mk in mk_te]
        union = np.zeros(test.X.shape[0], bool)
        for mk in mk_te: union |= mk
        res["union_cov_test"].append(float(union.mean()))
        print(f"seed {seed}: SCRL in={res['scrl']['in'][-1]:.3f} oos={res['scrl']['oos'][-1]:.3f} mods={res['scrl']['mods'][-1]}/5", flush=True)

    # --- Causal Tree: top-5 leaves by predicted leaf CATE ---
    ct = CausalTreeBaseline(max_depth=3, min_samples_leaf=20, seed=seed)
    ct.fit(train.X, train.T, train.Y)
    pr_tr = ct.predict_cate(train.X); pr_te = ct.predict_cate(test.X)
    leaves = sorted(np.unique(pr_tr), reverse=True)[:5]
    mk_tr = [pr_tr == lv for lv in leaves]
    mk_te = [pr_te == lv for lv in leaves]
    res["ct"]["in"].append(oos_coverage_fidelity(train.X, tau_tr, train.X, tau_tr, mk_tr))
    res["ct"]["oos"].append(oos_coverage_fidelity(train.X, tau_tr, test.X, tau_te, mk_te))
    tree_feats = {int(f) for f in ct.tree.tree_.feature if f >= 0}
    res["ct"]["mods"].append(len(tree_feats & mods))

    # --- CURLS ---
    cu = CURLSBaseline(cate_model=m, n_rules=5, max_depth=3, min_samples_leaf=30, n_bootstrap=50, seed=seed)
    cu.fit(train.X, train.T, train.Y)
    cmk_tr = cu.get_rule_masks(train.X); cmk_te = cu.get_rule_masks(test.X)
    res["curls"]["in"].append(oos_coverage_fidelity(train.X, tau_tr, train.X, tau_tr, cmk_tr) if cmk_tr else 0.0)
    res["curls"]["oos"].append(oos_coverage_fidelity(train.X, tau_tr, test.X, tau_te, cmk_te) if cmk_te else 0.0)
    res["curls"]["mods"].append(0)
    print(f"seed {seed}: CT(top5) in={res['ct']['in'][-1]:.3f} oos={res['ct']['oos'][-1]:.3f} mods={res['ct']['mods'][-1]} | CURLS in={res['curls']['in'][-1]:.3f} oos={res['curls']['oos'][-1]:.3f}", flush=True)

def ms(a): a = np.asarray(a, float); return f"{a.mean():.3f}+/-{a.std():.3f}"
print("\n=== REBUILT TABLE 4 (10 seeds, current code, documented protocol) ===")
print(f"CT(top5)  in={ms(res['ct']['in'])} oos={ms(res['ct']['oos'])} mods={np.mean(res['ct']['mods']):.1f}/5")
print(f"CURLS     in={ms(res['curls']['in'])} oos={ms(res['curls']['oos'])}")
print(f"SCRL      in={ms(res['scrl']['in'])} oos={ms(res['scrl']['oos'])} mods={np.mean(res['scrl']['mods']):.1f}/5")
summary = {"table": {k: {"in": ms(v["in"]), "oos": ms(v["oos"]), "mods": float(np.mean(v["mods"]))} for k, v in res.items() if k in ("scrl", "ct", "curls")}}
for name in ("ct", "curls"):
    t, p = stats.ttest_rel(res["scrl"]["oos"], res[name]["oos"])
    t2, p2 = stats.ttest_rel(res["scrl"]["in"], res[name]["in"])
    summary[f"scrl_vs_{name}"] = {"p_oos": float(p), "p_in": float(p2)}
    print(f"SCRL vs {name}: paired-t OOS p={p:.4f}, in p={p2:.4f}")
print(f"per-rule OOS coverage range: {min(res['per_rule_cov_test']):.3f}-{max(res['per_rule_cov_test']):.3f}")
print(f"union OOS coverage: {ms(res['union_cov_test'])}")
summary["per_rule_cov_range"] = [min(res["per_rule_cov_test"]), max(res["per_rule_cov_test"])]
summary["union_cov"] = ms(res["union_cov_test"])
res["summary"] = summary
json.dump(res, open("results/paper_tables/table4_rebuild_10seed.json", "w"), indent=1)
print("saved -> results/paper_tables/table4_rebuild_10seed.json")
