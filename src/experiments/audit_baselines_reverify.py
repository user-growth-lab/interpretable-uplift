"""Audit part 2: Causal Tree + CURLS on the same protocol/seeds as audit_scrl_reverify."""
import json
import numpy as np
from scipy import stats
from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner
from src.models.baselines import CausalTreeBaseline, CURLSBaseline
from src.metrics.fidelity import oos_coverage_fidelity

res = {"ct": {"in": [], "oos": [], "mods": []}, "curls": {"in": [], "oos": [], "mods": []}}
for seed in range(42, 47):
    train, test = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed)
    mods = set(int(i) for i in train.effect_modifier_indices)
    m = TLearner(); m.fit(train.X, train.T, train.Y)
    tau_tr = m.predict_cate(train.X); tau_te = m.predict_cate(test.X)
    # Causal Tree (leaf -> rule conversion, as in paper)
    ct = CausalTreeBaseline(max_depth=3, min_samples_leaf=20, seed=seed)
    ct.fit(train.X, train.T, train.Y)
    pr_tr = ct.predict_cate(train.X); pr_te = ct.predict_cate(test.X)
    mk_tr = [pr_tr == leaf for leaf in np.unique(pr_tr)]
    mk_te = [pr_te == leaf for leaf in np.unique(pr_te)]
    in_f = oos_coverage_fidelity(train.X, tau_tr, train.X, tau_tr, mk_tr)
    oos_f = oos_coverage_fidelity(train.X, tau_tr, test.X, tau_te, mk_te)
    tree_feats = {int(f) for f in ct.tree.tree_.feature if f >= 0}
    res["ct"]["in"].append(in_f); res["ct"]["oos"].append(oos_f); res["ct"]["mods"].append(len(tree_feats & mods))
    # CURLS
    cu = CURLSBaseline(cate_model=m, n_rules=5, max_depth=3, min_samples_leaf=30, n_bootstrap=50, seed=seed)
    cu.fit(train.X, train.T, train.Y)
    cmk_tr = cu.get_rule_masks(train.X); cmk_te = cu.get_rule_masks(test.X)
    cin = oos_coverage_fidelity(train.X, tau_tr, train.X, tau_tr, cmk_tr) if cmk_tr else 0.0
    coos = oos_coverage_fidelity(train.X, tau_tr, test.X, tau_te, cmk_te) if cmk_te else 0.0
    res["curls"]["in"].append(cin); res["curls"]["oos"].append(coos); res["curls"]["mods"].append(0)
    print(f"seed {seed}: CT in={in_f:.3f} oos={oos_f:.3f} mods={len(tree_feats & mods)} | CURLS in={cin:.3f} oos={coos:.3f}", flush=True)

scrl = json.load(open("results/paper_tables/audit_scrl_reverify.json"))["scrl"]
def ms(a): a = np.asarray(a, float); return f"{a.mean():.3f}±{a.std():.3f}"
print("\n=== Rebuilt Table 4 (seeds 42-46, current code) ===")
print("CT   :", ms(res["ct"]["in"]), ms(res["ct"]["oos"]), np.mean(res["ct"]["mods"]))
print("CURLS:", ms(res["curls"]["in"]), ms(res["curls"]["oos"]))
print("SCRL :", ms(scrl["in"]), ms(scrl["oos"]), np.mean(scrl["mods"]))
for name, base in [("ct", res["ct"]), ("curls", res["curls"])]:
    t, p = stats.ttest_rel(scrl["oos"], base["oos"])
    t2, p2 = stats.ttest_rel(scrl["in"], base["in"])
    print(f"SCRL vs {name}: paired t OOS p={p:.4f}, in-sample p={p2:.4f}")
    ratio = np.mean(scrl["oos"]) / np.mean(base["oos"]) if np.mean(base["oos"]) != 0 else float("inf")
    print(f"  OOS ratio SCRL/{name}: {ratio:.2f}")
json.dump(res, open("results/paper_tables/audit_baselines_reverify.json", "w"), indent=1)
print("saved")
