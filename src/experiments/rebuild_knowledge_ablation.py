"""Honest rebuild of Table 8 (knowledge-component ablation on IHDP), seeds 42-46.

Protocol mirrors run_significance_and_scale.run_knowledge_ablation_crossdataset,
with one strengthening: on IHDP the targeting metric is evaluated against the
TRUE tau (ground truth available), not the predicted CATE.
  - Rule-based targeting: rules sorted by predicted subgroup tau_mean; accumulate
    covered test instances until 20% of the test population; report true-tau mean.
  - TESV-fallback targeting: top-20% instances by TESV attribution magnitude.
  - Actionable coverage: CF positive-gain fraction (Full / No-Rules);
    rule coverage fraction (No-CF); 0 for Attribution-Only.
"""
import json
import numpy as np
from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL
from src.models.tesv import TESVExplainer
from src.models.counterfactual import CausalCounterfactualExplainer

SEEDS = list(range(42, 47))
res = {k: {"targeting": [], "coverage": []} for k in ["full", "no_rules", "no_cf", "attr_only"]}

for seed in SEEDS:
    train, test = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed)
    m = TLearner(); m.fit(train.X, train.T, train.Y)
    true_tau = test.tau
    n = test.X.shape[0]
    n_top = max(1, int(0.2 * n))

    # SCRL rules
    s = SCRL(cate_model=m, n_rules=5, n_bootstrap=10, n_candidates=20, seed=seed)
    s.fit(train.X, train.T, train.Y)
    rules = sorted(s.rules, key=lambda r: r.tau_mean, reverse=True)
    sel = np.zeros(n, bool)
    for r in rules:
        if sel.sum() >= n_top:
            break
        sel |= r.covers(test.X)
    # truncate to exactly top-20% budget by predicted tau within selection
    if sel.sum() > n_top:
        tau_pred = m.predict_cate(test.X)
        idx = np.where(sel)[0]
        keep = idx[np.argsort(-tau_pred[idx])[:n_top]]
        sel = np.zeros(n, bool); sel[keep] = True
    targeting_rules = float(true_tau[sel].mean()) if sel.sum() else float(true_tau.mean())
    rule_cov = np.zeros(n, bool)
    for r in s.rules: rule_cov |= r.covers(test.X)

    # TESV fallback ranking
    n_exp = min(100, n)
    tesv = TESVExplainer(cate_model=m, background_data=train.X, n_background=50, n_permutations=10, seed=seed)
    attr = tesv.explain(test.X[:n_exp], show_progress=False)
    scores = np.abs(attr).sum(axis=1)
    top_idx = np.argsort(-scores)[:max(1, int(0.2 * n_exp))]
    targeting_tesv = float(true_tau[:n_exp][top_idx].mean())

    # CF actionable coverage
    cf = CausalCounterfactualExplainer(cate_model=m, actionable_features=list(range(train.X.shape[1])),
                                       max_changes=2, step_size=0.2, max_steps=10,
                                       proximity_weight=0.05, seed=seed)
    cfs = cf.explain(test.X[:n_exp])
    cf_cov = float(np.mean([e.tau_gain > 0 for e in cfs]))

    res["full"]["targeting"].append(targeting_rules);  res["full"]["coverage"].append(cf_cov)
    res["no_rules"]["targeting"].append(targeting_tesv); res["no_rules"]["coverage"].append(cf_cov)
    res["no_cf"]["targeting"].append(targeting_rules); res["no_cf"]["coverage"].append(float(rule_cov.mean()))
    res["attr_only"]["targeting"].append(targeting_tesv); res["attr_only"]["coverage"].append(0.0)
    print(f"seed {seed}: rules_target={targeting_rules:.3f} tesv_target={targeting_tesv:.3f} cf_cov={cf_cov:.2f} rule_cov={rule_cov.mean():.2f} pop_true_tau={true_tau.mean():.3f}", flush=True)

def ms(a): a = np.asarray(a, float); return f"{a.mean():.3f}+/-{a.std():.3f}"
print("\n=== REBUILT TABLE 8 (IHDP, 5 seeds, true-tau targeting) ===")
for k, v in res.items():
    print(f"{k}: targeting={ms(v['targeting'])} coverage={ms(v['coverage'])}")
json.dump(res, open("results/paper_tables/knowledge_ablation_rebuild.json", "w"), indent=1)
print("saved")
