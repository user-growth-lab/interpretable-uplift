"""Audit re-verification: Table 4 (rules) + example rules + coverage claims, seeds 42-46."""
import json
import numpy as np
from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL
from src.models.baselines import CausalTreeBaseline
from src.metrics.fidelity import oos_coverage_fidelity

res = {"scrl": {"in": [], "oos": [], "mods": []}, "ct": {"in": [], "oos": []},
       "per_rule_cov_train": [], "per_rule_cov_test": [], "union_cov_test": [], "example_rules_seed42": []}
for seed in range(42, 47):
    train, test = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed)
    mods = set(int(i) for i in train.effect_modifier_indices)
    m = TLearner(); m.fit(train.X, train.T, train.Y)
    tau_tr = m.predict_cate(train.X); tau_te = m.predict_cate(test.X)
    s = SCRL(cate_model=m, n_rules=5, n_bootstrap=10, n_candidates=20, seed=seed)
    s.fit(train.X, train.T, train.Y)
    mk_tr = [r.covers(train.X) for r in s.rules]
    mk_te = [r.covers(test.X) for r in s.rules]
    in_f = oos_coverage_fidelity(train.X, tau_tr, train.X, tau_tr, mk_tr)
    oos_f = oos_coverage_fidelity(train.X, tau_tr, test.X, tau_te, mk_te)
    hit = len({int(c[0]) for r in s.rules for c in r.conditions} & mods)
    union = np.zeros(test.X.shape[0], bool)
    for mk in mk_te: union |= mk
    res["scrl"]["in"].append(in_f); res["scrl"]["oos"].append(oos_f); res["scrl"]["mods"].append(hit)
    res["per_rule_cov_train"] += [float(mk.mean()) for mk in mk_tr]
    res["per_rule_cov_test"] += [float(mk.mean()) for mk in mk_te]
    res["union_cov_test"].append(float(union.mean()))
    if seed == 42:
        res["example_rules_seed42"] = [
            {"rule": str(r), "tau": float(r.tau_mean), "cov": float(r.coverage),
             "mods_hit": sorted(int(c[0]) for c in r.conditions if int(c[0]) in mods)}
            for r in s.rules]
    print(f"seed {seed}: in={in_f:.3f} oos={oos_f:.3f} mods={hit}/5 union_cov={union.mean():.3f}", flush=True)

def ms(a): a=np.asarray(a,float); return f"{a.mean():.3f}±{a.std():.3f}"
print("SCRL in:", ms(res["scrl"]["in"]), "oos:", ms(res["scrl"]["oos"]), "mods:", np.mean(res["scrl"]["mods"]))
print("per-rule cov range (train):", min(res["per_rule_cov_train"]), max(res["per_rule_cov_train"]))
print("per-rule cov range (test):", min(res["per_rule_cov_test"]), max(res["per_rule_cov_test"]))
print("union OOS cov:", ms(res["union_cov_test"]))
json.dump(res, open("results/paper_tables/audit_scrl_reverify.json", "w"), indent=1)
print("saved")
