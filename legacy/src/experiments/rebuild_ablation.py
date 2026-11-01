"""Honest rebuild of SCRL ablation (Table 7), seeds 42-46, current-code variants."""
import json
import numpy as np
from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL
from src.metrics.fidelity import oos_coverage_fidelity

SEEDS = list(range(42, 47))
VARIANTS = {
    "no_div_dedup": dict(use_diversity=False, use_dedup=False),
    "variance_based": dict(variance_scoring=True),
}
prev = json.load(open("results/paper_tables/audit_scrl_reverify.json"))
res = {"full": {"in": list(prev["scrl"]["in"]), "oos": list(prev["scrl"]["oos"]),
                "mods": list(prev["scrl"]["mods"]), "nfeat": []},
       "no_div_dedup": {"in": [], "oos": [], "mods": [], "nfeat": []},
       "variance_based": {"in": [], "oos": [], "mods": [], "nfeat": []}}

for seed in SEEDS:
    train, test = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed)
    mods = set(int(i) for i in train.effect_modifier_indices)
    m = TLearner(); m.fit(train.X, train.T, train.Y)
    tau_tr = m.predict_cate(train.X); tau_te = m.predict_cate(test.X)
    # full variant: recompute nfeat only (rules cheap to refit? no—reuse not possible for nfeat) -> refit full too for nfeat
    for name, kw in [("full", {})] + list(VARIANTS.items()):
        if name == "full" and res["full"]["nfeat"] and len(res["full"]["nfeat"]) > SEEDS.index(seed):
            continue
        s = SCRL(cate_model=m, n_rules=5, n_bootstrap=10, n_candidates=20, seed=seed, **kw)
        s.fit(train.X, train.T, train.Y)
        mk_tr = [r.covers(train.X) for r in s.rules]
        mk_te = [r.covers(test.X) for r in s.rules]
        in_f = oos_coverage_fidelity(train.X, tau_tr, train.X, tau_tr, mk_tr) if mk_tr else 0.0
        oos_f = oos_coverage_fidelity(train.X, tau_tr, test.X, tau_te, mk_te) if mk_te else 0.0
        feats = {int(c[0]) for r in s.rules for c in r.conditions}
        hit = len(feats & mods)
        if name == "full":
            res["full"]["nfeat"].append(len(feats))  # in/oos/mods already from prev
        else:
            res[name]["in"].append(in_f); res[name]["oos"].append(oos_f)
            res[name]["mods"].append(hit); res[name]["nfeat"].append(len(feats))
        print(f"seed {seed} {name}: in={in_f:.3f} oos={oos_f:.3f} mods={hit}/5 nfeat={len(feats)} nrules={len(s.rules)}", flush=True)

def ms(a): a = np.asarray(a, float); return f"{a.mean():.3f}+/-{a.std():.3f}"
print("\n=== REBUILT ABLATION (5 seeds) ===")
for name in ("full", "no_div_dedup", "variance_based"):
    v = res[name]
    print(f"{name}: in={ms(v['in'])} oos={ms(v['oos'])} mods={np.mean(v['mods']):.1f}/5 nfeat={np.mean(v['nfeat']):.1f}")
json.dump(res, open("results/paper_tables/ablation_rebuild.json", "w"), indent=1)
print("saved")
