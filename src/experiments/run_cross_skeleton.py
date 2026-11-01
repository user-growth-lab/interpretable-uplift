"""Cross-skeleton replication of zero-attribution and modifier recovery.

Addresses reviewer M3: headline semi-synthetic evidence (zero attribution,
modifier recovery) is concentrated on the IHDP covariate skeleton. Here we
replicate the core protocol on two NON-IHDP skeletons: real covariates from
Criteo Uplift v2.1 and Hillstrom, with a synthetic RCT and a known additive
tau built exactly as in make_semisynthetic_ihdp (signal 2.0, noise 0.3,
5 modifiers, Scenario B adds 5 confounded proxies at rho=0.9).

Protocol mirrors run_all_tables.run_table1 / run_table2_3:
  - Attribution: 10 seeds, TESV vs Interventional SHAP vs Conditional SHAP,
    Top-5/Top-10 recall (Scenario A) and spurious mass (Scenario B).
  - Rules: 5 seeds, SCRL modifier hit + coverage fidelity vs Causal Tree.
"""
from __future__ import annotations
import json, os, time
import numpy as np

from src.data.datasets import load_criteo, load_hillstrom, CATEDataset
from src.models.cate_estimators import TLearner, CATEEstimator
from src.models.tesv import TESVExplainer
from src.models.baselines import (
    SHAPCATEBaseline, InterventionalSHAPBaseline, CausalTreeBaseline,
)
from src.models.scrl import SCRL
from src.metrics.fidelity import (
    top_k_effect_modifier_recall, rule_coverage_fidelity,
)

ATTR_SEEDS = list(range(10))
RULE_SEEDS = list(range(5))
N_PERM = 20
N_BG = 50
N_TEST_EXPLAIN = 100
N_SUB = 5000          # covariate rows drawn per skeleton
SIGNAL = 2.0
TAU_NOISE = 0.3
MU0_NOISE = 0.5
N_MODIFIERS = 5
CONF_CORR = 0.9
OUT_JSON = "results/paper_tables/cross_skeleton.json"


class _OrigFeatureWrapper(CATEEstimator):
    def __init__(self, model, n_orig):
        self.model = model; self.n_orig = n_orig
    def fit(self, X, T, Y): return self
    def predict_cate(self, X): return self.model.predict_cate(X[:, :self.n_orig])


def _spurious_attr(attr, n_orig):
    g = np.abs(attr).mean(axis=0)
    total = g.sum()
    return float(g[n_orig:].sum() / total) if total > 0 else 0.0


def load_skeleton_covariates(name: str) -> np.ndarray:
    """Load real (standardized) covariates once; discard original T/Y."""
    if name == "criteo":
        train, test = load_criteo(sample_n=N_SUB + 1000, seed=0)
    elif name == "hillstrom":
        train, test = load_hillstrom(seed=0)
    else:
        raise ValueError(name)
    X = np.vstack([train.X, test.X])
    return X


def make_semisynthetic_from_skeleton(
    X_all: np.ndarray,
    seed: int,
    add_confounded_features: bool = False,
):
    """Synthetic RCT + additive tau on real covariates (IHDP-matched DGP)."""
    rng = np.random.default_rng(seed)
    n_take = min(N_SUB, X_all.shape[0])
    idx = rng.choice(X_all.shape[0], size=n_take, replace=False)
    X = X_all[idx].copy()
    n, d = X.shape

    modifiers = sorted(rng.choice(d, size=N_MODIFIERS, replace=False))

    # Randomised treatment (synthetic RCT)
    T = rng.binomial(1, 0.5, size=n)

    # Baseline outcome mu0 = X @ w + noise (analogue of IHDP's mu0 surface)
    w = rng.normal(0, 0.5, size=d)
    mu0 = X @ w + rng.normal(0, MU0_NOISE, size=n)

    # Additive tau on the true modifiers (same form as make_semisynthetic_ihdp)
    tau = np.zeros(n)
    for j in modifiers:
        tau += SIGNAL * X[:, j]
    tau += rng.normal(0, TAU_NOISE, size=n)

    Y = T * (mu0 + tau) + (1 - T) * mu0

    feature_names = [f"x{i}" for i in range(d)]
    if add_confounded_features:
        noise = rng.normal(0, 1, size=(n, N_MODIFIERS))
        conf = CONF_CORR * X[:, modifiers] + np.sqrt(1 - CONF_CORR ** 2) * noise
        X = np.hstack([X, conf])
        feature_names += [f"x{d + i}_conf" for i in range(N_MODIFIERS)]

    # 80/20 split
    n_tr = int(0.8 * n)
    perm = rng.permutation(n)
    tr, te = perm[:n_tr], perm[n_tr:]
    train_ds = CATEDataset(X=X[tr], T=T[tr], Y=Y[tr], tau=tau[tr], feature_names=feature_names)
    test_ds = CATEDataset(X=X[te], T=T[te], Y=Y[te], tau=tau[te], feature_names=feature_names)
    train_ds.effect_modifier_indices = modifiers
    test_ds.effect_modifier_indices = modifiers
    train_ds.n_orig = d
    test_ds.n_orig = d
    return train_ds, test_ds


def run_attribution(skeleton: str, X_all: np.ndarray):
    print(f"\n=== Attribution on {skeleton} skeleton ({len(ATTR_SEEDS)} seeds) ===", flush=True)
    res = {"A": {m: {"top5": [], "top10": []} for m in ["cond", "interv", "tesv"]},
           "B": {m: {"spur": []} for m in ["cond", "interv", "tesv"]}}
    for seed in ATTR_SEEDS:
        for scenario, add_conf in [("A", False), ("B", True)]:
            train, test = make_semisynthetic_from_skeleton(X_all, seed=seed, add_confounded_features=add_conf)
            mods = train.effect_modifier_indices
            n_orig = train.n_orig
            model = TLearner()
            model.fit(train.X[:, :n_orig], train.T, train.Y)
            cate_m = _OrigFeatureWrapper(model, n_orig) if add_conf else model
            X_bg = train.X
            X_te = test.X[:N_TEST_EXPLAIN]
            explainers = {
                "cond": SHAPCATEBaseline(cate_model=cate_m, background_data=X_bg, n_background=N_BG, k_neighbors=10, seed=seed),
                "interv": InterventionalSHAPBaseline(cate_model=cate_m, background_data=X_bg, n_background=N_BG, seed=seed),
                "tesv": TESVExplainer(cate_model=cate_m, background_data=X_bg, n_background=N_BG, n_permutations=N_PERM, seed=seed),
            }
            for name, exp in explainers.items():
                if name == "tesv":
                    attr = exp.explain(X_te, show_progress=False)
                else:
                    attr = exp.explain(X_te, n_permutations=N_PERM)
                if scenario == "A":
                    r5 = top_k_effect_modifier_recall(attr, mods, k=5)
                    r10 = top_k_effect_modifier_recall(attr, mods, k=10)
                    res["A"][name]["top5"].append(r5)
                    res["A"][name]["top10"].append(r10)
                    print(f"  seed {seed} A {name}: top5={r5:.3f} top10={r10:.3f}", flush=True)
                else:
                    sp = _spurious_attr(attr, n_orig)
                    res["B"][name]["spur"].append(sp)
                    print(f"  seed {seed} B {name}: spur={sp:.4f}", flush=True)
    return res


def run_rules(skeleton: str, X_all: np.ndarray):
    print(f"\n=== Rules on {skeleton} skeleton ({len(RULE_SEEDS)} seeds) ===", flush=True)
    res = {m: {"fid": [], "mods": []} for m in ["scrl", "ctree"]}
    for seed in RULE_SEEDS:
        train, test = make_semisynthetic_from_skeleton(X_all, seed=seed, add_confounded_features=False)
        mods = set(train.effect_modifier_indices)
        model = TLearner()
        model.fit(train.X, train.T, train.Y)
        tau_test = model.predict_cate(test.X)
        scrl = SCRL(cate_model=model, n_rules=5, max_rule_length=2, min_support=0.1,
                    overlap_ratio=0.1, n_candidates=10, n_bootstrap=5, seed=seed)
        scrl.fit(train.X, train.T, train.Y)
        masks = [r.covers(test.X) for r in scrl.rules]
        fid = rule_coverage_fidelity(test.X, tau_test, masks) if masks else 0.0
        hits = len({c[0] for r in scrl.rules for c in r.conditions} & mods)
        res["scrl"]["fid"].append(fid)
        res["scrl"]["mods"].append(hits)
        ctree = CausalTreeBaseline(max_depth=3, min_samples_leaf=20, seed=seed)
        ctree.fit(train.X, train.T, train.Y)
        ct_pred = ctree.predict_cate(test.X)
        ct_masks = [ct_pred == leaf for leaf in np.unique(ct_pred)]
        fid_ct = rule_coverage_fidelity(test.X, tau_test, ct_masks)
        tree_feats = {f for f in ctree.tree.tree_.feature if f >= 0}
        res["ctree"]["fid"].append(fid_ct)
        res["ctree"]["mods"].append(len(tree_feats & mods))
        print(f"  seed {seed}: SCRL fid={fid:.4f} mods={hits}/5 | CT fid={fid_ct:.4f} mods={len(tree_feats & mods)}/5", flush=True)
    return res


def summarize(res):
    out = {}
    for k, v in res.items():
        if isinstance(v, dict):
            out[k] = summarize(v)
        else:
            arr = np.asarray(v, dtype=float)
            out[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}
    return out


def main():
    t0 = time.time()
    all_results = {}
    for skeleton in ["hillstrom", "criteo"]:
        print(f"\n########## Skeleton: {skeleton} ##########", flush=True)
        X_all = load_skeleton_covariates(skeleton)
        print(f"covariates: {X_all.shape}", flush=True)
        all_results[skeleton] = {
            "d": int(X_all.shape[1]),
            "attribution": run_attribution(skeleton, X_all),
            "rules": run_rules(skeleton, X_all),
        }
        # incremental save
        with open(OUT_JSON, "w") as f:
            json.dump({"raw": all_results, "summary": {k: {kk: summarize(vv) if kk != "d" else vv for kk, vv in v.items()} for k, v in all_results.items()},
                       "config": {"attr_seeds": ATTR_SEEDS, "rule_seeds": RULE_SEEDS, "n_sub": N_SUB,
                                  "signal": SIGNAL, "tau_noise": TAU_NOISE, "n_modifiers": N_MODIFIERS,
                                  "conf_corr": CONF_CORR, "n_perm": N_PERM, "n_bg": N_BG}}, f, indent=2)
        print(f"saved -> {OUT_JSON}", flush=True)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
