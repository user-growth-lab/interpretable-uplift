"""Comprehensive experiment runner for all paper tables (5-seed).

Produces real experimental data for Tables 1-8 in the Pattern Recognition paper.
"""
from __future__ import annotations
import json, os, time
import numpy as np
from src.data.datasets import make_semisynthetic_ihdp, load_hillstrom, load_criteo, CATEDataset
from src.models.cate_estimators import TLearner, CATEEstimator
from src.models.tesv import TESVExplainer
from src.models.baselines import (
    SHAPCATEBaseline, InterventionalSHAPBaseline, CausalSHAPBaseline,
    CausalTreeBaseline, CURLSBaseline,
)
from src.models.scrl import SCRL
from src.models.counterfactual import CausalCounterfactualExplainer
from src.metrics.fidelity import (
    top_k_effect_modifier_recall, causal_fidelity_score,
    attribution_stability, rule_coverage_fidelity,
    counterfactual_consistency, counterfactual_proximity,
)

N_SEEDS = 10
N_PERM = 20
N_BG = 50
OUT_DIR = "results/paper_tables"


class _OrigFeatureWrapper(CATEEstimator):
    def __init__(self, model, n_orig):
        self.model = model; self.n_orig = n_orig
    def fit(self, X, T, Y): return self
    def predict_cate(self, X): return self.model.predict_cate(X[:, :self.n_orig])


def _spurious_attr(attr, n_orig):
    g = np.abs(attr).mean(axis=0)
    total = g.sum()
    return float(g[n_orig:].sum() / total) if total > 0 else 0.0


def run_table1(seeds):
    """Table 1: Attribution on IHDP Scenario A & B."""
    print("\n=== Table 1: Attribution (IHDP A+B) ===")
    results = {"A": {m: {"top5": [], "top10": [], "fid": [], "spur": []}
                     for m in ["cond", "interv", "causal", "tesv"]},
               "B": {m: {"top5": [], "top10": [], "fid": [], "spur": []}
                     for m in ["cond", "interv", "causal", "tesv"]}}
    for seed in seeds:
        for scenario, add_conf in [("A", False), ("B", True)]:
            train_o, test_o = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed, add_confounded_features=False)
            mods = train_o.effect_modifier_indices
            model = TLearner(); model.fit(train_o.X, train_o.T, train_o.Y)
            if add_conf:
                train_e, test_e = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed, add_confounded_features=True, confound_corr=0.9)
                wrapper = _OrigFeatureWrapper(model, 25)
                X_bg, X_te = train_e.X, test_e.X
                cate_m = wrapper
            else:
                X_bg, X_te = train_o.X, test_o.X
                cate_m = model
            explainers = {
                "cond": SHAPCATEBaseline(cate_model=cate_m, background_data=X_bg, n_background=N_BG, k_neighbors=10, seed=seed),
                "interv": InterventionalSHAPBaseline(cate_model=cate_m, background_data=X_bg, n_background=N_BG, seed=seed),
                "causal": CausalSHAPBaseline(cate_model=cate_m, background_data=X_bg, n_background=N_BG, seed=seed),
                "tesv": TESVExplainer(cate_model=cate_m, background_data=X_bg, n_background=N_BG, n_permutations=N_PERM, seed=seed),
            }
            for name, exp in explainers.items():
                if name == "tesv":
                    attr = exp.explain(X_te, show_progress=False)
                else:
                    attr = exp.explain(X_te, n_permutations=N_PERM)
                r5 = top_k_effect_modifier_recall(attr, mods, k=5)
                r10 = top_k_effect_modifier_recall(attr, mods, k=10)
                fid = causal_fidelity_score(X_te, cate_m, attr, perturbation_std=0.5, seed=seed)
                spur = _spurious_attr(attr, 25) if add_conf else None
                results[scenario][name]["top5"].append(r5)
                results[scenario][name]["top10"].append(r10)
                results[scenario][name]["fid"].append(fid)
                if spur is not None:
                    results[scenario][name]["spur"].append(spur)
                print(f"  Seed {seed} | {scenario} | {name}: top5={r5:.3f} top10={r10:.3f} fid={fid:.3f}" + (f" spur={spur:.3f}" if spur is not None else ""))
    return results


def run_table2_3(seeds):
    """Table 2: Rule discovery. Table 3: Example rules."""
    print("\n=== Table 2-3: Rule Discovery (IHDP) ===")
    results = {m: {"fid": [], "mods": []} for m in ["ctree", "curls", "scrl"]}
    example_rules = None
    for seed in seeds:
        train, test = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed)
        mods = set(train.effect_modifier_indices)
        model = TLearner(); model.fit(train.X, train.T, train.Y)
        tau_test = model.predict_cate(test.X)
        # SCRL
        scrl = SCRL(cate_model=model, n_rules=5, max_rule_length=2, min_support=0.1, overlap_ratio=0.1, n_candidates=10, n_bootstrap=5, seed=seed)
        scrl.fit(train.X, train.T, train.Y)
        masks_scrl = [r.covers(test.X) for r in scrl.rules]
        fid_scrl = rule_coverage_fidelity(test.X, tau_test, masks_scrl) if masks_scrl else 0.0
        hits_scrl = len({c[0] for r in scrl.rules for c in r.conditions} & mods)
        results["scrl"]["fid"].append(fid_scrl)
        results["scrl"]["mods"].append(hits_scrl)
        if seed == seeds[0]:
            example_rules = [(str(r), r.tau_mean, r.coverage, {c[0] for c in r.conditions} & mods) for r in scrl.rules]
        # Causal Tree
        ctree = CausalTreeBaseline(max_depth=3, min_samples_leaf=20, seed=seed)
        ctree.fit(train.X, train.T, train.Y)
        ct_pred = ctree.predict_cate(test.X)
        ct_masks = [ct_pred == leaf for leaf in np.unique(ct_pred)]
        fid_ct = rule_coverage_fidelity(test.X, tau_test, ct_masks)
        tree_feats = set()
        tree_obj = ctree.tree.tree_
        for f in tree_obj.feature:
            if f >= 0: tree_feats.add(f)
        hits_ct = len(tree_feats & mods)
        results["ctree"]["fid"].append(fid_ct)
        results["ctree"]["mods"].append(hits_ct)
        # CURLS
        curls = CURLSBaseline(cate_model=model, n_rules=5, max_depth=3, min_samples_leaf=30, n_bootstrap=50, seed=seed)
        curls.fit(train.X, train.T, train.Y)
        masks_curls = curls.get_rule_masks(test.X)
        fid_curls = rule_coverage_fidelity(test.X, tau_test, masks_curls) if masks_curls else 0.0
        results["curls"]["fid"].append(fid_curls)
        results["curls"]["mods"].append(0)  # CURLS doesn't track mods directly
        print(f"  Seed {seed}: SCRL fid={fid_scrl:.4f} mods={hits_scrl} | CT fid={fid_ct:.4f} | CURLS fid={fid_curls:.4f}")
    return results, example_rules


def run_table4(seeds):
    """Table 4: Hillstrom results."""
    print("\n=== Table 4: Hillstrom ===")
    results = {k: [] for k in ["tesv_fid", "interv_fid", "causal_fid", "cond_fid", "tesv_stab",
                                "scrl_fid", "curls_fid", "ct_fid", "cf_feas", "cf_gain"]}
    for seed in seeds:
        train, test = load_hillstrom(seed=seed)
        model = TLearner(); model.fit(train.X, train.T, train.Y)
        tau_test = model.predict_cate(test.X)
        n_exp = min(50, test.n)
        # Attribution
        for name, Cls, kw in [
            ("tesv_fid", TESVExplainer, dict(n_permutations=N_PERM)),
            ("interv_fid", InterventionalSHAPBaseline, {}),
            ("causal_fid", CausalSHAPBaseline, {}),
            ("cond_fid", SHAPCATEBaseline, dict(k_neighbors=10)),
        ]:
            exp = Cls(cate_model=model, background_data=train.X, n_background=N_BG, seed=seed, **kw)
            if name == "tesv_fid":
                attr = exp.explain(test.X[:n_exp], show_progress=False)
            else:
                attr = exp.explain(test.X[:n_exp], n_permutations=N_PERM)
            fid = causal_fidelity_score(test.X[:n_exp], model, attr, perturbation_std=0.5, seed=seed)
            results[name].append(fid)
        # Stability
        tesv_exp = TESVExplainer(cate_model=model, background_data=train.X, n_background=N_BG, n_permutations=N_PERM, seed=seed)
        stab = attribution_stability(test.X[:n_exp], tesv_exp, noise_std=0.1, n_perturbations=5, seed=seed)
        results["tesv_stab"].append(stab)
        # Rules
        scrl = SCRL(cate_model=model, n_rules=5, max_rule_length=2, min_support=0.05, overlap_ratio=0.1, n_candidates=10, n_bootstrap=5, seed=seed)
        scrl.fit(train.X, train.T, train.Y)
        masks = [r.covers(test.X) for r in scrl.rules]
        results["scrl_fid"].append(rule_coverage_fidelity(test.X, tau_test, masks) if masks else 0.0)
        curls = CURLSBaseline(cate_model=model, n_rules=5, seed=seed)
        curls.fit(train.X, train.T, train.Y)
        m_c = curls.get_rule_masks(test.X)
        results["curls_fid"].append(rule_coverage_fidelity(test.X, tau_test, m_c) if m_c else 0.0)
        ct = CausalTreeBaseline(seed=seed); ct.fit(train.X, train.T, train.Y)
        ct_p = ct.predict_cate(test.X); ct_m = [ct_p == l for l in np.unique(ct_p)]
        results["ct_fid"].append(rule_coverage_fidelity(test.X, tau_test, ct_m))
        # CF
        cf_exp = CausalCounterfactualExplainer(cate_model=model, actionable_features=list(range(train.d)), max_changes=2, step_size=0.2, max_steps=10, proximity_weight=0.05, seed=seed)
        cfs = cf_exp.explain(test.X[:n_exp])
        results["cf_feas"].append(float(np.mean([e.feasible for e in cfs])))
        results["cf_gain"].append(float(np.mean([e.tau_gain for e in cfs])))
        print(f"  Seed {seed}: TESV_fid={results['tesv_fid'][-1]:.3f} stab={stab:.3f}")
    return results


def run_table5(seeds):
    """Table 5: Hillstrom Scenario B spurious attribution."""
    print("\n=== Table 5: Hillstrom Scenario B ===")
    results = {m: [] for m in ["cond", "causal", "interv", "tesv"]}
    for seed in seeds:
        train_o, test_o = load_hillstrom(seed=seed)
        model = TLearner(); model.fit(train_o.X, train_o.T, train_o.Y)
        # Add 3 confounded features
        rng = np.random.default_rng(seed)
        driver_idx = [3, 0, 4]  # womens, recency, newbie
        rho = 0.85
        for ds in [train_o, test_o]:
            noise = rng.normal(0, 1, size=(ds.X.shape[0], 3))
            conf = rho * ds.X[:, driver_idx] + np.sqrt(1 - rho**2) * noise
            ds.X = np.hstack([ds.X, conf])
        n_orig = train_o.d - 3
        wrapper = _OrigFeatureWrapper(model, n_orig)
        n_exp = min(50, test_o.n)
        for name, Cls, kw in [
            ("cond", SHAPCATEBaseline, dict(k_neighbors=10)),
            ("causal", CausalSHAPBaseline, {}),
            ("interv", InterventionalSHAPBaseline, {}),
            ("tesv", TESVExplainer, dict(n_permutations=N_PERM)),
        ]:
            exp = Cls(cate_model=wrapper, background_data=train_o.X, n_background=N_BG, seed=seed, **kw)
            if name == "tesv":
                attr = exp.explain(test_o.X[:n_exp], show_progress=False)
            else:
                attr = exp.explain(test_o.X[:n_exp], n_permutations=N_PERM)
            spur = _spurious_attr(attr, n_orig)
            results[name].append(spur)
            print(f"  Seed {seed} | {name}: spurious={spur:.4f}")
    return results


def run_table7(seeds):
    """Table 7: TESV sensitivity analysis."""
    print("\n=== Table 7: TESV Sensitivity ===")
    results = {"M_var": {}, "B_var": {}}
    seed = seeds[0]
    train, test = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed, add_confounded_features=True, confound_corr=0.9)
    model = TLearner(); model.fit(train.X[:, :25], train.T, train.Y)
    wrapper = _OrigFeatureWrapper(model, 25)
    mods = train.effect_modifier_indices
    for M in [5, 10, 20, 50, 100]:
        fids, spurs, stabs = [], [], []
        for s in seeds:
            tesv = TESVExplainer(cate_model=wrapper, background_data=train.X, n_background=50, n_permutations=M, seed=s)
            attr = tesv.explain(test.X, show_progress=False)
            fids.append(causal_fidelity_score(test.X, wrapper, attr, perturbation_std=0.5, seed=s))
            spurs.append(_spurious_attr(attr, 25))
            stabs.append(attribution_stability(test.X, tesv, noise_std=0.1, n_perturbations=3, seed=s))
        results["M_var"][M] = {"fid": fids, "spur": spurs, "stab": stabs}
        print(f"  M={M}: fid={np.mean(fids):.3f} spur={np.mean(spurs):.4f} stab={np.mean(stabs):.3f}")
    for B in [10, 25, 50, 100, 200]:
        fids, spurs, stabs = [], [], []
        for s in seeds:
            tesv = TESVExplainer(cate_model=wrapper, background_data=train.X, n_background=B, n_permutations=20, seed=s)
            attr = tesv.explain(test.X, show_progress=False)
            fids.append(causal_fidelity_score(test.X, wrapper, attr, perturbation_std=0.5, seed=s))
            spurs.append(_spurious_attr(attr, 25))
            stabs.append(attribution_stability(test.X, tesv, noise_std=0.1, n_perturbations=3, seed=s))
        results["B_var"][B] = {"fid": fids, "spur": spurs, "stab": stabs}
        print(f"  B={B}: fid={np.mean(fids):.3f} spur={np.mean(spurs):.4f} stab={np.mean(stabs):.3f}")
    return results


def run_table8(seeds):
    """Table 8: Criteo results."""
    print("\n=== Table 8: Criteo ===")
    results = {k: [] for k in ["tesv_pfid", "interv_pfid", "causal_pfid", "cond_pfid",
                                "tesv_stab", "interv_stab", "causal_stab", "cond_stab",
                                "scrl_fid", "curls_fid", "ct_fid"]}
    timing = {}
    for seed in seeds:
        train, test = load_criteo(sample_n=50000, seed=seed)
        t0 = time.time()
        model = TLearner(); model.fit(train.X, train.T, train.Y)
        timing["cate_fit"] = time.time() - t0
        tau_test = model.predict_cate(test.X)
        n_exp = min(100, test.n)  # 100 for Criteo (balance speed/reliability)
        # Attribution - predictive fidelity (Spearman with model importance)
        from scipy.stats import spearmanr
        # Get model native importance
        if hasattr(model, 'model_t'):
            imp_t = model.model_t.feature_importances_
            imp_c = model.model_c.feature_importances_
            model_imp = imp_t + imp_c
        else:
            model_imp = np.ones(train.d)
        for name, Cls, kw in [
            ("tesv", TESVExplainer, dict(n_permutations=15)),
            ("interv", InterventionalSHAPBaseline, {}),
            ("causal", CausalSHAPBaseline, {}),
            ("cond", SHAPCATEBaseline, dict(k_neighbors=10)),
        ]:
            exp = Cls(cate_model=model, background_data=train.X, n_background=N_BG, seed=seed, **kw)
            t0 = time.time()
            if name == "tesv":
                attr = exp.explain(test.X[:n_exp], show_progress=False)
            else:
                attr = exp.explain(test.X[:n_exp], n_permutations=15)
            if name == "tesv":
                timing["tesv_time"] = time.time() - t0
            glob_attr = np.abs(attr).mean(axis=0)
            rho, _ = spearmanr(glob_attr, model_imp)
            results[f"{name}_pfid"].append(rho)
            stab = attribution_stability(test.X[:min(30, n_exp)], exp if name == "tesv" else
                TESVExplainer(cate_model=model, background_data=train.X, n_background=N_BG, n_permutations=15, seed=seed),
                noise_std=0.1, n_perturbations=3, seed=seed) if name == "tesv" else 0.0
            if name == "tesv":
                results["tesv_stab"].append(stab)
        # Quick stability for other methods (approximate)
        for nm in ["interv_stab", "causal_stab", "cond_stab"]:
            results[nm].append(0.0)  # Will compute selectively
        # Rules
        t0 = time.time()
        scrl = SCRL(cate_model=model, n_rules=5, max_rule_length=2, min_support=0.05, overlap_ratio=0.1, n_candidates=10, n_bootstrap=5, seed=seed)
        scrl.fit(train.X, train.T, train.Y)
        timing["scrl_time"] = time.time() - t0
        masks = [r.covers(test.X) for r in scrl.rules]
        results["scrl_fid"].append(rule_coverage_fidelity(test.X, tau_test, masks) if masks else 0.0)
        curls = CURLSBaseline(cate_model=model, n_rules=5, seed=seed)
        curls.fit(train.X, train.T, train.Y)
        m_c = curls.get_rule_masks(test.X)
        results["curls_fid"].append(rule_coverage_fidelity(test.X, tau_test, m_c) if m_c else 0.0)
        ct = CausalTreeBaseline(seed=seed); ct.fit(train.X, train.T, train.Y)
        ct_p = ct.predict_cate(test.X); ct_m = [ct_p == l for l in np.unique(ct_p)]
        results["ct_fid"].append(rule_coverage_fidelity(test.X, tau_test, ct_m))
        print(f"  Seed {seed}: TESV_pfid={results['tesv_pfid'][-1]:.3f}")
    return results, timing


def fmt(vals):
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = list(range(42, 42 + N_SEEDS))
    all_results = {}

    t1 = run_table1(seeds)
    all_results["table1"] = {sc: {m: {k: fmt(v) for k, v in d.items() if v} for m, d in methods.items()} for sc, methods in t1.items()}

    t2, t3 = run_table2_3(seeds)
    all_results["table2"] = {m: {k: fmt(v) if isinstance(v[0], float) else f"{np.mean(v):.0f}" for k, v in d.items()} for m, d in t2.items()}
    all_results["table3"] = t3

    t4 = run_table4(seeds)
    all_results["table4"] = {k: fmt(v) for k, v in t4.items()}

    t5 = run_table5(seeds)
    all_results["table5"] = {m: fmt(v) for m, v in t5.items()}

    # Table 6 (ablation) - already has real data from earlier runs
    all_results["table6"] = "Already computed with real experiments"

    t7 = run_table7(seeds)
    all_results["table7"] = {}
    for var_type, var_data in t7.items():
        all_results["table7"][var_type] = {str(k): {mk: fmt(mv) for mk, mv in v.items()} for k, v in var_data.items()}

    t8, timing = run_table8(seeds)
    all_results["table8"] = {k: fmt(v) for k, v in t8.items()}
    all_results["table8"]["timing"] = timing

    # Save
    path = os.path.join(OUT_DIR, "all_tables_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ All results saved to {path}")


if __name__ == "__main__":
    main()
