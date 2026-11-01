"""Hostile-review probes: RCT-truth validation of the paper's two load-bearing real-data claims.

H3 (Hillstrom): Table 21's "1.8x uplift, 68% cost reduction" was computed on the model's own
predicted CATE. Hillstrom is an RCT, so we re-compute subgroup ATE by difference-in-means
(valid under randomisation) for SCRL rules vs treat-all vs Top-20%-by-tau, with SEs and
paired tests across seeds.

H2 (MegaFon): the "-0.59x Causal Tree" safety contrast rests on a single baseline
configuration (min_samples_leaf=50 on 100K = 0.05%). We sweep honest configurations
(min_samples_leaf in {50, 500, 1000, 2500, 5000}, max_depth in {3, 5}) and check whether
the catastrophic negative targeting gain survives.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.datasets import load_hillstrom, load_megafon
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL
from src.models.baselines import CausalTreeBaseline


def subgroup_ate(Y, T, mask):
    """Difference-in-means ATE within a subgroup (valid under RCT). Returns (ate, se, n, coverage)."""
    if mask.sum() < 10:
        return np.nan, np.nan, int(mask.sum()), float(mask.mean())
    Y1, Y0 = Y[mask & (T == 1)], Y[mask & (T == 0)]
    if len(Y1) < 5 or len(Y0) < 5:
        return np.nan, np.nan, int(mask.sum()), float(mask.mean())
    ate = Y1.mean() - Y0.mean()
    se = np.sqrt(Y1.var(ddof=1) / len(Y1) + Y0.var(ddof=1) / len(Y0))
    return float(ate), float(se), int(mask.sum()), float(mask.mean())


def run_hillstrom_probe(n_seeds=5):
    print("=" * 70)
    print("PROBE H3: Hillstrom RCT-truth subgroup ATE (predicted-CATE vs truth)")
    print("=" * 70)
    per_seed = []
    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        train_ds, test_ds = load_hillstrom(seed=seed)
        base = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        cate_model = TLearner(base_learner=base)
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)
        tau_test = cate_model.predict_cate(test_ds.X)

        scrl = SCRL(cate_model=cate_model, n_rules=5, max_rule_length=2,
                    n_bootstrap=10, n_candidates=20, max_coverage=0.40,
                    sample_split=True, seed=seed)
        scrl.fit(train_ds.X, train_ds.T, train_ds.Y)

        masks = {}
        masks["treat_all"] = np.ones(test_ds.n, dtype=bool)
        if scrl.rules:
            masks["scrl_rule1"] = scrl.rules[0].covers(test_ds.X)
        else:
            masks["scrl_rule1"] = np.zeros(test_ds.n, dtype=bool)
        union = np.zeros(test_ds.n, dtype=bool)
        for r in scrl.rules:
            union |= r.covers(test_ds.X)
        masks["scrl_union"] = union
        thr = np.percentile(tau_test, 80)
        masks["top20_tau"] = tau_test >= thr

        row = {"seed": seed}
        pop_ate, pop_se, _, _ = subgroup_ate(test_ds.Y, test_ds.T, masks["treat_all"])
        row["pop_ate"] = pop_ate
        for name, mask in masks.items():
            ate, se, n, cov = subgroup_ate(test_ds.Y, test_ds.T, mask)
            row[f"{name}_ate"] = ate
            row[f"{name}_se"] = se
            row[f"{name}_cov"] = cov
            row[f"{name}_gain"] = (ate / pop_ate) if (pop_ate and not np.isnan(ate) and pop_ate > 0) else np.nan
            # Model's own predicted-CATE mean within the mask (the Table-21 metric)
            row[f"{name}_pred"] = float(tau_test[mask].mean()) if mask.sum() > 0 else np.nan
        per_seed.append(row)

    # Aggregate + paired tests (rule1 vs pop, top20 vs rule1, across seeds)
    def col(key):
        return np.array([r[key] for r in per_seed], dtype=float)

    summary = {}
    for name in ["treat_all", "scrl_rule1", "scrl_union", "top20_tau"]:
        ates, gains, covs = col(f"{name}_ate"), col(f"{name}_gain"), col(f"{name}_cov")
        summary[name] = {
            "ate_mean": float(np.nanmean(ates)), "ate_std": float(np.nanstd(ates)),
            "gain_mean": float(np.nanmean(gains)), "gain_std": float(np.nanstd(gains)),
            "cov_mean": float(np.nanmean(covs)),
        }
    pop = col("pop_ate")
    for name in ["scrl_rule1", "top20_tau"]:
        diff = col(f"{name}_ate") - pop
        t, p = stats.ttest_rel(col(f"{name}_ate"), pop)
        summary[f"{name}_vs_pop"] = {"mean_diff": float(np.nanmean(diff)), "p_value": float(p)}
    t, p = stats.ttest_rel(col("top20_tau_ate"), col("scrl_rule1_ate"))
    summary["top20_vs_rule1"] = {"mean_diff": float(np.nanmean(col("top20_tau_ate") - col("scrl_rule1_ate"))),
                                 "p_value": float(p)}
    # The model-vs-truth gap for the paper's Table-21 headline
    summary["rule1_pred_mean"] = float(np.nanmean(col("scrl_rule1_pred")))
    summary["pop_pred_mean"] = float(np.nanmean(col("treat_all_pred")))
    return {"per_seed": per_seed, "summary": summary}


def _fit_ct_rct(ct, train_ds, e_rct):
    """Fit CausalTreeBaseline's tree using RCT propensity (constant e) instead of LR estimate."""
    from sklearn.linear_model import Ridge
    outcome_model = Ridge()
    outcome_model.fit(train_ds.X, train_ds.Y)
    m = outcome_model.predict(train_ds.X)
    pseudo = (train_ds.T - e_rct) * (train_ds.Y - m)
    ct.tree.fit(train_ds.X, pseudo)
    return ct


def run_megafon_ct_probe(n_seeds=3):
    print("\n" + "=" * 70)
    print("PROBE H2: MegaFon Causal Tree honest-configuration sweep")
    print("=" * 70)
    configs = [
        {"max_depth": 3, "min_samples_leaf": 50},    # ~ current baseline (0.05% of 100K)
        {"max_depth": 3, "min_samples_leaf": 500},
        {"max_depth": 3, "min_samples_leaf": 1000},
        {"max_depth": 3, "min_samples_leaf": 2500},
        {"max_depth": 3, "min_samples_leaf": 5000},
        {"max_depth": 5, "min_samples_leaf": 2500},
    ]
    results = {f"d{c['max_depth']}_msl{c['min_samples_leaf']}": [] for c in configs}
    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        train_ds, test_ds = load_megafon(sample_n=100000, seed=seed)
        pop_ate, pop_se, _, _ = subgroup_ate(test_ds.Y, test_ds.T, np.ones(test_ds.n, dtype=bool))
        # MegaFon is an RCT with e ~= 0.5: use the RCT propensity directly instead of
        # estimating it (LogisticRegression on 100Kx50 dominates runtime for no benefit).
        e_rct = float(train_ds.T.mean())
        for c in configs:
            key = f"d{c['max_depth']}_msl{c['min_samples_leaf']}"
            ct = CausalTreeBaseline(max_depth=c["max_depth"],
                                    min_samples_leaf=c["min_samples_leaf"], seed=seed)
            # Patch propensity estimation: monkey-patch fit to use RCT propensity
            _fit_ct_rct(ct, train_ds, e_rct)
            # Target subgroup: the train leaf with the highest mean pseudo-outcome
            leaves_train = ct.tree.apply(train_ds.X)
            pred_train = ct.tree.predict(train_ds.X)
            best_leaf, best_val = None, -np.inf
            for leaf in np.unique(leaves_train):
                m = leaves_train == leaf
                v = pred_train[m].mean()
                if v > best_val:
                    best_val, best_leaf = v, leaf
            leaf_test = ct.tree.apply(test_ds.X)
            mask = leaf_test == best_leaf
            ate, se, n, cov = subgroup_ate(test_ds.Y, test_ds.T, mask)
            gain = (ate / pop_ate) if (pop_ate and not np.isnan(ate) and pop_ate > 0) else np.nan
            results[key].append({"seed": seed, "ate": ate, "se": se, "coverage": cov,
                                 "gain": gain, "pop_ate": pop_ate})
    summary = {}
    for key, rows in results.items():
        gains = np.array([r["gain"] for r in rows], dtype=float)
        ates = np.array([r["ate"] for r in rows], dtype=float)
        covs = np.array([r["coverage"] for r in rows], dtype=float)
        summary[key] = {"gain_mean": float(np.nanmean(gains)), "gain_std": float(np.nanstd(gains)),
                        "ate_mean": float(np.nanmean(ates)), "cov_mean": float(np.nanmean(covs))}
    return {"per_config": results, "summary": summary}


def run_scrl_megafon_truth(seeds=(0, 1, 2)):
    """Re-verify SCRL's MegaFon targeting gains under the fixed loader, with RCT truth,
    plus a coverage-matched Causal Tree comparison (fair like-for-like)."""
    print("\n" + "=" * 70)
    print("PROBE H2b: MegaFon SCRL vs coverage-matched Causal Tree (RCT truth)")
    print("=" * 70)
    per_seed = []
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i + 1}/{len(seeds)})")
        train_ds, test_ds = load_megafon(sample_n=100000, seed=seed)
        base = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        cate_model = TLearner(base_learner=base)
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)
        tau_test = cate_model.predict_cate(test_ds.X)
        pop_ate, pop_se, _, _ = subgroup_ate(test_ds.Y, test_ds.T, np.ones(test_ds.n, dtype=bool))

        scrl = SCRL(cate_model=cate_model, n_rules=5, max_rule_length=2,
                    n_bootstrap=10, n_candidates=20, max_coverage=0.40,
                    sample_split=True, seed=seed)
        scrl.fit(train_ds.X, train_ds.T, train_ds.Y)

        masks = {}
        if scrl.rules:
            masks["scrl_rule1"] = scrl.rules[0].covers(test_ds.X)
            if len(scrl.rules) > 1:
                masks["scrl_rule2"] = scrl.rules[1].covers(test_ds.X)
        union = np.zeros(test_ds.n, dtype=bool)
        for r in scrl.rules:
            union |= r.covers(test_ds.X)
        masks["scrl_union"] = union
        thr = np.percentile(tau_test, 80)
        masks["top20_tau"] = tau_test >= thr

        # Coverage-matched Causal Tree: honest config (msl=2500), take top leaves by
        # train pseudo-outcome until reaching rule1's coverage
        ct = CausalTreeBaseline(max_depth=3, min_samples_leaf=2500, seed=seed)
        e_rct = float(train_ds.T.mean())
        _fit_ct_rct(ct, train_ds, e_rct)
        leaves_train = ct.tree.apply(train_ds.X)
        pred_train = ct.tree.predict(train_ds.X)
        leaf_order = sorted(np.unique(leaves_train),
                            key=lambda l: -pred_train[leaves_train == l].mean())
        leaf_test = ct.tree.apply(test_ds.X)
        target_cov = masks["scrl_rule1"].mean() if "scrl_rule1" in masks else 0.3
        ct_mask = np.zeros(test_ds.n, dtype=bool)
        for l in leaf_order:
            if ct_mask.mean() >= target_cov:
                break
            ct_mask |= (leaf_test == l)
        masks["ct_matched"] = ct_mask
        # CT single best leaf (tiny subgroup reference)
        masks["ct_best_leaf"] = (leaf_test == leaf_order[0])

        row = {"seed": seed, "pop_ate": pop_ate}
        for name, mask in masks.items():
            ate, se, n, cov = subgroup_ate(test_ds.Y, test_ds.T, mask)
            row[f"{name}_ate"] = ate
            row[f"{name}_se"] = se
            row[f"{name}_cov"] = cov
            row[f"{name}_gain"] = (ate / pop_ate) if (pop_ate and not np.isnan(ate) and pop_ate > 0) else np.nan
        per_seed.append(row)

    def col(key):
        return np.array([r[key] for r in per_seed if not np.isnan(r.get(key, np.nan))], dtype=float)

    summary = {}
    names = {k[len('scrl_'):] if k.startswith('scrl_') else k for k in per_seed[0] if k.endswith('_ate')}
    for name in ["scrl_rule1", "scrl_rule2", "scrl_union", "top20_tau", "ct_matched", "ct_best_leaf"]:
        if f"{name}_ate" not in per_seed[0]:
            continue
        ates, gains, covs = col(f"{name}_ate"), col(f"{name}_gain"), col(f"{name}_cov")
        if len(ates) == 0:
            continue
        summary[name] = {"ate_mean": float(np.nanmean(ates)), "ate_std": float(np.nanstd(ates)),
                         "gain_mean": float(np.nanmean(gains)), "gain_std": float(np.nanstd(gains)),
                         "cov_mean": float(np.nanmean(covs)), "n_seeds": int(len(ates))}
    pop = np.array([r["pop_ate"] for r in per_seed])
    for name in ["scrl_rule1", "ct_matched"]:
        if f"{name}_ate" in per_seed[0]:
            a = np.array([r[f"{name}_ate"] for r in per_seed])
            t, p = stats.ttest_rel(a, pop)
            summary[f"{name}_vs_pop_p"] = float(p)
    return {"per_seed": per_seed, "summary": summary}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/paper_tables", exist_ok=True)

    if "--scrl-megafon-one" in sys.argv:
        idx = sys.argv.index("--scrl-megafon-one")
        one_seed = int(sys.argv[idx + 1])
        output_path = f"results/paper_tables/hostile_probes_h2b_seed{one_seed}.json"
        h2b = run_scrl_megafon_truth(seeds=(one_seed,))
        with open(output_path, "w") as f:
            json.dump({"h2b_megafon_scrl_truth": h2b}, f, indent=2)
        print(f"\nSeed {one_seed} saved to {output_path}")
        return

    if "--scrl-megafon" in sys.argv:
        output_path = f"results/paper_tables/hostile_probes_h2b_{ts}.json"
        h2b = run_scrl_megafon_truth(seeds=(0, 1, 2))
        print("\nH2b summary:")
        for k, v in h2b["summary"].items():
            print(f"  {k}: {v}")
        with open(output_path, "w") as f:
            json.dump({"h2b_megafon_scrl_truth": h2b}, f, indent=2)
        print(f"\nResults saved to {output_path}")
        return

    if "--h2-only" in sys.argv:
        output_path = f"results/paper_tables/hostile_probes_h2_{ts}.json"
        h2 = run_megafon_ct_probe(n_seeds=3)
        print("\nH2 summary:")
        for k, v in h2["summary"].items():
            print(f"  {k}: {v}")
        with open(output_path, "w") as f:
            json.dump({"h2_megafon_honest_ct": h2}, f, indent=2)
        print(f"\nResults saved to {output_path}")
        return

    output_path = f"results/paper_tables/hostile_probes_{ts}.json"
    h3 = run_hillstrom_probe(n_seeds=5)
    print("\nH3 summary:")
    for k, v in h3["summary"].items():
        print(f"  {k}: {v}")
    # Save H3 immediately so a later crash cannot lose the 55-min Hillstrom run
    with open(output_path, "w") as f:
        json.dump({"h3_hillstrom_rct_truth": h3}, f, indent=2)

    try:
        h2 = run_megafon_ct_probe(n_seeds=3)
    except Exception as ex:
        print(f"MegaFon probe failed: {ex}")
        h2 = None
    if h2 is not None:
        print("\nH2 summary:")
        for k, v in h2["summary"].items():
            print(f"  {k}: {v}")
        with open(output_path, "w") as f:
            json.dump({"h3_hillstrom_rct_truth": h3, "h2_megafon_honest_ct": h2}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
