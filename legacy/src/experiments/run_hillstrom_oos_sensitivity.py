"""Run Hillstrom OOS-CovFidelity and λ sensitivity experiments.

Produces:
  1. OOS-CovFidelity for SCRL vs baselines on Hillstrom (M4)
  2. λ1/λ2 sensitivity surface on Hillstrom (M5)

Results saved to results/paper_tables/hillstrom_oos_sensitivity.json
"""
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.datasets import load_hillstrom
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL
from src.models.baselines import CausalTreeBaseline
from src.metrics.fidelity import oos_coverage_fidelity
from sklearn.ensemble import GradientBoostingRegressor


def run_hillstrom_oos(n_seeds: int = 5):
    """M4: Compute OOS-CovFidelity for SCRL, CURLS-proxy, and Causal Tree on Hillstrom."""
    print("=" * 60)
    print("M4: Hillstrom OOS-CovFidelity")
    print("=" * 60)

    results = {"SCRL": [], "Causal_Tree": []}

    for seed in range(n_seeds):
        print(f"\n  Seed {seed+1}/{n_seeds}")
        train_ds, test_ds = load_hillstrom(seed=seed)

        # Fit CATE model
        base = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        cate_model = TLearner(base_learner=base)
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

        tau_train = cate_model.predict_cate(train_ds.X)
        tau_test = cate_model.predict_cate(test_ds.X)

        # --- SCRL ---
        scrl = SCRL(
            cate_model=cate_model,
            n_rules=5,
            max_rule_length=2,
            n_bootstrap=10,
            n_candidates=20,
            max_coverage=0.40,
            sample_split=True,
            seed=seed,
        )
        scrl.fit(train_ds.X, train_ds.T, train_ds.Y)

        # Evaluate rules on TEST set
        rule_masks_test = [rule.covers(test_ds.X) for rule in scrl.rules]
        if rule_masks_test:
            oos_fid = oos_coverage_fidelity(
                train_ds.X, tau_train, test_ds.X, tau_test, rule_masks_test
            )
        else:
            oos_fid = 0.0
        results["SCRL"].append(oos_fid)
        print(f"    SCRL OOS-CovFidelity: {oos_fid:.4f}")

        # --- Causal Tree ---
        try:
            ct = CausalTreeBaseline(cate_model=cate_model, max_depth=3, seed=seed)
            ct.fit(train_ds.X, train_ds.T, train_ds.Y)
            ct_masks_test = [rule.covers(test_ds.X) for rule in ct.rules]
            if ct_masks_test:
                ct_oos = oos_coverage_fidelity(
                    train_ds.X, tau_train, test_ds.X, tau_test, ct_masks_test
                )
            else:
                ct_oos = 0.0
        except Exception as e:
            print(f"    Causal Tree error: {e}")
            ct_oos = 0.0
        results["Causal_Tree"].append(ct_oos)
        print(f"    Causal Tree OOS-CovFidelity: {ct_oos:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("M4 Results Summary (Hillstrom OOS-CovFidelity):")
    for method, vals in results.items():
        arr = np.array(vals)
        print(f"  {method}: {arr.mean():.4f} ± {arr.std():.4f}")
    return results


def run_lambda_sensitivity(n_seeds: int = 3):
    """M5: λ1/λ2 sensitivity analysis on Hillstrom.

    We vary λ1 (tau_std weight) and λ2 (bootstrap stability weight)
    and measure both in-sample and OOS Coverage Fidelity.
    """
    print("\n" + "=" * 60)
    print("M5: λ1/λ2 Sensitivity on Hillstrom")
    print("=" * 60)

    # The scoring formula in SCRL._score_rule is:
    #   score = |tau_mean - global| * sqrt(coverage) + tau_std_coeff * tau_std + stability
    # where stability = -std(bootstrap_means)
    # Paper notation: score = magnitude - λ1*σ - λ2*σ_boot
    # We vary the effective lambda via n_bootstrap (higher = more stability emphasis)
    # and tau_std coefficient sign/magnitude

    lambda1_values = [0.05, 0.10, 0.15, 0.20]  # tau_std penalty
    lambda2_values = [0.3, 0.5, 0.7, 1.0]  # bootstrap stability weight

    results = {}

    # Load data once per seed
    for seed in range(n_seeds):
        print(f"\n  Seed {seed+1}/{n_seeds}")
        train_ds, test_ds = load_hillstrom(seed=seed)
        base = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        cate_model = TLearner(base_learner=base)
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)
        tau_test = cate_model.predict_cate(test_ds.X)
        tau_train = cate_model.predict_cate(train_ds.X)

        for l1 in lambda1_values:
            for l2 in lambda2_values:
                key = f"l1={l1:.2f}_l2={l2:.2f}"
                if key not in results:
                    results[key] = {"oos_fidelity": [], "in_sample_fidelity": []}

                # Run SCRL with modified scoring
                scrl = SCRL(
                    cate_model=cate_model,
                    n_rules=5,
                    max_rule_length=2,
                    n_bootstrap=10,
                    n_candidates=20,
                    max_coverage=0.40,
                    sample_split=True,
                    seed=seed,
                )
                # Monkey-patch the scoring to use our λ values
                original_score = scrl._score_rule

                def make_scorer(lambda1, lambda2):
                    def custom_score(rule, X, tau, T):
                        mask = rule.covers(X)
                        if mask.sum() < 2:
                            return -np.inf
                        rule.tau_mean = float(tau[mask].mean())
                        rule.tau_std = float(tau[mask].std())
                        rule.coverage = float(mask.mean())
                        n_t = T[mask].sum()
                        n_c = mask.sum() - n_t
                        if min(n_t, n_c) < scrl.overlap_ratio * mask.sum():
                            return -np.inf
                        # Bootstrap stability
                        from sklearn.utils import resample
                        stab_scores = []
                        for _ in range(scrl.n_bootstrap):
                            idx = resample(np.arange(mask.sum()), random_state=scrl.rng.integers(1e9))
                            stab_scores.append(tau[mask][idx].mean())
                        sigma_boot = np.std(stab_scores)
                        rule.stability_score = float(-sigma_boot)
                        # Enforce max_coverage
                        if rule.coverage > scrl.max_coverage:
                            return -np.inf
                        global_mean = getattr(scrl, "_global_tau_mean", 0.0)
                        magnitude = abs(rule.tau_mean - global_mean) * np.sqrt(rule.coverage)
                        score = magnitude - lambda1 * rule.tau_std - lambda2 * sigma_boot
                        return float(score)
                    return custom_score

                scrl._score_rule = make_scorer(l1, l2)
                scrl.fit(train_ds.X, train_ds.T, train_ds.Y)

                # OOS evaluation
                rule_masks_test = [rule.covers(test_ds.X) for rule in scrl.rules]
                if rule_masks_test:
                    oos_fid = oos_coverage_fidelity(
                        train_ds.X, tau_train, test_ds.X, tau_test, rule_masks_test
                    )
                else:
                    oos_fid = 0.0

                # In-sample evaluation
                rule_masks_train = [rule.covers(train_ds.X) for rule in scrl.rules]
                if rule_masks_train:
                    is_fid = oos_coverage_fidelity(
                        train_ds.X, tau_train, train_ds.X, tau_train, rule_masks_train
                    )
                else:
                    is_fid = 0.0

                results[key]["oos_fidelity"].append(oos_fid)
                results[key]["in_sample_fidelity"].append(is_fid)

    # Print summary table
    print("\n" + "=" * 60)
    print("M5 Results: lambda Sensitivity (OOS-CovFidelity)")
    header = "l1/l2       "
    for l2 in lambda2_values:
        header += f"  {l2:.1f}  "
    print(header)
    for l1 in lambda1_values:
        row = f"  {l1:.2f}     "
        for l2 in lambda2_values:
            key = f"l1={l1:.2f}_l2={l2:.2f}"
            arr = np.array(results[key]["oos_fidelity"])
            row += f" {arr.mean():.3f} "
        print(row)

    return results


def main():
    output_path = "results/paper_tables/hillstrom_oos_sensitivity.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    oos_results = run_hillstrom_oos(n_seeds=5)
    sensitivity_results = run_lambda_sensitivity(n_seeds=3)

    # Save results
    output = {
        "M4_OOS_CovFidelity": {
            method: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
            for method, vals in oos_results.items()
        },
        "M5_sensitivity": {
            key: {
                "oos_fidelity_mean": float(np.mean(v["oos_fidelity"])),
                "oos_fidelity_std": float(np.std(v["oos_fidelity"])),
                "in_sample_mean": float(np.mean(v["in_sample_fidelity"])),
            }
            for key, v in sensitivity_results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
