"""Honest out-of-sample evaluation of SCRL rules.

Performs train/test split, discovers rules on train, evaluates on test
using ground-truth tau to provide honest CATE estimates within rules.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL


def run_honest_evaluation(seeds=range(42, 47), n_modifiers=5):
    """Run honest out-of-sample rule evaluation."""
    print("=" * 60)
    print("SCRL Honest Out-of-Sample Evaluation")
    print("=" * 60)
    print(f"Seeds: {list(seeds)}")
    sys.stdout.flush()

    results = {
        "scrl_train_cate": [],
        "scrl_test_ground_truth_tau": [],
        "scrl_test_predicted_cate": [],
        "scrl_coverage": [],
        "naive_top20_test_tau": [],
        "population_mean_tau": [],
    }

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        sys.stdout.flush()

        # Generate semi-synthetic data with known effect modifiers
        train, test = make_semisynthetic_ihdp(
            n_modifiers=n_modifiers, signal_strength=2.0,
            noise_scale=0.3, seed=seed, add_confounded_features=False
        )

        # Fit CATE model on training data
        model = TLearner()
        model.fit(train.X, train.T, train.Y)

        # Predict CATE on both splits
        tau_hat_train = model.predict_cate(train.X)
        tau_hat_test = model.predict_cate(test.X)

        # Discover rules on TRAINING data only (reduced candidates for speed)
        scrl = SCRL(
            cate_model=model,
            n_rules=5,
            n_candidates=8,
            max_rule_length=2,
            n_bootstrap=10,
            min_support=0.1,
            seed=seed,
        )
        scrl.fit(train.X, train.T, train.Y)
        rules = scrl.rules

        # Evaluate rules on TEST data (honest evaluation)
        if rules:
            # Get rule masks on test data
            test_covered = np.zeros(test.n, dtype=bool)
            for rule in rules:
                mask = rule.covers(test.X)
                test_covered |= mask

            coverage = test_covered.mean()

            if coverage > 0:
                # Ground-truth tau within rules (HONEST - using test.tau)
                mean_tau_in_rules = test.tau[test_covered].mean()
                # Predicted CATE within rules on test
                mean_cate_in_rules = tau_hat_test[test_covered].mean()
                # Predicted CATE within rules on train (what SCRL optimized)
                train_covered = np.zeros(train.n, dtype=bool)
                for rule in rules:
                    train_covered |= rule.covers(train.X)
                mean_cate_train = tau_hat_train[train_covered].mean() if train_covered.any() else 0.0
            else:
                mean_tau_in_rules = 0.0
                mean_cate_in_rules = 0.0
                mean_cate_train = 0.0
        else:
            coverage = 0.0
            mean_tau_in_rules = 0.0
            mean_cate_in_rules = 0.0
            mean_cate_train = 0.0

        # Naive baseline: top-20% by CATE prediction on test
        top20_idx = np.argsort(-tau_hat_test)[:int(0.2 * test.n)]
        naive_top20_tau = test.tau[top20_idx].mean()

        # Population mean tau
        pop_mean_tau = test.tau.mean()

        results["scrl_train_cate"].append(float(mean_cate_train))
        results["scrl_test_ground_truth_tau"].append(float(mean_tau_in_rules))
        results["scrl_test_predicted_cate"].append(float(mean_cate_in_rules))
        results["scrl_coverage"].append(float(coverage))
        results["naive_top20_test_tau"].append(float(naive_top20_tau))
        results["population_mean_tau"].append(float(pop_mean_tau))

        print(f"  Coverage: {coverage:.3f}")
        print(f"  Train CATE (in-sample): {mean_cate_train:.3f}")
        print(f"  Test tau (ground-truth, honest): {mean_tau_in_rules:.3f}")
        print(f"  Test CATE (predicted): {mean_cate_in_rules:.3f}")
        print(f"  Naive top-20% tau: {naive_top20_tau:.3f}")
        print(f"  Population mean tau: {pop_mean_tau:.3f}")
        sys.stdout.flush()

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std over seeds)")
    print("=" * 60)

    for key, values in results.items():
        arr = np.array(values)
        print(f"  {key}: {arr.mean():.3f} ± {arr.std():.3f}")

    # Compute 95% CI for honest evaluation
    honest_tau = np.array(results["scrl_test_ground_truth_tau"])
    ci_low, ci_high = stats.t.interval(
        0.95, len(honest_tau) - 1,
        loc=honest_tau.mean(), scale=stats.sem(honest_tau)
    )
    print(f"\n  SCRL honest tau 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

    # Is SCRL honest tau > population mean?
    pop_tau = np.array(results["population_mean_tau"])
    diff = honest_tau - pop_tau
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    print(f"  SCRL tau vs population mean: t={t_stat:.3f}, p={p_val:.4f}")

    results["summary"] = {
        "honest_tau_mean": float(honest_tau.mean()),
        "honest_tau_std": float(honest_tau.std()),
        "honest_tau_ci95": [float(ci_low), float(ci_high)],
        "vs_population_p": float(p_val),
        "n_seeds": len(seeds),
    }

    return results


def main():
    t0 = time.time()
    results = run_honest_evaluation(seeds=range(42, 47))
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s")

    out_path = Path("results/paper_tables/honest_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
