"""Counterfactual baseline comparison experiments.

Compares our greedy coordinate ascent against:
- DiCE (existing baseline)
- GrowingSpheres (new: minimal-distance guarantee)
- RandomSearch (new: structured weighted sampling)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.datasets import make_semisynthetic_ihdp, load_hillstrom
from src.models.cate_estimators import TLearner
from src.models.counterfactual import CausalCounterfactualExplainer
from src.models.baselines import DiCEUpliftBaseline, GrowingSpheresUplift, RandomSearchUplift

OUTPUT_DIR = Path("results/paper_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_cf_method(method, X_test, method_name: str) -> dict:
    """Evaluate a counterfactual method on test instances."""
    results = method.explain(X_test)

    consistencies = []
    feasibilities = []
    gains = []
    proximities = []

    for r in results:
        if isinstance(r, dict):
            consistencies.append(1.0 if r.get("consistent", r.get("tau_gain", 0) > 0) else 0.0)
            feasibilities.append(1.0 if r.get("feasible", True) else 0.0)
            gains.append(r.get("tau_gain", 0.0))
            proximities.append(r.get("proximity", 0.0))
        else:
            # CounterfactualExplanation object
            consistencies.append(1.0 if r.tau_gain > 0 else 0.0)
            feasibilities.append(1.0 if r.feasible else 0.0)
            gains.append(r.tau_gain)
            proximities.append(r.proximity)

    return {
        "method": method_name,
        "consistency": float(np.mean(consistencies)),
        "feasibility": float(np.mean(feasibilities)),
        "mean_gain": float(np.mean(gains)),
        "gain_std": float(np.std(gains)),
        "proximity": float(np.mean(proximities)),
        "proximity_std": float(np.std(proximities)),
    }


def run_cf_comparison(n_seeds: int = 5, n_test: int = 100):
    """Run counterfactual comparison across methods and datasets."""
    print("=" * 60)
    print("Counterfactual Baseline Comparison")
    print("=" * 60)

    all_results = {}

    # Dataset 1: IHDP
    print("\n--- IHDP ---")
    for seed in range(n_seeds):
        print(f"  Seed {seed}/{n_seeds}...")
        train_ds, test_ds = make_semisynthetic_ihdp(
            effect_modifier_indices=[1, 5, 10, 15, 24],
            seed=seed + 42,
        )

        cate_model = TLearner()
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

        # Feature bounds from training data
        bounds = np.column_stack([train_ds.X.min(axis=0), train_ds.X.max(axis=0)])
        test_X = test_ds.X[:n_test]

        # Our method
        ours = CausalCounterfactualExplainer(
            cate_model=cate_model,
            feature_bounds=bounds,
            max_steps=20,
            step_size=0.1,
            seed=seed,
        )

        # DiCE baseline
        dice = DiCEUpliftBaseline(
            cate_model=cate_model,
            feature_bounds=bounds,
            n_counterfactuals=100,
            seed=seed,
        )

        # GrowingSpheres
        gs = GrowingSpheresUplift(
            cate_model=cate_model,
            feature_bounds=bounds,
            initial_radius=0.1,
            n_samples_per_sphere=200,
            max_iterations=20,
            seed=seed,
        )

        # RandomSearch
        rs = RandomSearchUplift(
            cate_model=cate_model,
            feature_bounds=bounds,
            n_candidates=500,
            max_distance=2.0,
            seed=seed,
        )

        # Evaluate
        ours_results = []
        for x in test_X:
            cf = ours.explain_instance(x)
            ours_results.append({
                "consistent": cf.tau_gain > 0,
                "feasible": cf.feasible,
                "tau_gain": cf.tau_gain,
                "proximity": cf.proximity,
            })

        key = f"ihdp_seed{seed}"
        all_results[key] = {
            "ours": evaluate_cf_method(ours, test_X, "Ours (Greedy)"),
            "dice": evaluate_cf_method(dice, test_X, "DiCE"),
            "growing_spheres": evaluate_cf_method(gs, test_X, "GrowingSpheres"),
            "random_search": evaluate_cf_method(rs, test_X, "RandomSearch"),
        }

    # Aggregate across seeds
    methods = ["ours", "dice", "growing_spheres", "random_search"]
    ihdp_agg = {}
    for method in methods:
        metrics = {}
        for metric in ["consistency", "feasibility", "mean_gain", "proximity"]:
            values = [all_results[f"ihdp_seed{s}"][method][metric] for s in range(n_seeds)]
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        ihdp_agg[method] = metrics

    all_results["ihdp_aggregated"] = ihdp_agg

    # Print summary
    print("\n  IHDP Results (mean ± std):")
    print(f"  {'Method':<20} {'Consistency':>12} {'Feasibility':>12} {'Mean Gain':>12} {'Proximity':>12}")
    for method in methods:
        m = ihdp_agg[method]
        print(f"  {method:<20} "
              f"{m['consistency']['mean']:.3f}±{m['consistency']['std']:.3f}  "
              f"{m['feasibility']['mean']:.3f}±{m['feasibility']['std']:.3f}  "
              f"{m['mean_gain']['mean']:.3f}±{m['mean_gain']['std']:.3f}  "
              f"{m['proximity']['mean']:.3f}±{m['proximity']['std']:.3f}")

    return all_results


if __name__ == "__main__":
    results = run_cf_comparison(n_seeds=5, n_test=100)

    output_path = OUTPUT_DIR / "cf_baselines_extended.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
