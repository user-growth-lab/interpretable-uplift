"""Compare TESV vs baseline explanation methods on semi-synthetic IHDP.

Runs two scenarios:
  A) Standard: original features only (25 features)
  B) Confounded: adds correlated non-causal features (30 features)

Scenario B demonstrates TESV's causal robustness: it gives zero
attribution to confounded features, while conditional SHAP leaks
attribution to them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

from src.data.datasets import make_semisynthetic_ihdp
from src.metrics.fidelity import (
    attribution_stability,
    causal_fidelity_score,
    top_k_effect_modifier_recall,
)
from src.models.baselines import SHAPCATEBaseline
from src.models.cate_estimators import TLearner, CATEEstimator
from src.models.tesv import TESVExplainer


class _OriginalFeatureWrapper(CATEEstimator):
    """Wrapper that only uses the first n_orig features."""
    def __init__(self, model: CATEEstimator, n_orig: int):
        self.model = model
        self.n_orig = n_orig
    def fit(self, X, T, Y): return self
    def predict_cate(self, X):
        return self.model.predict_cate(X[:, :self.n_orig])


def run_scenario(scenario_name, train, test, model, true_modifiers, n_perm=20):
    """Run TESV vs SHAP comparison on a given scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    print(f"Features: {test.X.shape[1]}, True modifiers: {true_modifiers}")

    results = {}

    # TESV
    print("Computing TESV...")
    tesv = TESVExplainer(
        cate_model=model, background_data=train.X,
        n_background=50, n_permutations=n_perm, seed=42,
    )
    attr_tesv = tesv.explain(test.X, show_progress=True)

    # Conditional SHAP
    print("Computing conditional SHAP...")
    shap_bl = SHAPCATEBaseline(
        cate_model=model, background_data=train.X,
        n_background=50, k_neighbors=10, seed=42,
    )
    attr_shap = shap_bl.explain(test.X, n_permutations=n_perm)

    # Recall
    results["tesv_top5_recall"] = top_k_effect_modifier_recall(attr_tesv, true_modifiers, k=5)
    results["shap_top5_recall"] = top_k_effect_modifier_recall(attr_shap, true_modifiers, k=5)
    results["tesv_top10_recall"] = top_k_effect_modifier_recall(attr_tesv, true_modifiers, k=10)
    results["shap_top10_recall"] = top_k_effect_modifier_recall(attr_shap, true_modifiers, k=10)

    # Causal fidelity
    results["tesv_fidelity"] = causal_fidelity_score(
        test.X, model, attr_tesv, perturbation_std=0.5, seed=42
    )
    results["shap_fidelity"] = causal_fidelity_score(
        test.X, model, attr_shap, perturbation_std=0.5, seed=42
    )

    # Stability
    results["tesv_stability"] = attribution_stability(
        test.X, tesv, noise_std=0.1, n_perturbations=5, seed=42
    )

    # Spurious attribution (if extended features exist)
    n_orig = 25
    if test.X.shape[1] > n_orig:
        glob_tesv = np.abs(attr_tesv).mean(axis=0)
        glob_shap = np.abs(attr_shap).mean(axis=0)
        results["tesv_spurious_attr"] = float(glob_tesv[n_orig:].sum())
        results["shap_spurious_attr"] = float(glob_shap[n_orig:].sum())

    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results


def main():
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # Train model on original features
    train_orig, test_orig = make_semisynthetic_ihdp(
        n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=42,
        add_confounded_features=False,
    )
    true_modifiers = train_orig.effect_modifier_indices

    model = TLearner()
    model.fit(train_orig.X, train_orig.T, train_orig.Y)
    tau_pred = model.predict_cate(test_orig.X)
    mse = float(np.mean((tau_pred - test_orig.tau) ** 2))
    print(f"CATE test MSE: {mse:.4f}")

    # Scenario A: Standard (25 features)
    results_a = run_scenario(
        "A: Standard (no confounders)",
        train_orig, test_orig, model, true_modifiers,
    )

    # Scenario B: Confounded (30 features, corr=0.9)
    train_ext, test_ext = make_semisynthetic_ihdp(
        n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=42,
        add_confounded_features=True, confound_corr=0.9,
    )
    wrapper = _OriginalFeatureWrapper(model, n_orig=25)
    results_b = run_scenario(
        "B: Confounded (corr=0.9)",
        train_ext, test_ext, wrapper, true_modifiers,
    )

    # Save
    all_results = {
        "cate_mse": mse,
        "scenario_a": results_a,
        "scenario_b": results_b,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"comparison_results_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
