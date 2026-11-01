"""Empirical variance comparison: TESV (single-model) vs Outcome Differencing.

This experiment validates Theorem 2 by measuring the actual estimation variance
of TESV vs the outcome-differencing approach across different numbers of
permutations M.

For each M value, we run 30 independent estimations (with different random seeds)
and compute the empirical variance of feature attributions.
"""

import json
import sys
import os
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner, clone
from src.models.tesv import TESVExplainer


class OutcomeDifferencingExplainer:
    """Interventional SHAP via outcome differencing: phi_j(f1) - phi_j(f0).
    
    Computes separate interventional Shapley values for the treated outcome
    model f1 and control model f0, then differences them.
    """

    def __init__(self, model_t, model_c, n_background=50, n_permutations=20, seed=42):
        self.model_t = model_t
        self.model_c = model_c
        self.n_background = n_background
        self.n_permutations = n_permutations
        self.seed = seed

    def _interventional_expectation(self, model, x, feature_subset, background):
        if len(feature_subset) == 0:
            return float(model.predict(background).mean())
        X_perturbed = background.copy()
        X_perturbed[:, feature_subset] = x[feature_subset]
        return float(model.predict(X_perturbed).mean())

    def explain_instance(self, x, background_data):
        rng = np.random.default_rng(self.seed)
        n = background_data.shape[0]
        if n > self.n_background:
            idx = rng.choice(n, size=self.n_background, replace=False)
            background = background_data[idx]
        else:
            background = background_data

        d = x.shape[0]
        phi_t = np.zeros(d)
        phi_c = np.zeros(d)

        for _ in range(self.n_permutations):
            perm = rng.permutation(d)
            subset = []
            prev_t = self._interventional_expectation(self.model_t, x, subset, background)
            prev_c = self._interventional_expectation(self.model_c, x, subset, background)
            for j in perm:
                subset.append(j)
                curr_t = self._interventional_expectation(self.model_t, x, subset, background)
                curr_c = self._interventional_expectation(self.model_c, x, subset, background)
                phi_t[j] += curr_t - prev_t
                phi_c[j] += curr_c - prev_c
                prev_t = curr_t
                prev_c = curr_c

        phi_t /= self.n_permutations
        phi_c /= self.n_permutations
        return phi_t - phi_c  # differencing


def run_variance_experiment(M_values=(5, 10, 20, 50), n_repeats=20, n_bg=30,
                            n_test_instances=5, data_seed=42):
    """Run the variance comparison for TESV vs differencing."""
    
    print("=" * 60)
    print("TESV vs Outcome Differencing: Empirical Variance Comparison")
    print("=" * 60)
    print(f"Config: M_values={M_values}, n_repeats={n_repeats}, n_bg={n_bg}, n_test={n_test_instances}")
    sys.stdout.flush()

    # Load semi-synthetic IHDP with confounded features (Scenario B)
    train, test = make_semisynthetic_ihdp(
        n_modifiers=5, signal_strength=2.0, noise_scale=0.3,
        seed=data_seed, add_confounded_features=True, confound_corr=0.9
    )

    # Fit T-Learner
    from sklearn.ensemble import GradientBoostingRegressor
    model_t = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=data_seed)
    model_c = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=data_seed)
    model_t.fit(train.X[train.T == 1], train.Y[train.T == 1])
    model_c.fit(train.X[train.T == 0], train.Y[train.T == 0])

    # Wrap as TLearner for TESV
    tlearner = TLearner()
    tlearner.model_t = model_t
    tlearner.model_c = model_c

    # Select test instances
    test_instances = test.X[:n_test_instances]
    background = train.X

    # Compute C, C1, C0 bounds for context
    tau_preds = tlearner.predict_cate(train.X)
    f1_preds = model_t.predict(train.X)
    f0_preds = model_c.predict(train.X)
    C = np.max(np.abs(tau_preds))
    C1 = np.max(np.abs(f1_preds))
    C0 = np.max(np.abs(f0_preds))
    print(f"\nBound parameters: C={C:.4f}, C1={C1:.4f}, C0={C0:.4f}")
    print(f"C/(C1+C0) = {C/(C1+C0):.4f} (variance ratio upper bound)")
    print(f"C^2/(C1^2+C0^2) = {C**2/(C1**2+C0**2):.4f}")
    sys.stdout.flush()

    results = {}
    
    for M in M_values:
        print(f"\n--- M = {M} permutations ---")
        sys.stdout.flush()
        
        # Collect attributions from n_repeats independent runs
        tesv_runs = []
        diff_runs = []
        
        for repeat in range(n_repeats):
            seed_r = data_seed * 1000 + repeat
            
            # TESV (single model)
            explainer_tesv = TESVExplainer(
                cate_model=tlearner,
                background_data=background,
                n_background=n_bg,
                n_permutations=M,
                seed=seed_r,
            )
            
            # Differencing (two models)
            explainer_diff = OutcomeDifferencingExplainer(
                model_t=model_t,
                model_c=model_c,
                n_background=n_bg,
                n_permutations=M,
                seed=seed_r,
            )
            
            tesv_attrs = []
            diff_attrs = []
            for x in test_instances:
                tesv_attrs.append(explainer_tesv.explain_instance(x))
                diff_attrs.append(explainer_diff.explain_instance(x, background))
            
            tesv_runs.append(np.array(tesv_attrs))
            diff_runs.append(np.array(diff_attrs))
        
        # Shape: (n_repeats, n_instances, d)
        tesv_runs = np.array(tesv_runs)
        diff_runs = np.array(diff_runs)
        
        # Compute variance across repeats for each (instance, feature)
        var_tesv = np.var(tesv_runs, axis=0)  # (n_instances, d)
        var_diff = np.var(diff_runs, axis=0)  # (n_instances, d)
        
        mean_var_tesv = var_tesv.mean()
        mean_var_diff = var_diff.mean()
        ratio = mean_var_tesv / mean_var_diff if mean_var_diff > 0 else float('inf')
        
        print(f"  TESV mean variance:  {mean_var_tesv:.6f}")
        print(f"  Diff mean variance:  {mean_var_diff:.6f}")
        print(f"  Variance ratio (TESV/Diff): {ratio:.4f}")
        sys.stdout.flush()
        
        results[str(M)] = {
            "tesv_mean_var": float(mean_var_tesv),
            "diff_mean_var": float(mean_var_diff),
            "variance_ratio": float(ratio),
            "tesv_max_var": float(var_tesv.max()),
            "diff_max_var": float(var_diff.max()),
        }
    
    # Summary
    results["meta"] = {
        "C": float(C),
        "C1": float(C1),
        "C0": float(C0),
        "theoretical_ratio_bound": float(C**2 / (C1**2 + C0**2)),
        "n_repeats": n_repeats,
        "n_bg": n_bg,
        "n_test_instances": n_test_instances,
        "dataset": "IHDP Scenario B (confounded)",
    }
    
    return results


def main():
    t0 = time.time()
    results = run_variance_experiment(
        M_values=(5, 10, 20, 50),
        n_repeats=20,
        n_bg=30,
        n_test_instances=5,
        data_seed=42,
    )
    elapsed = time.time() - t0
    
    print(f"\n{'='*60}")
    print(f"Experiment completed in {elapsed:.1f}s")
    print(f"{'='*60}")
    
    # Save results
    out_dir = Path("results/paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "variance_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
