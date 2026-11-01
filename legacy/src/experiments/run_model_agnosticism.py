"""Extended model-agnosticism experiment: T-Learner, R-Learner, DR-Learner.

Demonstrates TESV's unique applicability to non-decomposable CATE estimators.
For R-Learner and DR-Learner, no separate outcome models (f_1, f_0) exist,
making the standard Interventional SHAP differencing approach inapplicable.
TESV operates identically on any estimator via the tau_hat(X) interface.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner, RLearner, DRLearner, clone
from src.models.tesv import TESVExplainer
from src.metrics.fidelity import (
    top_k_effect_modifier_recall,
    causal_fidelity_score,
)

N_SEEDS = 5
N_PERM = 20
N_BG = 50


def _spurious_attr(attr, n_orig):
    """Compute fraction of attribution mass on spurious features."""
    g = np.abs(attr).mean(axis=0)
    total = g.sum()
    return float(g[n_orig:].sum() / total) if total > 0 else 0.0


class _OrigFeatureWrapper:
    """Wrapper that uses only original features for CATE prediction,
    enabling TESV to operate on augmented feature space (with confounders)."""

    def __init__(self, model, n_orig):
        self.model = model
        self.n_orig = n_orig

    def predict_cate(self, X):
        return self.model.predict_cate(X[:, :self.n_orig])


class NeuralTLearner:
    """Neural T-Learner: 2-layer MLP for each outcome model."""

    def __init__(self, seed=42):
        self.model_t = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500,
            early_stopping=True, random_state=seed
        )
        self.model_c = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500,
            early_stopping=True, random_state=seed + 1
        )

    def fit(self, X, T, Y):
        self.model_t.fit(X[T == 1], Y[T == 1])
        self.model_c.fit(X[T == 0], Y[T == 0])
        return self

    def predict_cate(self, X):
        return self.model_t.predict(X) - self.model_c.predict(X)

    def predict_outcomes(self, X):
        return self.model_c.predict(X), self.model_t.predict(X)


def run_full_model_agnosticism():
    """Run TESV on 4 CATE model types: GBT T-Learner, Neural T-Learner, R-Learner, DR-Learner."""
    seeds = list(range(42, 42 + N_SEEDS))

    model_configs = [
        ("GBT T-Learner", lambda: TLearner(), True),
        ("Neural T-Learner", lambda: NeuralTLearner(seed=42), True),
        ("R-Learner", lambda: RLearner(), False),
        ("DR-Learner", lambda: DRLearner(), False),
    ]

    results = {name: {"top5": [], "fid": [], "spur": [], "mse": []}
               for name, _, _ in model_configs}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # Scenario A (standard) + Scenario B (confounded)
        train_o, test_o = make_semisynthetic_ihdp(
            n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed,
            add_confounded_features=False,
        )
        train_e, test_e = make_semisynthetic_ihdp(
            n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed,
            add_confounded_features=True, confound_corr=0.9,
        )
        mods = train_o.effect_modifier_indices

        for model_name, model_factory, decomposable in model_configs:
            print(f"\n  {model_name} (decomposable={decomposable})")
            model = model_factory()
            model.fit(train_o.X, train_o.T, train_o.Y)

            # CATE quality
            tau_pred = model.predict_cate(test_o.X)
            mse = float(np.mean((tau_pred - test_o.tau) ** 2))
            results[model_name]["mse"].append(mse)
            print(f"    CATE MSE: {mse:.3f}")

            # TESV attribution on Scenario B (with confounded features)
            # Wrap model to handle augmented feature space
            n_orig = train_o.X.shape[1]
            wrapper = _OrigFeatureWrapper(model, n_orig)

            tesv = TESVExplainer(
                cate_model=wrapper, background_data=train_e.X,
                n_background=N_BG, n_permutations=N_PERM, seed=seed,
            )
            attr = tesv.explain(test_e.X, show_progress=False)

            r5 = top_k_effect_modifier_recall(attr, mods, k=5)
            fid = causal_fidelity_score(test_e.X, wrapper, attr, perturbation_std=0.5, seed=seed)
            spur = _spurious_attr(attr, n_orig)

            results[model_name]["top5"].append(r5)
            results[model_name]["fid"].append(fid)
            results[model_name]["spur"].append(spur)
            print(f"    TESV: top5={r5:.3f} fid={fid:.3f} spur={spur:.4f}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Model-Agnosticism Results (mean ± std over 5 seeds)")
    print("=" * 70)
    print(f"{'Model':<20} {'Decomp.':<10} {'MSE↓':<12} {'Top-5↑':<12} {'Fidelity↑':<12} {'Spurious(B)↓':<12}")
    print("-" * 70)
    for model_name, _, decomposable in model_configs:
        d = results[model_name]
        dec_str = "Yes" if decomposable else "No†"
        print(f"{model_name:<20} {dec_str:<10} "
              f"{np.mean(d['mse']):.3f}±{np.std(d['mse']):.3f}  "
              f"{np.mean(d['top5']):.2f}±{np.std(d['top5']):.2f}    "
              f"{np.mean(d['fid']):.3f}±{np.std(d['fid']):.3f}  "
              f"{np.mean(d['spur']):.3f}±{np.std(d['spur']):.3f}")
    print("\n†Non-decomposable: Interv. SHAP differencing not directly applicable.")

    # Save results
    os.makedirs("results/paper_tables", exist_ok=True)
    out = {}
    for model_name, _, decomposable in model_configs:
        d = results[model_name]
        out[model_name] = {
            "decomposable": decomposable,
            "mse": f"{np.mean(d['mse']):.3f}±{np.std(d['mse']):.3f}",
            "top5": f"{np.mean(d['top5']):.3f}±{np.std(d['top5']):.3f}",
            "fidelity": f"{np.mean(d['fid']):.3f}±{np.std(d['fid']):.3f}",
            "spurious_B": f"{np.mean(d['spur']):.4f}±{np.std(d['spur']):.4f}",
        }

    with open("results/paper_tables/model_agnosticism_extended.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to results/paper_tables/model_agnosticism_extended.json")
    return results


if __name__ == "__main__":
    run_full_model_agnosticism()
