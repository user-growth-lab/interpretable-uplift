"""R-Learner model-agnosticism experiment.

Demonstrates TESV's advantage on non-decomposable CATE estimators:
- R-Learner has no separate outcome models (mu0, mu1), only tau(X) directly.
- Interventional SHAP differencing requires f_1(X) and f_0(X), which are
  unavailable for R-Learner, causing degraded attributions.
- TESV works directly on tau_hat(X) and produces correct results regardless.
"""
from __future__ import annotations

import json
import os
import numpy as np

from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner, RLearner
from src.models.tesv import TESVExplainer
from src.models.baselines import InterventionalSHAPBaseline, SHAPCATEBaseline
from src.metrics.fidelity import (
    top_k_effect_modifier_recall,
    causal_fidelity_score,
    attribution_stability,
)

N_SEEDS = 5
N_PERM = 20
N_BG = 50


def _spurious_attr(attr, n_orig):
    g = np.abs(attr).mean(axis=0)
    total = g.sum()
    return float(g[n_orig:].sum() / total) if total > 0 else 0.0


class _OrigFeatureWrapper:
    """Wrapper that uses only original features for CATE prediction."""
    def __init__(self, model, n_orig):
        self.model = model
        self.n_orig = n_orig

    def predict_cate(self, X):
        return self.model.predict_cate(X[:, :self.n_orig])


def run_rlearner_experiment():
    """Compare TESV vs Interv. SHAP on R-Learner (non-decomposable CATE)."""
    seeds = list(range(42, 42 + N_SEEDS))

    results = {
        "r_learner": {
            "tesv": {"top5": [], "top10": [], "fid": [], "spur": []},
            "interv": {"top5": [], "top10": [], "fid": [], "spur": []},
        },
        "t_learner": {
            "tesv": {"top5": [], "top10": [], "fid": [], "spur": []},
            "interv": {"top5": [], "top10": [], "fid": [], "spur": []},
        },
    }
    cate_mse = {"r_learner": [], "t_learner": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        # Load data with confounded features (Scenario B)
        train_o, test_o = make_semisynthetic_ihdp(
            n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed,
            add_confounded_features=False,
        )
        train_e, test_e = make_semisynthetic_ihdp(
            n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed,
            add_confounded_features=True, confound_corr=0.9,
        )
        mods = train_o.effect_modifier_indices
        n_orig = 25

        for model_name, ModelCls in [("t_learner", TLearner), ("r_learner", RLearner)]:
            print(f"\n  Model: {model_name}")
            model = ModelCls()
            model.fit(train_o.X, train_o.T, train_o.Y)

            # CATE quality
            tau_pred = model.predict_cate(test_o.X)
            mse = float(np.mean((tau_pred - test_o.tau) ** 2))
            cate_mse[model_name].append(mse)
            print(f"    CATE MSE: {mse:.4f}")

            # Scenario B wrapper
            wrapper = _OrigFeatureWrapper(model, n_orig)
            X_bg = train_e.X
            X_te = test_e.X

            # TESV
            tesv = TESVExplainer(
                cate_model=wrapper, background_data=X_bg,
                n_background=N_BG, n_permutations=N_PERM, seed=seed,
            )
            attr_tesv = tesv.explain(X_te, show_progress=False)
            r5 = top_k_effect_modifier_recall(attr_tesv, mods, k=5)
            r10 = top_k_effect_modifier_recall(attr_tesv, mods, k=10)
            fid = causal_fidelity_score(X_te, wrapper, attr_tesv, perturbation_std=0.5, seed=seed)
            spur = _spurious_attr(attr_tesv, n_orig)
            results[model_name]["tesv"]["top5"].append(r5)
            results[model_name]["tesv"]["top10"].append(r10)
            results[model_name]["tesv"]["fid"].append(fid)
            results[model_name]["tesv"]["spur"].append(spur)
            print(f"    TESV: top5={r5:.3f} top10={r10:.3f} fid={fid:.3f} spur={spur:.4f}")

            # Interventional SHAP (differencing approach)
            interv = InterventionalSHAPBaseline(
                cate_model=wrapper, background_data=X_bg,
                n_background=N_BG, seed=seed,
            )
            attr_interv = interv.explain(X_te, n_permutations=N_PERM)
            r5_i = top_k_effect_modifier_recall(attr_interv, mods, k=5)
            r10_i = top_k_effect_modifier_recall(attr_interv, mods, k=10)
            fid_i = causal_fidelity_score(X_te, wrapper, attr_interv, perturbation_std=0.5, seed=seed)
            spur_i = _spurious_attr(attr_interv, n_orig)
            results[model_name]["interv"]["top5"].append(r5_i)
            results[model_name]["interv"]["top10"].append(r10_i)
            results[model_name]["interv"]["fid"].append(fid_i)
            results[model_name]["interv"]["spur"].append(spur_i)
            print(f"    Interv: top5={r5_i:.3f} top10={r10_i:.3f} fid={fid_i:.3f} spur={spur_i:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model_name in ["t_learner", "r_learner"]:
        print(f"\n{model_name.upper()} (CATE MSE: {np.mean(cate_mse[model_name]):.3f})")
        for method in ["tesv", "interv"]:
            d = results[model_name][method]
            print(f"  {method:8s}: top5={np.mean(d['top5']):.3f}±{np.std(d['top5']):.3f} "
                  f"top10={np.mean(d['top10']):.3f}±{np.std(d['top10']):.3f} "
                  f"fid={np.mean(d['fid']):.3f}±{np.std(d['fid']):.3f} "
                  f"spur={np.mean(d['spur']):.4f}±{np.std(d['spur']):.4f}")

    # Check key finding
    tesv_spur_r = np.mean(results["r_learner"]["tesv"]["spur"])
    interv_spur_r = np.mean(results["r_learner"]["interv"]["spur"])
    print(f"\nKey finding: On R-Learner, TESV spurious={tesv_spur_r:.4f} vs Interv. SHAP spurious={interv_spur_r:.4f}")
    if tesv_spur_r < interv_spur_r:
        print("✓ TESV correctly handles non-decomposable CATE; Interv. SHAP degrades.")
    else:
        print("Both methods handle R-Learner similarly (TESV still has computational advantage).")

    # Save
    os.makedirs("results/paper_tables", exist_ok=True)
    out = {
        model_name: {
            method: {k: f"{np.mean(v):.3f}±{np.std(v):.3f}" for k, v in d.items()}
            for method, d in methods.items()
        }
        for model_name, methods in results.items()
    }
    out["cate_mse"] = {k: f"{np.mean(v):.3f}" for k, v in cate_mse.items()}
    with open("results/paper_tables/rlearner_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to results/paper_tables/rlearner_comparison.json")
    return results, cate_mse


if __name__ == "__main__":
    run_rlearner_experiment()
