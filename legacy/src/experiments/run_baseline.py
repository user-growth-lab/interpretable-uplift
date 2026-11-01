"""Baseline experiment for Interpretable Uplift Modeling (Direction 35).

Runs the full pipeline on semi-synthetic IHDP:
  1. CATE estimation
  2. TESV attribution + evaluation
  3. SCRL rule discovery + evaluation
  4. Counterfactual explanation + evaluation
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
    counterfactual_consistency,
    counterfactual_proximity,
    rule_coverage_fidelity,
    top_k_effect_modifier_recall,
)
from src.models.cate_estimators import TLearner, get_estimator
from src.models.counterfactual import CausalCounterfactualExplainer
from src.models.scrl import SCRL
from src.models.tesv import TESVExplainer


def cate_metrics(dataset, model):
    tau_pred = model.predict_cate(dataset.X)
    mse = float(np.mean((tau_pred - dataset.tau) ** 2))
    mae = float(np.mean(np.abs(tau_pred - dataset.tau)))
    return {"cate_mse": mse, "cate_mae": mae}


def main():
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Direction 35: Interpretable Uplift Modeling - Baseline Experiment")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading semi-synthetic IHDP...")
    train, test = make_semisynthetic_ihdp(
        n_modifiers=5,
        signal_strength=2.0,
        noise_scale=0.3,
        seed=42,
    )
    true_modifiers = train.effect_modifier_indices
    print(f"  True effect modifiers: {true_modifiers}")

    # 2. Fit CATE model
    print("\n[2/5] Fitting CATE model...")
    model = TLearner()
    model.fit(train.X, train.T, train.Y)
    metrics = cate_metrics(test, model)
    print(f"  CATE test MSE: {metrics['cate_mse']:.4f}")
    print(f"  CATE test MAE: {metrics['cate_mae']:.4f}")

    # 3. TESV attribution
    print("\n[3/5] Computing TESV attributions...")
    explainer = TESVExplainer(
        cate_model=model,
        background_data=train.X,
        n_background=50,
        n_permutations=20,
        seed=42,
    )
    attributions = explainer.explain(test.X, show_progress=True)
    metrics["tesv_top5_recall"] = top_k_effect_modifier_recall(
        attributions, true_modifiers, k=5
    )
    metrics["tesv_top10_recall"] = top_k_effect_modifier_recall(
        attributions, true_modifiers, k=10
    )
    metrics["tesv_causal_fidelity"] = causal_fidelity_score(
        test.X, model, attributions, perturbation_std=0.5, seed=42
    )
    metrics["tesv_stability"] = attribution_stability(
        test.X, explainer, noise_std=0.1, n_perturbations=5, seed=42
    )
    print(f"  TESV Top-5 recall: {metrics['tesv_top5_recall']:.3f}")
    print(f"  TESV Top-10 recall: {metrics['tesv_top10_recall']:.3f}")
    print(f"  TESV causal fidelity: {metrics['tesv_causal_fidelity']:.3f}")
    print(f"  TESV stability: {metrics['tesv_stability']:.3f}")

    global_attr = np.abs(attributions).mean(axis=0)
    print("  Top 10 global TESV features:")
    for idx in np.argsort(-global_attr)[:10]:
        mark = "*" if idx in true_modifiers else " "
        print(f"    [{mark}] feature {idx}: {global_attr[idx]:.4f}")

    # 4. SCRL rule discovery
    print("\n[4/5] Discovering stable causal rules...")
    scrl = SCRL(
        cate_model=model,
        n_rules=5,
        max_rule_length=2,
        min_support=0.1,
        overlap_ratio=0.1,
        n_candidates=10,
        n_bootstrap=5,
        seed=42,
    )
    scrl.fit(train.X, train.T, train.Y)
    tau_pred_test = model.predict_cate(test.X)
    metrics["scrl_rule_fidelity"] = rule_coverage_fidelity(
        test.X, tau_pred_test, [rule.covers(test.X) for rule in scrl.rules]
    )
    print(f"  SCRL rule fidelity: {metrics['scrl_rule_fidelity']:.4f}")
    print("  Discovered rules:")
    for i, rule in enumerate(scrl.rules):
        print(
            f"    Rule {i+1}: {rule} | tau={rule.tau_mean:.3f} "
            f"| coverage={rule.coverage:.3f} | stability={rule.stability_score:.3f}"
        )

    # 5. Counterfactual explanations
    print("\n[5/5] Generating counterfactual explanations...")
    cf_explainer = CausalCounterfactualExplainer(
        cate_model=model,
        actionable_features=list(range(train.X.shape[1])),
        max_changes=2,
        step_size=0.2,
        max_steps=10,
        proximity_weight=0.05,
        seed=42,
    )
    cf_explanations = cf_explainer.explain(test.X[:50])
    X_cf = np.array([e.x_cf for e in cf_explanations])
    X_orig = np.array([e.x_orig for e in cf_explanations])
    metrics["cf_feasibility_rate"] = float(np.mean([e.feasible for e in cf_explanations]))
    metrics["cf_mean_proximity"] = counterfactual_proximity(X_orig, X_cf)
    metrics["cf_consistency"] = counterfactual_consistency(X_orig, X_cf, model)
    metrics["cf_mean_gain"] = float(np.mean([e.tau_gain for e in cf_explanations]))
    print(f"  CF feasibility rate: {metrics['cf_feasibility_rate']:.3f}")
    print(f"  CF mean proximity: {metrics['cf_mean_proximity']:.4f}")
    print(f"  CF consistency: {metrics['cf_consistency']:.3f}")
    print(f"  CF mean tau gain: {metrics['cf_mean_gain']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(out_dir, f"baseline_results_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {result_path}")

    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
