"""End-to-end sanity check: can TESV recover known effect modifiers?"""

from __future__ import annotations

import numpy as np

from src.data.datasets import make_semisynthetic_ihdp
from src.metrics.fidelity import (
    attribution_rank_correlation,
    causal_fidelity_score,
    top_k_effect_modifier_recall,
)
from src.models.cate_estimators import TLearner
from src.models.tesv import TESVExplainer


def main():
    print("Loading semi-synthetic IHDP with 5 known effect modifiers...")
    train, test = make_semisynthetic_ihdp(
        n_modifiers=5,
        signal_strength=2.0,
        noise_scale=0.3,
        seed=42,
    )
    true_modifiers = train.effect_modifier_indices
    print(f"True effect modifiers: {true_modifiers}")

    print("\nFitting T-Learner CATE model...")
    model = TLearner()
    model.fit(train.X, train.T, train.Y)

    tau_pred = model.predict_cate(test.X)
    mse = np.mean((tau_pred - test.tau) ** 2)
    print(f"CATE test MSE: {mse:.4f}")

    print("\nComputing TESV attributions (this may take a minute)...")
    explainer = TESVExplainer(
        cate_model=model,
        background_data=train.X,
        n_background=50,
        n_permutations=20,
        seed=42,
    )
    attributions = explainer.explain(test.X[:50], show_progress=True)

    # Ground-truth importance: features with nonzero contribution to tau
    true_importance = np.zeros(train.X.shape[1])
    true_importance[list(true_modifiers)] = 1.0

    global_attr = np.abs(attributions).mean(axis=0)
    print("\nGlobal TESV attributions (top 10):")
    top_idx = np.argsort(-global_attr)[:10]
    for i in top_idx:
        mark = " *" if i in true_modifiers else ""
        print(f"  feature {i}: {global_attr[i]:.4f}{mark}")

    recall = top_k_effect_modifier_recall(
        attributions, true_modifiers, k=len(true_modifiers)
    )
    print(f"\nTop-{len(true_modifiers)} recall: {recall:.3f}")

    rank_corr = attribution_rank_correlation(attributions, true_importance)
    print(f"Rank correlation with ground truth: {rank_corr:.3f}")

    fidelity = causal_fidelity_score(
        test.X[:50], model, attributions, perturbation_std=0.5, seed=42
    )
    print(f"Causal fidelity score: {fidelity:.3f}")


if __name__ == "__main__":
    main()
