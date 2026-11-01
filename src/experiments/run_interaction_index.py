"""TESV Interaction Index experiments.

Validates that TESV-I can recover known feature interactions in CATE,
providing a capability that standard Interventional SHAP differencing
cannot directly achieve.

Experiments:
1. IHDP with injected interaction X_1 * X_15 -> verify interaction_recall@3
2. Hillstrom: report top-5 interaction pairs (qualitative)
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.datasets import make_semisynthetic_ihdp, load_hillstrom
from src.models.cate_estimators import TLearner
from src.models.tesv import TESVInteractionExplainer
from src.metrics.fidelity import interaction_recall

OUTPUT_DIR = Path("results/paper_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_ihdp_interaction_experiment(n_seeds: int = 5, n_test: int = 50):
    """Run TESV-I on semi-synthetic IHDP with known interaction terms."""
    print("=" * 60)
    print("Experiment 1: IHDP with injected interaction X_1 * X_15")
    print("=" * 60)

    # Ground-truth: modifiers = [1, 5, 10, 15, 24], interaction = (1, 15)
    effect_modifiers = [1, 5, 10, 15, 24]
    interaction_pairs = [(1, 15)]

    results = []
    for seed in range(n_seeds):
        print(f"\n  Seed {seed}/{n_seeds}...")
        train_ds, test_ds = make_semisynthetic_ihdp(
            effect_modifier_indices=effect_modifiers,
            interaction_terms=interaction_pairs,
            interaction_strength=1.5,
            signal_strength=1.0,
            noise_scale=0.3,
            add_confounded_features=False,
            seed=seed + 42,
        )

        # Fit CATE model
        cate_model = TLearner()
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

        # Compute TESV-I on test subset
        test_X = test_ds.X[:n_test]
        explainer = TESVInteractionExplainer(
            cate_model=cate_model,
            background_data=train_ds.X,
            n_background=50,
            n_interaction_samples=100,  # MC samples for speed
            seed=seed + 42,
        )

        # Get interaction matrix
        interaction_tensor = explainer.explain_interactions(test_X, show_progress=False)

        # Evaluate recall
        recall_at_3 = interaction_recall(interaction_tensor, interaction_pairs, k=3)
        recall_at_5 = interaction_recall(interaction_tensor, interaction_pairs, k=5)

        # Get top interactions
        mean_matrix = np.abs(interaction_tensor).mean(axis=0)
        d = mean_matrix.shape[0]
        pairs_ranked = []
        for i in range(d):
            for j in range(i + 1, d):
                pairs_ranked.append((i, j, mean_matrix[i, j]))
        pairs_ranked.sort(key=lambda x: x[2], reverse=True)

        top5 = [(p[0], p[1], float(p[2])) for p in pairs_ranked[:5]]
        target_rank = next(
            (idx + 1 for idx, p in enumerate(pairs_ranked) if (p[0], p[1]) == (1, 15)),
            -1,
        )

        results.append({
            "seed": seed,
            "recall@3": recall_at_3,
            "recall@5": recall_at_5,
            "target_rank": target_rank,
            "top5_interactions": top5,
        })
        print(f"    recall@3={recall_at_3:.3f}, recall@5={recall_at_5:.3f}, "
              f"target (1,15) rank={target_rank}")

    # Summary
    recalls_3 = [r["recall@3"] for r in results]
    recalls_5 = [r["recall@5"] for r in results]
    ranks = [r["target_rank"] for r in results]

    summary = {
        "experiment": "IHDP_interaction",
        "n_seeds": n_seeds,
        "n_test": n_test,
        "effect_modifiers": effect_modifiers,
        "interaction_pairs": interaction_pairs,
        "results": results,
        "summary": {
            "recall@3_mean": float(np.mean(recalls_3)),
            "recall@3_std": float(np.std(recalls_3)),
            "recall@5_mean": float(np.mean(recalls_5)),
            "recall@5_std": float(np.std(recalls_5)),
            "target_rank_mean": float(np.mean(ranks)),
            "target_rank_std": float(np.std(ranks)),
        },
    }

    print(f"\n  Summary: recall@3 = {np.mean(recalls_3):.3f}±{np.std(recalls_3):.3f}")
    print(f"           recall@5 = {np.mean(recalls_5):.3f}±{np.std(recalls_5):.3f}")
    print(f"           target rank = {np.mean(ranks):.1f}±{np.std(ranks):.1f}")

    return summary


def run_hillstrom_interaction_experiment(n_test: int = 30):
    """Run TESV-I on Hillstrom (no ground truth, qualitative analysis)."""
    print("\n" + "=" * 60)
    print("Experiment 2: Hillstrom top-5 interactions (qualitative)")
    print("=" * 60)

    try:
        train_ds, test_ds = load_hillstrom()
    except FileNotFoundError:
        print("  [SKIP] Hillstrom data not found locally")
        return None

    # Fit CATE model
    cate_model = TLearner()
    cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

    # TESV-I (MC approximation for d=15)
    explainer = TESVInteractionExplainer(
        cate_model=cate_model,
        background_data=train_ds.X,
        n_background=50,
        n_interaction_samples=100,
        seed=42,
    )

    test_X = test_ds.X[:n_test]
    top5 = explainer.top_interactions(test_X, k=5, show_progress=True)

    feature_names = train_ds.feature_names or [f"f{i}" for i in range(train_ds.d)]
    print("\n  Top-5 feature interactions (Hillstrom):")
    for i, j, strength in top5:
        name_i = feature_names[i] if i < len(feature_names) else f"f{i}"
        name_j = feature_names[j] if j < len(feature_names) else f"f{j}"
        print(f"    ({name_i}, {name_j}): strength = {strength:.6f}")

    result = {
        "experiment": "Hillstrom_interaction",
        "n_test": n_test,
        "top5_interactions": [
            {"feature_i": i, "feature_j": j, "strength": float(s)}
            for i, j, s in top5
        ],
    }
    return result


if __name__ == "__main__":
    print("TESV Interaction Index Experiments")
    print("=" * 60)

    all_results = {}

    # Experiment 1: IHDP
    ihdp_results = run_ihdp_interaction_experiment(n_seeds=5, n_test=50)
    all_results["ihdp"] = ihdp_results

    # Experiment 2: Hillstrom
    hillstrom_results = run_hillstrom_interaction_experiment(n_test=30)
    if hillstrom_results:
        all_results["hillstrom"] = hillstrom_results

    # Save results
    output_path = OUTPUT_DIR / "tesv_interaction_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to {output_path}")

    # Sanity check
    recall = ihdp_results["summary"]["recall@3_mean"]
    if recall >= 0.67:
        print(f"\n✅ SANITY CHECK PASSED: recall@3 = {recall:.3f} >= 0.67")
    else:
        print(f"\n⚠️  SANITY CHECK: recall@3 = {recall:.3f} < 0.67 (may need tuning)")
