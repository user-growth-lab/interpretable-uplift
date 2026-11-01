"""Enhanced statistical significance experiments.

Runs all method comparisons with increased seeds (IHDP: 20, Hillstrom: 10)
and reports permutation test p-values + Cohen's d effect sizes.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.datasets import make_semisynthetic_ihdp, load_hillstrom
from src.models.cate_estimators import TLearner
from src.models.tesv import TESVExplainer
from src.models.scrl import SCRL
from src.models.baselines import SHAPCATEBaseline, CausalTreeBaseline
from src.metrics.fidelity import (
    top_k_effect_modifier_recall,
    causal_fidelity_score,
    attribution_stability,
    rule_coverage_fidelity,
    permutation_test,
    cohens_d,
)

OUTPUT_DIR = Path("results/paper_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_ihdp_attribution_significance(n_seeds: int = 20, n_test: int = 50):
    """TESV vs baselines on IHDP with 20 seeds."""
    print("=" * 60)
    print(f"IHDP Attribution: {n_seeds} seeds")
    print("=" * 60)

    effect_modifiers = [1, 5, 10, 15, 24]
    results_tesv = {"fidelity": [], "top5": [], "spurious": []}
    results_cond = {"fidelity": [], "top5": [], "spurious": []}

    for seed in range(n_seeds):
        if seed % 5 == 0:
            print(f"  Seed {seed}/{n_seeds}...")

        train_ds, test_ds = make_semisynthetic_ihdp(
            effect_modifier_indices=effect_modifiers,
            add_confounded_features=True,
            confound_corr=0.9,
            seed=seed + 42,
        )

        cate_model = TLearner()
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

        test_X = test_ds.X[:n_test]

        # TESV
        tesv = TESVExplainer(
            cate_model=cate_model,
            background_data=train_ds.X,
            n_background=50,
            n_permutations=20,
            seed=seed,
        )
        tesv_attr = tesv.explain(test_X, show_progress=False)

        # Conditional SHAP
        cond = SHAPCATEBaseline(cate_model=cate_model, background_data=train_ds.X)
        cond_attr = cond.explain(test_X)

        # Metrics
        d = test_X.shape[1]
        n_orig = 25  # original features (not confounded)
        confounded_indices = list(range(n_orig, d))

        # Fidelity (correlation with perturbation response)
        fid_tesv = causal_fidelity_score(test_X, cate_model, tesv_attr, seed=seed)
        fid_cond = causal_fidelity_score(test_X, cate_model, cond_attr, seed=seed)

        # Top-5 recall
        top5_tesv = top_k_effect_modifier_recall(tesv_attr, effect_modifiers, k=5)
        top5_cond = top_k_effect_modifier_recall(cond_attr, effect_modifiers, k=5)

        # Spurious attribution
        total_attr_tesv = np.abs(tesv_attr).sum()
        spurious_tesv = np.abs(tesv_attr[:, confounded_indices]).sum() / total_attr_tesv if total_attr_tesv > 0 else 0
        total_attr_cond = np.abs(cond_attr).sum()
        spurious_cond = np.abs(cond_attr[:, confounded_indices]).sum() / total_attr_cond if total_attr_cond > 0 else 0

        results_tesv["fidelity"].append(fid_tesv)
        results_tesv["top5"].append(top5_tesv)
        results_tesv["spurious"].append(spurious_tesv)
        results_cond["fidelity"].append(fid_cond)
        results_cond["top5"].append(top5_cond)
        results_cond["spurious"].append(spurious_cond)

    # Statistical tests
    output = {"ihdp_attribution": {}}
    for metric_name in ["fidelity", "top5", "spurious"]:
        a = np.array(results_tesv[metric_name])
        b = np.array(results_cond[metric_name])
        t_stat, t_pval = stats.ttest_rel(a, b)
        perm_p = permutation_test(a, b, n_permutations=1000, seed=42)
        d_val = cohens_d(a, b)

        output["ihdp_attribution"][metric_name] = {
            "tesv_mean": float(np.mean(a)),
            "tesv_std": float(np.std(a)),
            "cond_mean": float(np.mean(b)),
            "cond_std": float(np.std(b)),
            "t_test_p": float(t_pval),
            "permutation_p": float(perm_p),
            "cohens_d": float(d_val),
        }
        print(f"  {metric_name}: TESV={np.mean(a):.3f}±{np.std(a):.3f} vs "
              f"Cond={np.mean(b):.3f}±{np.std(b):.3f} | "
              f"perm_p={perm_p:.4f} | d={d_val:.3f}")

    return output


def run_ihdp_scrl_significance(n_seeds: int = 20, n_test: int = 100):
    """SCRL vs Causal Tree on IHDP with 20 seeds."""
    print("\n" + "=" * 60)
    print(f"IHDP Rule Discovery: {n_seeds} seeds")
    print("=" * 60)

    effect_modifiers = [1, 5, 10, 15, 24]
    results_scrl = []
    results_ct = []

    for seed in range(n_seeds):
        if seed % 5 == 0:
            print(f"  Seed {seed}/{n_seeds}...")

        train_ds, test_ds = make_semisynthetic_ihdp(
            effect_modifier_indices=effect_modifiers,
            seed=seed + 42,
        )

        cate_model = TLearner()
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)
        tau_pred = cate_model.predict_cate(test_ds.X[:n_test])

        # SCRL
        scrl = SCRL(cate_model=cate_model, n_rules=5, seed=seed)
        scrl.fit(train_ds.X, train_ds.T, train_ds.Y)
        scrl_masks = [rule.covers(test_ds.X[:n_test]) for rule in scrl.rules]
        fid_scrl = rule_coverage_fidelity(
            test_ds.X[:n_test], tau_pred, scrl_masks, metric="mean_uplift"
        )

        # Causal Tree
        ct = CausalTreeBaseline(max_depth=3, min_samples_leaf=20)
        ct.fit(train_ds.X, train_ds.T, train_ds.Y)
        ct_masks = [rule.covers(test_ds.X[:n_test]) for rule in ct.rules]
        fid_ct = rule_coverage_fidelity(
            test_ds.X[:n_test], tau_pred, ct_masks, metric="mean_uplift"
        ) if ct_masks else 0.0

        results_scrl.append(fid_scrl)
        results_ct.append(fid_ct)

    a = np.array(results_scrl)
    b = np.array(results_ct)
    t_stat, t_pval = stats.ttest_rel(a, b)
    perm_p = permutation_test(a, b, n_permutations=1000, seed=42)
    d_val = cohens_d(a, b)

    output = {
        "ihdp_scrl": {
            "scrl_mean": float(np.mean(a)),
            "scrl_std": float(np.std(a)),
            "ct_mean": float(np.mean(b)),
            "ct_std": float(np.std(b)),
            "t_test_p": float(t_pval),
            "permutation_p": float(perm_p),
            "cohens_d": float(d_val),
        }
    }
    print(f"  SCRL={np.mean(a):.3f}±{np.std(a):.3f} vs CT={np.mean(b):.3f}±{np.std(b):.3f}")
    print(f"  perm_p={perm_p:.4f} | Cohen's d={d_val:.3f}")

    return output


if __name__ == "__main__":
    print("Enhanced Statistical Significance Tests")
    print("=" * 60)

    all_results = {}

    # IHDP attribution (20 seeds)
    attr_results = run_ihdp_attribution_significance(n_seeds=20, n_test=50)
    all_results.update(attr_results)

    # IHDP SCRL (20 seeds)
    scrl_results = run_ihdp_scrl_significance(n_seeds=20, n_test=100)
    all_results.update(scrl_results)

    # Save
    output_path = OUTPUT_DIR / "enhanced_significance_20seeds.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
