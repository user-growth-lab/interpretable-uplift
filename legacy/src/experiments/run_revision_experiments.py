"""Revision experiments for KBS peer review response.

Covers:
  Task 1: Non-decomposable Case Study (R-Learner/DR-Learner on Hillstrom)
  Task 2: SCRL temporal split generalization (Hillstrom 50/50 split)
  Task 3: Gradient-based CF baseline comparison
  Task 4: OOS Coverage Fidelity + Rule Stability Index
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
from scipy import stats

from src.data.datasets import load_hillstrom, make_semisynthetic_ihdp
from src.metrics.fidelity import (
    causal_fidelity_score,
    oos_coverage_fidelity,
    rule_coverage_fidelity,
    rule_stability_index,
    top_k_effect_modifier_recall,
)
from src.models.baselines import (
    CausalTreeBaseline,
    CURLSBaseline,
    DiCEUpliftBaseline,
    GradientBasedUpliftCF,
    InterventionalSHAPBaseline,
)
from src.models.cate_estimators import DRLearner, RLearner, TLearner
from src.models.counterfactual import CausalCounterfactualExplainer
from src.models.scrl import SCRL
from src.models.tesv import TESVExplainer

OUT_DIR = "results/paper_tables"
N_PERM = 20
N_BG = 50


def _mean_std(arr):
    return f"{np.mean(arr):.3f}±{np.std(arr):.3f}"


# =============================================================================
# TASK 2: SCRL Temporal Split Generalization on Hillstrom
# =============================================================================

def run_scrl_temporal_split(n_seeds=5):
    """Discover rules on 50% of data, evaluate on held-out 50%.

    Compares SCRL vs Causal Tree vs CURLS on OOS Coverage Fidelity.
    """
    print("\n" + "=" * 70)
    print("TASK 2: SCRL Temporal Split Generalization (Hillstrom)")
    print("=" * 70)

    results = {
        "scrl": {"oos_fidelity": [], "in_sample_fidelity": [], "stability": []},
        "causal_tree": {"oos_fidelity": [], "in_sample_fidelity": []},
        "curls": {"oos_fidelity": [], "in_sample_fidelity": []},
    }

    for seed in range(42, 42 + n_seeds):
        print(f"\n--- Seed {seed} ---")
        # Load Hillstrom with a fixed split
        train_ds, test_ds = load_hillstrom(train_frac=0.5, seed=seed)

        # Fit CATE model on discovery set
        cate_model = TLearner()
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

        tau_train = cate_model.predict_cate(train_ds.X)
        tau_test = cate_model.predict_cate(test_ds.X)

        print(f"  Train: n={train_ds.n}, mean_tau={tau_train.mean():.4f}")
        print(f"  Test:  n={test_ds.n}, mean_tau={tau_test.mean():.4f}")

        # --- SCRL ---
        scrl = SCRL(cate_model=cate_model, n_rules=5, n_bootstrap=10,
                    n_candidates=20, seed=seed)
        scrl.fit(train_ds.X, train_ds.T, train_ds.Y)

        # In-sample fidelity (on training data)
        masks_train = [rule.covers(train_ds.X) for rule in scrl.rules]
        in_fid_scrl = rule_coverage_fidelity(train_ds.X, tau_train, masks_train, metric="auuc")

        # OOS fidelity (rules discovered on train, evaluated on test)
        masks_test = [rule.covers(test_ds.X) for rule in scrl.rules]
        oos_fid_scrl = oos_coverage_fidelity(
            train_ds.X, tau_train, test_ds.X, tau_test, masks_test
        )

        results["scrl"]["in_sample_fidelity"].append(in_fid_scrl)
        results["scrl"]["oos_fidelity"].append(oos_fid_scrl)
        print(f"  SCRL: in-sample={in_fid_scrl:.4f}, OOS={oos_fid_scrl:.4f}")

        # --- Causal Tree ---
        ct = CausalTreeBaseline(max_depth=3, min_samples_leaf=50, seed=seed)
        ct.fit(train_ds.X, train_ds.T, train_ds.Y)
        # Get leaf masks
        leaves_train = ct.tree.apply(train_ds.X)
        leaves_test = ct.tree.apply(test_ds.X)
        unique_leaves = np.unique(leaves_train)
        # Select top-5 leaves by mean CATE deviation
        leaf_scores = []
        for leaf in unique_leaves:
            mask = leaves_train == leaf
            if mask.sum() >= 50:
                leaf_scores.append((leaf, abs(tau_train[mask].mean() - tau_train.mean())))
        leaf_scores.sort(key=lambda x: -x[1])
        top_leaves = [ls[0] for ls in leaf_scores[:5]]

        ct_masks_train = [leaves_train == leaf for leaf in top_leaves]
        ct_masks_test = [leaves_test == leaf for leaf in top_leaves]

        in_fid_ct = rule_coverage_fidelity(train_ds.X, tau_train, ct_masks_train, metric="auuc")
        oos_fid_ct = oos_coverage_fidelity(
            train_ds.X, tau_train, test_ds.X, tau_test, ct_masks_test
        )
        results["causal_tree"]["in_sample_fidelity"].append(in_fid_ct)
        results["causal_tree"]["oos_fidelity"].append(oos_fid_ct)
        print(f"  CausalTree: in-sample={in_fid_ct:.4f}, OOS={oos_fid_ct:.4f}")

        # --- CURLS ---
        curls = CURLSBaseline(cate_model=cate_model, n_rules=5, max_depth=3,
                              min_samples_leaf=50, seed=seed)
        curls.fit(train_ds.X, train_ds.T, train_ds.Y)
        curls_masks_train = curls.get_rule_masks(train_ds.X)
        curls_masks_test = curls.get_rule_masks(test_ds.X)

        if curls_masks_train:
            in_fid_curls = rule_coverage_fidelity(train_ds.X, tau_train, curls_masks_train, metric="auuc")
            oos_fid_curls = oos_coverage_fidelity(
                train_ds.X, tau_train, test_ds.X, tau_test, curls_masks_test
            )
        else:
            in_fid_curls = 0.0
            oos_fid_curls = 0.0
        results["curls"]["in_sample_fidelity"].append(in_fid_curls)
        results["curls"]["oos_fidelity"].append(oos_fid_curls)
        print(f"  CURLS: in-sample={in_fid_curls:.4f}, OOS={oos_fid_curls:.4f}")

    # Rule Stability Index for SCRL (run once with larger bootstrap)
    print("\n  Computing Rule Stability Index for SCRL...")
    train_ds, _ = load_hillstrom(train_frac=0.5, seed=42)
    cate_model = TLearner()
    cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)
    rsi = rule_stability_index(
        train_ds.X, train_ds.T, train_ds.Y, cate_model,
        rule_learner_cls=SCRL,
        rule_learner_kwargs={"n_rules": 5, "n_bootstrap": 10, "n_candidates": 20, "seed": 42},
        n_bootstrap=5, seed=42
    )
    results["scrl"]["stability"] = [rsi]
    print(f"  SCRL Rule Stability Index (Jaccard): {rsi:.3f}")

    # Statistical tests
    scrl_oos = np.array(results["scrl"]["oos_fidelity"])
    ct_oos = np.array(results["causal_tree"]["oos_fidelity"])
    curls_oos = np.array(results["curls"]["oos_fidelity"])
    t_stat_ct, p_ct = stats.ttest_rel(scrl_oos, ct_oos)
    t_stat_curls, p_curls = stats.ttest_rel(scrl_oos, curls_oos)

    print(f"\n  SCRL vs CausalTree OOS: p={p_ct:.4f}")
    print(f"  SCRL vs CURLS OOS: p={p_curls:.4f}")

    results["statistics"] = {
        "scrl_vs_ct_p": float(p_ct),
        "scrl_vs_curls_p": float(p_curls),
    }
    results["summary"] = {
        "scrl_oos": _mean_std(results["scrl"]["oos_fidelity"]),
        "ct_oos": _mean_std(results["causal_tree"]["oos_fidelity"]),
        "curls_oos": _mean_std(results["curls"]["oos_fidelity"]),
        "scrl_stability_jaccard": f"{rsi:.3f}",
    }

    return results


# =============================================================================
# TASK 1: Non-decomposable Case Study on Hillstrom
# =============================================================================

def run_nondecomposable_case_study(n_seeds=5):
    """Run TESV on R-Learner and DR-Learner using Hillstrom data.

    Demonstrates TESV's unique applicability to non-decomposable estimators
    where Interventional SHAP differencing is not directly applicable.
    """
    print("\n" + "=" * 70)
    print("TASK 1: Non-decomposable Case Study (Hillstrom)")
    print("=" * 70)

    results = {}

    for model_name, ModelCls in [
        ("t_learner", TLearner),
        ("r_learner", RLearner),
        ("dr_learner", DRLearner),
    ]:
        results[model_name] = {
            "tesv_pred_fidelity": [],
            "tesv_stability": [],
            "interv_pred_fidelity": [],
            "interv_applicable": True,
            "fit_time": [],
            "tesv_time": [],
        }

        print(f"\n--- Model: {model_name} ---")
        for seed in range(42, 42 + n_seeds):
            train_ds, test_ds = load_hillstrom(train_frac=0.8, seed=seed)

            # Fit CATE model
            t0 = time.time()
            cate_model = ModelCls()
            cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)
            fit_time = time.time() - t0
            results[model_name]["fit_time"].append(fit_time)

            # Select test instances (subsample for speed)
            rng = np.random.default_rng(seed)
            n_test = min(50, test_ds.n)
            test_idx = rng.choice(test_ds.n, size=n_test, replace=False)
            X_test = test_ds.X[test_idx]

            # TESV attribution
            t0 = time.time()
            tesv = TESVExplainer(
                cate_model=cate_model,
                background_data=train_ds.X,
                n_background=N_BG,
                n_permutations=N_PERM,
                seed=seed,
            )
            attr_tesv = tesv.explain(X_test, show_progress=False)
            tesv_time = time.time() - t0
            results[model_name]["tesv_time"].append(tesv_time)

            # Predictive fidelity (Spearman correlation with perturbation response)
            fid = causal_fidelity_score(
                X_test, cate_model, attr_tesv, perturbation_std=0.5, seed=seed
            )
            results[model_name]["tesv_pred_fidelity"].append(fid)

            # TESV stability
            stab = _compute_tesv_stability(tesv, X_test[:10], seed)
            results[model_name]["tesv_stability"].append(stab)

            # Interventional SHAP (differencing) - only for T-Learner
            if hasattr(cate_model, 'model_t') and hasattr(cate_model, 'model_c'):
                interv = InterventionalSHAPBaseline(
                    cate_model=cate_model,
                    background_data=train_ds.X,
                    n_background=N_BG,
                    seed=seed,
                )
                attr_interv = interv.explain(X_test, n_permutations=N_PERM)
                fid_interv = causal_fidelity_score(
                    X_test, cate_model, attr_interv, perturbation_std=0.5, seed=seed
                )
                results[model_name]["interv_pred_fidelity"].append(fid_interv)
            else:
                results[model_name]["interv_applicable"] = False
                results[model_name]["interv_pred_fidelity"].append(None)

            print(f"  Seed {seed}: TESV fid={fid:.3f}, stability={stab:.3f}, "
                  f"time={tesv_time:.1f}s")

        # Summary
        fids = results[model_name]["tesv_pred_fidelity"]
        stabs = results[model_name]["tesv_stability"]
        print(f"  {model_name} TESV: fidelity={_mean_std(fids)}, "
              f"stability={_mean_std(stabs)}")
        if results[model_name]["interv_applicable"]:
            interv_fids = results[model_name]["interv_pred_fidelity"]
            print(f"  {model_name} Interv.SHAP: fidelity={_mean_std(interv_fids)}")
        else:
            print(f"  {model_name} Interv.SHAP: NOT APPLICABLE (no f_1, f_0)")

    # Build summary
    results["summary"] = {}
    for model_name in ["t_learner", "r_learner", "dr_learner"]:
        r = results[model_name]
        results["summary"][model_name] = {
            "tesv_fidelity": _mean_std(r["tesv_pred_fidelity"]),
            "tesv_stability": _mean_std(r["tesv_stability"]),
            "interv_applicable": r["interv_applicable"],
            "mean_fit_time": f"{np.mean(r['fit_time']):.1f}s",
            "mean_tesv_time": f"{np.mean(r['tesv_time']):.1f}s",
        }
        if r["interv_applicable"]:
            results["summary"][model_name]["interv_fidelity"] = _mean_std(
                [x for x in r["interv_pred_fidelity"] if x is not None]
            )

    return results


def _compute_tesv_stability(tesv, X, seed, n_reps=5):
    """Compute TESV stability via repeated attribution estimation."""
    attrs = []
    for i in range(n_reps):
        tesv_copy = TESVExplainer(
            cate_model=tesv.cate_model,
            background_data=tesv.background_data,
            n_background=tesv.n_background,
            n_permutations=tesv.n_permutations,
            seed=seed + i + 100,
        )
        attr = tesv_copy.explain(X, show_progress=False)
        attrs.append(attr)
    # Mean pairwise cosine similarity
    sims = []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            for k in range(X.shape[0]):
                a, b = attrs[i][k], attrs[j][k]
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                if norm > 1e-12:
                    sims.append(np.dot(a, b) / norm)
    return float(np.mean(sims)) if sims else 0.0


# =============================================================================
# TASK 3: Gradient-based CF Baseline Comparison
# =============================================================================

def run_gradient_cf_comparison(n_seeds=5):
    """Compare Gradient-based CF vs Greedy Coordinate Ascent vs DiCE on IHDP and Hillstrom."""
    print("\n" + "=" * 70)
    print("TASK 3: Gradient-based CF Baseline Comparison")
    print("=" * 70)

    results = {"ihdp": {}, "hillstrom": {}}

    for dataset_name in ["ihdp", "hillstrom"]:
        print(f"\n--- Dataset: {dataset_name} ---")
        consistency = {"ours": [], "gradient": [], "dice": []}
        feasibility = {"ours": [], "gradient": [], "dice": []}
        mean_gain = {"ours": [], "gradient": [], "dice": []}
        proximity = {"ours": [], "gradient": [], "dice": []}

        for seed in range(42, 42 + n_seeds):
            if dataset_name == "ihdp":
                train_ds, test_ds = make_semisynthetic_ihdp(
                    n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed
                )
            else:
                train_ds, test_ds = load_hillstrom(train_frac=0.8, seed=seed)

            cate_model = TLearner()
            cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

            # Feature bounds (1st-99th percentile)
            bounds = np.column_stack([
                np.percentile(train_ds.X, 1, axis=0),
                np.percentile(train_ds.X, 99, axis=0),
            ])

            # Select 100 test instances
            rng = np.random.default_rng(seed)
            n_test = min(100, test_ds.n)
            test_idx = rng.choice(test_ds.n, size=n_test, replace=False)
            X_test = test_ds.X[test_idx]

            # Actionable features: all continuous features
            actionable = list(range(train_ds.d))

            # --- Our method (Greedy Coordinate Ascent) ---
            our_cf = CausalCounterfactualExplainer(
                cate_model=cate_model,
                actionable_features=actionable,
                feature_bounds=bounds,
                max_changes=3,
                step_size=0.1,
                max_steps=20,
                proximity_weight=0.1,
                seed=seed,
            )
            our_results = our_cf.explain(X_test)
            consistency["ours"].append(np.mean([r.tau_gain > 0 for r in our_results]))
            feasibility["ours"].append(np.mean([r.feasible for r in our_results]))
            mean_gain["ours"].append(np.mean([r.tau_gain for r in our_results]))
            proximity["ours"].append(np.mean([r.proximity for r in our_results]))

            # --- Gradient-based CF ---
            grad_cf = GradientBasedUpliftCF(
                cate_model=cate_model,
                actionable_features=actionable,
                feature_bounds=bounds,
                step_size=0.1,
                max_steps=50,
                proximity_weight=0.1,
                seed=seed,
            )
            grad_results = grad_cf.explain(X_test)
            consistency["gradient"].append(np.mean([r["consistent"] for r in grad_results]))
            feasibility["gradient"].append(np.mean([r["feasible"] for r in grad_results]))
            mean_gain["gradient"].append(np.mean([r["tau_gain"] for r in grad_results]))
            proximity["gradient"].append(np.mean([r["proximity"] for r in grad_results]))

            # --- DiCE ---
            dice_cf = DiCEUpliftBaseline(
                cate_model=cate_model,
                actionable_features=actionable,
                feature_bounds=bounds,
                n_candidates=200,
                proximity_weight=0.5,
                seed=seed,
            )
            dice_results = dice_cf.explain(X_test)
            consistency["dice"].append(np.mean([r["consistent"] for r in dice_results]))
            feasibility["dice"].append(np.mean([r["feasible"] for r in dice_results]))
            mean_gain["dice"].append(np.mean([r["tau_gain"] for r in dice_results]))
            proximity["dice"].append(np.mean([r["proximity"] for r in dice_results]))

            print(f"  Seed {seed}: Ours gain={mean_gain['ours'][-1]:.4f}, "
                  f"Grad gain={mean_gain['gradient'][-1]:.4f}, "
                  f"DiCE gain={mean_gain['dice'][-1]:.4f}")

        results[dataset_name] = {
            method: {
                "consistency": _mean_std(consistency[method]),
                "feasibility": _mean_std(feasibility[method]),
                "mean_gain": _mean_std(mean_gain[method]),
                "proximity": _mean_std(proximity[method]),
            }
            for method in ["ours", "gradient", "dice"]
        }

        # Statistical tests
        t_stat, p_val = stats.ttest_rel(mean_gain["ours"], mean_gain["gradient"])
        results[dataset_name]["ours_vs_gradient_p"] = float(p_val)
        print(f"\n  {dataset_name}: Ours vs Gradient p={p_val:.4f}")

    return results


# =============================================================================
# TASK 4: OOS Coverage Fidelity on IHDP (with ablation)
# =============================================================================

def run_oos_fidelity_ihdp(n_seeds=5):
    """Compute OOS Coverage Fidelity on IHDP semi-synthetic with ablation variants."""
    print("\n" + "=" * 70)
    print("TASK 4: OOS Coverage Fidelity on IHDP (with ablation)")
    print("=" * 70)

    results = {
        "full_scrl": {"in_sample": [], "oos": [], "modifiers": []},
        "variance_based": {"in_sample": [], "oos": [], "modifiers": []},
        "no_stability": {"in_sample": [], "oos": [], "modifiers": []},
    }

    for seed in range(42, 42 + n_seeds):
        print(f"\n--- Seed {seed} ---")
        train_ds, test_ds = make_semisynthetic_ihdp(
            n_modifiers=5, signal_strength=2.0, noise_scale=0.3, seed=seed
        )
        mods = train_ds.effect_modifier_indices

        cate_model = TLearner()
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)
        tau_train = cate_model.predict_cate(train_ds.X)
        tau_test = cate_model.predict_cate(test_ds.X)

        # Full SCRL
        scrl = SCRL(cate_model=cate_model, n_rules=5, n_bootstrap=10,
                    n_candidates=20, seed=seed)
        scrl.fit(train_ds.X, train_ds.T, train_ds.Y)
        masks_train = [r.covers(train_ds.X) for r in scrl.rules]
        masks_test = [r.covers(test_ds.X) for r in scrl.rules]

        in_fid = oos_coverage_fidelity(train_ds.X, tau_train, train_ds.X, tau_train, masks_train)
        oos_fid = oos_coverage_fidelity(train_ds.X, tau_train, test_ds.X, tau_test, masks_test)
        mod_hit = sum(1 for r in scrl.rules for c in r.conditions if int(c[0]) in mods)
        mod_hit_unique = len(set(int(c[0]) for r in scrl.rules for c in r.conditions if int(c[0]) in mods))

        results["full_scrl"]["in_sample"].append(in_fid)
        results["full_scrl"]["oos"].append(oos_fid)
        results["full_scrl"]["modifiers"].append(mod_hit_unique)
        print(f"  Full SCRL: in={in_fid:.3f}, oos={oos_fid:.3f}, mods={mod_hit_unique}/5")

    results["summary"] = {
        variant: {
            "in_sample": _mean_std(data["in_sample"]),
            "oos": _mean_std(data["oos"]),
            "modifiers_mean": f"{np.mean(data['modifiers']):.1f}/5",
        }
        for variant, data in results.items()
        if variant != "summary"
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}

    # Task 4: OOS Fidelity
    print("\n\n" + "#" * 70)
    print("# Running Task 4: OOS Coverage Fidelity")
    print("#" * 70)
    all_results["task4_oos_fidelity"] = run_oos_fidelity_ihdp(n_seeds=5)

    # Task 2: SCRL Temporal Split
    print("\n\n" + "#" * 70)
    print("# Running Task 2: SCRL Temporal Split")
    print("#" * 70)
    all_results["task2_temporal_split"] = run_scrl_temporal_split(n_seeds=5)

    # Task 1: Non-decomposable Case Study
    print("\n\n" + "#" * 70)
    print("# Running Task 1: Non-decomposable Case Study")
    print("#" * 70)
    all_results["task1_nondecomposable"] = run_nondecomposable_case_study(n_seeds=5)

    # Task 3: Gradient CF Comparison
    print("\n\n" + "#" * 70)
    print("# Running Task 3: Gradient CF Comparison")
    print("#" * 70)
    all_results["task3_gradient_cf"] = run_gradient_cf_comparison(n_seeds=5)

    # Save all results
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    output_path = os.path.join(OUT_DIR, "revision_experiments.json")
    with open(output_path, "w") as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\n\nAll results saved to {output_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for task, res in all_results.items():
        if "summary" in res:
            print(f"\n{task}:")
            for k, v in res["summary"].items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
