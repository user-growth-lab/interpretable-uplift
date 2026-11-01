"""Counterfactual-gain robustness diagnostics (reviewer M3).

M3 questions whether the large reported CF gains are a genuine property of the
CATE surface or an artefact of a single fitted model / boundary extrapolation.
We probe this with three diagnostics:

  (a) Model-perturbation robustness. We refit the CATE model on B bootstrap
      resamples of the training data and re-evaluate the gain of the SAME
      recommended counterfactual under each perturbed model. If the gain is a
      real property, it should persist (high retention, high positive rate);
      if it is an artefact of one fitted surface, it should collapse.

  (b) Boundary / perturbation-magnitude analysis. For each recommendation we
      record how many features change, the L2 magnitude, and whether the
      changed feature values land on the plausibility bounds (1st/99th pct).
      Few boundary-hitting changes indicate the gain is not obtained by pushing
      features to clipping edges.

  (c) Stronger baseline. Beyond the conservative Gradient-CF, we add a
      high-budget structured random search (RandomSearchUplift) and report the
      same robustness diagnostics, showing our gain is not merely an artefact of
      weak baselines and that robustness is method-comparable.

Outputs results/paper_tables/cf_robustness.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datasets import load_hillstrom, make_semisynthetic_ihdp
from src.models.baselines import GradientBasedUpliftCF, RandomSearchUplift
from src.models.cate_estimators import TLearner
from src.models.counterfactual import CausalCounterfactualExplainer

OUT_DIR = "results/paper_tables"


def feature_bounds_from(X, lo=1, hi=99):
    """Plausibility bounds = [1st, 99th] percentile per feature."""
    return np.stack([np.percentile(X, lo, axis=0), np.percentile(X, hi, axis=0)], axis=1)


def fit_bootstrap_models(X, T, Y, B, seed):
    """Refit T-Learner on B bootstrap resamples; return list of models."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    models = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        m = TLearner()
        m.fit(X[idx], T[idx], Y[idx])
        models.append(m)
    return models


def robustness_of_pairs(pairs, boot_models):
    """Given (x, x_cf, gain_orig) triples, evaluate gain under each bootstrap model.

    Returns dict with retention, positive_rate, and rank correlation.
    """
    if not pairs:
        return {"retention": 0.0, "positive_rate": 0.0, "mean_gain_boot": 0.0,
                "rank_corr": 0.0, "n": 0}
    X_orig = np.array([p[0] for p in pairs])
    X_cf = np.array([p[1] for p in pairs])
    gains_orig = np.array([p[2] for p in pairs])

    boot_gains = []  # (B, n_pairs)
    for m in boot_models:
        g = m.predict_cate(X_cf) - m.predict_cate(X_orig)
        boot_gains.append(g)
    boot_gains = np.array(boot_gains)  # (B, n)

    mean_gain_boot = boot_gains.mean(axis=0)  # per pair
    # Retention: bootstrap gain relative to original gain (only where orig gain > 0)
    valid = gains_orig > 1e-9
    retention = float(np.mean(mean_gain_boot[valid] / gains_orig[valid])) if valid.any() else 0.0
    positive_rate = float((boot_gains > 0).mean())  # over all (B, pairs)
    # Rank correlation between original gain and mean bootstrap gain (Spearman-lite)
    if valid.sum() >= 3:
        a = gains_orig[valid]
        b = mean_gain_boot[valid]
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        rank_corr = float(np.corrcoef(ra, rb)[0, 1])
    else:
        rank_corr = 0.0
    return {
        "retention": retention,
        "positive_rate": positive_rate,
        "mean_gain_orig": float(gains_orig.mean()),
        "mean_gain_boot": float(mean_gain_boot.mean()),
        "rank_corr": rank_corr,
        "n": int(len(pairs)),
    }


def boundary_analysis(x, x_cf, bounds, eps=1e-3):
    """Per-instance: #changed, L2, and #changed-values at a plausibility bound."""
    diff = x_cf - x
    changed = np.where(np.abs(diff) > 1e-6)[0]
    at_bound = 0
    for j in changed:
        near_lo = abs(x_cf[j] - bounds[j, 0]) <= eps + 1e-6
        near_hi = abs(x_cf[j] - bounds[j, 1]) <= eps + 1e-6
        if near_lo or near_hi:
            at_bound += 1
    return {
        "n_changed": int(len(changed)),
        "l2": float(np.linalg.norm(diff)),
        "n_at_bound": int(at_bound),
    }


def run_dataset(label, train_ds, test_ds, n_instances=50, B=20, seed=42):
    X, T, Y = train_ds.X, train_ds.T, train_ds.Y
    bounds = feature_bounds_from(X)

    shared = TLearner()
    shared.fit(X, T, Y)

    rng = np.random.default_rng(seed)
    n_test = min(n_instances, test_ds.n)
    sel = rng.choice(test_ds.n, size=n_test, replace=False)
    X_test = test_ds.X[sel]

    boot_models = fit_bootstrap_models(X, T, Y, B, seed)

    methods = {
        "Ours": CausalCounterfactualExplainer(
            cate_model=shared, feature_bounds=bounds, max_changes=3,
            step_size=0.1, max_steps=20, seed=seed),
        "Gradient-CF": GradientBasedUpliftCF(
            cate_model=shared, feature_bounds=bounds, step_size=0.1,
            max_steps=50, seed=seed),
        "RandomSearch (strong)": RandomSearchUplift(
            cate_model=shared, feature_bounds=bounds, n_candidates=2000,
            max_distance=3.0, seed=seed),
    }

    out = {}
    for name, explainer in methods.items():
        pairs = []
        bnd = {"n_changed": [], "l2": [], "n_at_bound": [], "n_changed_total": 0}
        consistent = 0
        for x in X_test:
            if name == "Ours":
                exp = explainer.explain_instance(x, direction="increase")
                x_cf, gain = exp.x_cf, exp.tau_gain
            else:
                res = explainer.explain_instance(x)
                x_cf, gain = res["x_cf"], res["tau_gain"]
            pairs.append((x, x_cf, gain))
            if gain >= -1e-9:
                consistent += 1
            b = boundary_analysis(x, x_cf, bounds)
            bnd["n_changed"].append(b["n_changed"])
            bnd["l2"].append(b["l2"])
            bnd["n_at_bound"].append(b["n_at_bound"])
            bnd["n_changed_total"] += b["n_changed"]

        rob = robustness_of_pairs(pairs, boot_models)
        frac_at_bound = (sum(bnd["n_at_bound"]) / bnd["n_changed_total"]
                         if bnd["n_changed_total"] > 0 else 0.0)
        out[name] = {
            "mean_gain": float(np.mean([p[2] for p in pairs])),
            "consistency": consistent / n_test,
            "mean_n_changed": float(np.mean(bnd["n_changed"])),
            "mean_l2": float(np.mean(bnd["l2"])),
            "frac_changes_at_bound": float(frac_at_bound),
            "robustness": rob,
        }
        print(f"  {name:22s} gain={out[name]['mean_gain']:.3f} "
              f"retention={rob['retention']:.2f} pos_rate={rob['positive_rate']:.2f} "
              f"rankcorr={rob['rank_corr']:.2f} @bound={frac_at_bound:.2f} "
              f"nchg={out[name]['mean_n_changed']:.1f}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print("COUNTERFACTUAL-GAIN ROBUSTNESS DIAGNOSTICS (M3)")
    print("=" * 74)

    results = {}

    print("\n--- IHDP (semi-synthetic) ---")
    tr, te = make_semisynthetic_ihdp(replication=0, n_modifiers=5, seed=42)
    results["IHDP"] = run_dataset("IHDP", tr, te, n_instances=50, B=20, seed=42)

    print("\n--- Hillstrom (real) ---")
    tr, te = load_hillstrom(train_frac=0.8, seed=42)
    # subsample training for tractable bootstrap refits
    if tr.n > 8000:
        r = np.random.default_rng(42)
        s = r.choice(tr.n, 8000, replace=False)
        tr.X, tr.T, tr.Y = tr.X[s], tr.T[s], tr.Y[s]
    results["Hillstrom"] = run_dataset("Hillstrom", tr, te, n_instances=50, B=20, seed=42)

    out = {
        "experiment": "counterfactual_gain_robustness",
        "description": (
            "Bootstrap model-perturbation retention, boundary analysis, and a "
            "high-budget random-search baseline for CF gain robustness."
        ),
        "n_bootstrap": 20,
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"cf_robustness_{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "cf_robustness.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")
    print(f"Done in {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
