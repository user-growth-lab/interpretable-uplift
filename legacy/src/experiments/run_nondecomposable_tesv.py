"""Non-decomposable TESV as a first-class, multi-dataset experiment (reviewers M2/C2/C3).

Reviewer M2 asks that TESV's core differentiator---direct applicability to
non-decomposable CATE estimators (R-/DR-Learner), where the outcome-differencing
approach must re-train separate outcome models and thereby lose fidelity to the
deployed estimator---be established as a first-class experiment across multiple
datasets and seeds, not a single-point number.

Reviewers C2/C3 (round 4) additionally require: (i) TUNED non-decomposable
estimators (the earlier R-Learner used a linear Ridge outcome nuisance, giving a
poor CATE fit, MSE~9; we replace it with a flexible GBT outcome nuisance and
report the resulting CATE MSE against ground truth on IHDP); (ii) higher
statistical power (>=20 seeds); and (iii) confidence intervals and effect sizes
alongside the paired t-test, so the flagship claim is not resting on two
marginal cells at 5 seeds.

Protocol. For each non-decomposable estimator $\\hat\\tau$ (tuned R-Learner,
tuned DR-Learner) fit on (X, T, Y):
  * TESV: attribute directly on $\\hat\\tau(\\cdot)$.
  * Re-train-and-difference: fit a SEPARATE T-Learner $(\\tilde f_1,\\tilde f_0)$ on
    the same data and difference its interventional Shapley values.
Both attribution sets are scored by Causal Fidelity---the Pearson correlation
between attribution magnitudes and the perturbation response of the ORIGINAL
non-decomposable surface $\\hat\\tau$ (not the re-trained surface). Because
$\\tilde f_1-\\tilde f_0\\neq\\hat\\tau$ in general, differencing is less faithful to the
deployed estimator. We report mean$\\pm$std over 20 seeds, a paired $t$-test, the
95% CI of the paired fidelity difference, and Cohen's d; on semi-synthetic IHDP
we additionally report Top-5 modifier recall and the estimator's CATE MSE.

Datasets: semi-synthetic IHDP, Hillstrom (real), Criteo (real subsample).
Outputs results/paper_tables/nondecomposable_tesv.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datasets import load_criteo, load_hillstrom, make_semisynthetic_ihdp
from src.metrics.fidelity import causal_fidelity_score, top_k_effect_modifier_recall
from src.models.baselines import InterventionalSHAPBaseline
from src.models.cate_estimators import DRLearner, RLearner, TLearner
from src.models.tesv import TESVExplainer

OUT_DIR = "results/paper_tables"
N_BG = 40
N_PERM = 15
N_INSTANCES = 40


def make_rlearner():
    """Tuned R-Learner: a flexible GBT outcome nuisance replaces the default
    linear Ridge, which was the main source of the high-variance pseudo-outcome
    and the poor CATE fit flagged by reviewer C3."""
    return RLearner(
        outcome_learner=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
    )


def make_drlearner():
    """Tuned DR-Learner: flexible GBT outcome nuisances (n=100)."""
    return DRLearner(
        outcome_learner=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
    )


def run_case(train, test, est_factory, seed, mods=None):
    """Return (fid_tesv, fid_diff, recall_tesv, recall_diff, mse) for one case."""
    target = est_factory()
    target.fit(train.X, train.T, train.Y)
    X_bg = train.X

    rng = np.random.default_rng(seed)
    n_te = min(N_INSTANCES, test.n)
    idx = rng.choice(test.n, size=n_te, replace=False)
    X_te = test.X[idx]

    # TESV: attribute directly on the non-decomposable estimator.
    tesv = TESVExplainer(cate_model=target, background_data=X_bg,
                         n_background=N_BG, n_permutations=N_PERM, seed=seed)
    attr_tesv = tesv.explain(X_te, show_progress=False)
    fid_tesv = causal_fidelity_score(X_te, target, attr_tesv, perturbation_std=0.5, seed=seed)

    # Re-train-and-difference: separate T-Learner, differenced; scored against the
    # ORIGINAL non-decomposable surface (target), not the re-trained one.
    tl = TLearner()
    tl.fit(train.X, train.T, train.Y)
    diff = InterventionalSHAPBaseline(cate_model=tl, background_data=X_bg,
                                      n_background=N_BG, seed=seed)
    attr_diff = diff.explain(X_te, n_permutations=N_PERM)
    fid_diff = causal_fidelity_score(X_te, target, attr_diff, perturbation_std=0.5, seed=seed)

    # CATE fit quality (MSE vs ground-truth tau) where available (IHDP).
    mse = None
    if getattr(test, "tau", None) is not None:
        mse = float(np.mean((target.predict_cate(test.X) - test.tau) ** 2))

    rec_tesv = rec_diff = None
    if mods is not None:
        rec_tesv = top_k_effect_modifier_recall(attr_tesv, mods, k=5)
        rec_diff = top_k_effect_modifier_recall(attr_diff, mods, k=5)
    return fid_tesv, fid_diff, rec_tesv, rec_diff, mse


def _subsample(ds, n, seed):
    if ds.n > n:
        r = np.random.default_rng(seed)
        sel = r.choice(ds.n, n, replace=False)
        ds.X, ds.T, ds.Y = ds.X[sel], ds.T[sel], ds.Y[sel]
        if ds.tau is not None:
            ds.tau = ds.tau[sel]
    return ds


def load_case(key, seed):
    """Return (train, test, mods) for a dataset key."""
    if key == "IHDP":
        train, test = make_semisynthetic_ihdp(n_modifiers=5, signal_strength=2.0,
                                              noise_scale=0.3, seed=seed)
        return train, test, train.effect_modifier_indices
    if key == "Hillstrom":
        train, test = load_hillstrom(train_frac=0.8, seed=seed)
        return _subsample(train, 3000, seed), _subsample(test, 1500, seed), None
    if key == "Criteo":
        train, test = load_criteo(sample_n=6000, train_frac=0.8, seed=seed)
        return _subsample(train, 3000, seed), _subsample(test, 1500, seed), None
    raise ValueError(key)


def paired_stats(fid_t, fid_d):
    """Paired t-test + 95% CI of the mean difference + Cohen's d (paired)."""
    a, b = np.asarray(fid_t), np.asarray(fid_d)
    d = a - b
    n = len(d)
    mean_diff = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 0 else 0.0
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    ci = (mean_diff - tcrit * sem, mean_diff + tcrit * sem)
    cohen_d = mean_diff / sd if sd > 0 else 0.0
    tstat, pval = stats.ttest_rel(a, b)
    return mean_diff, ci, float(cohen_d), float(tstat), float(pval)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 78)
    print("C2/C3: TUNED NON-DECOMPOSABLE TESV vs RE-TRAIN-AND-DIFFERENCE (20 seeds)")
    print("Fidelity to the original non-decomposable surface; paired t-test + 95% CI + d")
    print("=" * 78)

    seeds = list(range(42, 62))  # 20 seeds
    estimators = [("R-Learner", make_rlearner), ("DR-Learner", make_drlearner)]
    datasets = ["IHDP", "Hillstrom", "Criteo"]

    results = {}
    for ds_key in datasets:
        for est_name, est_factory in estimators:
            fid_t, fid_d, rec_t, rec_d, mses = [], [], [], [], []
            for seed in seeds:
                try:
                    train, test, mods = load_case(ds_key, seed)
                except Exception as e:  # pragma: no cover - data guard
                    print(f"  [skip {ds_key}/{est_name} seed {seed}] {e}")
                    continue
                ft, fd, rt, rd, mse = run_case(train, test, est_factory, seed, mods=mods)
                fid_t.append(ft); fid_d.append(fd)
                if rt is not None:
                    rec_t.append(rt); rec_d.append(rd)
                if mse is not None:
                    mses.append(mse)
            if not fid_t:
                continue
            mean_diff, ci, cohen_d, tstat, pval = paired_stats(fid_t, fid_d)
            gain = (np.mean(fid_t) - np.mean(fid_d)) / abs(np.mean(fid_d)) * 100 if np.mean(fid_d) != 0 else 0.0
            key = f"{ds_key}/{est_name}"
            results[key] = {
                "fid_tesv_mean": float(np.mean(fid_t)), "fid_tesv_std": float(np.std(fid_t)),
                "fid_diff_mean": float(np.mean(fid_d)), "fid_diff_std": float(np.std(fid_d)),
                "rel_gain_pct": float(gain),
                "mean_diff": float(mean_diff),
                "ci95_low": float(ci[0]), "ci95_high": float(ci[1]),
                "cohen_d": float(cohen_d),
                "paired_t": float(tstat), "p_value": float(pval),
                "cate_mse_mean": float(np.mean(mses)) if mses else None,
                "recall_tesv": float(np.mean(rec_t)) if rec_t else None,
                "recall_diff": float(np.mean(rec_d)) if rec_d else None,
            }
            mse_str = f" | MSE={np.mean(mses):.3f}" if mses else ""
            rec_str = (f" | recall TESV={np.mean(rec_t):.2f} vs diff={np.mean(rec_d):.2f}"
                       if rec_t else "")
            print(f"  {key:22s} TESV={np.mean(fid_t):.3f}±{np.std(fid_t):.3f}"
                  f" vs diff={np.mean(fid_d):.3f}±{np.std(fid_d):.3f}"
                  f" (+{gain:.1f}%, p={pval:.3f}, d={cohen_d:.2f},"
                  f" 95%CI[{ci[0]:.3f},{ci[1]:.3f}]){mse_str}{rec_str}", flush=True)

    out = {
        "experiment": "nondecomposable_tesv_tuned_20seed",
        "metric": "Causal Fidelity to the original non-decomposable CATE surface",
        "n_seeds": len(seeds),
        "tuning": "R-/DR-Learner with flexible GBT outcome nuisance (n=100)",
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"nondecomposable_tesv_tuned_{stamp}.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "nondecomposable_tesv_tuned.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
