"""Feature-independence diagnostics for the interventional (TESV) assumption.

The interventional Shapley value function underlying TESV assumes that features
can be intervened upon independently---i.e., setting X_S = x_S does not
structurally force changes in X_{\bar S}. This assumption is appropriate when
covariates are statistically *correlated* but not *causally linked*.

While causal direction cannot be tested from observational data, we can bound
the concern empirically by quantifying, on each real benchmark:

  (1) Marginal dependence   -- pairwise Pearson correlation |r|. High marginal
      dependence is *expected* and is exactly what the interventional
      (correlation-blocking) value function is designed to neutralise.
  (2) Conditional dependence -- pairwise partial correlation controlling for all
      other covariates (via the precision matrix). This isolates the *direct*
      linear dependence not explained by other features; a large partial
      correlation flags a pair that may share a direct (possibly causal) link
      warranting domain scrutiny.
  (3) Nonlinear dependence   -- distance correlation (dCor) on a subsample, which
      detects nonlinear statistical dependence (an HSIC-style measure).

The key diagnostic is the *fraction of feature pairs with high partial /
nonlinear dependence*: when this fraction is small, most covariates are
statistically correlated but not directly linked, which is precisely the regime
where interventional TESV is appropriate; the small residual fraction is where
the correlation-blocking caveat of Section 4.1 applies.

Outputs a JSON table to results/paper_tables/feature_independence.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datasets import load_criteo, load_hillstrom, load_ihdp

OUT_DIR = "results/paper_tables"


# ---------------------------------------------------------------------------
# Dependence measures
# ---------------------------------------------------------------------------

def marginal_correlation(X: np.ndarray) -> np.ndarray:
    """Return the (d, d) Pearson correlation matrix (constant cols -> 0)."""
    d = X.shape[1]
    std = X.std(axis=0)
    keep = std > 1e-12
    corr = np.zeros((d, d))
    if keep.sum() >= 2:
        sub = np.corrcoef(X[:, keep], rowvar=False)
        idx = np.where(keep)[0]
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                corr[ia, ib] = sub[a, b]
    np.fill_diagonal(corr, 1.0)
    return corr


def partial_correlation(X: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """Partial correlation matrix via the (regularised) precision matrix.

    partial_corr(i, j) = -P_ij / sqrt(P_ii * P_jj), where P = inv(cov(X)).
    Measures the direct linear dependence between i and j after removing the
    linear influence of every other covariate. A ridge is added to the
    covariance to guarantee a well-conditioned inverse under rank-deficiency
    (e.g., collinear binary IHDP covariates).
    """
    d = X.shape[1]
    cov = np.cov(X, rowvar=False)
    # Scale the ridge to the covariance magnitude for stable conditioning.
    ridge_abs = ridge * float(np.mean(np.diag(cov)) + 1e-12)
    cov = cov + ridge_abs * np.eye(d)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prec = np.linalg.pinv(cov)
        dp = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
        pcorr = -prec / np.outer(dp, dp)
    pcorr = np.nan_to_num(pcorr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(pcorr, 1.0)
    return np.clip(pcorr, -1.0, 1.0)


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Szekely distance correlation between two 1-D samples (nonlinear dep.)."""
    n = x.shape[0]
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom))


def _offdiag(M: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle of a symmetric matrix."""
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def diagnose_dataset(
    name: str,
    X: np.ndarray,
    feature_names,
    n_dcor_rows: int = 800,
    n_dcor_pairs: int = 150,
    seed: int = 42,
) -> dict:
    """Compute marginal / partial / nonlinear dependence diagnostics for X."""
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # (1) Marginal Pearson correlation
    corr = marginal_correlation(X)
    marg = np.abs(_offdiag(corr))

    # (2) Partial (conditional) correlation
    pcorr = partial_correlation(X)
    part = np.abs(_offdiag(pcorr))

    # (3) Distance correlation on a subsample of rows and feature pairs
    ridx = rng.choice(n, size=min(n_dcor_rows, n), replace=False)
    Xs = X[ridx]
    all_pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    if len(all_pairs) > n_dcor_pairs:
        sel = rng.choice(len(all_pairs), size=n_dcor_pairs, replace=False)
        pairs = [all_pairs[k] for k in sel]
    else:
        pairs = all_pairs
    dcors = np.array([distance_correlation(Xs[:, i], Xs[:, j]) for i, j in pairs])

    n_pairs = int(d * (d - 1) / 2)
    result = {
        "dataset": name,
        "n_samples": int(n),
        "n_features": int(d),
        "n_pairs": n_pairs,
        # Marginal dependence
        "marg_mean_abs": float(marg.mean()),
        "marg_max_abs": float(marg.max()),
        "marg_frac_gt_0.3": float((marg > 0.3).mean()),
        "marg_frac_gt_0.5": float((marg > 0.5).mean()),
        "marg_frac_gt_0.7": float((marg > 0.7).mean()),
        # Conditional (partial) dependence -- the key diagnostic
        "part_mean_abs": float(part.mean()),
        "part_max_abs": float(part.max()),
        "part_frac_gt_0.1": float((part > 0.1).mean()),
        "part_frac_gt_0.3": float((part > 0.3).mean()),
        "part_frac_gt_0.5": float((part > 0.5).mean()),
        # Nonlinear dependence (subsampled)
        "dcor_n_rows": int(len(ridx)),
        "dcor_n_pairs": int(len(pairs)),
        "dcor_mean": float(dcors.mean()),
        "dcor_max": float(dcors.max()),
        "dcor_frac_gt_0.3": float((dcors > 0.3).mean()),
        "dcor_frac_gt_0.5": float((dcors > 0.5).mean()),
    }
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print("FEATURE-INDEPENDENCE DIAGNOSTICS FOR THE INTERVENTIONAL ASSUMPTION")
    print("=" * 74)

    datasets = []

    # IHDP (real covariates; semi-synthetic outcome). Use the raw covariates.
    try:
        tr, _ = load_ihdp(replication=0, standardize=True, seed=42)
        datasets.append(("IHDP", tr.X, tr.feature_names))
    except Exception as e:  # pragma: no cover - data availability guard
        print(f"[skip] IHDP: {e}")

    # Hillstrom (real email-marketing covariates)
    try:
        tr, _ = load_hillstrom(train_frac=0.8, standardize=True, seed=42)
        datasets.append(("Hillstrom", tr.X, tr.feature_names))
    except Exception as e:  # pragma: no cover
        print(f"[skip] Hillstrom: {e}")

    # Criteo (real ad-RCT covariates; subsample for tractability)
    try:
        tr, _ = load_criteo(sample_n=50000, train_frac=0.8, standardize=True, seed=42)
        datasets.append(("Criteo", tr.X, tr.feature_names))
    except Exception as e:  # pragma: no cover
        print(f"[skip] Criteo: {e}")

    results = []
    for name, X, fnames in datasets:
        print(f"\n--- {name}: n={X.shape[0]}, d={X.shape[1]} ---")
        res = diagnose_dataset(name, X, fnames)
        results.append(res)
        print(
            f"  marginal |r|: mean={res['marg_mean_abs']:.3f} max={res['marg_max_abs']:.3f} "
            f"| frac>0.3={res['marg_frac_gt_0.3']:.2f} frac>0.5={res['marg_frac_gt_0.5']:.2f}"
        )
        print(
            f"  partial  |r|: mean={res['part_mean_abs']:.3f} max={res['part_max_abs']:.3f} "
            f"| frac>0.1={res['part_frac_gt_0.1']:.2f} frac>0.3={res['part_frac_gt_0.3']:.2f}"
        )
        print(
            f"  dCor (nonlin): mean={res['dcor_mean']:.3f} max={res['dcor_max']:.3f} "
            f"| frac>0.3={res['dcor_frac_gt_0.3']:.2f} (rows={res['dcor_n_rows']}, pairs={res['dcor_n_pairs']})"
        )

    out = {
        "experiment": "feature_independence_diagnostic",
        "description": (
            "Marginal (Pearson), conditional (partial correlation), and nonlinear "
            "(distance correlation) dependence diagnostics quantifying the degree to "
            "which the feature-independence assumption of interventional TESV holds."
        ),
        "elapsed_sec": round(time.time() - t0, 1),
        "datasets": results,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"feature_independence_{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    # Also write a stable-named copy for easy reference.
    with open(os.path.join(OUT_DIR, "feature_independence.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")
    print(f"Done in {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
