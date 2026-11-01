"""Head-to-head subgroup comparison: SCRL vs CDT vs Causal Tree (reviewer M2).

We provide a numerical head-to-head against the concurrent subgroup methods on a
common, reviewer-endorsed metric: the Jaccard Subgroup Similarity Index (from
CDT, Huang et al. 2025, and already borrowed for Definition 5b).

Fair protocol:
  * On EACH bootstrap resample we REFIT the CATE teacher (T-Learner). This is
    essential: with a single fixed shared teacher, a deterministic distillation
    (CDT) would be trivially reproducible (Jaccard ~ 1) simply because it
    re-distils the same frozen function, which does not reflect subgroup-
    discovery stability. Refitting exposes ALL methods to the same estimation +
    discovery variability.
  * SCRL discovers its natural positive-uplift rule union; the tree baselines
    (CDT = distillation of the refit teacher; Causal Tree = standalone pseudo-
    outcome partition) are matched to SCRL's coverage per bootstrap by taking
    their highest-uplift samples up to the same fraction, so the Jaccard
    overlaps are directly comparable.

Two Jaccard-based metrics are reported:
  * Stability -- mean pairwise Jaccard of the discovered subgroup across
    bootstraps (reproducibility under resampling).
  * Targeting precision (semi-synthetic IHDP only) -- fraction of the discovered
    subgroup that lies in the ground-truth top-30% high-uplift set (correctness).

Outputs results/paper_tables/subgroup_stability.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datasets import load_criteo, load_hillstrom, make_semisynthetic_ihdp
from src.models.baselines import CausalTreeBaseline
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL

OUT_DIR = "results/paper_tables"


def topk_mask(scores, k):
    """Boolean mask of the top-k highest-scoring samples."""
    k = max(1, min(int(k), len(scores)))
    order = np.argsort(-scores)
    m = np.zeros(len(scores), dtype=bool)
    m[order[:k]] = True
    return m


def scrl_subgroup(scrl, tau_full, X_full):
    """SCRL natural high-uplift subgroup = union of positive-uplift rules."""
    gm = float(tau_full.mean())
    pos = [r for r in scrl.rules if r.tau_mean >= gm] or scrl.rules
    cov = np.zeros(X_full.shape[0], dtype=bool)
    for r in pos:
        cov |= r.covers(X_full)
    return cov


def mean_pairwise_jaccard(masks):
    """Return (mean, std) of pairwise Jaccard similarities across subgroups."""
    sims = []
    B = len(masks)
    for i in range(B):
        for j in range(i + 1, B):
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            sims.append(inter / union if union > 0 else 1.0)
    if not sims:
        return 0.0, 0.0
    return float(np.mean(sims)), float(np.std(sims))


def evaluate_seed(train_ds, seed, B, scrl_kwargs, tree_kwargs, true_tau=None):
    X, T, Y = train_ds.X, train_ds.T, train_ds.Y
    n = X.shape[0]

    # Reference teacher on full data: used to score predicted uplift when no
    # ground truth is available.
    ref = TLearner()
    ref.fit(X, T, Y)
    tau_ref = ref.predict_cate(X)

    # Ground-truth top-30% high-uplift set for the precision metric.
    true_top = None
    if true_tau is not None:
        true_top = topk_mask(true_tau, int(round(0.30 * n)))

    rng = np.random.default_rng(seed)
    methods = ["SCRL", "CDT", "CausalTree"]
    masks = {m: [] for m in methods}
    cov = {m: [] for m in methods}
    upl = {m: [] for m in methods}
    prec = {m: [] for m in methods}

    for b in range(B):
        _t = time.time()
        idx = rng.choice(n, size=n, replace=True)
        Xb, Tb, Yb = X[idx], T[idx], Y[idx]

        # Refit the CATE teacher on this bootstrap (shared by SCRL and CDT).
        teacher = TLearner()
        teacher.fit(Xb, Tb, Yb)
        tau_b_full = teacher.predict_cate(X)

        # SCRL: natural positive-uplift rule union.
        scrl = SCRL(cate_model=teacher, seed=int(rng.integers(1e9)), **scrl_kwargs)
        scrl.fit(Xb, Tb, Yb)
        m_scrl = scrl_subgroup(scrl, tau_b_full, X)
        c = float(m_scrl.mean())
        k = int(round(c * n))  # coverage budget matched to SCRL for this bootstrap

        # CDT: distil the refit teacher into a tree; match SCRL coverage.
        cdt = DecisionTreeRegressor(random_state=int(rng.integers(1e9)), **tree_kwargs)
        cdt.fit(Xb, teacher.predict_cate(Xb))
        m_cdt = topk_mask(cdt.predict(X), k)

        # Causal Tree: standalone pseudo-outcome partition; match SCRL coverage.
        ct = CausalTreeBaseline(seed=int(rng.integers(1e9)), **tree_kwargs)
        ct.fit(Xb, Tb, Yb)
        m_ct = topk_mask(ct.tree.predict(X), k)

        for name, mask in [("SCRL", m_scrl), ("CDT", m_cdt), ("CausalTree", m_ct)]:
            masks[name].append(mask)
            cov[name].append(float(mask.mean()))
            if mask.sum() > 0:
                if true_tau is not None:
                    upl[name].append(float(true_tau[mask].mean()))
                    prec[name].append(float(true_top[mask].mean()))
                else:
                    upl[name].append(float(tau_ref[mask].mean()))
        print(f"      [b {b + 1}/{B}] {time.time() - _t:.1f}s", flush=True)

    out = {}
    for m in methods:
        jm, js = mean_pairwise_jaccard(masks[m])
        out[m] = {
            "jaccard": jm,
            "jaccard_pairstd": js,
            "mean_subgroup_uplift": float(np.mean(upl[m])) if upl[m] else 0.0,
            "mean_coverage": float(np.mean(cov[m])),
            "precision": float(np.mean(prec[m])) if prec[m] else None,
        }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print("SUBGROUP HEAD-TO-HEAD: SCRL vs CDT vs Causal Tree")
    print("Metric: Jaccard Subgroup Similarity Index (fair: refit teacher +")
    print("coverage-matched) + ground-truth targeting precision on IHDP")
    print("=" * 74)

    B = 6
    scrl_kwargs = dict(n_rules=5, max_rule_length=2, min_support=0.05,
                       max_coverage=0.40, overlap_ratio=0.1, n_candidates=6,
                       n_bootstrap=3, sample_split=True)
    tree_kwargs = dict(max_depth=3, min_samples_leaf=30)
    methods = ["SCRL", "CDT", "CausalTree"]

    dataset_specs = [
        ("IHDP (semi-synth)", "ihdp"),
        ("Hillstrom", "hillstrom"),
        ("Criteo", "criteo"),
    ]

    all_results = {}
    for label, key in dataset_specs:
        print(f"\n--- {label} ---")
        try:
            if key == "ihdp":
                train_ds, _ = make_semisynthetic_ihdp(replication=0, n_modifiers=5, seed=42)
                true_tau = train_ds.tau
            elif key == "hillstrom":
                train_ds, _ = load_hillstrom(train_frac=0.8, seed=42)
                if train_ds.n > 2500:
                    r = np.random.default_rng(42)
                    sel = r.choice(train_ds.n, 2500, replace=False)
                    train_ds.X, train_ds.T, train_ds.Y = (
                        train_ds.X[sel], train_ds.T[sel], train_ds.Y[sel])
                true_tau = None
            else:
                train_ds, _ = load_criteo(sample_n=6000, train_frac=0.8, seed=42)
                if train_ds.n > 2500:
                    r = np.random.default_rng(42)
                    sel = r.choice(train_ds.n, 2500, replace=False)
                    train_ds.X, train_ds.T, train_ds.Y = (
                        train_ds.X[sel], train_ds.T[sel], train_ds.Y[sel])
                true_tau = None
        except Exception as e:  # pragma: no cover - data guard
            print(f"  [skip] {e}")
            continue

        res = evaluate_seed(train_ds, 42, B, scrl_kwargs, tree_kwargs, true_tau=true_tau)
        all_results[label] = res
        for m in methods:
            p = res[m]["precision"]
            ps = f", precision={p:.3f}" if p is not None else ""
            print(f"    {m:11s} J={res[m]['jaccard']:.3f}+/-{res[m]['jaccard_pairstd']:.3f}"
                  f" | uplift={res[m]['mean_subgroup_uplift']:.3f}"
                  f" | cov={res[m]['mean_coverage']:.2f}{ps}")

    out = {
        "experiment": "subgroup_head_to_head",
        "metric": "Jaccard Subgroup Similarity Index + IHDP targeting precision",
        "protocol": "refit CATE teacher per bootstrap; coverage matched to SCRL",
        "n_bootstrap": B,
        "elapsed_sec": round(time.time() - t0, 1),
        "results": all_results,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"subgroup_stability_{stamp}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "subgroup_stability.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")
    print(f"Done in {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
