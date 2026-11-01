"""Robustness of TESV and SCRL to non-additive effect-modifier structure (M6).

Reviewer M6 notes that the semi-synthetic IHDP construction $\\tau(X)=\\sum_{j\\in
S^*} X_j+\\varepsilon$ is linear-additive, a structure that may favour Shapley-style
marginal-contribution attribution (TESV) and threshold-conjunction rules (SCRL).
We test whether the method's ability to recover the true effect modifiers survives
non-additive ground truth by re-running under four constructions of increasing
non-additivity (fixed modifier set $S^*$ for comparability):

  * linear-additive   : tau = sum_j X_j                       (paper default)
  * nonlinear         : tau = sum_j sin(pi X_j)
  * interaction       : tau = sum_j X_j + sum_{(i,k)} X_i X_k
  * nonlinear+interact: tau = sum_j sin(pi X_j) + sum_{(i,k)} X_i X_k

Metrics (higher is better): TESV Top-5 recall of the true modifiers, the fraction
of total TESV attribution mass concentrated on the true modifiers, and SCRL's
recovery of the true modifiers in its rule set.

Outputs results/paper_tables/nonlinear_modifiers.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.datasets import make_semisynthetic_ihdp
from src.metrics.fidelity import top_k_effect_modifier_recall
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL
from src.models.tesv import TESVExplainer

OUT_DIR = "results/paper_tables"

# Fixed modifier set + interaction pairs (among the modifiers) for comparability.
MODIFIERS = [1, 5, 10, 15, 20]
INTERACTIONS = [(1, 5), (10, 15)]


def modifier_mass(mean_abs_attr, modifiers):
    """Fraction of total attribution mass concentrated on the true modifiers."""
    total = float(mean_abs_attr.sum())
    if total <= 0:
        return 0.0
    return float(mean_abs_attr[modifiers].sum() / total)


def scrl_modifier_hit(scrl, modifiers):
    """Fraction of true modifiers that appear in at least one SCRL rule condition."""
    used = set()
    for rule in scrl.rules:
        for feat_idx, _, _ in rule.conditions:
            used.add(int(feat_idx))
    return len(used & set(modifiers)) / len(modifiers)


def run_condition(label, nonlinear, interactions, n_seeds, tesv_kwargs, scrl_kwargs):
    recalls, masses, hits = [], [], []
    for s in range(n_seeds):
        seed = 42 + s
        train, test = make_semisynthetic_ihdp(
            replication=s,
            effect_modifier_indices=MODIFIERS,
            nonlinear=nonlinear,
            interaction_terms=interactions,
            signal_strength=2.0,
            noise_scale=0.3,
            seed=seed,
        )
        model = TLearner()
        model.fit(train.X, train.T, train.Y)

        # --- TESV attribution ---
        rng = np.random.default_rng(seed)
        n_expl = min(40, test.n)
        idx = rng.choice(test.n, size=n_expl, replace=False)
        tesv = TESVExplainer(cate_model=model, background_data=train.X, **tesv_kwargs)
        attr = tesv.explain(test.X[idx], show_progress=False)  # (n_expl, d)
        mean_abs = np.abs(attr).mean(axis=0)

        recalls.append(top_k_effect_modifier_recall(attr, MODIFIERS, k=len(MODIFIERS)))
        masses.append(modifier_mass(mean_abs, MODIFIERS))

        # --- SCRL modifier recovery ---
        scrl = SCRL(cate_model=model, seed=seed, **scrl_kwargs)
        scrl.fit(train.X, train.T, train.Y)
        hits.append(scrl_modifier_hit(scrl, MODIFIERS))

    def ms(a):
        return {"mean": float(np.mean(a)), "std": float(np.std(a))}

    res = {
        "condition": label,
        "tesv_top5_recall": ms(recalls),
        "tesv_modifier_mass": ms(masses),
        "scrl_modifier_hit": ms(hits),
    }
    print(f"  {label:20s} recall={res['tesv_top5_recall']['mean']:.2f}"
          f"+/-{res['tesv_top5_recall']['std']:.2f}"
          f" | mod-mass={res['tesv_modifier_mass']['mean']:.2f}"
          f" | scrl_hit={res['scrl_modifier_hit']['mean']:.2f}", flush=True)
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print("M6: ROBUSTNESS TO NON-ADDITIVE EFFECT-MODIFIER STRUCTURE")
    print("=" * 74)

    N_SEEDS = 3
    tesv_kwargs = dict(n_background=40, n_permutations=20, seed=42)
    scrl_kwargs = dict(n_rules=5, max_rule_length=2, min_support=0.05,
                       max_coverage=0.40, overlap_ratio=0.1, n_candidates=6,
                       n_bootstrap=3, sample_split=True)

    conditions = [
        ("linear-additive", False, None),
        ("nonlinear", True, None),
        ("interaction", False, INTERACTIONS),
        ("nonlinear+interact", True, INTERACTIONS),
    ]

    results = []
    for label, nonlinear, interactions in conditions:
        res = run_condition(label, nonlinear, interactions, N_SEEDS,
                            tesv_kwargs, scrl_kwargs)
        results.append(res)

    out = {
        "experiment": "nonlinear_modifiers_robustness",
        "description": "TESV modifier recall / mass and SCRL modifier recovery "
                       "under additive vs non-additive ground-truth CATE.",
        "modifiers": MODIFIERS,
        "interactions": INTERACTIONS,
        "n_seeds": N_SEEDS,
        "elapsed_sec": round(time.time() - t0, 1),
        "conditions": results,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"nonlinear_modifiers_{stamp}.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "nonlinear_modifiers.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
