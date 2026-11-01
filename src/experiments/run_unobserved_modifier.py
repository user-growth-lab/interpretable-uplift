"""Conditionality of the zero-attribution guarantee: unobserved modifiers (M4).

Reviewer M4 correctly notes that TESV's zero-attribution property is stated for
the *true* CATE $\\tau(X)=g(X_{S^*})$, but attribution operates on the *fitted*
$\\hat\\tau$. The interventional value function blocks correlation-induced leakage
only at attribution time; if the true modifier is \\emph{unobserved} and only a
correlated proxy is available, the fitted model must lean on the proxy, and TESV
will faithfully attribute to it---so the zero-attribution guarantee is conditional
on the true modifiers being observed.

This experiment demonstrates that boundary honestly. For each proxy correlation
$\\rho$ we compare two settings on semi-synthetic IHDP:

  * observed   : true modifiers are in X (standard setting); proxies are appended
                 but non-causal. Expected: TESV assigns ~0 mass to proxies.
  * unobserved : the true modifier columns are dropped from X, leaving only the
                 correlated proxies; the CATE model is refit. Expected: TESV
                 assigns substantial mass to the proxies the model now depends on,
                 and CATE-fit quality degrades.

Outputs results/paper_tables/unobserved_modifier.json.
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
from src.models.cate_estimators import TLearner
from src.models.tesv import TESVExplainer

OUT_DIR = "results/paper_tables"

MODIFIERS = [1, 5, 10, 15, 20]


def proxy_mass(model, train_X, test_X, proxy_indices, tesv_kwargs, seed):
    """Fraction of TESV attribution mass assigned to the proxy columns."""
    rng = np.random.default_rng(seed)
    n_expl = min(40, test_X.shape[0])
    idx = rng.choice(test_X.shape[0], size=n_expl, replace=False)
    tesv = TESVExplainer(cate_model=model, background_data=train_X, **tesv_kwargs)
    attr = tesv.explain(test_X[idx], show_progress=False)
    mean_abs = np.abs(attr).mean(axis=0)
    total = float(mean_abs.sum())
    return float(mean_abs[proxy_indices].sum() / total) if total > 0 else 0.0


def cate_fit_r2(model, X, tau_true):
    """R^2 of the fitted CATE against the ground-truth tau."""
    pred = model.predict_cate(X)
    ss_res = float(np.sum((tau_true - pred) ** 2))
    ss_tot = float(np.sum((tau_true - tau_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def run_rho(rho, n_seeds, tesv_kwargs):
    obs_mass, unobs_mass, obs_r2, unobs_r2 = [], [], [], []
    for s in range(n_seeds):
        seed = 42 + s
        train, test = make_semisynthetic_ihdp(
            replication=s,
            effect_modifier_indices=MODIFIERS,
            add_confounded_features=True,
            confound_corr=rho,
            signal_strength=2.0,
            noise_scale=0.3,
            seed=seed,
        )
        d_full = train.X.shape[1]
        n_mod = len(MODIFIERS)
        proxy_full = list(range(d_full - n_mod, d_full))  # appended proxy columns

        # --- Observed setting: true modifiers present ---
        m_obs = TLearner()
        m_obs.fit(train.X, train.T, train.Y)
        obs_mass.append(proxy_mass(m_obs, train.X, test.X, proxy_full, tesv_kwargs, seed))
        obs_r2.append(cate_fit_r2(m_obs, test.X, test.tau))

        # --- Unobserved setting: drop the true modifier columns ---
        keep = [i for i in range(d_full) if i not in MODIFIERS]
        proxy_unobs = [keep.index(p) for p in proxy_full]
        Xtr_u, Xte_u = train.X[:, keep], test.X[:, keep]
        m_unobs = TLearner()
        m_unobs.fit(Xtr_u, train.T, train.Y)
        unobs_mass.append(proxy_mass(m_unobs, Xtr_u, Xte_u, proxy_unobs, tesv_kwargs, seed))
        unobs_r2.append(cate_fit_r2(m_unobs, Xte_u, test.tau))

    def ms(a):
        return {"mean": float(np.mean(a)), "std": float(np.std(a))}

    res = {
        "confound_corr": rho,
        "proxy_mass_observed": ms(obs_mass),
        "proxy_mass_unobserved": ms(unobs_mass),
        "cate_r2_observed": ms(obs_r2),
        "cate_r2_unobserved": ms(unobs_r2),
    }
    print(f"  rho={rho:.2f} | proxy-mass obs={res['proxy_mass_observed']['mean']:.3f}"
          f" vs unobs={res['proxy_mass_unobserved']['mean']:.3f}"
          f" | CATE R2 obs={res['cate_r2_observed']['mean']:.2f}"
          f" vs unobs={res['cate_r2_unobserved']['mean']:.2f}", flush=True)
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print("M4: ZERO-ATTRIBUTION CONDITIONALITY UNDER UNOBSERVED MODIFIERS")
    print("=" * 74)

    N_SEEDS = 3
    tesv_kwargs = dict(n_background=40, n_permutations=12, seed=42)
    rhos = [0.5, 0.7, 0.9]

    results = [run_rho(rho, N_SEEDS, tesv_kwargs) for rho in rhos]

    out = {
        "experiment": "unobserved_modifier_conditionality",
        "description": "TESV proxy-attribution mass and CATE fit when the true "
                       "modifiers are observed vs dropped (only proxies available).",
        "modifiers": MODIFIERS,
        "n_seeds": N_SEEDS,
        "elapsed_sec": round(time.time() - t0, 1),
        "results": results,
    }
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"unobserved_modifier_{stamp}.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT_DIR, "unobserved_modifier.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone in {out['elapsed_sec']}s")


if __name__ == "__main__":
    main()
