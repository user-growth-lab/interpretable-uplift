"""IPW Offline Policy Evaluation for SCRL-targeted strategies.

Computes the policy value of different targeting strategies using
Inverse Propensity Weighting on RCT data (MegaFon and Hillstrom).

For RCT data with known propensity e(X) = P(T=1), the IPW policy value is:
    V(pi) = (1/n) * sum_i [ pi(Xi)*Ti*Yi/e(Xi) + (1-pi(Xi))*(1-Ti)*Yi/(1-e(Xi)) ]

This gives an unbiased estimate of the expected outcome under policy pi.
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.datasets import load_hillstrom
from src.models.cate_estimators import TLearner
from src.models.scrl import SCRL


def ipw_policy_value(Y, T, policy_mask, propensity):
    """Compute IPW policy value.

    Args:
        Y: Observed outcomes (n,)
        T: Treatment indicators (n,)
        policy_mask: Boolean mask - True = treat, False = don't treat (n,)
        propensity: P(T=1) for each unit (n,) or scalar for RCT

    Returns:
        IPW estimated policy value (expected outcome under policy)
    """
    n = len(Y)
    if np.isscalar(propensity):
        propensity = np.full(n, propensity)

    # IPW: for treated units in policy, weight by 1/e; for control units not in policy, weight by 1/(1-e)
    value = 0.0
    for i in range(n):
        if policy_mask[i]:  # Policy says treat
            if T[i] == 1:  # Actually treated
                value += Y[i] / propensity[i]
        else:  # Policy says don't treat
            if T[i] == 0:  # Actually control
                value += Y[i] / (1 - propensity[i])
    return value / n


def ipw_policy_uplift(Y, T, policy_mask, propensity):
    """Compute the incremental value of a targeting policy over treat-none.

    Returns V(policy) - V(treat-none), representing the causal value added by targeting.
    """
    v_policy = ipw_policy_value(Y, T, policy_mask, propensity)
    v_none = ipw_policy_value(Y, T, np.zeros(len(Y), dtype=bool), propensity)
    v_all = ipw_policy_value(Y, T, np.ones(len(Y), dtype=bool), propensity)
    return {
        "policy_value": float(v_policy),
        "treat_none_value": float(v_none),
        "treat_all_value": float(v_all),
        "incremental_vs_none": float(v_policy - v_none),
        "incremental_vs_all": float(v_policy - v_all),
        "policy_coverage": float(policy_mask.mean()),
    }


def run_megafon_ipw(n_seeds=3):
    """IPW policy evaluation on MegaFon RCT data."""
    print("=" * 60)
    print("IPW Policy Evaluation: MegaFon")
    print("=" * 60)

    # Load MegaFon data
    try:
        from src.data.datasets import load_megafon
        results_all = {"treat_all": [], "treat_none": [], "scrl_rule1": [], "scrl_union": [], "top20_tau": []}

        for seed in range(n_seeds):
            print(f"  Seed {seed+1}/{n_seeds}")
            train_ds, test_ds = load_megafon(sample_n=100000, seed=seed)

            # Fit CATE
            base = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
            cate_model = TLearner(base_learner=base)
            cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

            # Propensity (RCT: approximately 0.5)
            e = train_ds.T.mean()

            # Predict CATE on test
            tau_test = cate_model.predict_cate(test_ds.X)

            # Run SCRL on training data
            scrl = SCRL(cate_model=cate_model, n_rules=5, max_rule_length=2,
                        n_bootstrap=10, n_candidates=20, max_coverage=0.40,
                        sample_split=True, seed=seed)
            scrl.fit(train_ds.X, train_ds.T, train_ds.Y)

            # Policies on TEST set
            # 1. Treat all
            policy_all = np.ones(test_ds.n, dtype=bool)
            # 2. Treat none
            policy_none = np.zeros(test_ds.n, dtype=bool)
            # 3. SCRL Rule 1 (top rule)
            if scrl.rules:
                policy_rule1 = scrl.rules[0].covers(test_ds.X)
            else:
                policy_rule1 = policy_none.copy()
            # 4. SCRL union
            policy_union = np.zeros(test_ds.n, dtype=bool)
            for rule in scrl.rules:
                policy_union |= rule.covers(test_ds.X)
            # 5. Top-20% by predicted tau
            threshold = np.percentile(tau_test, 80)
            policy_top20 = tau_test >= threshold

            # Evaluate all policies
            r_all = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_all, e)
            r_none = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_none, e)
            r_rule1 = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_rule1, e)
            r_union = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_union, e)
            r_top20 = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_top20, e)

            results_all["treat_all"].append(r_all)
            results_all["treat_none"].append(r_none)
            results_all["scrl_rule1"].append(r_rule1)
            results_all["scrl_union"].append(r_union)
            results_all["top20_tau"].append(r_top20)

        return results_all
    except Exception as ex:
        print(f"  MegaFon not available: {ex}")
        return None


def run_hillstrom_ipw(n_seeds=3):
    """IPW policy evaluation on Hillstrom RCT data."""
    print("\n" + "=" * 60)
    print("IPW Policy Evaluation: Hillstrom")
    print("=" * 60)

    results_all = {"treat_all": [], "treat_none": [], "scrl_rule1": [], "scrl_union": [], "top20_tau": []}

    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}")
        train_ds, test_ds = load_hillstrom(seed=seed)

        # Fit CATE
        base = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed)
        cate_model = TLearner(base_learner=base)
        cate_model.fit(train_ds.X, train_ds.T, train_ds.Y)

        # Propensity (RCT: 2/3 treated in Hillstrom - Mens + Womens email vs No Email)
        e = train_ds.T.mean()

        # Predict CATE on test
        tau_test = cate_model.predict_cate(test_ds.X)

        # Run SCRL
        scrl = SCRL(cate_model=cate_model, n_rules=5, max_rule_length=2,
                    n_bootstrap=10, n_candidates=20, max_coverage=0.40,
                    sample_split=True, seed=seed)
        scrl.fit(train_ds.X, train_ds.T, train_ds.Y)

        # Policies on TEST set
        policy_all = np.ones(test_ds.n, dtype=bool)
        policy_none = np.zeros(test_ds.n, dtype=bool)
        if scrl.rules:
            policy_rule1 = scrl.rules[0].covers(test_ds.X)
        else:
            policy_rule1 = policy_none.copy()
        policy_union = np.zeros(test_ds.n, dtype=bool)
        for rule in scrl.rules:
            policy_union |= rule.covers(test_ds.X)
        threshold = np.percentile(tau_test, 80)
        policy_top20 = tau_test >= threshold

        # Evaluate
        r_all = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_all, e)
        r_rule1 = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_rule1, e)
        r_union = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_union, e)
        r_top20 = ipw_policy_uplift(test_ds.Y, test_ds.T, policy_top20, e)

        results_all["treat_all"].append(r_all)
        results_all["scrl_rule1"].append(r_rule1)
        results_all["scrl_union"].append(r_union)
        results_all["top20_tau"].append(r_top20)

    return results_all


def summarize(results, dataset_name):
    """Print summary table."""
    print(f"\n{'='*60}")
    print(f"Summary: {dataset_name}")
    print(f"{'Policy':<20} {'Value':<12} {'vs None':<12} {'Coverage':<10}")
    print("-" * 54)
    for policy_name, runs in results.items():
        if not runs:
            continue
        values = [r["policy_value"] for r in runs]
        inc = [r["incremental_vs_none"] for r in runs]
        cov = [r["policy_coverage"] for r in runs]
        print(f"{policy_name:<20} {np.mean(values):.4f}±{np.std(values):.4f}  "
              f"{np.mean(inc):.4f}±{np.std(inc):.4f}  {np.mean(cov):.2f}")


def main():
    output_path = "results/paper_tables/ipw_policy_eval.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    hillstrom_results = run_hillstrom_ipw(n_seeds=3)
    summarize(hillstrom_results, "Hillstrom")

    megafon_results = run_megafon_ipw(n_seeds=3)
    if megafon_results:
        summarize(megafon_results, "MegaFon")

    # Save
    output = {"hillstrom": hillstrom_results}
    if megafon_results:
        output["megafon"] = megafon_results

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
