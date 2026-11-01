"""Fidelity metrics for CATE explanations.

Implements evaluation criteria for feature attributions, rule sets, and
counterfactual explanations when ground-truth effect modifiers are known or
when only observational fidelity tests are possible.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.metrics import auc, roc_curve


def top_k_effect_modifier_recall(
    attribution_scores: np.ndarray,
    true_effect_modifiers: List[int],
    k: Optional[int] = None,
) -> float:
    """Fraction of true effect modifiers recovered in top-K attributions.

    Args:
        attribution_scores: Absolute attribution scores of shape (d,) or (n, d).
            If 2D, scores are averaged across instances.
        true_effect_modifiers: List of ground-truth effect modifier indices.
        k: Number of top features to consider. Defaults to len(true_effect_modifiers).

    Returns:
        Recall in [0, 1].
    """
    if attribution_scores.ndim == 2:
        scores = np.abs(attribution_scores).mean(axis=0)
    else:
        scores = np.abs(attribution_scores)

    if k is None:
        k = len(true_effect_modifiers)
    k = min(k, scores.shape[0])

    top_k = set(np.argsort(-scores)[:k])
    true_set = set(true_effect_modifiers)
    return len(top_k & true_set) / len(true_set)


def attribution_rank_correlation(
    attribution_scores: np.ndarray,
    true_importance: np.ndarray,
) -> float:
    """Spearman rank correlation between estimated and true feature importance.

    Args:
        attribution_scores: Shape (d,) or (n, d).
        true_importance: Shape (d,) with ground-truth importance scores.

    Returns:
        Spearman correlation in [-1, 1].
    """
    from scipy.stats import spearmanr

    if attribution_scores.ndim == 2:
        scores = np.abs(attribution_scores).mean(axis=0)
    else:
        scores = np.abs(attribution_scores)

    corr, _ = spearmanr(scores, true_importance)
    return float(corr)


def causal_fidelity_score(
    X: np.ndarray,
    cate_model,
    attributions: np.ndarray,
    n_bins: int = 10,
    perturbation_std: float = 1.0,
    seed: int = 42,
) -> float:
    """Measure consistency between attribution ranking and CATE perturbation response.

    For each feature, perturb it by Gaussian noise and measure the change in
    predicted CATE. The Pearson correlation between attribution magnitudes and
    CATE change magnitudes is the fidelity score.

    Args:
        X: Instances of shape (n, d).
        cate_model: Fitted CATE model.
        attributions: Attribution matrix of shape (n, d).
        n_bins: Not used; kept for API compatibility.
        perturbation_std: Standard deviation of Gaussian perturbation.
        seed: Random seed.

    Returns:
        Pearson correlation in [-1, 1].
    """
    from scipy.stats import pearsonr

    rng = np.random.default_rng(seed)
    base_cate = cate_model.predict_cate(X)

    # Average absolute attribution per feature
    mean_attr = np.abs(attributions).mean(axis=0)

    cate_changes = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        X_pert = X.copy()
        X_pert[:, j] += rng.normal(0, perturbation_std, size=X.shape[0])
        pert_cate = cate_model.predict_cate(X_pert)
        cate_changes[j] = np.abs(pert_cate - base_cate).mean()

    corr, _ = pearsonr(mean_attr, cate_changes)
    return float(corr)


def attribution_stability(
    X: np.ndarray,
    explainer,
    noise_std: float = 0.1,
    n_perturbations: int = 10,
    seed: int = 42,
) -> float:
    """Average cosine similarity of attributions under small input perturbations.

    Args:
        X: Instances of shape (n, d).
        explainer: Explainer with `explain` method.
        noise_std: Standard deviation of Gaussian noise.
        n_perturbations: Number of perturbations per instance.
        seed: Random seed.

    Returns:
        Mean cosine similarity in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    base_attr = explainer.explain(X, show_progress=False)

    similarities = []
    for _ in range(n_perturbations):
        X_noisy = X + rng.normal(0, noise_std, size=X.shape)
        noisy_attr = explainer.explain(X_noisy, show_progress=False)
        for i in range(X.shape[0]):
            a = base_attr[i]
            b = noisy_attr[i]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm > 1e-12:
                similarities.append(np.dot(a, b) / norm)

    return float(np.mean(similarities))


def rule_coverage_fidelity(
    X: np.ndarray,
    tau_pred: np.ndarray,
    rule_masks: List[np.ndarray],
    metric: str = "auuc",
) -> float:
    """Evaluate whether rule-selected subgroups rank high in uplift.

    Args:
        X: Instances of shape (n, d).
        tau_pred: Predicted CATE of shape (n,).
        rule_masks: List of boolean masks, each of shape (n,), indicating rule coverage.
        metric: Either "auuc" or "mean_uplift".

    Returns:
        Scalar fidelity score.
    """
    if metric == "mean_uplift":
        scores = [tau_pred[mask].mean() for mask in rule_masks]
        return float(np.mean(scores))

    if metric == "auuc":
        # Treat rule index as a score and compute AUUC-like curve
        # Higher score = covered by more rules
        score = np.zeros(X.shape[0])
        for mask in rule_masks:
            score[mask] += 1
        order = np.argsort(-score)
        sorted_tau = tau_pred[order]
        cumsum = np.cumsum(sorted_tau) / np.arange(1, len(sorted_tau) + 1)
        return float(cumsum.mean())

    raise ValueError(f"Unknown rule fidelity metric: {metric}")


def counterfactual_consistency(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
    cate_model,
    threshold: float = 0.01,
) -> float:
    """Fraction of counterfactuals whose predicted CATE change aligns with the explanation goal.

    Args:
        X_orig: Original instances of shape (n, d).
        X_cf: Counterfactual instances of shape (n, d).
        cate_model: Fitted CATE model.
        threshold: Minimum absolute CATE change to be considered meaningful.

    Returns:
        Consistency rate in [0, 1].
    """
    tau_orig = cate_model.predict_cate(X_orig)
    tau_cf = cate_model.predict_cate(X_cf)
    consistent = np.abs(tau_cf - tau_orig) >= threshold
    return float(consistent.mean())


def counterfactual_proximity(
    X_orig: np.ndarray,
    X_cf: np.ndarray,
    continuous_features: Optional[List[int]] = None,
) -> float:
    """Average L2 distance between original and counterfactual instances.

    Args:
        X_orig: Original instances of shape (n, d).
        X_cf: Counterfactual instances of shape (n, d).
        continuous_features: Indices of continuous features. If None, all features
            are treated as continuous.

    Returns:
        Mean proximity score (lower is closer).
    """
    if continuous_features is None:
        diff = X_orig - X_cf
    else:
        diff = X_orig[:, continuous_features] - X_cf[:, continuous_features]
    return float(np.linalg.norm(diff, axis=1).mean())


def interaction_recall(
    interaction_matrix: np.ndarray,
    true_interaction_pairs: List[tuple],
    k: Optional[int] = None,
) -> float:
    """Fraction of true interaction pairs recovered in top-K detected interactions.

    Args:
        interaction_matrix: Interaction matrix of shape (d, d) or (n, d, d).
            If 3D, values are averaged across instances.
        true_interaction_pairs: List of (feature_i, feature_j) ground-truth pairs.
        k: Number of top interactions to consider. Defaults to len(true_interaction_pairs).

    Returns:
        Recall in [0, 1].
    """
    if interaction_matrix.ndim == 3:
        mean_matrix = np.abs(interaction_matrix).mean(axis=0)
    else:
        mean_matrix = np.abs(interaction_matrix)

    d = mean_matrix.shape[0]
    if k is None:
        k = len(true_interaction_pairs)

    # Extract upper triangle and rank by magnitude
    pairs_with_scores = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs_with_scores.append((i, j, mean_matrix[i, j]))

    pairs_with_scores.sort(key=lambda x: x[2], reverse=True)
    top_k_pairs = set((p[0], p[1]) for p in pairs_with_scores[:k])

    # Normalize ground truth to canonical (min, max) form
    true_set = set((min(i, j), max(i, j)) for i, j in true_interaction_pairs)

    return len(top_k_pairs & true_set) / len(true_set) if true_set else 0.0


def permutation_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test for paired samples.

    Tests H0: mean(metric_a) == mean(metric_b).

    Args:
        metric_a: Scores from method A, shape (n_seeds,).
        metric_b: Scores from method B, shape (n_seeds,).
        n_permutations: Number of random sign flips.
        seed: Random seed.

    Returns:
        p-value.
    """
    rng = np.random.default_rng(seed)
    observed_diff = np.mean(metric_a) - np.mean(metric_b)
    diffs = np.asarray(metric_a) - np.asarray(metric_b)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_permutations


def cohens_d(metric_a: np.ndarray, metric_b: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples.

    Args:
        metric_a: Scores from method A, shape (n_seeds,).
        metric_b: Scores from method B, shape (n_seeds,).

    Returns:
        Cohen's d (positive means A > B).
    """
    diff = np.asarray(metric_a) - np.asarray(metric_b)
    std = np.std(diff, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(diff) / std)


def oos_coverage_fidelity(
    X_train: np.ndarray,
    tau_train: np.ndarray,
    X_test: np.ndarray,
    tau_test: np.ndarray,
    rule_masks_test: List[np.ndarray],
    p: Optional[float] = None,
) -> float:
    """Out-of-sample coverage fidelity for rules discovered on training data.

    Rules are discovered on (X_train, tau_train) but evaluated on (X_test, tau_test).
    This metric is robust against in-sample overfitting and cannot be "hacked"
    by exploiting high-noise training regions.

    Definition:
        OOS_CovFidelity = (mean_tau_covered_test - mean_tau_all_test) /
                          (mean_tau_top_p_test - mean_tau_all_test)

    where p equals the coverage proportion of rules on the test set.

    Args:
        X_train: Training instances (used for rule discovery, not used here).
        tau_train: Training CATE predictions (not used in computation).
        X_test: Test instances of shape (n_test, d).
        tau_test: Test CATE predictions of shape (n_test,).
        rule_masks_test: Boolean masks on the *test* set indicating rule coverage.
        p: Top fraction for normalisation. If None, uses rule coverage proportion.

    Returns:
        OOS Coverage Fidelity score. Values near 1.0 indicate perfect alignment
        between rule-covered subgroups and the highest-CATE population on unseen data.
    """
    # Combine rule masks: a test instance is "covered" if any rule covers it
    n_test = X_test.shape[0]
    covered = np.zeros(n_test, dtype=bool)
    for mask in rule_masks_test:
        covered |= mask

    n_covered = covered.sum()
    if n_covered == 0 or n_covered == n_test:
        return 0.0

    mean_tau_covered = tau_test[covered].mean()
    mean_tau_all = tau_test.mean()

    # Top-p fraction for normalisation
    if p is None:
        p = n_covered / n_test
    top_k = int(np.ceil(p * n_test))
    top_k = max(1, min(top_k, n_test))
    sorted_indices = np.argsort(-tau_test)
    mean_tau_top_p = tau_test[sorted_indices[:top_k]].mean()

    denom = mean_tau_top_p - mean_tau_all
    if abs(denom) < 1e-12:
        return 0.0

    return float((mean_tau_covered - mean_tau_all) / denom)


def rule_stability_index(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    cate_model,
    rule_learner_cls,
    rule_learner_kwargs: dict,
    n_bootstrap: int = 10,
    seed: int = 42,
) -> float:
    """Rule Stability Index: Jaccard similarity of rule-covered populations across bootstraps.

    Measures how stable the discovered rule set is across different bootstrap
    resamples of the training data. Inspired by CDT's Jaccard Subgroup Similarity Index.

    For each pair of bootstrap resamples (b_i, b_j), compute the Jaccard similarity
    of the populations covered by the respective rule sets (evaluated on the full data).
    The Rule Stability Index is the mean pairwise Jaccard similarity.

    Args:
        X: Full dataset covariates of shape (n, d).
        T: Treatment indicators of shape (n,).
        Y: Outcomes of shape (n,).
        cate_model: Fitted CATE model (shared across bootstraps).
        rule_learner_cls: Class to instantiate (e.g., SCRL).
        rule_learner_kwargs: kwargs for rule_learner_cls (excluding cate_model).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Mean pairwise Jaccard similarity in [0, 1]. Higher = more stable.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # Collect coverage masks for each bootstrap run
    coverage_masks = []
    for b in range(n_bootstrap):
        # Bootstrap resample
        idx = rng.choice(n, size=n, replace=True)
        X_b, T_b, Y_b = X[idx], T[idx], Y[idx]

        # Fit rule learner on bootstrap sample
        learner = rule_learner_cls(cate_model=cate_model, **rule_learner_kwargs)
        learner.fit(X_b, T_b, Y_b)

        # Evaluate coverage on the FULL dataset (not bootstrap)
        covered = np.zeros(n, dtype=bool)
        for rule in learner.rules:
            covered |= rule.covers(X)
        coverage_masks.append(covered)

    # Compute pairwise Jaccard similarity
    jaccard_sims = []
    for i in range(n_bootstrap):
        for j in range(i + 1, n_bootstrap):
            intersection = (coverage_masks[i] & coverage_masks[j]).sum()
            union = (coverage_masks[i] | coverage_masks[j]).sum()
            if union > 0:
                jaccard_sims.append(intersection / union)
            else:
                jaccard_sims.append(1.0)

    return float(np.mean(jaccard_sims)) if jaccard_sims else 0.0


def rct_subgroup_ate(
    T: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Compute the true ATE within a subgroup using RCT data (simple difference-in-means).

    Because treatment is randomly assigned in an RCT, the subgroup ATE is
    identified by the simple difference: E[Y|T=1, X in S] - E[Y|T=0, X in S].
    No propensity weighting is needed.

    Args:
        T: Treatment indicators of shape (n,).
        Y: Outcomes of shape (n,).
        mask: Boolean mask indicating subgroup membership, shape (n,).

    Returns:
        Dict with 'ate', 'se', 'ci_lower', 'ci_upper', 'n_treated', 'n_control'.
    """
    T_s = T[mask]
    Y_s = Y[mask]

    treated = T_s == 1
    control = T_s == 0
    n_t = treated.sum()
    n_c = control.sum()

    if n_t < 2 or n_c < 2:
        return {"ate": 0.0, "se": np.inf, "ci_lower": -np.inf,
                "ci_upper": np.inf, "n_treated": int(n_t), "n_control": int(n_c)}

    y_t = Y_s[treated]
    y_c = Y_s[control]

    ate = float(y_t.mean() - y_c.mean())
    se = float(np.sqrt(y_t.var(ddof=1) / n_t + y_c.var(ddof=1) / n_c))
    ci_lower = ate - 1.96 * se
    ci_upper = ate + 1.96 * se

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_treated": int(n_t),
        "n_control": int(n_c),
    }


def rct_targeting_gain(
    T: np.ndarray,
    Y: np.ndarray,
    rule_masks: List[np.ndarray],
) -> dict:
    """Compute targeting gain: ratio of rule-covered subgroup ATE to population ATE.

    Args:
        T: Treatment indicators.
        Y: Outcomes.
        rule_masks: List of boolean masks for rule coverage.

    Returns:
        Dict with 'population_ate', 'subgroup_ate', 'targeting_gain', 'subgroup_ci'.
    """
    # Population ATE
    pop_result = rct_subgroup_ate(T, Y, np.ones(len(T), dtype=bool))

    # Combined rule coverage
    covered = np.zeros(len(T), dtype=bool)
    for mask in rule_masks:
        covered |= mask

    if covered.sum() == 0:
        return {"population_ate": pop_result["ate"], "subgroup_ate": 0.0,
                "targeting_gain": 0.0, "subgroup_ci": (0, 0)}

    sub_result = rct_subgroup_ate(T, Y, covered)

    gain = sub_result["ate"] / pop_result["ate"] if abs(pop_result["ate"]) > 1e-10 else 0.0

    return {
        "population_ate": pop_result["ate"],
        "subgroup_ate": sub_result["ate"],
        "targeting_gain": gain,
        "subgroup_ci": (sub_result["ci_lower"], sub_result["ci_upper"]),
        "subgroup_coverage": float(covered.mean()),
        "subgroup_n_treated": sub_result["n_treated"],
        "subgroup_n_control": sub_result["n_control"],
    }
