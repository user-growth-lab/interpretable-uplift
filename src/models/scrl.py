"""Stable Causal Rule Learning (SCRL).

Discovers a small set of interpretable if-then rules that characterize
subgroups of a fitted treatment-effect predictor. The rule learning
objective incorporates:
  1. CATE heterogeneity within the rule
  2. Treated/control overlap constraints
  3. Stability regularization via bootstrap variance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

import numpy as np
from sklearn.utils import resample

from src.models.cate_estimators import CATEEstimator

# Optional exact-index acceleration. The default reference path needs no Numba.
# NumPy still performs every bootstrap mean and the final standard deviation.
_BOOTSTRAP_BACKEND = os.environ.get("SCRL_BOOTSTRAP_BACKEND", "numpy")
if _BOOTSTRAP_BACKEND not in {"numpy", "numba"}:
    raise ValueError("SCRL_BOOTSTRAP_BACKEND must be numpy or numba")


def _bootstrap_index_matrix(seeds, n):
    out = np.empty((len(seeds), n), dtype=np.int64)
    for b in range(len(seeds)):
        np.random.seed(seeds[b])
        for j in range(n):
            out[b, j] = np.random.randint(0, n)
    return out


if _BOOTSTRAP_BACKEND == "numba":
    from numba import njit
    _bootstrap_index_matrix = njit(_bootstrap_index_matrix)


@dataclass
class CausalRule:
    """A single if-then rule with estimated uplift."""

    conditions: List[Tuple[int, str, float]]
    """List of (feature_index, operator, threshold). Operator is one of '<=', '>='."""

    tau_mean: float
    tau_std: float
    coverage: float
    n_treated: int
    n_control: int
    stability_score: float

    @property
    def _canonical_key(self) -> tuple:
        """Canonical key for deduplication (sorted by feature index)."""
        return tuple(sorted((int(c[0]), c[1], round(float(c[2]), 6)) for c in self.conditions))

    def is_duplicate_of(self, other: "CausalRule") -> bool:
        """Check if two rules have the same canonical condition set."""
        return self._canonical_key == other._canonical_key

    def covers(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask of rows covered by this rule."""
        mask = np.ones(X.shape[0], dtype=bool)
        for feat_idx, op, thresh in self.conditions:
            if op == "<=":
                mask &= X[:, feat_idx] <= thresh
            elif op == ">=":
                mask &= X[:, feat_idx] >= thresh
            elif op == "<":
                mask &= X[:, feat_idx] < thresh
            elif op == ">":
                mask &= X[:, feat_idx] > thresh
            else:
                raise ValueError(f"Unknown operator: {op}")
        return mask

    def __str__(self) -> str:
        if not self.conditions:
            return "TRUE"
        parts = []
        for feat_idx, op, thresh in self.conditions:
            parts.append(f"x{feat_idx} {op} {thresh:.3f}")
        return " AND ".join(parts)


class SCRL:
    """Stable Causal Rule Learning."""

    def __init__(
        self,
        cate_model: CATEEstimator,
        n_rules: int = 5,
        max_rule_length: int = 3,
        min_support: float = 0.05,
        max_coverage: float = 0.40,
        overlap_ratio: float = 0.1,
        n_candidates: int = 100,
        n_bootstrap: int = 20,
        coverage_bonus_weight: float = 0.3,
        sample_split: bool = True,
        feature_prior_weights: Optional[np.ndarray] = None,
        interaction_seeds: Optional[List[Tuple[int, int]]] = None,
        seed: int = 42,
        # Ablation switches (defaults reproduce full SCRL)
        use_diversity: bool = True,
        use_dedup: bool = True,
        variance_scoring: bool = False,
        support_mode: str = "hard",
    ):
        """Initialize SCRL.

        Args:
            cate_model: Fitted CATE estimator.
            n_rules: Maximum number of rules to select; hard filtering may return fewer.
            max_rule_length: Maximum number of conditions per rule.
            min_support: Minimum fraction of data covered by a rule.
            max_coverage: Maximum fraction of data a single rule may cover.
            overlap_ratio: Minimum ratio of min(treated, control) / rule_size.
            n_candidates: Number of candidate thresholds per feature.
            n_bootstrap: Number of bootstrap samples for stability estimation.
            coverage_bonus_weight: Weight for coverage bonus in selection (0-1).
            sample_split: If True, use 50/50 discovery/selection halves. Both
                halves belong to the teacher-fitting sample, not an independent teacher test.
            feature_prior_weights: Prior weights per feature from TESV (knowledge-guided).
            interaction_seeds: Feature interaction pairs from TESV-I for candidate generation.
            seed: Random seed.
        """
        self.cate_model = cate_model
        self.n_rules = n_rules
        self.max_rule_length = max_rule_length
        self.min_support = min_support
        self.max_coverage = max_coverage
        self.overlap_ratio = overlap_ratio
        self.n_candidates = n_candidates
        self.n_bootstrap = n_bootstrap
        self.coverage_bonus_weight = coverage_bonus_weight
        self.sample_split = sample_split
        self.feature_prior_weights = feature_prior_weights
        self.interaction_seeds = interaction_seeds
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.rules: List[CausalRule] = []
        self.use_diversity = use_diversity
        self.use_dedup = use_dedup
        self.variance_scoring = variance_scoring
        if support_mode not in {"hard", "legacy"}:
            raise ValueError("support_mode must be 'hard' or 'legacy'")
        self.support_mode = support_mode
        self.algorithm_version_ = ("scrl-r1-hard-support-v1" if support_mode == "hard"
                                   else "scrl-legacy-20260725")

    def empirical_support(self, rule, X, T):
        """Sample-only support diagnostics; no population or causal guarantee."""
        mask = rule.covers(X)
        n_r = int(mask.sum())
        n_t = int(T[mask].sum())
        n_c = n_r - n_t
        coverage = n_r / len(X) if len(X) else 0.0
        return {"n": n_r, "n_treated": n_t, "n_control": n_c,
                "coverage": coverage,
                "treated_fraction": n_t / n_r if n_r else None,
                "support_ok": bool(n_r >= self.min_support * len(X) and n_r >= 2),
                "overlap_ok": bool(n_r > 0 and min(n_t, n_c) >= self.overlap_ratio * n_r),
                "max_coverage_ok": bool(coverage <= self.max_coverage)}

    def _generate_candidate_rules(
        self, X: np.ndarray, tau: np.ndarray, T: np.ndarray
    ) -> List[CausalRule]:
        """Generate candidate single-condition rules from feature thresholds."""
        n, d = X.shape
        candidates = []
        quantiles = np.linspace(0.1, 0.9, self.n_candidates)

        for j in range(d):
            thresholds = np.quantile(X[:, j], quantiles)
            for thr in thresholds:
                for op in [">=", "<="]:
                    rule = CausalRule(
                        conditions=[(j, op, float(thr))],
                        tau_mean=0.0,
                        tau_std=0.0,
                        coverage=0.0,
                        n_treated=0,
                        n_control=0,
                        stability_score=0.0,
                    )
                    mask = rule.covers(X)
                    if mask.sum() < self.min_support * n:
                        continue

                    n_t = T[mask].sum()
                    n_c = mask.sum() - n_t
                    if min(n_t, n_c) < self.overlap_ratio * mask.sum():
                        continue

                    tau_vals = tau[mask]
                    rule.tau_mean = float(tau_vals.mean())
                    rule.tau_std = float(tau_vals.std())
                    rule.coverage = float(mask.mean())
                    rule.n_treated = int(n_t)
                    rule.n_control = int(n_c)
                    candidates.append(rule)

        return candidates

    def _score_rule(
        self, rule: CausalRule, X: np.ndarray, tau: np.ndarray, T: np.ndarray
    ) -> float:
        """Score a rule by subgroup effect magnitude, heterogeneity, overlap, and stability."""
        mask = rule.covers(X)
        n_r = int(mask.sum())
        if n_r < 2:
            return -np.inf
        tau_vals = tau[mask]
        rule.tau_mean = float(tau_vals.mean())
        rule.tau_std = float(tau_vals.std())
        rule.coverage = float(mask.mean())

        # Overlap penalty
        n_t = T[mask].sum()
        n_c = n_r - n_t
        overlap_penalty = 0.0
        if min(n_t, n_c) < self.overlap_ratio * n_r:
            overlap_penalty = -1e6

        # Stability via bootstrap
        stability_scores = []
        if _BOOTSTRAP_BACKEND == "numba" and n_r <= 10000:
            # Draw seeds sequentially, exactly as in the reference path.  The
            # local MT19937 generators do not alter the learner's Generator.
            seeds = np.array([self.rng.integers(1e9) for _ in range(self.n_bootstrap)])
            indices = _bootstrap_index_matrix(seeds, n_r)
            for idx in indices:
                stability_scores.append(tau_vals[idx].mean())
        else:
            for _ in range(self.n_bootstrap):
                # Exact sklearn 1.6 resample(replace=True, stratify=None) path.
                rng_boot = np.random.RandomState(self.rng.integers(1e9))
                idx = rng_boot.randint(0, n_r, size=n_r)
                stability_scores.append(tau_vals[idx].mean())
        stability = -np.std(stability_scores)  # lower std = higher stability
        rule.stability_score = float(stability)
        rule.n_treated, rule.n_control = int(n_t), int(n_c)

        # Reward rules whose subgroup treatment effect deviates strongly from the global mean
        # Use sqrt(coverage) instead of linear coverage to avoid systematic preference for broad rules
        global_mean = getattr(self, "_global_tau_mean", 0.0)
        if self.variance_scoring:
            # Ablation variant: score by within-rule CATE variance instead of effect magnitude
            effect_magnitude = rule.tau_std * np.sqrt(rule.coverage)
        else:
            effect_magnitude = abs(rule.tau_mean - global_mean) * np.sqrt(rule.coverage)
        score = effect_magnitude + 0.05 * rule.tau_std + stability + overlap_penalty
        return float(score)

    def _extend_rules(
        self,
        rules: List[CausalRule],
        X: np.ndarray,
        tau: np.ndarray,
        T: np.ndarray,
        all_candidates: List[CausalRule],
    ) -> List[CausalRule]:
        """Extend existing rules by adding one more condition."""
        extended = []
        for rule in rules:
            if len(rule.conditions) >= self.max_rule_length:
                continue
            for cand in all_candidates:
                new_cond = cand.conditions[0]
                # Avoid duplicate conditions on same feature
                if any(cond[0] == new_cond[0] for cond in rule.conditions):
                    continue
                new_rule = CausalRule(
                    conditions=rule.conditions + [new_cond],
                    tau_mean=0.0,
                    tau_std=0.0,
                    coverage=0.0,
                    n_treated=0,
                    n_control=0,
                    stability_score=0.0,
                )
                mask = new_rule.covers(X)
                if mask.sum() < self.min_support * X.shape[0]:
                    continue
                n_t = T[mask].sum()
                n_c = mask.sum() - n_t
                if min(n_t, n_c) < self.overlap_ratio * mask.sum():
                    continue

                tau_vals = tau[mask]
                new_rule.tau_mean = float(tau_vals.mean())
                new_rule.tau_std = float(tau_vals.std())
                new_rule.coverage = float(mask.mean())
                new_rule.n_treated = int(n_t)
                new_rule.n_control = int(n_c)
                extended.append(new_rule)
        return extended

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "SCRL":
        """Discover stable causal rules.

        Args:
            X: Covariates of shape (n, d).
            T: Treatment indicators of shape (n,).
            Y: Outcomes of shape (n,).

        Returns:
            Self with fitted `self.rules`.
        """
        tau = self.cate_model.predict_cate(X)
        self._global_tau_mean = float(tau.mean())

        # Sample-splitting: discover candidates on train half, score on validation half
        n = X.shape[0]
        if self.sample_split and n >= 100:
            idx = np.arange(n)
            self.rng.shuffle(idx)
            mid = n // 2
            train_idx, val_idx = idx[:mid], idx[mid:]
            X_disc, T_disc, tau_disc = X[train_idx], T[train_idx], tau[train_idx]
            X_val, T_val, tau_val = X[val_idx], T[val_idx], tau[val_idx]
        else:
            X_disc, T_disc, tau_disc = X, T, tau
            X_val, T_val, tau_val = X, T, tau
            train_idx = val_idx = np.arange(n)
        self.discovery_indices_ = train_idx.copy()
        self.selection_indices_ = val_idx.copy()

        # Generate candidates on discovery set
        candidates = self._generate_candidate_rules(X_disc, tau_disc, T_disc)

        # Knowledge-guided: add interaction seed candidates
        if self.interaction_seeds:
            for i_feat, j_feat in self.interaction_seeds:
                for q in np.linspace(0.25, 0.75, 3):
                    thr_i = np.quantile(X_disc[:, i_feat], q)
                    thr_j = np.quantile(X_disc[:, j_feat], q)
                    for op_i, op_j in [(">=", ">="), ("<=", "<="), (">=", "<="), ("<=", ">=")]:
                        rule = CausalRule(
                            conditions=[(i_feat, op_i, float(thr_i)), (j_feat, op_j, float(thr_j))],
                            tau_mean=0.0, tau_std=0.0, coverage=0.0,
                            n_treated=0, n_control=0, stability_score=0.0,
                        )
                        mask = rule.covers(X_disc)
                        if mask.sum() >= self.min_support * X_disc.shape[0]:
                            n_t = T_disc[mask].sum()
                            n_c = mask.sum() - n_t
                            if min(n_t, n_c) >= self.overlap_ratio * mask.sum():
                                rule.tau_mean = float(tau_disc[mask].mean())
                                rule.tau_std = float(tau_disc[mask].std())
                                rule.coverage = float(mask.mean())
                                rule.n_treated = int(n_t)
                                rule.n_control = int(n_c)
                                candidates.append(rule)

        # Build rules of increasing length, but only extend top-K single rules
        all_rules = candidates.copy()
        current = candidates
        for length in range(1, self.max_rule_length):
            # Score and keep top single rules for extension to limit combinatorial blowup
            if len(current) > 100:
                scores = [self._score_rule(r, X_disc, tau_disc, T_disc) for r in current]
                top_idx = np.argsort(scores)[-100:]
                current = [current[i] for i in top_idx]
            current = self._extend_rules(current, X_disc, tau_disc, T_disc, candidates)
            all_rules.extend(current)

        # Adaptively score/select on the selection half, which shares the fitted teacher.
        selected = []
        remaining = all_rules.copy()
        covered = np.zeros(X_val.shape[0], dtype=bool)
        selected_feature_counts: dict = {}
        self.selection_trace_ = []
        self.legacy_hard_equivalent_ = True

        for _ in range(self.n_rules):
            if not remaining:
                break

            # Score away from candidate construction; this is not an independent teacher test.
            base_scores = np.array([self._score_rule(r, X_val, tau_val, T_val) for r in remaining])

            # Reduced coverage bonus (controlled by coverage_bonus_weight)
            coverage_bonuses = np.array(
                [
                    float(((~covered) & r.covers(X_val)).sum()) / X_val.shape[0]
                    for r in remaining
                ]
            )

            # Feature diversity penalty: discourage reusing already-selected features
            feature_diversity_penalties = np.array(
                [
                    sum(selected_feature_counts.get(int(cond[0]), 0) for cond in r.conditions)
                    for r in remaining
                ]
            )

            scores = base_scores + self.coverage_bonus_weight * coverage_bonuses
            if self.use_diversity:
                scores = scores - 0.25 * feature_diversity_penalties

            # Apply max_coverage constraint: reject over-broad rules
            for i, r in enumerate(remaining):
                r_cov = r.covers(X_val).mean()
                if r_cov > self.max_coverage:
                    scores[i] = -np.inf
                if self.support_mode == "hard":
                    support = self.empirical_support(r, X_val, T_val)
                    if not (support["support_ok"] and support["overlap_ok"]):
                        scores[i] = -np.inf

            best_idx = int(np.argmax(scores))
            if scores[best_idx] == -np.inf:
                break

            best_rule = remaining.pop(best_idx)
            winner_support = self.empirical_support(best_rule, X_val, T_val)
            self.selection_trace_.append(winner_support)
            if not all(winner_support[k] for k in ("support_ok", "overlap_ok", "max_coverage_ok")):
                self.legacy_hard_equivalent_ = False

            # Hard filter: rule must still satisfy minimum support
            if best_rule.covers(X_val).sum() < self.min_support * X_val.shape[0]:
                continue
            if self.support_mode == "hard":
                support = self.empirical_support(best_rule, X_val, T_val)
                if not all(support[k] for k in ("support_ok", "overlap_ok", "max_coverage_ok")):
                    raise AssertionError("hard support filter failed")

            selected.append(best_rule)
            covered |= best_rule.covers(X_val)
            for cond in best_rule.conditions:
                selected_feature_counts[int(cond[0])] = selected_feature_counts.get(int(cond[0]), 0) + 1

            # Remove duplicates of the selected rule from remaining candidates
            if self.use_dedup:
                remaining = [
                    r for r in remaining if not r.is_duplicate_of(best_rule)
                ]

        self.rules = selected
        self.support_audit_ = [
            {"discovery": self.empirical_support(r, X_disc, T_disc),
             "selection": self.empirical_support(r, X_val, T_val)} for r in selected]
        return self

    def predict_rules(self, X: np.ndarray) -> np.ndarray:
        """Return rule coverage matrix of shape (n, n_rules)."""
        if not self.rules:
            return np.zeros((X.shape[0], 0), dtype=bool)
        return np.array([rule.covers(X) for rule in self.rules]).T

    def predict_subgroup_uplift(self, X: np.ndarray) -> np.ndarray:
        """For each instance, return the mean tau of the highest-scoring covering rule."""
        tau = self.cate_model.predict_cate(X)
        coverage = self.predict_rules(X)
        # Assign each instance to the first covering rule; default to overall tau
        assigned = np.full(X.shape[0], tau.mean())
        for i, rule in enumerate(self.rules):
            mask = coverage[:, i] & (assigned == tau.mean())
            assigned[mask] = rule.tau_mean
        return assigned
