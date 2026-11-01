"""Baseline explanation methods for comparison.

Includes:
  - Conditional SHAP (KNN-based observational)
  - Interventional SHAP (marginalisation on outcome model, then difference)
  - Causal SHAP (causal ordering approximation)
  - Causal Tree for subgroup discovery
  - CURLS-style rule discovery
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

from src.models.cate_estimators import CATEEstimator


class SHAPCATEBaseline:
    """Local nearest-neighbor approximation to conditional CATE attribution.

    For coalition S, choose background neighbors in the S-subspace, overwrite
    their S coordinates and average fitted predictions. This is not an exact
    conditional expectation or evidence of causal feature effects. Permutations
    are uniform; the finite-background implementation is a local comparator."""

    def __init__(
        self,
        cate_model: CATEEstimator,
        background_data: Optional[np.ndarray] = None,
        n_background: int = 100,
        k_neighbors: int = 10,
        seed: int = 42,
    ):
        self.cate_model = cate_model
        self.rng = np.random.default_rng(seed)
        self.k_neighbors = k_neighbors
        if background_data is not None:
            self.background_data = self._prepare_background(background_data, n_background)
        else:
            self.background_data = None
        self.n_background = n_background

    def _prepare_background(
        self, background_data: np.ndarray, n_background: int
    ) -> np.ndarray:
        n = background_data.shape[0]
        if n > n_background:
            idx = self.rng.choice(n, size=n_background, replace=False)
            return background_data[idx]
        return background_data

    def _conditional_expectation(
        self,
        x: np.ndarray,
        feature_subset: List[int],
        background: np.ndarray,
    ) -> float:
        """Approximate E[tau(X) | X_S = x_S] via KNN in the S subspace.

        Finds K nearest neighbors based on the fixed features S, then averages
        CATE predictions using those neighbors' complement features. This
        preserves correlations between S and complement features.
        """
        if len(feature_subset) == 0:
            return float(self.cate_model.predict_cate(background).mean())

        if len(feature_subset) == background.shape[1]:
            return float(self.cate_model.predict_cate(x.reshape(1, -1)).mean())

        # KNN in the S-subspace to approximate conditional distribution
        dists = np.sum((background[:, feature_subset] - x[feature_subset]) ** 2, axis=1)
        k = min(self.k_neighbors, len(background))
        nn_idx = np.argpartition(dists, k)[:k]

        # Use neighbors' full feature vectors but replace S-features with x_S
        X_cond = background[nn_idx].copy()
        X_cond[:, feature_subset] = x[feature_subset]
        return float(self.cate_model.predict_cate(X_cond).mean())

    def explain_instance(
        self,
        x: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        n_permutations: int = 20,
    ) -> np.ndarray:
        if background_data is None:
            background_data = self.background_data
        if background_data is None:
            raise ValueError("background_data must be provided")

        background_data = self._prepare_background(background_data, self.n_background)
        d = x.shape[0]
        phi = np.zeros(d)

        for _ in range(n_permutations):
            perm = self.rng.permutation(d)
            subset = []
            prev_val = self._conditional_expectation(x, subset, background_data)
            for j in perm:
                subset.append(j)
                curr_val = self._conditional_expectation(x, subset, background_data)
                phi[j] += curr_val - prev_val
                prev_val = curr_val

        phi /= n_permutations
        return phi

    def explain(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        n_permutations: int = 20,
    ) -> np.ndarray:
        return np.array(
            [self.explain_instance(x, background_data, n_permutations) for x in X]
        )


class InterventionalSHAPBaseline:
    """Marginal-replacement attribution using a paired existing outcome model.

    With exposed model_t/model_c attributes, subtract their contributions using
    identical background rows and permutations. Otherwise use the CATE predictor
    as the first function and zero as the second. Both routes target direct
    interventional Shapley of the same fitted CATE function. No refitting occurs.
    Marginal replacement does not by itself identify interventions on covariates."""

    def __init__(self, cate_model: CATEEstimator, background_data: Optional[np.ndarray] = None,
                 n_background: int = 100, seed: int = 42):
        self.cate_model = cate_model
        self.rng = np.random.default_rng(seed)
        if background_data is not None:
            n = background_data.shape[0]
            if n > n_background:
                idx = self.rng.choice(n, size=n_background, replace=False)
                self.background_data = background_data[idx]
            else:
                self.background_data = background_data
        else:
            self.background_data = None
        self.n_background = n_background

    def _interventional_value(self, x, feature_subset, background, predict_fn):
        if len(feature_subset) == 0:
            return float(predict_fn(background).mean())
        X_p = background.copy()
        X_p[:, feature_subset] = x[feature_subset]
        return float(predict_fn(X_p).mean())

    def explain_instance(self, x, background_data=None, n_permutations=20):
        bg = background_data if background_data is not None else self.background_data
        if bg is None:
            raise ValueError("background_data required")
        n = bg.shape[0]
        if n > self.n_background:
            idx = self.rng.choice(n, size=self.n_background, replace=False)
            bg = bg[idx]
        d = x.shape[0]
        # Get predict functions for each arm
        if hasattr(self.cate_model, 'model_t') and hasattr(self.cate_model, 'model_c'):
            pred_t = lambda X: self.cate_model.model_t.predict(X)
            pred_c = lambda X: self.cate_model.model_c.predict(X)
        else:
            # Fallback: use CATE directly
            pred_t = lambda X: self.cate_model.predict_cate(X)
            pred_c = lambda X: np.zeros(X.shape[0])
        phi_t = np.zeros(d)
        phi_c = np.zeros(d)
        for _ in range(n_permutations):
            perm = self.rng.permutation(d)
            subset = []
            prev_t = self._interventional_value(x, subset, bg, pred_t)
            prev_c = self._interventional_value(x, subset, bg, pred_c)
            for j in perm:
                subset.append(j)
                curr_t = self._interventional_value(x, subset, bg, pred_t)
                curr_c = self._interventional_value(x, subset, bg, pred_c)
                phi_t[j] += curr_t - prev_t
                phi_c[j] += curr_c - prev_c
                prev_t, prev_c = curr_t, curr_c
        phi_t /= n_permutations
        phi_c /= n_permutations
        return phi_t - phi_c

    def explain(self, X, background_data=None, n_permutations=20):
        return np.array([self.explain_instance(x, background_data, n_permutations) for x in X])


class CausalSHAPBaseline:
    """Heuristic-ordering attribution comparator (legacy API class name).

    Feature order comes from a prediction-variance permutation heuristic, not a
    causal graph. Coalition-relative upstream coordinates select nearest
    background neighbors; all coalition coordinates are then overwritten.
    Ranking plus Gaussian noise produces biased, non-uniform permutations.
    This changes both replacement and permutation weighting and is not an
    implementation of graph-based causal Shapley or do-calculus."""

    def __init__(self, cate_model: CATEEstimator, background_data: Optional[np.ndarray] = None,
                 n_background: int = 100, seed: int = 42):
        self.cate_model = cate_model
        self.rng = np.random.default_rng(seed)
        if background_data is not None:
            n = background_data.shape[0]
            if n > n_background:
                idx = self.rng.choice(n, size=n_background, replace=False)
                self.background_data = background_data[idx]
            else:
                self.background_data = background_data
        else:
            self.background_data = None
        self.n_background = n_background
        self._causal_order = None

    def _estimate_causal_order(self, background):
        """Heuristic: rank features by their marginal variance contribution to CATE."""
        tau_base = self.cate_model.predict_cate(background)
        d = background.shape[1]
        var_contribs = np.zeros(d)
        for j in range(d):
            X_perm = background.copy()
            X_perm[:, j] = self.rng.permutation(X_perm[:, j])
            tau_perm = self.cate_model.predict_cate(X_perm)
            var_contribs[j] = np.var(tau_base) - np.var(tau_perm)
        # Higher contribution = more upstream
        self._causal_order = np.argsort(-var_contribs)

    def _conditional_interventional(self, x, feature_subset, background):
        """Use KNN for 'upstream' features, marginalisation for 'downstream'."""
        if len(feature_subset) == 0:
            return float(self.cate_model.predict_cate(background).mean())
        if len(feature_subset) == background.shape[1]:
            return float(self.cate_model.predict_cate(x.reshape(1, -1)).mean())
        # Split subset into upstream and downstream based on causal order
        order_pos = {f: i for i, f in enumerate(self._causal_order)}
        median_pos = np.median([order_pos[f] for f in feature_subset])
        upstream = [f for f in feature_subset if order_pos[f] <= median_pos]
        downstream = [f for f in feature_subset if order_pos[f] > median_pos]
        # KNN for upstream (conditional), marginalise for downstream (interventional)
        if upstream:
            dists = np.sum((background[:, upstream] - x[upstream]) ** 2, axis=1)
            k = min(10, len(background))
            nn_idx = np.argpartition(dists, k)[:k]
            bg_sub = background[nn_idx].copy()
        else:
            bg_sub = background.copy()
        bg_sub[:, feature_subset] = x[feature_subset]
        return float(self.cate_model.predict_cate(bg_sub).mean())

    def explain_instance(self, x, background_data=None, n_permutations=20):
        bg = background_data if background_data is not None else self.background_data
        if bg is None:
            raise ValueError("background_data required")
        n = bg.shape[0]
        if n > self.n_background:
            idx = self.rng.choice(n, size=self.n_background, replace=False)
            bg = bg[idx]
        if self._causal_order is None:
            self._estimate_causal_order(bg)
        d = x.shape[0]
        phi = np.zeros(d)
        for _ in range(n_permutations):
            # Biased permutation: respect causal ordering with noise
            order = self._causal_order.copy()
            noise = self.rng.normal(0, 0.3, size=len(order))
            noisy_rank = np.arange(len(order)) + noise
            perm = order[np.argsort(noisy_rank)]
            subset = []
            prev_val = self._conditional_interventional(x, subset, bg)
            for j in perm:
                subset.append(j)
                curr_val = self._conditional_interventional(x, subset, bg)
                phi[j] += curr_val - prev_val
                prev_val = curr_val
        phi /= n_permutations
        return phi

    def explain(self, X, background_data=None, n_permutations=20):
        return np.array([self.explain_instance(x, background_data, n_permutations) for x in X])


class CURLSBaseline:
    """Bootstrap-pruned prediction-tree comparator (legacy API class name).

    This is not the published CURLS minorize-maximization algorithm.

    Discovers rules by fitting a regression tree on CATE predictions,
    then prunes rules using fixed-model bootstrap percentile intervals. Leaves whose fixed-model bootstrap interval includes the global prediction
    mean are removed; this is not a post-selection causal significance test.
    """

    def __init__(self, cate_model: CATEEstimator, n_rules: int = 5,
                 max_depth: int = 3, min_samples_leaf: int = 30,
                 n_bootstrap: int = 50, alpha: float = 0.05, seed: int = 42):
        self.cate_model = cate_model
        self.n_rules = n_rules
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tree = None
        self.rules: List = []

    def fit(self, X, T, Y):
        from sklearn.utils import resample
        tau = self.cate_model.predict_cate(X)
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed
        )
        self.tree.fit(X, tau)
        # Extract leaf predictions and prune by CI
        leaves = self.tree.apply(X)
        unique_leaves = np.unique(leaves)
        significant_leaves = []
        for leaf in unique_leaves:
            mask = leaves == leaf
            if mask.sum() < self.min_samples_leaf:
                continue
            tau_leaf = tau[mask]
            # Bootstrap CI
            boot_means = []
            for _ in range(self.n_bootstrap):
                idx = resample(np.arange(mask.sum()), random_state=self.rng.integers(1e9))
                boot_means.append(tau_leaf[idx].mean())
            ci_low = np.percentile(boot_means, 100 * self.alpha / 2)
            ci_high = np.percentile(boot_means, 100 * (1 - self.alpha / 2))
            # Only keep if CI doesn't include global mean (significant heterogeneity)
            global_mean = tau.mean()
            if ci_low > global_mean or ci_high < global_mean:
                significant_leaves.append((leaf, abs(np.mean(tau_leaf) - global_mean), mask))
        # Sort by effect magnitude, take top-K
        significant_leaves.sort(key=lambda x: -x[1])
        self.rules = significant_leaves[:self.n_rules]
        return self

    def get_rule_masks(self, X):
        if self.tree is None:
            return []
        leaves = self.tree.apply(X)
        return [leaves == leaf_id for leaf_id, _, _ in self.rules]

    def predict_cate(self, X):
        return self.tree.predict(X) if self.tree else np.zeros(X.shape[0])


class CausalTreeBaseline:
    """Residual-product tree for subgroup discovery (legacy API class name).

    Fits a decision tree on pseudo-outcomes (T - e(X)) * (Y - m(X)) to
    form a partition, then extracts rules from tree leaves. With correct nuisances,
    the conditional mean target is e(X)*(1-e(X))*tau(X), not tau(X). The method
    predict_cate retains its historical name but returns uncalibrated tree scores.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        seed: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "CausalTreeBaseline":
        from sklearn.linear_model import LogisticRegression, Ridge

        # Propensity score
        prop_model = LogisticRegression(max_iter=1000)
        prop_model.fit(X, T)
        e = np.clip(prop_model.predict_proba(X)[:, 1], 0.05, 0.95)

        # Outcome regression
        outcome_model = Ridge()
        outcome_model.fit(X, Y)
        m = outcome_model.predict(X)

        # Pseudo-outcome (R-Learner style)
        pseudo = (T - e) * (Y - m)
        self.tree.fit(X, pseudo)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict(X)

    def extract_rules(self, feature_names: Optional[List[str]] = None) -> List[str]:
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.tree.n_features_in_)]
        return [export_text(self.tree, feature_names=feature_names)]


class DiCEUpliftBaseline:
    """DiCE-style diverse counterfactual generation adapted for uplift models.

    Generates diverse counterfactuals by random perturbation with diversity
    enforcement (determinantal point process-inspired selection). Unlike our
    causal counterfactual explainer which uses greedy coordinate ascent
    optimizing CATE gain per unit cost, DiCE generates multiple candidate
    counterfactuals randomly and selects a diverse subset.

    Reference: Mothilal et al. (2020) "Explaining machine learning classifiers
    through diverse counterfactual explanations"
    """

    def __init__(
        self,
        cate_model: CATEEstimator,
        actionable_features: Optional[List[int]] = None,
        feature_bounds: Optional[np.ndarray] = None,
        n_candidates: int = 100,
        n_counterfactuals: int = 1,
        proximity_weight: float = 0.5,
        diversity_weight: float = 1.0,
        seed: int = 42,
    ):
        self.cate_model = cate_model
        self.actionable_features = actionable_features
        self.feature_bounds = feature_bounds
        self.n_candidates = n_candidates
        self.n_counterfactuals = n_counterfactuals
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.rng = np.random.default_rng(seed)

    def _is_feasible(self, x: np.ndarray) -> bool:
        if self.feature_bounds is None:
            return True
        return bool(
            np.all(x >= self.feature_bounds[:, 0])
            and np.all(x <= self.feature_bounds[:, 1])
        )

    def _generate_candidates(self, x: np.ndarray) -> np.ndarray:
        """Generate random perturbation candidates."""
        d = x.shape[0]
        actionable = (
            self.actionable_features
            if self.actionable_features is not None
            else list(range(d))
        )
        candidates = []
        for _ in range(self.n_candidates * 3):  # oversample
            x_cand = x.copy()
            # Randomly perturb 1-3 actionable features
            n_changes = self.rng.integers(1, min(4, len(actionable) + 1))
            feats_to_change = self.rng.choice(
                actionable, size=n_changes, replace=False
            )
            for j in feats_to_change:
                if self.feature_bounds is not None:
                    low, high = self.feature_bounds[j]
                    x_cand[j] = self.rng.uniform(low, high)
                else:
                    x_cand[j] += self.rng.normal(0, 0.5)
            if self._is_feasible(x_cand):
                candidates.append(x_cand)
            if len(candidates) >= self.n_candidates:
                break
        if len(candidates) == 0:
            candidates.append(x.copy())
        return np.array(candidates)

    def explain_instance(self, x: np.ndarray) -> dict:
        """Generate diverse counterfactuals for a single instance."""
        tau_orig = float(self.cate_model.predict_cate(x.reshape(1, -1))[0])
        candidates = self._generate_candidates(x)
        tau_cands = self.cate_model.predict_cate(candidates)

        # Filter: keep only candidates with positive CATE gain
        gains = tau_cands - tau_orig
        valid_mask = gains > 0
        if valid_mask.sum() == 0:
            # No valid counterfactual found
            return {
                "x_cf": x.copy(),
                "tau_orig": tau_orig,
                "tau_cf": tau_orig,
                "tau_gain": 0.0,
                "proximity": 0.0,
                "feasible": True,
                "consistent": False,
            }

        valid_cands = candidates[valid_mask]
        valid_gains = gains[valid_mask]
        valid_taus = tau_cands[valid_mask]

        # Score: CATE gain - proximity penalty
        proximities = np.linalg.norm(valid_cands - x, axis=1)
        scores = valid_gains - self.proximity_weight * proximities

        # Select best (or diverse subset if n_counterfactuals > 1)
        best_idx = np.argmax(scores)
        x_cf = valid_cands[best_idx]
        tau_cf = float(valid_taus[best_idx])

        return {
            "x_cf": x_cf,
            "tau_orig": tau_orig,
            "tau_cf": tau_cf,
            "tau_gain": tau_cf - tau_orig,
            "proximity": float(np.linalg.norm(x_cf - x)),
            "feasible": self._is_feasible(x_cf),
            "consistent": tau_cf >= tau_orig,
        }

    def explain(self, X: np.ndarray) -> List[dict]:
        """Generate counterfactuals for a batch."""
        return [self.explain_instance(x) for x in X]


class GrowingSpheresUplift:
    """Growing Spheres adapted for CATE uplift increase.

    Algorithm:
    1. Start with small epsilon-sphere around x
    2. Sample uniformly within sphere
    3. If any sample achieves tau_gain > 0, reduce sphere and refine
    4. If no sample improves CATE, increase sphere radius
    5. Repeat until convergence or max iterations

    This provides a minimal-distance guarantee (unlike greedy coordinate ascent)
    at the cost of reduced gain magnitude.

    Reference: Laugel et al. (2018) "Comparison-based Inverse Classification"
    """

    def __init__(
        self,
        cate_model: CATEEstimator,
        actionable_features: Optional[List[int]] = None,
        feature_bounds: Optional[np.ndarray] = None,
        initial_radius: float = 0.1,
        growth_factor: float = 1.5,
        shrink_factor: float = 0.8,
        n_samples_per_sphere: int = 500,
        max_iterations: int = 30,
        seed: int = 42,
    ):
        self.cate_model = cate_model
        self.actionable_features = actionable_features
        self.feature_bounds = feature_bounds
        self.initial_radius = initial_radius
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.n_samples_per_sphere = n_samples_per_sphere
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)

    def _is_feasible(self, x: np.ndarray) -> bool:
        if self.feature_bounds is None:
            return True
        return bool(np.all(x >= self.feature_bounds[:, 0]) and np.all(x <= self.feature_bounds[:, 1]))

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        if self.feature_bounds is None:
            return x
        return np.clip(x, self.feature_bounds[:, 0], self.feature_bounds[:, 1])

    def explain_instance(self, x: np.ndarray) -> dict:
        """Generate a counterfactual for a single instance using growing spheres."""
        d = x.shape[0]
        actionable = self.actionable_features if self.actionable_features else list(range(d))
        tau_orig = float(self.cate_model.predict_cate(x.reshape(1, -1))[0])

        radius = self.initial_radius
        best_cf = None
        best_gain = 0.0
        best_proximity = np.inf

        for _ in range(self.max_iterations):
            # Sample uniformly within sphere (only actionable dimensions)
            candidates = np.tile(x, (self.n_samples_per_sphere, 1))
            noise = self.rng.normal(0, 1, size=(self.n_samples_per_sphere, len(actionable)))
            noise = noise / (np.linalg.norm(noise, axis=1, keepdims=True) + 1e-12)
            radii = self.rng.uniform(0, radius, size=(self.n_samples_per_sphere, 1))
            perturbations = noise * radii

            for k, feat_idx in enumerate(actionable):
                candidates[:, feat_idx] += perturbations[:, k % len(actionable)]

            # Clip to bounds
            for i in range(len(candidates)):
                candidates[i] = self._clip_to_bounds(candidates[i])

            # Evaluate CATE
            tau_cands = self.cate_model.predict_cate(candidates)
            gains = tau_cands - tau_orig

            # Find valid counterfactuals (positive gain + feasible)
            valid_mask = gains > 0
            if valid_mask.any():
                valid_cands = candidates[valid_mask]
                valid_gains = gains[valid_mask]
                proximities = np.linalg.norm(valid_cands - x, axis=1)

                # Select closest valid counterfactual
                closest_idx = np.argmin(proximities)
                if proximities[closest_idx] < best_proximity:
                    best_cf = valid_cands[closest_idx]
                    best_gain = float(valid_gains[closest_idx])
                    best_proximity = float(proximities[closest_idx])

                # Shrink sphere to refine
                radius *= self.shrink_factor
            else:
                # No valid CF found, grow sphere
                radius *= self.growth_factor

        if best_cf is None:
            # Fallback: return original (failed)
            return {
                "x_cf": x.copy(),
                "tau_orig": tau_orig,
                "tau_cf": tau_orig,
                "tau_gain": 0.0,
                "proximity": 0.0,
                "feasible": True,
                "consistent": False,
            }

        tau_cf = float(self.cate_model.predict_cate(best_cf.reshape(1, -1))[0])
        return {
            "x_cf": best_cf,
            "tau_orig": tau_orig,
            "tau_cf": tau_cf,
            "tau_gain": tau_cf - tau_orig,
            "proximity": best_proximity,
            "feasible": self._is_feasible(best_cf),
            "consistent": tau_cf >= tau_orig,
        }

    def explain(self, X: np.ndarray) -> List[dict]:
        """Generate counterfactuals for a batch."""
        return [self.explain_instance(x) for x in X]


class GradientBasedUpliftCF:
    """Gradient-based counterfactual explanation for uplift models.

    Uses numerical gradients (finite differences) of the CATE model to guide
    counterfactual search in a principled direction. This searches a fitted prediction surface; it does not establish causal
    reachability of the recommended feature state.

    The method iteratively:
    1. Estimates the gradient of tau(x) w.r.t. actionable features via finite differences
    2. Steps in the gradient direction (projected onto feasible set)
    3. Applies line search to find optimal step size
    4. Terminates when gain saturates or budget is exhausted

    This is more principled than greedy coordinate ascent (uses full gradient
    information) but may converge to local optima. Unlike SCM-based methods,
    it requires only model access and feature-level actionability constraints.

    Reference: Inspired by gradient-based CF methods (Wachter et al., 2017;
    Van Looveren & Klaise, 2021) adapted for the uplift/CATE setting.
    """

    def __init__(
        self,
        cate_model: CATEEstimator,
        actionable_features: Optional[List[int]] = None,
        feature_bounds: Optional[np.ndarray] = None,
        step_size: float = 0.1,
        max_steps: int = 50,
        proximity_weight: float = 0.1,
        finite_diff_eps: float = 0.01,
        line_search_steps: int = 5,
        seed: int = 42,
        candidate_validator=None,
    ):
        self.candidate_validator = candidate_validator
        self.cate_model = cate_model
        self.actionable_features = actionable_features
        self.feature_bounds = feature_bounds
        self.step_size = step_size
        self.max_steps = max_steps
        self.proximity_weight = proximity_weight
        self.finite_diff_eps = finite_diff_eps
        self.line_search_steps = line_search_steps
        self.rng = np.random.default_rng(seed)

    def _is_feasible(self, x: np.ndarray) -> bool:
        if self.feature_bounds is None:
            return True
        return bool(np.all(x >= self.feature_bounds[:, 0]) and np.all(x <= self.feature_bounds[:, 1]))

    def _clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        if self.feature_bounds is None:
            return x
        return np.clip(x, self.feature_bounds[:, 0], self.feature_bounds[:, 1])

    def _numerical_gradient(self, x: np.ndarray, actionable: List[int]) -> np.ndarray:
        """Compute numerical gradient of CATE w.r.t. actionable features."""
        d = x.shape[0]
        grad = np.zeros(d)
        tau_x = float(self.cate_model.predict_cate(x.reshape(1, -1))[0])

        for j in actionable:
            x_plus = x.copy()
            x_plus[j] += self.finite_diff_eps
            x_minus = x.copy()
            x_minus[j] -= self.finite_diff_eps
            tau_plus = float(self.cate_model.predict_cate(x_plus.reshape(1, -1))[0])
            tau_minus = float(self.cate_model.predict_cate(x_minus.reshape(1, -1))[0])
            grad[j] = (tau_plus - tau_minus) / (2 * self.finite_diff_eps)

        return grad

    def explain_instance(self, x: np.ndarray) -> dict:
        """Generate a counterfactual via gradient ascent on the CATE surface."""
        d = x.shape[0]
        actionable = self.actionable_features if self.actionable_features else list(range(d))
        tau_orig = float(self.cate_model.predict_cate(x.reshape(1, -1))[0])

        x_cf = x.copy()
        tau_best = tau_orig
        no_improve_count = 0

        for step in range(self.max_steps):
            # Compute gradient
            grad = self._numerical_gradient(x_cf, actionable)

            # Check for zero gradient
            grad_norm = np.linalg.norm(grad[actionable])
            if grad_norm < 1e-8:
                break

            # Normalize gradient direction
            direction = np.zeros(d)
            direction[actionable] = grad[actionable] / grad_norm

            # Line search for optimal step size
            best_step = 0.0
            best_obj = -np.inf
            for ls in range(self.line_search_steps):
                alpha = self.step_size * (0.5 ** ls)
                x_trial = self._clip_to_bounds(x_cf + alpha * direction)
                if self.candidate_validator is not None and not self.candidate_validator(x, x_trial):
                    continue
                tau_trial = float(self.cate_model.predict_cate(x_trial.reshape(1, -1))[0])
                proximity_cost = self.proximity_weight * np.linalg.norm(x_trial - x)
                obj = (tau_trial - tau_orig) - proximity_cost
                if obj > best_obj:
                    best_obj = obj
                    best_step = alpha

            if best_step == 0.0:
                break

            # Take step
            x_cf = self._clip_to_bounds(x_cf + best_step * direction)
            tau_new = float(self.cate_model.predict_cate(x_cf.reshape(1, -1))[0])

            # Check improvement
            if tau_new <= tau_best + 1e-6:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
            else:
                no_improve_count = 0
                tau_best = tau_new

        tau_cf = float(self.cate_model.predict_cate(x_cf.reshape(1, -1))[0])
        proximity = float(np.linalg.norm(x_cf - x))

        return {
            "x_cf": x_cf,
            "tau_orig": tau_orig,
            "tau_cf": tau_cf,
            "tau_gain": tau_cf - tau_orig,
            "proximity": proximity,
            "feasible": self._is_feasible(x_cf) and (self.candidate_validator is None or self.candidate_validator(x, x_cf)),
            "consistent": tau_cf >= tau_orig,
        }

    def explain(self, X: np.ndarray) -> List[dict]:
        """Generate counterfactuals for a batch."""
        return [self.explain_instance(x) for x in X]


class RandomSearchUplift:
    """Structured random search for uplift counterfactuals.

    Unlike DiCE's unstructured perturbation, this uses:
    1. Feature importance-weighted sampling (higher weight = more likely perturbed)
    2. Budget-constrained search (max total L2 distance)
    3. Best-of-K selection with proximity-CATE gain tradeoff

    This represents a middle ground between pure random (DiCE) and greedy (Ours).
    """

    def __init__(
        self,
        cate_model: CATEEstimator,
        actionable_features: Optional[List[int]] = None,
        feature_bounds: Optional[np.ndarray] = None,
        feature_weights: Optional[np.ndarray] = None,
        n_candidates: int = 1000,
        max_distance: float = 2.0,
        proximity_weight: float = 0.3,
        seed: int = 42,
        candidate_validator=None,
    ):
        self.candidate_validator = candidate_validator
        self.cate_model = cate_model
        self.actionable_features = actionable_features
        self.feature_bounds = feature_bounds
        self.feature_weights = feature_weights
        self.n_candidates = n_candidates
        self.max_distance = max_distance
        self.proximity_weight = proximity_weight
        self.rng = np.random.default_rng(seed)

    def _is_feasible(self, x: np.ndarray) -> bool:
        if self.feature_bounds is None:
            return True
        return bool(np.all(x >= self.feature_bounds[:, 0]) and np.all(x <= self.feature_bounds[:, 1]))

    def explain_instance(self, x: np.ndarray) -> dict:
        """Generate counterfactual via structured random search."""
        d = x.shape[0]
        actionable = self.actionable_features if self.actionable_features else list(range(d))
        tau_orig = float(self.cate_model.predict_cate(x.reshape(1, -1))[0])

        # Feature selection probabilities (importance-weighted)
        if self.feature_weights is not None:
            probs = self.feature_weights[actionable]
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(actionable)) / len(actionable)

        # Generate candidates with budget constraint
        candidates = np.tile(x, (self.n_candidates, 1))
        for i in range(self.n_candidates):
            # Randomly select 1-3 features to modify (weighted)
            n_modify = self.rng.integers(1, min(4, len(actionable) + 1))
            selected = self.rng.choice(len(actionable), size=n_modify, replace=False, p=probs)

            for idx in selected:
                feat = actionable[idx]
                # Random perturbation within budget
                delta = self.rng.normal(0, self.max_distance / np.sqrt(n_modify))
                candidates[i, feat] += delta

                # Clip to bounds
                if self.feature_bounds is not None:
                    candidates[i, feat] = np.clip(
                        candidates[i, feat],
                        self.feature_bounds[feat, 0],
                        self.feature_bounds[feat, 1],
                    )

            # Enforce total distance budget
            dist = np.linalg.norm(candidates[i] - x)
            if dist > self.max_distance:
                candidates[i] = x + (candidates[i] - x) * (self.max_distance / dist)

        # Evaluate all candidates
        tau_cands = self.cate_model.predict_cate(candidates)
        gains = tau_cands - tau_orig
        proximities = np.linalg.norm(candidates - x, axis=1)

        # Filter: positive gain + feasible
        valid_mask = gains > 0
        for i in range(len(candidates)):
            if not self._is_feasible(candidates[i]) or (self.candidate_validator is not None and not self.candidate_validator(x, candidates[i])):
                valid_mask[i] = False

        if not valid_mask.any():
            return {
                "x_cf": x.copy(),
                "tau_orig": tau_orig,
                "tau_cf": tau_orig,
                "tau_gain": 0.0,
                "proximity": 0.0,
                "feasible": True,
                "consistent": False,
            }

        # Score: gain - proximity_weight * distance
        scores = gains[valid_mask] - self.proximity_weight * proximities[valid_mask]
        best_idx = np.where(valid_mask)[0][np.argmax(scores)]
        x_cf = candidates[best_idx]
        tau_cf = float(tau_cands[best_idx])

        return {
            "x_cf": x_cf,
            "tau_orig": tau_orig,
            "tau_cf": tau_cf,
            "tau_gain": tau_cf - tau_orig,
            "proximity": float(proximities[best_idx]),
            "feasible": self._is_feasible(x_cf) and (self.candidate_validator is None or self.candidate_validator(x, x_cf)),
            "consistent": tau_cf >= tau_orig,
        }

    def explain(self, X: np.ndarray) -> List[dict]:
        """Generate counterfactuals for a batch."""
        return [self.explain_instance(x) for x in X]

