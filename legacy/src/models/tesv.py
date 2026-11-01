"""Treatment Effect Shapley Value (TESV) for CATE explanation.

TESV measures the contribution of each feature to the conditional average
treatment effect (CATE) by replacing the standard Shapley value function with
one that targets the treatment effect:

    v_tau(S) = E[ tau(X) | do(X_S = x_S) ] - E[ tau(X) ]

In practice, we approximate the do-operator by marginalizing over the
complement features using samples from the observed distribution:

    E[ tau(X) | do(X_S = x_S) ] ~= (1/M) sum_{m=1}^M tau(x_S, X_{~S}^{(m)})
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from src.models.cate_estimators import CATEEstimator


class TESVExplainer:
    """Treatment Effect Shapley Value explainer for a fitted CATE model."""

    def __init__(
        self,
        cate_model: CATEEstimator,
        background_data: Optional[np.ndarray] = None,
        n_background: int = 100,
        n_permutations: int = 50,
        seed: int = 42,
    ):
        """Initialize TESV explainer.

        Args:
            cate_model: Fitted CATE estimator with `predict_cate` method.
            background_data: Reference samples used to marginalize complement features.
                If None, must be provided at `explain` time.
            n_background: Number of background samples to draw if background_data
                has more rows than this.
            n_permutations: Number of Monte Carlo permutations for Shapley approximation.
            seed: Random seed.
        """
        self.cate_model = cate_model
        self.rng = np.random.default_rng(seed)

        if background_data is not None:
            self.background_data = self._prepare_background(background_data, n_background)
        else:
            self.background_data = None
        self.n_background = n_background
        self.n_permutations = n_permutations

    def _prepare_background(
        self, background_data: np.ndarray, n_background: int
    ) -> np.ndarray:
        n = background_data.shape[0]
        if n > n_background:
            idx = self.rng.choice(n, size=n_background, replace=False)
            return background_data[idx]
        return background_data

    def _interventional_expectation(
        self,
        x: np.ndarray,
        feature_subset: List[int],
        background: np.ndarray,
    ) -> float:
        """Estimate E[tau(X) | do(X_S = x_S)] by marginalizing over background.

        Args:
            x: Single instance of shape (d,).
            feature_subset: Indices S to intervene on.
            background: Background samples of shape (M, d).

        Returns:
            Scalar estimate of the interventional expectation.
        """
        if len(feature_subset) == 0:
            return float(self.cate_model.predict_cate(background).mean())

        X_perturbed = background.copy()
        X_perturbed[:, feature_subset] = x[feature_subset]
        return float(self.cate_model.predict_cate(X_perturbed).mean())

    def explain_instance(
        self,
        x: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute TESV attribution for a single instance.

        Args:
            x: Instance of shape (d,).
            background_data: Optional background samples; overrides constructor value.

        Returns:
            TESV values of shape (d,).
        """
        if background_data is None:
            background_data = self.background_data
        if background_data is None:
            raise ValueError("background_data must be provided either at init or explain time")

        background_data = self._prepare_background(background_data, self.n_background)
        d = x.shape[0]
        phi = np.zeros(d)

        # Monte Carlo permutation sampling
        for _ in range(self.n_permutations):
            perm = self.rng.permutation(d)
            subset = []
            prev_val = self._interventional_expectation(x, subset, background_data)
            for j in perm:
                subset.append(j)
                curr_val = self._interventional_expectation(x, subset, background_data)
                phi[j] += curr_val - prev_val
                prev_val = curr_val

        phi /= self.n_permutations
        return phi

    def explain(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute TESV attributions for a batch of instances.

        Args:
            X: Instances of shape (n, d).
            background_data: Optional background samples.
            show_progress: Whether to show a progress bar.

        Returns:
            TESV matrix of shape (n, d).
        """
        iterator = tqdm(X) if show_progress else X
        return np.array([self.explain_instance(x, background_data) for x in iterator])

    def global_attribution(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute mean absolute TESV attribution across the dataset.

        Args:
            X: Instances of shape (n, d).
            background_data: Optional background samples.
            show_progress: Whether to show a progress bar.

        Returns:
            Mean absolute TESV vector of shape (d,).
        """
        attributions = self.explain(X, background_data, show_progress)
        return np.abs(attributions).mean(axis=0)


class TESVWithCI(TESVExplainer):
    """TESV with confidence intervals via bootstrap resampling of background data.

    Provides attribution uncertainty quantification specific to the CATE setting.
    Because treatment effects are typically small relative to outcome levels,
    standard Shapley CI methods designed for f(X) have different signal-to-noise
    characteristics than those needed for tau(X). TESVWithCI addresses this by:
    1. Bootstrap-resampling the background data to capture marginalisation uncertainty
    2. Providing per-feature significance testing (CI excludes zero)
    3. Enabling practitioners to distinguish reliable attributions from noise
    """

    def __init__(
        self,
        cate_model,
        background_data: Optional[np.ndarray] = None,
        n_background: int = 100,
        n_permutations: int = 50,
        n_bootstrap_ci: int = 30,
        alpha: float = 0.05,
        seed: int = 42,
    ):
        """Initialize TESV with CI.

        Args:
            cate_model: Fitted CATE estimator.
            background_data: Reference samples for marginalisation.
            n_background: Background sample size.
            n_permutations: MC permutations per bootstrap.
            n_bootstrap_ci: Number of bootstrap resamples for CI construction.
            alpha: Significance level (default 0.05 for 95% CI).
            seed: Random seed.
        """
        super().__init__(
            cate_model=cate_model,
            background_data=background_data,
            n_background=n_background,
            n_permutations=n_permutations,
            seed=seed,
        )
        self.n_bootstrap_ci = n_bootstrap_ci
        self.alpha = alpha
        self._seed = seed

    def explain_with_ci(
        self,
        x: np.ndarray,
        full_background: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute TESV attribution with confidence intervals for a single instance.

        Args:
            x: Instance of shape (d,).
            full_background: Full background pool to resample from.

        Returns:
            Dict with keys: 'phi_mean', 'phi_lower', 'phi_upper', 'is_significant'
        """
        if full_background is None:
            full_background = self.background_data
        if full_background is None:
            raise ValueError("full_background must be provided")

        d = x.shape[0]
        bootstrap_phis = np.zeros((self.n_bootstrap_ci, d))

        for b in range(self.n_bootstrap_ci):
            # Resample background data
            rng_b = np.random.default_rng(self._seed + b * 1000)
            n_bg = full_background.shape[0]
            idx = rng_b.choice(n_bg, size=min(self.n_background, n_bg), replace=True)
            bg_b = full_background[idx]

            # Compute TESV with this bootstrap background
            phi_b = np.zeros(d)
            for _ in range(self.n_permutations):
                perm = rng_b.permutation(d)
                subset = []
                prev_val = self._interventional_expectation(x, subset, bg_b)
                for j in perm:
                    subset.append(j)
                    curr_val = self._interventional_expectation(x, subset, bg_b)
                    phi_b[j] += curr_val - prev_val
                    prev_val = curr_val
            phi_b /= self.n_permutations
            bootstrap_phis[b] = phi_b

        phi_mean = bootstrap_phis.mean(axis=0)
        phi_lower = np.percentile(bootstrap_phis, 100 * self.alpha / 2, axis=0)
        phi_upper = np.percentile(bootstrap_phis, 100 * (1 - self.alpha / 2), axis=0)
        is_significant = (phi_lower > 0) | (phi_upper < 0)

        return {
            "phi_mean": phi_mean,
            "phi_lower": phi_lower,
            "phi_upper": phi_upper,
            "is_significant": is_significant,
            "ci_width": phi_upper - phi_lower,
        }

    def explain_batch_with_ci(
        self,
        X: np.ndarray,
        full_background: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> dict:
        """Compute TESV-CI for a batch of instances.

        Returns:
            Dict with arrays of shape (n, d) for phi_mean, phi_lower, phi_upper
            and boolean array is_significant.
        """
        results = []
        iterator = tqdm(X, desc="TESV-CI") if show_progress else X
        for x in iterator:
            results.append(self.explain_with_ci(x, full_background))

        return {
            "phi_mean": np.array([r["phi_mean"] for r in results]),
            "phi_lower": np.array([r["phi_lower"] for r in results]),
            "phi_upper": np.array([r["phi_upper"] for r in results]),
            "is_significant": np.array([r["is_significant"] for r in results]),
            "ci_width": np.array([r["ci_width"] for r in results]),
        }

    def significant_features(
        self,
        X: np.ndarray,
        full_background: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Return indices of features with significant global attribution.

        A feature is significant if its mean |phi| has a CI that excludes zero
        across the majority of instances.
        """
        batch_result = self.explain_batch_with_ci(X, full_background, show_progress=False)
        # Feature is globally significant if significant in >50% of instances
        sig_rate = batch_result["is_significant"].mean(axis=0)
        return sorted(np.where(sig_rate > 0.5)[0].tolist())


class TESVInteractionExplainer(TESVExplainer):
    """TESV Interaction Index: quantifies pairwise feature synergy in CATE.

    Computes the Shapley Interaction Index on the interventional CATE value
    function v_tau, identifying pairs of features that jointly drive treatment
    effect heterogeneity. This is a capability unique to the TESV framework —
    outcome-level differencing cannot directly compute CATE-specific interactions.

    Formula:
        Phi_{ij}^tau(x) = Sum_{S subset N\\{i,j}}
            [|S|! (d-|S|-2)! / (d-1)!] * Delta_ij(S)

        where Delta_ij(S) = v_tau(S u {i,j}) - v_tau(S u {i}) - v_tau(S u {j}) + v_tau(S)
    """

    def __init__(
        self,
        cate_model: "CATEEstimator",
        background_data: Optional[np.ndarray] = None,
        n_background: int = 100,
        n_permutations: int = 50,
        n_interaction_samples: int = 200,
        seed: int = 42,
    ):
        super().__init__(cate_model, background_data, n_background, n_permutations, seed)
        self.n_interaction_samples = n_interaction_samples

    def _shapley_interaction_weight(self, s_size: int, d: int) -> float:
        """Compute the Shapley interaction index weight for coalition size |S|."""
        from math import factorial
        return factorial(s_size) * factorial(d - s_size - 2) / factorial(d - 1)

    def explain_interactions_instance(
        self,
        x: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute pairwise TESV Interaction Index for a single instance.

        Args:
            x: Instance of shape (d,).
            background_data: Optional background samples.

        Returns:
            Interaction matrix of shape (d, d). Entry (i, j) is the
            interaction strength between features i and j for the CATE.
        """
        if background_data is None:
            background_data = self.background_data
        if background_data is None:
            raise ValueError("background_data must be provided")

        background_data = self._prepare_background(background_data, self.n_background)
        d = x.shape[0]
        interactions = np.zeros((d, d))

        if d <= 15:
            # Exact enumeration for small d
            interactions = self._exact_interactions(x, d, background_data)
        else:
            # Monte Carlo approximation for large d
            interactions = self._mc_interactions(x, d, background_data)

        return interactions

    def _exact_interactions(
        self, x: np.ndarray, d: int, background: np.ndarray
    ) -> np.ndarray:
        """Exact interaction computation via exhaustive subset enumeration."""
        interactions = np.zeros((d, d))

        for i in range(d):
            for j in range(i + 1, d):
                phi_ij = 0.0
                others = [k for k in range(d) if k != i and k != j]
                for s_size in range(len(others) + 1):
                    weight = self._shapley_interaction_weight(s_size, d)
                    for S in combinations(others, s_size):
                        S = list(S)
                        v_ij = self._interventional_expectation(x, S + [i, j], background)
                        v_i = self._interventional_expectation(x, S + [i], background)
                        v_j = self._interventional_expectation(x, S + [j], background)
                        v_none = self._interventional_expectation(x, S, background)
                        delta = v_ij - v_i - v_j + v_none
                        phi_ij += weight * delta
                interactions[i, j] = phi_ij
                interactions[j, i] = phi_ij

        return interactions

    def _mc_interactions(
        self, x: np.ndarray, d: int, background: np.ndarray
    ) -> np.ndarray:
        """Monte Carlo approximation for high-dimensional interaction computation."""
        interactions = np.zeros((d, d))
        n_samples = self.n_interaction_samples

        for i in range(d):
            for j in range(i + 1, d):
                phi_ij = 0.0
                others = [k for k in range(d) if k != i and k != j]

                for _ in range(n_samples):
                    # Sample a random coalition size and subset
                    s_size = self.rng.integers(0, len(others) + 1)
                    if s_size > 0:
                        S = list(self.rng.choice(others, size=s_size, replace=False))
                    else:
                        S = []

                    v_ij = self._interventional_expectation(x, S + [i, j], background)
                    v_i = self._interventional_expectation(x, S + [i], background)
                    v_j = self._interventional_expectation(x, S + [j], background)
                    v_none = self._interventional_expectation(x, S, background)
                    delta = v_ij - v_i - v_j + v_none
                    phi_ij += delta

                interactions[i, j] = phi_ij / n_samples
                interactions[j, i] = phi_ij / n_samples

        return interactions

    def explain_interactions(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute TESV interaction matrices for a batch of instances.

        Args:
            X: Instances of shape (n, d).
            background_data: Optional background samples.
            show_progress: Whether to show progress bar.

        Returns:
            Interaction tensor of shape (n, d, d).
        """
        iterator = tqdm(X) if show_progress else X
        return np.array(
            [self.explain_interactions_instance(x, background_data) for x in iterator]
        )

    def top_interactions(
        self,
        X: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        k: int = 5,
        show_progress: bool = True,
    ) -> List[tuple]:
        """Return the top-K feature interaction pairs ranked by mean absolute strength.

        Args:
            X: Instances of shape (n, d).
            background_data: Optional background samples.
            k: Number of top interactions to return.
            show_progress: Whether to show progress.

        Returns:
            List of (feature_i, feature_j, mean_abs_interaction) tuples,
            sorted descending by interaction strength.
        """
        interaction_tensor = self.explain_interactions(X, background_data, show_progress)
        # Average absolute interaction across instances
        mean_interactions = np.abs(interaction_tensor).mean(axis=0)
        d = mean_interactions.shape[0]

        # Extract upper triangle entries
        pairs = []
        for i in range(d):
            for j in range(i + 1, d):
                pairs.append((i, j, mean_interactions[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:k]


class ExactTESVExplainer(TESVExplainer):
    """Exact TESV via exhaustive subset enumeration (feasible only for small d)."""

    def explain_instance(
        self,
        x: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if background_data is None:
            background_data = self.background_data
        if background_data is None:
            raise ValueError("background_data must be provided")

        background_data = self._prepare_background(background_data, self.n_background)
        d = x.shape[0]
        phi = np.zeros(d)

        for j in range(d):
            other = [k for k in range(d) if k != j]
            for s in range(d):
                subsets = list(combinations(other, s))
                weight = 1.0 / (len(subsets) * d) if len(subsets) > 0 else 0.0
                for S in subsets:
                    S = list(S)
                    v_without = self._interventional_expectation(x, S, background_data)
                    v_with = self._interventional_expectation(x, S + [j], background_data)
                    phi[j] += weight * (v_with - v_without)

        return phi
