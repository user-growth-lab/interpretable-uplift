"""Constrained feature-space target recommendations for a fitted CATE model.

Bounds and an optional candidate validator constrain outputs. They do not imply
causal interventions, treatment positivity, joint support or real-world reachability.
The legacy class name is retained for source compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.models.cate_estimators import CATEEstimator


@dataclass
class CounterfactualExplanation:
    """A counterfactual explanation for a single instance."""

    x_orig: np.ndarray
    x_cf: np.ndarray
    tau_orig: float
    tau_cf: float
    changed_features: List[int]
    proximity: float
    feasible: bool

    @property
    def tau_gain(self) -> float:
        return self.tau_cf - self.tau_orig


class CausalCounterfactualExplainer:
    """Search for a nearby feature state with greater predicted uplift."""

    def __init__(
        self,
        cate_model: CATEEstimator,
        actionable_features: Optional[List[int]] = None,
        feature_bounds: Optional[np.ndarray] = None,
        max_changes: int = 3,
        step_size: float = 0.1,
        max_steps: int = 20,
        proximity_weight: float = 0.1,
        seed: int = 42,
        candidate_validator=None,
    ):
        """Initialize counterfactual explainer.

        Args:
            cate_model: Fitted CATE model.
            actionable_features: Indices of features that can be changed.
                If None, all features are actionable.
            feature_bounds: Array of shape (d, 2) with [min, max] for each feature.
            max_changes: Maximum number of features to modify.
            step_size: Step size for each modification.
            max_steps: Maximum number of greedy steps.
            proximity_weight: Weight for proximity penalty in objective.
            seed: Random seed.
        """
        self.candidate_validator = candidate_validator
        self.cate_model = cate_model
        self.actionable_features = actionable_features
        self.feature_bounds = feature_bounds
        self.max_changes = max_changes
        self.step_size = step_size
        self.max_steps = max_steps
        self.proximity_weight = proximity_weight
        self.rng = np.random.default_rng(seed)

    def _is_feasible(self, x: np.ndarray) -> bool:
        """Check whether x respects feature bounds."""
        if self.feature_bounds is None:
            return True
        return bool(np.all(x >= self.feature_bounds[:, 0]) and np.all(x <= self.feature_bounds[:, 1]))

    def explain_instance(
        self,
        x: np.ndarray,
        direction: str = "increase",
    ) -> CounterfactualExplanation:
        """Generate a counterfactual for a single instance.

        Args:
            x: Original instance of shape (d,).
            direction: "increase" or "decrease" uplift.

        Returns:
            CounterfactualExplanation object.
        """
        d = x.shape[0]
        actionable = self.actionable_features if self.actionable_features is not None else list(range(d))
        sign = 1.0 if direction == "increase" else -1.0

        x_cf = x.copy()
        changed_features = set()
        tau_orig = float(self.cate_model.predict_cate(x.reshape(1, -1))[0])
        tau_best = tau_orig

        for _ in range(self.max_steps):
            if len(changed_features) >= self.max_changes:
                break

            best_feature = None
            best_obj = -np.inf
            best_x = None
            best_tau = tau_best

            for j in actionable:
                for delta in [self.step_size, -self.step_size]:
                    x_temp = x_cf.copy()
                    x_temp[j] += sign * delta
                    if not self._is_feasible(x_temp) or (self.candidate_validator is not None and not self.candidate_validator(x, x_temp)):
                        continue

                    tau_temp = float(self.cate_model.predict_cate(x_temp.reshape(1, -1))[0])
                    gain = sign * (tau_temp - tau_best)
                    proximity = -self.proximity_weight * np.linalg.norm(x_temp - x)
                    obj = gain + proximity

                    if obj > best_obj:
                        best_obj = obj
                        best_feature = j
                        best_x = x_temp
                        best_tau = tau_temp

            if best_feature is None or best_obj <= 0:
                break

            x_cf = best_x
            changed_features.add(best_feature)
            tau_best = best_tau

        tau_cf = float(self.cate_model.predict_cate(x_cf.reshape(1, -1))[0])
        feasible = self._is_feasible(x_cf) and (self.candidate_validator is None or self.candidate_validator(x, x_cf))
        proximity = float(np.linalg.norm(x_cf - x))

        return CounterfactualExplanation(
            x_orig=x,
            x_cf=x_cf,
            tau_orig=tau_orig,
            tau_cf=tau_cf,
            changed_features=sorted(changed_features),
            proximity=proximity,
            feasible=feasible,
        )

    def explain(
        self,
        X: np.ndarray,
        direction: str = "increase",
    ) -> List[CounterfactualExplanation]:
        """Generate counterfactuals for a batch of instances."""
        return [self.explain_instance(x, direction) for x in X]
