"""Wrappers for CATE estimators used as the black-box models to be explained."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


class CATEEstimator(ABC):
    """Abstract base class for CATE estimators."""

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "CATEEstimator":
        return self

    @abstractmethod
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """Return estimated CATE tau(x) for each row of X."""
        raise NotImplementedError

    def predict_outcomes(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu0(x), mu1(x)) if available, else (tau(x), tau(x))."""
        tau = self.predict_cate(X)
        return tau, tau


class SLearner(CATEEstimator):
    """S-Learner: treatment as a feature."""

    def __init__(self, base_learner: Optional[object] = None):
        if base_learner is None:
            base_learner = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        self.model = base_learner

    def fit(self, X, T, Y):
        X_aug = np.hstack([X, T.reshape(-1, 1)])
        self.model.fit(X_aug, Y)
        return self

    def predict_cate(self, X):
        X0 = np.hstack([X, np.zeros((X.shape[0], 1))])
        X1 = np.hstack([X, np.ones((X.shape[0], 1))])
        return self.model.predict(X1) - self.model.predict(X0)


class TLearner(CATEEstimator):
    """T-Learner: separate models for treated and control."""

    def __init__(self, base_learner: Optional[object] = None):
        if base_learner is None:
            base_learner = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        self.model_t = base_learner
        self.model_c = clone(base_learner)

    def fit(self, X, T, Y):
        self.model_t.fit(X[T == 1], Y[T == 1])
        self.model_c.fit(X[T == 0], Y[T == 0])
        return self

    def predict_cate(self, X):
        return self.model_t.predict(X) - self.model_c.predict(X)

    def predict_outcomes(self, X):
        return self.model_c.predict(X), self.model_t.predict(X)


class XLearner(CATEEstimator):
    """X-Learner with propensity-weighted pseudo-outcome."""

    def __init__(
        self,
        base_learner: Optional[object] = None,
        propensity_learner: Optional[object] = None,
    ):
        if base_learner is None:
            base_learner = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        if propensity_learner is None:
            propensity_learner = LogisticRegression(max_iter=1000)
        self.model_tau1 = clone(base_learner)
        self.model_tau0 = clone(base_learner)
        self.propensity_model = propensity_learner

    def fit(self, X, T, Y):
        self.model_tau1.fit(X[T == 1], Y[T == 1])
        self.model_tau0.fit(X[T == 0], Y[T == 0])
        D1 = Y[T == 1] - self.model_tau0.predict(X[T == 1])
        D0 = self.model_tau1.predict(X[T == 0]) - Y[T == 0]
        self.model_tau1.fit(X[T == 1], D1)
        self.model_tau0.fit(X[T == 0], D0)
        self.propensity_model.fit(X, T)
        return self

    def predict_cate(self, X):
        e = self.propensity_model.predict_proba(X)[:, 1]
        tau1 = self.model_tau1.predict(X)
        tau0 = self.model_tau0.predict(X)
        return e * tau0 + (1 - e) * tau1


class RLearner(CATEEstimator):
    """R-Learner: residual-on-residual regression."""

    def __init__(
        self,
        base_learner: Optional[object] = None,
        propensity_learner: Optional[object] = None,
        outcome_learner: Optional[object] = None,
    ):
        if base_learner is None:
            base_learner = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        if propensity_learner is None:
            propensity_learner = LogisticRegression(max_iter=1000)
        if outcome_learner is None:
            outcome_learner = Ridge()
        self.cate_model = base_learner
        self.propensity_model = propensity_learner
        self.outcome_model = outcome_learner

    def fit(self, X, T, Y):
        self.outcome_model.fit(X, Y)
        m = self.outcome_model.predict(X)
        self.propensity_model.fit(X, T)
        e = np.clip(self.propensity_model.predict_proba(X)[:, 1], 0.05, 0.95)
        pseudo = (Y - m) / (T - e)
        weights = (T - e) ** 2
        self.cate_model.fit(X, pseudo, sample_weight=weights)
        return self

    def predict_cate(self, X):
        return self.cate_model.predict(X)


class DRLearner(CATEEstimator):
    """DR-Learner: doubly robust estimation of CATE (Kennedy 2023).

    Non-decomposable: directly estimates tau(X) without maintaining
    separate outcome models f_1(X) and f_0(X).
    """

    def __init__(
        self,
        base_learner: Optional[object] = None,
        propensity_learner: Optional[object] = None,
        outcome_learner: Optional[object] = None,
    ):
        if base_learner is None:
            base_learner = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        if propensity_learner is None:
            propensity_learner = LogisticRegression(max_iter=1000)
        if outcome_learner is None:
            outcome_learner = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        self.cate_model = base_learner
        self.propensity_model = propensity_learner
        self.outcome_learner_t = outcome_learner
        self.outcome_learner_c = clone(outcome_learner)

    def fit(self, X, T, Y):
        # Step 1: Nuisance estimation
        self.propensity_model.fit(X, T)
        e = np.clip(self.propensity_model.predict_proba(X)[:, 1], 0.05, 0.95)
        self.outcome_learner_t.fit(X[T == 1], Y[T == 1])
        self.outcome_learner_c.fit(X[T == 0], Y[T == 0])
        mu1 = self.outcome_learner_t.predict(X)
        mu0 = self.outcome_learner_c.predict(X)

        # Step 2: Doubly robust pseudo-outcome
        dr_pseudo = (
            (mu1 - mu0)
            + T * (Y - mu1) / e
            - (1 - T) * (Y - mu0) / (1 - e)
        )

        # Step 3: Regress pseudo-outcome on X
        self.cate_model.fit(X, dr_pseudo)
        return self

    def predict_cate(self, X):
        return self.cate_model.predict(X)


def get_estimator(name: str, **kwargs) -> CATEEstimator:
    """Factory for CATE estimators."""
    name = name.lower()
    if name == "s_learner":
        return SLearner(**kwargs)
    if name == "t_learner":
        return TLearner(**kwargs)
    if name == "x_learner":
        return XLearner(**kwargs)
    if name == "r_learner":
        return RLearner(**kwargs)
    if name == "dr_learner":
        return DRLearner(**kwargs)
    raise ValueError(f"Unknown estimator: {name}")


def clone(estimator):
    """Lightweight clone for sklearn-style estimators."""
    from sklearn.base import clone as sk_clone
    return sk_clone(estimator)
