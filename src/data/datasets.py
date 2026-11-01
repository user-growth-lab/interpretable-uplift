"""Unified data loaders for interpretable uplift modeling.

Supports:
    - IHDP (npci 100 replications, with ground-truth CATE)
    - Hillstrom Email Marketing
    - Criteo Uplift v2.1 (with optional subsampling)
    - Semi-synthetic CATE data with known effect modifiers
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class CATEDataset:
    """Container for a CATE dataset split."""

    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    tau: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    actionable_features: Optional[List[int]] = None
    effect_modifier_indices: Optional[List[int]] = None
    metadata: Optional[dict] = None

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]


def load_ihdp(
    path: str = "data/ihdp",
    replication: int = 0,
    train_frac: float = 0.8,
    standardize: bool = True,
    seed: int = 42,
) -> Tuple[CATEDataset, CATEDataset]:
    """Load IHDP npci 1-100 replication.

    Args:
        path: Directory containing `ihdp_npci_1-100.train.npz` and `.test.npz`.
        replication: Which replication to use (0-99).
        train_frac: Fraction of training data (the npz already has train/test,
            this flag is ignored; kept for API consistency).
        standardize: Whether to standardize continuous features.
        seed: Random seed.

    Returns:
        (train_dataset, test_dataset) with ground-truth CATE `tau`.
    """
    train_path = os.path.join(path, "ihdp_npci_1-100.train.npz")
    test_path = os.path.join(path, "ihdp_npci_1-100.test.npz")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"IHDP train file not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"IHDP test file not found at {test_path}")

    train = np.load(train_path)
    test = np.load(test_path)

    X_tr = train["x"][:, :, replication]  # (n, d)
    T_tr = train["t"][:, replication].astype(int)
    Y_tr = train["yf"][:, replication]
    mu1_tr = train["mu1"][:, replication]
    mu0_tr = train["mu0"][:, replication]
    tau_tr = mu1_tr - mu0_tr

    X_te = test["x"][:, :, replication]
    T_te = test["t"][:, replication].astype(int)
    Y_te = test["yf"][:, replication]
    mu1_te = test["mu1"][:, replication]
    mu0_te = test["mu0"][:, replication]
    tau_te = mu1_te - mu0_te

    feature_names = [f"x{i}" for i in range(X_tr.shape[1])]

    if standardize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    train_ds = CATEDataset(
        X=X_tr, T=T_tr, Y=Y_tr, tau=tau_tr, feature_names=feature_names
    )
    test_ds = CATEDataset(
        X=X_te, T=T_te, Y=Y_te, tau=tau_te, feature_names=feature_names
    )
    return train_ds, test_ds


def load_hillstrom(
    path: str = "data/hillstrom/hillstrom.csv",
    treatment_col: str = "segment",
    control_value: str = "No E-Mail",
    outcome_col: str = "visit",
    drop_cols: Optional[List[str]] = None,
    train_frac: float = 0.8,
    standardize: bool = True,
    seed: int = 42,
    preprocess_scope: str = "legacy_full",
) -> Tuple[CATEDataset, CATEDataset]:
    """Load Hillstrom email marketing data as binary treatment uplift dataset.

    By default uses `visit` as outcome and treats `No E-Mail` as control.
    The multi-treatment arms (Mens/Womens) are pooled into a single treatment.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hillstrom file not found at {path}")

    df = pd.read_csv(path)

    if drop_cols is None:
        drop_cols = ["segment", outcome_col, "conversion", "spend"]

    T = (df[treatment_col] != control_value).astype(int).values
    Y = df[outcome_col].astype(float).values
    if preprocess_scope not in {"legacy_full", "train_only"}:
        raise ValueError("preprocess_scope must be legacy_full or train_only")
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=1-train_frac,
                                      random_state=seed, stratify=T)
    fit_idx = np.arange(len(df)) if preprocess_scope == "legacy_full" else tr_idx
    prep = {"scope": preprocess_scope, "fit_row_indices": fit_idx.tolist(),
            "dataset": "hillstrom", "outcome": outcome_col}

    X_raw = df.drop(columns=drop_cols)
    feature_names = X_raw.columns.tolist()

    # One-hot encode categoricals
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    X_num = X_raw[num_cols].values.astype(float)
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        encoder.fit(X_raw.iloc[fit_idx][cat_cols])
        X_cat = encoder.transform(X_raw[cat_cols])
        prep["categorical_groups"] = {c: cats.tolist() for c, cats in zip(cat_cols, encoder.categories_)}
        X = np.hstack([X_num, X_cat])
        feature_names = num_cols + [
            f"{c}_{cat}" for c, cats in zip(cat_cols, encoder.categories_) for cat in cats[1:]
        ]
    else:
        X = X_num

    if standardize:
        scaler = StandardScaler()
        scaler.fit(X[fit_idx])
        X = scaler.transform(X)
        prep["mean"] = scaler.mean_.tolist()
        prep["scale"] = scaler.scale_.tolist()
    prep["numeric_fields"] = num_cols
    prep["feature_names"] = feature_names

    indices = np.arange(X.shape[0])
    tr_idx, te_idx = train_test_split(
        indices, test_size=1 - train_frac, random_state=seed, stratify=T
    )

    train_ds = CATEDataset(
        X=X[tr_idx], T=T[tr_idx], Y=Y[tr_idx], feature_names=feature_names
    )
    test_ds = CATEDataset(
        X=X[te_idx], T=T[te_idx], Y=Y[te_idx], feature_names=feature_names
    )
    train_ds.metadata = dict(prep, row_indices=tr_idx.tolist())
    test_ds.metadata = dict(prep, row_indices=te_idx.tolist())
    return train_ds, test_ds


def load_criteo(
    path: str = "data/criteo/criteo-uplift-v2.1.csv.gz",
    sample_n: Optional[int] = None,
    treatment_col: str = "treatment",
    outcome_col: str = "visit",
    train_frac: float = 0.8,
    standardize: bool = True,
    seed: int = 42,
    preprocess_scope: str = "legacy_full",
) -> Tuple[CATEDataset, CATEDataset]:
    """Load Criteo Uplift v2.1 with optional subsampling.

    The original file has columns: f0-f11, treatment, conversion, visit, exposure.
    We use `visit` as default outcome and `treatment` as treatment indicator.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Criteo file not found at {path}")

    if sample_n is not None:
        # Approximate total rows; use skiprows for random sampling
        n_total = 13_979_592  # known size of Criteo v2.1
        rng = np.random.default_rng(seed)
        skip = sorted(rng.choice(n_total - 1, size=n_total - sample_n, replace=False) + 1)
        df = pd.read_csv(path, skiprows=skip)
    else:
        df = pd.read_csv(path)

    T = df[treatment_col].astype(int).values
    Y = df[outcome_col].astype(float).values

    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values.astype(float)
    feature_names = feature_cols

    if preprocess_scope not in {"legacy_full", "train_only"}:
        raise ValueError("preprocess_scope must be legacy_full or train_only")
    tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=1-train_frac,
                                      random_state=seed, stratify=T)
    fit_idx = np.arange(len(X)) if preprocess_scope == "legacy_full" else tr_idx
    prep = {"scope": preprocess_scope, "fit_row_indices": fit_idx.tolist(),
            "dataset": "criteo", "row_index_scope": "loaded_subsample"}

    if standardize:
        scaler = StandardScaler()
        scaler.fit(X[fit_idx])
        X = scaler.transform(X)
        prep["mean"] = scaler.mean_.tolist()
        prep["scale"] = scaler.scale_.tolist()

    indices = np.arange(X.shape[0])
    tr_idx, te_idx = train_test_split(
        indices, test_size=1 - train_frac, random_state=seed, stratify=T
    )

    train_ds = CATEDataset(
        X=X[tr_idx], T=T[tr_idx], Y=Y[tr_idx], feature_names=feature_names
    )
    test_ds = CATEDataset(
        X=X[te_idx], T=T[te_idx], Y=Y[te_idx], feature_names=feature_names
    )
    train_ds.metadata = dict(prep, row_indices=tr_idx.tolist())
    test_ds.metadata = dict(prep, row_indices=te_idx.tolist())
    return train_ds, test_ds


def make_semisynthetic_ihdp(
    base_path: str = "data/ihdp",
    replication: int = 0,
    effect_modifier_indices: Optional[List[int]] = None,
    n_modifiers: int = 5,
    signal_strength: float = 1.0,
    noise_scale: float = 0.5,
    nonlinear: bool = False,
    add_confounded_features: bool = False,
    confound_corr: float = 0.7,
    interaction_terms: Optional[List[Tuple[int, int]]] = None,
    interaction_strength: float = 1.0,
    train_frac: float = 0.8,
    standardize: bool = True,
    seed: int = 42,
) -> Tuple[CATEDataset, CATEDataset]:
    """Create semi-synthetic IHDP with known effect modifiers.

    A noise-free conditional effect g(X) depends on a small feature set:
        g(X) = signal_strength * sum_{j in S} g_j(X_j)
             + interaction_strength * sum_{(i,j) in interactions} X_i * X_j.
    The returned legacy field ``tau`` stores the realized individual effect
    theta = g(X) + eta, where eta is an independent zero-mean Gaussian draw.
    Outcomes use that same theta. Thus a saved-tau targeting calculation
    evaluates realized simulated individual effects, not noise-free CATE g(X).

    When `standardize=True`, the CATE is built on standardized features so that
    the ground truth is aligned with the model inputs.

    When `add_confounded_features=True`, appends features that are correlated
    with true effect modifiers but do NOT causally affect the treatment effect.

    Args:
        interaction_terms: List of (feature_i, feature_j) pairs that have
            multiplicative interaction effects on CATE. These are ground-truth
            interactions for evaluating TESV-I.
        interaction_strength: Coefficient for interaction terms.

    Returns:
        Train/test datasets with realized simulated effects in ``tau`` and metadata.
    """
    train_path = os.path.join(base_path, "ihdp_npci_1-100.train.npz")
    test_path = os.path.join(base_path, "ihdp_npci_1-100.test.npz")

    train = np.load(train_path)
    test = np.load(test_path)

    X_tr_raw = train["x"][:, :, replication]
    T_tr = train["t"][:, replication].astype(int)
    mu1_tr = train["mu1"][:, replication]
    mu0_tr = train["mu0"][:, replication]

    X_te_raw = test["x"][:, :, replication]
    T_te = test["t"][:, replication].astype(int)
    mu1_te = test["mu1"][:, replication]
    mu0_te = test["mu0"][:, replication]

    rng = np.random.default_rng(seed)
    if effect_modifier_indices is None:
        effect_modifier_indices = sorted(rng.choice(X_tr_raw.shape[1], size=n_modifiers, replace=False))

    # Standardize first so that tau is built on the same scale the model sees
    if standardize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
    else:
        X_tr = X_tr_raw.copy()
        X_te = X_te_raw.copy()

    def _build_tau(X: np.ndarray) -> np.ndarray:
        tau = np.zeros(X.shape[0])
        for j in effect_modifier_indices:
            if nonlinear:
                tau += signal_strength * np.sin(np.pi * X[:, j])
            else:
                tau += signal_strength * X[:, j]
        # Add interaction terms
        if interaction_terms is not None:
            for i_feat, j_feat in interaction_terms:
                tau += interaction_strength * X[:, i_feat] * X[:, j_feat]
        tau += rng.normal(0, noise_scale, size=X.shape[0])
        return tau

    tau_tr = _build_tau(X_tr)
    tau_te = _build_tau(X_te)

    # Generate outcomes under treatment/control using the original mu0 baseline
    Y_tr = T_tr * (mu0_tr + tau_tr) + (1 - T_tr) * mu0_tr
    Y_te = T_te * (mu0_te + tau_te) + (1 - T_te) * mu0_te

    feature_names = [f"x{i}" for i in range(X_tr.shape[1])]

    # Optionally add confounded features correlated with true modifiers
    if add_confounded_features:
        n_confounded = len(effect_modifier_indices)
        noise_tr = rng.normal(0, 1, size=(X_tr.shape[0], n_confounded))
        noise_te = rng.normal(0, 1, size=(X_te.shape[0], n_confounded))
        confounded_tr = confound_corr * X_tr[:, effect_modifier_indices] + np.sqrt(1 - confound_corr**2) * noise_tr
        confounded_te = confound_corr * X_te[:, effect_modifier_indices] + np.sqrt(1 - confound_corr**2) * noise_te
        X_tr = np.hstack([X_tr, confounded_tr])
        X_te = np.hstack([X_te, confounded_te])
        for i in range(n_confounded):
            feature_names.append(f"x{X_tr.shape[1] - n_confounded + i}_conf")

    train_ds = CATEDataset(
        X=X_tr,
        T=T_tr,
        Y=Y_tr,
        tau=tau_tr,
        feature_names=feature_names,
    )
    test_ds = CATEDataset(
        X=X_te,
        T=T_te,
        Y=Y_te,
        tau=tau_te,
        feature_names=feature_names,
    )

    # Attach metadata for evaluation
    train_ds.effect_modifier_indices = effect_modifier_indices
    test_ds.effect_modifier_indices = effect_modifier_indices
    train_ds.interaction_terms = interaction_terms if interaction_terms else []
    test_ds.interaction_terms = interaction_terms if interaction_terms else []

    return train_ds, test_ds


def load_dataset(name: str, **kwargs) -> Tuple[CATEDataset, CATEDataset]:
    """Dispatch loader by dataset name."""
    name = name.lower()
    if name == "ihdp":
        return load_ihdp(**kwargs)
    if name == "hillstrom":
        return load_hillstrom(**kwargs)
    if name == "criteo":
        return load_criteo(**kwargs)
    if name == "megafon":
        return load_megafon(**kwargs)
    if name == "semisynthetic_ihdp":
        return make_semisynthetic_ihdp(**kwargs)
    raise ValueError(f"Unknown dataset: {name}")


def load_megafon(
    sample_n: Optional[int] = 100000,
    train_frac: float = 0.8,
    standardize: bool = True,
    seed: int = 42,
    preprocess_scope: str = "legacy_full",
) -> Tuple[CATEDataset, CATEDataset]:
    """Load the provider-generated synthetic MegaFon competition benchmark.

    The Megafon dataset contains ~600K samples with 50 anonymized features,
    binary treatment, and binary outcome. The provider describes these as
    synthetic data approximating a real case. Treatment randomization and
    pre-treatment timing of the anonymous features are not independently
    verified; this loader does not establish a second real-world RCT.

    Args:
        sample_n: Number of samples to subsample (None = use all).
        train_frac: Train/test split ratio.
        standardize: Whether to standardize features.
        seed: Random seed.
        preprocess_scope: ``legacy_full`` replays full-sample imputation/scaling;
            ``train_only`` fits those transformations on outer training rows only.

    Returns:
        (train_dataset, test_dataset).
    """
    try:
        from sklift.datasets import fetch_megafon
    except ImportError:
        raise ImportError("scikit-uplift is required: pip install scikit-uplift>=0.5.0")

    bunch = fetch_megafon()
    df = bunch.data if hasattr(bunch, "data") else bunch

    # Extract features, treatment, outcome
    if hasattr(bunch, "target_name"):
        outcome_col = bunch.target_name
        treatment_col = bunch.treatment_name
    else:
        # Default column names for Megafon dataset
        outcome_col = "target"
        treatment_col = "treatment_group"

    if isinstance(df, pd.DataFrame) and treatment_col in df.columns and outcome_col in df.columns:
        T = df[treatment_col].astype(int).values
        Y = df[outcome_col].astype(float).values
        feature_cols = [c for c in df.columns if c not in [treatment_col, outcome_col]]
        X = df[feature_cols].values.astype(float)
        feature_names = feature_cols
    else:
        # sklift>=0.5 schema: bunch.data holds features only; treatment/target are
        # separate attributes and treatment is a string ('treatment'/'control').
        X = df.values.astype(float) if isinstance(df, pd.DataFrame) else np.asarray(bunch.data, dtype=float)
        feature_names = list(df.columns) if isinstance(df, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]
        t_raw = np.asarray(bunch.treatment)
        if t_raw.dtype.kind in {"U", "S", "O"}:
            T = (t_raw.astype(str) == "treatment").astype(int)
        else:
            T = t_raw.astype(int)
        Y = np.asarray(bunch.target).astype(float)

    # Subsample if requested
    rng = np.random.default_rng(seed)
    source_rows = np.arange(len(X))
    if sample_n is not None and X.shape[0] > sample_n:
        idx = rng.choice(X.shape[0], size=sample_n, replace=False)
        X, T, Y = X[idx], T[idx], Y[idx]
        source_rows = source_rows[idx]

    if preprocess_scope not in {"legacy_full", "train_only"}:
        raise ValueError("preprocess_scope must be legacy_full or train_only")
    tr_idx, te_idx = train_test_split(np.arange(len(X)), test_size=1-train_frac,
                                      random_state=seed, stratify=T)
    fit_idx = np.arange(len(X)) if preprocess_scope == "legacy_full" else tr_idx
    prep = {"scope": preprocess_scope, "fit_row_indices": source_rows[fit_idx].tolist(),
            "dataset": "megafon_synthetic_competition", "randomization_verified": False}

    # Handle NaN
    if np.any(np.isnan(X)):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X[fit_idx])
        X = imputer.transform(X)
        prep["imputer_statistics"] = imputer.statistics_.tolist()

    if standardize:
        scaler = StandardScaler()
        scaler.fit(X[fit_idx])
        X = scaler.transform(X)
        prep["mean"] = scaler.mean_.tolist()
        prep["scale"] = scaler.scale_.tolist()

    indices = np.arange(X.shape[0])
    tr_idx, te_idx = train_test_split(
        indices, test_size=1 - train_frac, random_state=seed, stratify=T
    )

    train_ds = CATEDataset(
        X=X[tr_idx], T=T[tr_idx], Y=Y[tr_idx], feature_names=feature_names
    )
    test_ds = CATEDataset(
        X=X[te_idx], T=T[te_idx], Y=Y[te_idx], feature_names=feature_names
    )
    train_ds.metadata = dict(prep, row_indices=source_rows[tr_idx].tolist())
    test_ds.metadata = dict(prep, row_indices=source_rows[te_idx].tolist())
    return train_ds, test_ds
