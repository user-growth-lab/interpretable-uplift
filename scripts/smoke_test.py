#!/usr/bin/env python3
"""Fast, offline packaging checks using toy inputs; not paper reproduction.

Run from any directory: python scripts/smoke_test.py
Requires NumPy, SciPy, scikit-learn and tqdm (including their dependencies).
No benchmark data, model fitting, downloads, result files or bytecode caches.
The checks exercise the shipped algorithms without modifying their source.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import platform
import sys
from importlib.metadata import version

sys.dont_write_bytecode = True
# Use the reference path, independent of optional Numba/JIT availability.
os.environ["SCRL_BOOTSTRAP_BACKEND"] = "numpy"


def _read_only_audit(event, args):
    """Fail rather than write files, start child processes or open sockets."""
    if event == "open":
        _, mode, flags = args
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & write_flags
        ):
            raise PermissionError("Offline smoke test forbids filesystem writes")
    if event.startswith("socket.") or event in {
        "subprocess.Popen", "os.system", "os.mkdir", "os.remove", "os.rename",
        "os.rmdir", "os.link", "os.symlink", "os.truncate",
    }:
        # Import-time optional probes (e.g. NumPy's lscpu probe) handle OSError.
        raise PermissionError("Offline smoke test forbids this operation: " + event)


sys.addaudithook(_read_only_audit)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    from src.models.tesv import TESVExplainer
    from src.models.scrl import SCRL
    from src.models.counterfactual import CausalCounterfactualExplainer
except ImportError as exc:
    print(
        "Missing smoke-test dependency: {}. Install numpy scipy scikit-learn "
        "tqdm in your environment, then rerun; this script never installs packages."
        .format(exc), file=sys.stderr,
    )
    raise SystemExit(2)


ATOL = 1e-10


def require(condition, message):
    # Keep checks active under python -O as well.
    if not condition:
        raise AssertionError(message)


class LinearCATE:
    """Fixed analytic prediction function, not a fitted research estimator."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict_cate(self, X):
        return np.asarray(X) @ self.weights + 0.5


def check_tesv():
    background = np.array([
        [-1., 0., 2.], [0., 1., -3.], [1., -1., 4.], [2., 2., 5.],
    ])
    before = background.copy()
    model = LinearCATE([2., -3., 0.])
    explainer = TESVExplainer(model, background, n_background=len(background))
    permutations = np.array(list(itertools.permutations(range(3))))
    residuals = []
    for x in (np.array([.7, -.3, 8.]), np.array([-2., .5, -9.])):
        original = x.copy()
        phi = explainer.explain_instance(x, permutations=permutations)
        # Independent analytic attribution for an additive linear function.
        expected = model.weights * (x - background.mean(axis=0))
        np.testing.assert_allclose(phi, expected, atol=ATOL, rtol=0)
        difference = model.predict_cate(x[None])[0] - model.predict_cate(background).mean()
        residuals.append(float(abs(phi.sum() - difference)))
        require(residuals[-1] <= ATOL, "TESV efficiency identity failed")
        require(abs(phi[2]) <= ATOL, "Ignored coordinate received attribution")
        np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(background, before)
    return {
        "passed": True, "instances": 2, "background_rows": len(background),
        "permutations_per_instance": len(permutations),
        "max_efficiency_residual": max(residuals),
        "analytic_attributions_and_ignored_coordinate_checked": True,
    }


def check_scrl():
    # Every distinct toy row has both treatment assignments before splitting.
    grid = np.linspace(-1., 1., 60)
    X = np.repeat(np.column_stack((grid, np.sin(3 * grid), grid ** 2)), 2, axis=0)
    T = np.tile([0, 1], len(grid))
    Y = np.zeros(len(X))  # fit uses the supplied predictor, not these outcomes.
    before = (X.copy(), T.copy(), Y.copy())
    options = dict(
        n_rules=3, max_rule_length=2, min_support=.10, max_coverage=.45,
        overlap_ratio=.20, n_candidates=5, n_bootstrap=3,
        sample_split=True, support_mode="hard", seed=17,
    )
    model = LinearCATE([2., -.2, 0.])
    learner = SCRL(model, **options).fit(X, T, Y)
    discovery = learner.discovery_indices_
    selection = learner.selection_indices_
    require(not np.intersect1d(discovery, selection).size, "Expected separate rule subsets")
    require(len(discovery) + len(selection) == len(X), "Rule split lost rows")
    require(0 < len(learner.rules) <= options["n_rules"], "Non-vacuous rule selection failed")
    checked = []
    for rule in learner.rules:
        mask = rule.covers(X[selection])
        count = int(mask.sum())
        treated = int(T[selection][mask].sum())
        control = count - treated
        coverage = count / len(selection)
        # Recompute support from the actual selected rows, not audit flags.
        require(count >= max(2, options["min_support"] * len(selection)), "Too little support")
        require(min(treated, control) >= options["overlap_ratio"] * count, "Arm support failed")
        require(coverage <= options["max_coverage"], "Maximum coverage failed")
        require(len(rule.conditions) <= options["max_rule_length"], "Rule length limit failed")
        checked.append(dict(n=count, treated=treated, control=control, coverage=coverage))
    # A sample containing no controls must not produce an admissible rule.
    no_controls = SCRL(model, **options).fit(X, np.ones(len(X), dtype=int), Y)
    require(len(no_controls.rules) == 0, "Single-arm sample produced a rule")
    require(no_controls.predict_rules(X).shape == (len(X), 0), "Empty-rule output shape failed")
    for actual, original in zip((X, T, Y), before):
        np.testing.assert_array_equal(actual, original)
    return {
        "passed": True, "toy_rows": len(X), "selection_rows": len(selection),
        "selected_rules": checked, "single_arm_rejected": True,
    }


def check_recourse():
    model = LinearCATE([2., 1., 5.])
    x = np.array([0., 0., .4])
    original = x.copy()
    bounds = np.array([[0., .5], [0., .5], [0., 1.]])
    counts = {"rejected": 0}

    def validator(start, candidate):
        valid = bool(candidate[0] + candidate[1] <= .5 + ATOL and candidate[2] == start[2])
        counts["rejected"] += int(not valid)
        return valid

    options = dict(
        cate_model=model, actionable_features=[0, 1], feature_bounds=bounds,
        max_changes=2, step_size=.25, max_steps=8, proximity_weight=.05,
        candidate_validator=validator,
    )
    result = CausalCounterfactualExplainer(**options).explain_instance(x)
    changed = np.flatnonzero(np.abs(result.x_cf - x) > ATOL).tolist()
    require(result.feasible and validator(x, result.x_cf), "Candidate validator failed")
    require(np.all(result.x_cf >= bounds[:, 0]) and np.all(result.x_cf <= bounds[:, 1]), "Bounds failed")
    require(set(changed) <= {0, 1} and len(changed) <= 2, "Actionable/change constraints failed")
    require(changed == result.changed_features, "Changed-feature reporting failed")
    require(result.tau_gain > 0 and changed, "Expected a changed positive-gain toy state")
    require(counts["rejected"] > 0, "Toy test did not exercise candidate rejection")
    require(abs(result.tau_cf - model.predict_cate(result.x_cf[None])[0]) <= ATOL, "Prediction mismatch")
    limited = CausalCounterfactualExplainer(**dict(options, max_changes=1)).explain_instance(x)
    require(len(np.flatnonzero(abs(limited.x_cf - x) > ATOL)) == 1, "One-change cap failed")
    blocked = CausalCounterfactualExplainer(**dict(
        options, candidate_validator=lambda start, candidate: bool(np.array_equal(start, candidate))
    )).explain_instance(x)
    np.testing.assert_array_equal(blocked.x_cf, x)
    require(blocked.feasible and blocked.tau_gain == 0, "Blocked search should return unchanged")
    np.testing.assert_array_equal(x, original)
    return {
        "passed": True, "changed_features": changed, "predicted_gain": float(result.tau_gain),
        "bounds_actionable_and_joint_validator_checked": True,
        "one_change_cap_checked": True, "blocked_search_returns_unchanged": True,
    }


def main():
    report = {
        "purpose": "offline packaging smoke check; not reproduction or scientific validation",
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "tqdm")},
        "scrl_bootstrap_backend": "numpy",
        "filesystem_writes_and_network_forbidden": True,
        "checks": {"tesv": check_tesv(), "scrl": check_scrl(), "recourse": check_recourse()},
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
