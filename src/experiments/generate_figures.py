"""Generate figures for the Pattern Recognition paper."""

from __future__ import annotations

import os

import numpy as np

from src.data.datasets import make_semisynthetic_ihdp
from src.models.cate_estimators import TLearner
from src.models.counterfactual import CausalCounterfactualExplainer
from src.models.scrl import SCRL
from src.models.tesv import TESVExplainer
from src.visualization.plots import (
    plot_counterfactual_gains,
    plot_framework_diagram,
    plot_global_attributions,
    plot_rule_lift_curve,
)


def main():
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)

    print("Loading data and fitting model...")
    train, test = make_semisynthetic_ihdp(
        n_modifiers=5,
        signal_strength=2.0,
        noise_scale=0.3,
        seed=42,
    )
    true_modifiers = test.effect_modifier_indices
    model = TLearner()
    model.fit(train.X, train.T, train.Y)

    print("Generating framework diagram...")
    plot_framework_diagram(save_path=os.path.join(fig_dir, "framework.pdf"))

    print("Generating TESV attribution plot...")
    tesv = TESVExplainer(
        cate_model=model,
        background_data=train.X,
        n_background=50,
        n_permutations=20,
        seed=42,
    )
    attr = tesv.explain(test.X, show_progress=True)
    plot_global_attributions(
        attr,
        feature_names=test.feature_names,
        true_effect_modifiers=true_modifiers,
        top_k=15,
        save_path=os.path.join(fig_dir, "tesv_attributions.pdf"),
    )

    print("Generating SCRL rule lift curve...")
    scrl = SCRL(
        cate_model=model,
        n_rules=5,
        max_rule_length=2,
        min_support=0.1,
        overlap_ratio=0.1,
        n_candidates=10,
        n_bootstrap=5,
        seed=42,
    )
    scrl.fit(train.X, train.T, train.Y)
    tau_pred = model.predict_cate(test.X)
    plot_rule_lift_curve(
        tau_pred,
        [rule.covers(test.X) for rule in scrl.rules],
        save_path=os.path.join(fig_dir, "rule_lift_curve.pdf"),
    )

    print("Generating counterfactual gain plot...")
    cf = CausalCounterfactualExplainer(
        cate_model=model,
        actionable_features=list(range(train.X.shape[1])),
        max_changes=2,
        step_size=0.2,
        max_steps=10,
        proximity_weight=0.05,
        seed=42,
    )
    explanations = cf.explain(test.X[:100])
    plot_counterfactual_gains(
        explanations,
        save_path=os.path.join(fig_dir, "counterfactual_gains.pdf"),
    )

    print(f"All figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
