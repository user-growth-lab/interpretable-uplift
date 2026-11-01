"""Visualization utilities for interpretable uplift modeling.

Generates figures for the paper:
  - Global TESV attribution bar chart
  - Rule coverage vs. uplift lift curve
  - Counterfactual gain distribution
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_global_attributions(
    attribution_scores: np.ndarray,
    feature_names: Optional[List[str]] = None,
    true_effect_modifiers: Optional[List[int]] = None,
    top_k: int = 15,
    save_path: Optional[str] = None,
    title: str = "Global TESV Attributions",
):
    """Plot mean absolute attribution scores with true effect modifiers highlighted."""
    mean_attr = np.abs(attribution_scores).mean(axis=0)
    d = mean_attr.shape[0]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(d)]

    top_idx = np.argsort(-mean_attr)[:top_k]
    colors = ["red" if i in (true_effect_modifiers or []) else "steelblue" for i in top_idx]

    plt.figure(figsize=(8, 5))
    plt.barh(range(top_k), mean_attr[top_idx][::-1], color=colors[::-1])
    plt.yticks(range(top_k), [feature_names[i] for i in top_idx][::-1])
    plt.xlabel("Mean Absolute TESV")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_rule_lift_curve(
    tau_pred: np.ndarray,
    rule_masks: List[np.ndarray],
    save_path: Optional[str] = None,
):
    """Plot cumulative mean uplift when sorting by rule coverage score."""
    score = np.zeros(tau_pred.shape[0])
    for mask in rule_masks:
        score[mask] += 1

    order = np.argsort(-score)
    sorted_tau = tau_pred[order]
    cum_mean = np.cumsum(sorted_tau) / np.arange(1, len(sorted_tau) + 1)
    pct = np.arange(1, len(sorted_tau) + 1) / len(sorted_tau)

    plt.figure(figsize=(7, 5))
    plt.plot(pct, cum_mean, label="Rule-based ranking")
    plt.axhline(y=tau_pred.mean(), color="gray", linestyle="--", label="Global mean")
    plt.xlabel("Fraction of population selected")
    plt.ylabel("Cumulative mean CATE")
    plt.title("Rule Coverage vs. Uplift Lift")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_counterfactual_gains(
    explanations,
    save_path: Optional[str] = None,
):
    """Plot distribution of counterfactual uplift gains."""
    gains = np.array([e.tau_gain for e in explanations])

    plt.figure(figsize=(7, 5))
    plt.hist(gains, bins=20, color="steelblue", edgecolor="white")
    plt.xlabel("Predicted Uplift Gain")
    plt.ylabel("Count")
    plt.title("Distribution of Counterfactual Uplift Gains")
    plt.axvline(x=gains.mean(), color="red", linestyle="--", label=f"Mean={gains.mean():.3f}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_framework_diagram(save_path: Optional[str] = None):
    """Data-flow framework diagram centred on the shared CATE interface.

    Emphasises that all three explanation modules consume the *same* fitted
    CATE function tau_hat(.), each producing a distinct knowledge artefact for a
    distinct stakeholder (attribution/why, rules/whom, prescriptions/what-to-do).
    """
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="black", fontsize=10.5, tc="black",
            rotation=0, weight="normal"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            fill=True, facecolor=fc, edgecolor=ec, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=tc, rotation=rotation, weight=weight)

    def arrow(x0, y0, x1, y1, color="black", lw=1.8, style="-|>"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, lw=lw, color=color))

    # Column header labels (data-flow stages)
    headers = [(2.55, "Data & model"), (6.65, "Shared interface"),
               (9.6, "Explanation modules"), (12.6, "Knowledge artefacts"),
               (15.3, "Stakeholders")]
    for hx, ht in headers:
        ax.text(hx, 9.5, ht, ha="center", va="center", fontsize=10,
                style="italic", color="#444444")

    # Row colours for the three knowledge types
    c_attr, c_rule, c_presc = "#4C72B0", "#55A868", "#C44E52"
    rows = {"top": 7.55, "mid": 4.75, "bot": 1.95}

    # Stage 1: input data and CATE estimator
    box(0.4, 4.05, 2.3, 1.4, r"Input$(X, T, Y)$", "#E8EEF6")
    box(3.1, 4.05, 2.4, 1.4, "CATE estimator\n(S/T/X/R/DR-Learner)", "#CBD8EA")

    # Stage 2: the shared CATE interface (tall central spine)
    box(5.9, 1.4, 1.55, 7.0, r"Shared CATE interface   $\widehat{\tau}(\cdot):\mathbb{R}^d\!\to\!\mathbb{R}$",
        "#8C8C8C", fontsize=11, tc="white", rotation=90, weight="bold")

    # Stage 3: three explanation modules (each taps the same tau_hat)
    box(8.3, rows["top"] - 0.7, 2.7, 1.4, "TESV\n(interv. Shapley)", c_attr, tc="white", fontsize=9.8)
    box(8.3, rows["mid"] - 0.7, 2.7, 1.4, "SCRL\n(stable causal rules)", c_rule, tc="white", fontsize=9.8)
    box(8.3, rows["bot"] - 0.7, 2.7, 1.4, "Uplift Recourse\n(counterfactual)", c_presc, tc="white", fontsize=9.8)

    # Stage 4: knowledge artefacts
    box(11.5, rows["top"] - 0.6, 2.2, 1.2, "Attribution\n(why)", "#DCE4F0", ec=c_attr)
    box(11.5, rows["mid"] - 0.6, 2.2, 1.2, "Rules\n(whom)", "#DEEEE2", ec=c_rule)
    box(11.5, rows["bot"] - 0.6, 2.2, 1.2, "Prescriptions\n(what-to-do)", "#F3DEDE", ec=c_presc)

    # Stage 5: stakeholders
    box(14.1, rows["top"] - 0.55, 2.3, 1.1, "Model\nauditors", "white", ec=c_attr, fontsize=9.5)
    box(14.1, rows["mid"] - 0.55, 2.3, 1.1, "Campaign\nmanagers", "white", ec=c_rule, fontsize=9.5)
    box(14.1, rows["bot"] - 0.55, 2.3, 1.1, "Individual /\nCRM actions", "white", ec=c_presc, fontsize=9.5)

    # Arrows: data flow into the shared interface
    arrow(2.7, 4.75, 3.1, 4.75)
    arrow(5.5, 4.75, 5.9, 4.75)

    # Fan-out from the shared interface to each module (shared consumption)
    for r, c in [("top", c_attr), ("mid", c_rule), ("bot", c_presc)]:
        arrow(7.45, 4.75, 8.3, rows[r], color=c)
        arrow(11.0, rows[r], 11.5, rows[r], color=c)      # module -> artefact
        arrow(13.7, rows[r], 14.1, rows[r], color=c)      # artefact -> stakeholder

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
