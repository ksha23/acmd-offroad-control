#!/usr/bin/env python3
"""Generate the manuscript's terrain-estimator validation figure.

The figure is ``terrain_profile_validation.png``, labelled
``fig:terrain-profile``. It plots the locked 144-case confirmation of the
joint independent ``n``/``phi`` estimator against plant truth, together with
its paired improvement over the matched scalar estimator and the outcomes of
its application-readiness gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "my_paper" / "paper_figures"
RUNS = FIGURES / "terrain_joint_scored_runs.csv"
SUMMARY = FIGURES / "terrain_joint_summary.csv"
BOOTSTRAP = FIGURES / "terrain_joint_paired_bootstrap.csv"
DECISION = FIGURES / "terrain_joint_decision.json"
EVIDENCE = FIGURES / "terrain_joint_evidence.json"
OUTPUT = FIGURES / "terrain_profile_validation.png"


def _limits(truth: pd.Series, estimate: pd.Series, padding: float) -> tuple[float, float]:
    values = np.concatenate((truth.to_numpy(float), estimate.to_numpy(float)))
    return float(np.min(values) - padding), float(np.max(values) + padding)


def _truth(values: pd.Series) -> np.ndarray:
    return values.map(
        lambda value: str(value).strip().lower() in {"true", "1", "1.0"}
    ).to_numpy(bool)


def main() -> int:
    runs = pd.read_csv(RUNS)
    summary = pd.read_csv(SUMMARY).set_index("method")
    bootstrap = pd.read_csv(BOOTSTRAP).set_index("baseline")
    decision = json.loads(DECISION.read_text())
    evidence = json.loads(EVIDENCE.read_text())
    required = {
        "trace_id", "n_true", "phi_true_deg", "joint_final_n",
        "joint_final_phi_deg", "parent_final_n",
        "parent_manifold_phi_deg", "joint_final_fresh",
        "joint_final_confident", "joint_final_control_envelope_valid",
        "joint_final_publication_ready", "joint_final_boundary_limited",
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise ValueError("joint terrain figure input is missing: " + ", ".join(missing))
    if (
        len(runs) != 144
        or runs["trace_id"].duplicated().any()
        or decision.get("promotion_criteria_pass") is not True
        or evidence.get("accepted_for_paper") is not True
        or evidence.get("active_backend") != "grit"
        or set(summary.index.astype(str))
        != {"joint", "scalar_parent", "uniform_prior"}
    ):
        raise ValueError("joint terrain evidence has not passed its locked contract")

    figure, axes = plt.subplots(1, 4, figsize=(6.7, 1.72))
    color = "#17365D"
    parent_color = "#b46a55"

    # phi leads: it is the better-identified coordinate (larger force signal
    # per prior width) and carries the paper's primary estimation claim.
    phi_limits = _limits(
        runs["phi_true_deg"], runs["joint_final_phi_deg"], 1.0
    )
    axes[0].scatter(
        runs["phi_true_deg"], runs["joint_final_phi_deg"], s=12,
        alpha=0.70, color=color, edgecolor="none",
    )
    axes[0].plot(phi_limits, phi_limits, "k--", linewidth=0.75)
    axes[0].set(
        xlim=phi_limits, ylim=phi_limits,
        xlabel=r"true $\phi$ (deg)",
        ylabel=r"joint estimate $\hat\phi$ (deg)",
    )
    axes[0].set_title(
        rf"$\phi$: MAE {summary.loc['joint', 'phi_mae_deg']:.2f}$^\circ$",
        fontsize=8.6,
    )

    n_limits = _limits(runs["n_true"], runs["joint_final_n"], 0.025)
    axes[1].scatter(
        runs["n_true"], runs["joint_final_n"], s=12, alpha=0.70,
        color=color, edgecolor="none",
    )
    axes[1].plot(n_limits, n_limits, "k--", linewidth=0.75)
    axes[1].set(
        xlim=n_limits, ylim=n_limits,
        xlabel=r"true $n$ (dimensionless)",
        ylabel=r"joint estimate $\hat n$",
    )
    axes[1].set_title(
        f"$n$: MAE {summary.loc['joint', 'n_mae']:.3f}", fontsize=8.6
    )

    n_joint = np.abs(runs["joint_final_n"] - runs["n_true"])
    n_parent = np.abs(runs["parent_final_n"] - runs["n_true"])
    phi_joint = np.abs(runs["joint_final_phi_deg"] - runs["phi_true_deg"])
    phi_parent = np.abs(
        runs["parent_manifold_phi_deg"] - runs["phi_true_deg"]
    )
    wins = [
        100.0 * float((phi_joint < phi_parent).mean()),
        100.0 * float((n_joint < n_parent).mean()),
    ]
    axes[2].bar(
        [r"$\phi$", r"$n$"], wins,
        color=[color, parent_color], width=0.58,
    )
    axes[2].set_ylim(0.0, 142.0)
    axes[2].set_ylabel("joint lower error (%)")
    axes[2].set_xlabel("soil coordinate")
    axes[2].set_title("Paired vs. parent", fontsize=8.6)
    improvements = [
        float(
            bootstrap.loc[
                "scalar_parent", "phi_mae_improvement_deg"
            ]
        ),
        float(bootstrap.loc["scalar_parent", "n_mae_improvement"]),
    ]
    units = [r"$^\circ$", ""]
    for index, (win, improvement, unit) in enumerate(
        zip(wins, improvements, units)
    ):
        axes[2].text(
            index, win + 3.0,
            f"{win:.0f}%\n$\\Delta$MAE {improvement:.2f}{unit}",
            ha="center", va="bottom", fontsize=7.4,
        )

    labels = [
        "ready", "fresh", "conf.", r"$\phi$ env.", "boundary-\nlimited",
    ]
    rates = [
        100.0 * _truth(runs["joint_final_publication_ready"]).mean(),
        100.0 * _truth(runs["joint_final_fresh"]).mean(),
        100.0 * _truth(runs["joint_final_confident"]).mean(),
        100.0 * _truth(
            runs["joint_final_control_envelope_valid"]
        ).mean(),
        100.0 * _truth(runs["joint_final_boundary_limited"]).mean(),
    ]
    axes[3].bar(
        np.arange(len(labels)), rates,
        color=[color, color, color, color, parent_color], width=0.68,
    )
    axes[3].set_xticks(np.arange(len(labels)))
    axes[3].set_xticklabels(labels, rotation=32, ha="right")
    axes[3].set_ylim(0.0, 110.0)
    axes[3].set_ylabel("cases (%)")
    axes[3].set_title("Gate outcomes (144 cases)", fontsize=8.6)

    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes):
        axis.text(
            0.01, 0.98, label, transform=axis.transAxes,
            va="top", ha="left", fontsize=7.0, fontweight="bold",
        )
        axis.grid(axis="y", alpha=0.20)
        axis.tick_params(labelsize=6.4)
        axis.xaxis.label.set_size(6.8)
        axis.yaxis.label.set_size(6.8)
    figure.tight_layout(pad=0.42, w_pad=0.48)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {OUTPUT.relative_to(ROOT)} (144 joint n/phi cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
