#!/usr/bin/env python3
"""Score the joint terrain estimator as control actually experiences it.

This produces the terrain-estimator accuracy figure and table in the
manuscript. It compares the joint ``n``/``phi`` estimator against the matched
scalar estimator over 144 identical sensor traces, using publish-gated
operational estimates throughout: the value scored on each trace is the value
the controller would have used at that instant, not the estimator's best
internal belief.

Operational semantics extend to failure. A trace on which the estimator never
accepts a snapshot is scored as control experiences it, at the labelled
fallback (n=0.50, phi=13 deg), unpublished and not ready, rather than being
dropped from the population. Dropping such traces would report the accuracy of
a system that always converges. Each such cell must be declared explicitly on
the command line, so any undeclared failure aborts scoring instead of being
absorbed into the fallback.

``score_joint_estimator.py`` is the complementary instrument: it validates
per-cell accepted-snapshot timelines, which requires every cell to have
accepted a snapshot, and so applies to matrices without never-accepting cells.
This module implements the identical statistical protocol (paired bootstrap,
BOOTSTRAP_SEED=20260722, 100000 resamples) extended to operational cells, and
emits the artifact set that the evidence lock and ``make_fig_terrain_profile``
consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_RESAMPLES = 100_000
UNIFORM_N_MEAN = 0.8
UNIFORM_PHI_MEAN_DEG = 21.9
FALLBACK_N = 0.50
FALLBACK_PHI_DEG = 13.0
NO_ACCEPT_FAILURE_TOKEN = "no accepted snapshot"
READINESS_FLAGS = [
    "final_fresh",
    "final_confident",
    "final_control_envelope_valid",
    "final_publication_ready",
    "final_boundary_limited",
    "final_snapshot_was_published",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
    )


def _load_candidate(
    path: Path, truth: pd.DataFrame, declared: set[str]
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    failed = frame[frame["status"] != "ok"]
    failed_ids = set(failed["trace_id"].astype(str))
    if failed_ids != declared:
        raise SystemExit(
            f"candidate failed cells {sorted(failed_ids)} do not match the "
            f"declared no-accept list {sorted(declared)}"
        )
    bad_reason = failed[
        ~failed["failure"].astype(str).str.contains(NO_ACCEPT_FAILURE_TOKEN)
    ]
    if len(bad_reason):
        raise SystemExit(
            "declared cells must fail with the no-accepted-snapshot "
            f"signature; got: {bad_reason['failure'].tolist()}"
        )
    frame = frame.copy()
    mask = frame["trace_id"].astype(str).isin(declared)
    frame.loc[mask, "final_est_n"] = FALLBACK_N
    frame.loc[mask, "final_est_phi_deg"] = FALLBACK_PHI_DEG
    for flag in READINESS_FLAGS:
        frame[flag] = frame[flag].astype(object)
        frame.loc[mask, flag] = False
    # Operational reading of a never-accepting trace: no update arrived
    # within the trace, so the update age equals the trace duration and the
    # publication confidence is zero.
    frame["final_final_update_age_s"] = pd.to_numeric(
        frame["final_final_update_age_s"], errors="coerce"
    )
    frame["final_publication_confidence"] = pd.to_numeric(
        frame["final_publication_confidence"], errors="coerce"
    )
    duration = pd.to_numeric(
        frame["final_final_trace_time"], errors="coerce"
    ).max()
    frame.loc[mask, "final_final_update_age_s"] = float(duration)
    frame.loc[mask, "final_publication_confidence"] = 0.0
    frame["operational_fallback_cell"] = mask
    merged = truth.merge(
        frame[
            [
                "trace_id",
                "trace_sha256",
                "final_est_n",
                "final_est_phi_deg",
                "final_final_update_age_s",
                "final_publication_confidence",
                "operational_fallback_cell",
                *READINESS_FLAGS,
            ]
        ],
        on="trace_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(truth):
        raise SystemExit("candidate arm does not cover the truth matrix")
    return merged


def _load_parent(path: Path, truth: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if set(frame["status"].astype(str)) != {"ok"}:
        raise SystemExit("parent arm contains failed cells")
    merged = truth.merge(
        frame[["trace_id", "final_est_n", "final_est_phi_deg"]].rename(
            columns={
                "final_est_n": "parent_final_n",
                "final_est_phi_deg": "parent_manifold_phi_deg",
            }
        ),
        on="trace_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(truth):
        raise SystemExit("parent arm does not cover the truth matrix")
    return merged


def _paired_bootstrap(
    rng: np.random.Generator,
    joint_err: np.ndarray,
    parent_err: np.ndarray,
) -> tuple[float, float, float]:
    delta = parent_err - joint_err
    point = float(np.mean(delta))
    draws = rng.integers(0, len(delta), size=(BOOTSTRAP_RESAMPLES, len(delta)))
    samples = delta[draws].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return point, float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument(
        "--declared-no-accept-cells",
        default="",
        help="Comma-separated trace ids allowed to be never-accepting.",
    )
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--endpoint-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    declared = {
        item.strip()
        for item in str(args.declared_no_accept_cells).split(",")
        if item.strip()
    }
    truth = pd.read_csv(
        args.collection_dir / "truth.csv", float_precision="round_trip"
    )
    if len(truth) != args.expected_count:
        raise SystemExit(
            f"truth has {len(truth)} rows; expected {args.expected_count}"
        )
    candidate = _load_candidate(
        args.candidate_dir / "estimates.csv", truth, declared
    )
    parent = _load_parent(args.parent_dir / "estimates.csv", truth)

    scored = candidate.merge(
        parent[["trace_id", "parent_final_n", "parent_manifold_phi_deg"]],
        on="trace_id",
        validate="one_to_one",
    ).rename(
        columns={
            "final_est_n": "joint_final_n",
            "final_est_phi_deg": "joint_final_phi_deg",
            "final_fresh": "joint_final_fresh",
            "final_confident": "joint_final_confident",
            "final_control_envelope_valid": (
                "joint_final_control_envelope_valid"
            ),
            "final_publication_ready": "joint_final_publication_ready",
            "final_boundary_limited": "joint_final_boundary_limited",
            "final_snapshot_was_published": "joint_final_published",
            "final_final_update_age_s": "joint_final_update_age_s",
            "final_publication_confidence": (
                "joint_final_publication_confidence"
            ),
        }
    )
    scored["uniform_prior_n"] = UNIFORM_N_MEAN
    scored["uniform_prior_phi_deg"] = UNIFORM_PHI_MEAN_DEG
    for flag in (
        "joint_final_fresh",
        "joint_final_confident",
        "joint_final_control_envelope_valid",
        "joint_final_publication_ready",
        "joint_final_boundary_limited",
        "joint_final_published",
    ):
        scored[flag] = _as_bool(scored[flag])

    joint_n_err = (scored["joint_final_n"] - scored["n_true"]).to_numpy()
    joint_phi_err = (
        scored["joint_final_phi_deg"] - scored["phi_true_deg"]
    ).to_numpy()
    parent_n_err = (scored["parent_final_n"] - scored["n_true"]).to_numpy()
    parent_phi_err = (
        scored["parent_manifold_phi_deg"] - scored["phi_true_deg"]
    ).to_numpy()

    def _method_row(
        method: str, est_n: pd.Series, est_phi: pd.Series
    ) -> dict[str, object]:
        n_err = (est_n - scored["n_true"]).to_numpy(dtype=float)
        phi_err = (est_phi - scored["phi_true_deg"]).to_numpy(dtype=float)
        n_true = scored["n_true"].to_numpy(dtype=float)
        constant_n = bool(pd.Series(est_n).nunique() <= 1)
        constant_phi = bool(pd.Series(est_phi).nunique() <= 1)
        return {
            "method": method,
            "n": len(scored),
            "n_mae": float(np.abs(n_err).mean()),
            "n_bias": float(n_err.mean()),
            "n_rmse": float(np.sqrt((n_err**2).mean())),
            "n_pct_within_20": float(
                100.0 * (np.abs(n_err) / n_true < 0.20).mean()
            ),
            "n_spearman": (
                float("nan")
                if constant_n
                else float(stats.spearmanr(n_true, est_n).statistic)
            ),
            "phi_mae_deg": float(np.abs(phi_err).mean()),
            "phi_bias_deg": float(phi_err.mean()),
            "phi_rmse_deg": float(np.sqrt((phi_err**2).mean())),
            "phi_pct_within_5_deg": float(
                100.0 * (np.abs(phi_err) < 5.0).mean()
            ),
            "phi_spearman": (
                float("nan")
                if constant_phi
                else float(
                    stats.spearmanr(
                        scored["phi_true_deg"], est_phi
                    ).statistic
                )
            ),
        }

    summary = pd.DataFrame(
        [
            _method_row(
                "joint", scored["joint_final_n"], scored["joint_final_phi_deg"]
            ),
            _method_row(
                "scalar_parent",
                scored["parent_final_n"],
                scored["parent_manifold_phi_deg"],
            ),
            _method_row(
                "uniform_prior",
                scored["uniform_prior_n"],
                scored["uniform_prior_phi_deg"],
            ),
        ]
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_point, n_low, n_high = _paired_bootstrap(
        rng, np.abs(joint_n_err), np.abs(parent_n_err)
    )
    phi_point, phi_low, phi_high = _paired_bootstrap(
        rng, np.abs(joint_phi_err), np.abs(parent_phi_err)
    )
    uniform_err = np.abs(UNIFORM_N_MEAN - scored["n_true"].to_numpy())
    u_point, u_low, u_high = _paired_bootstrap(
        rng, np.abs(joint_n_err), uniform_err
    )
    bootstrap = pd.DataFrame(
        [
            {
                "baseline": "scalar_parent",
                "n_units": len(scored),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "n_mae_improvement": n_point,
                "n_mae_improvement_ci_low": n_low,
                "n_mae_improvement_ci_high": n_high,
                "phi_mae_improvement_deg": phi_point,
                "phi_mae_improvement_ci_low_deg": phi_low,
                "phi_mae_improvement_ci_high_deg": phi_high,
            },
            {
                "baseline": "uniform_prior",
                "n_units": len(scored),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "n_mae_improvement": u_point,
                "n_mae_improvement_ci_low": u_low,
                "n_mae_improvement_ci_high": u_high,
                "phi_mae_improvement_deg": float("nan"),
                "phi_mae_improvement_ci_low_deg": float("nan"),
                "phi_mae_improvement_ci_high_deg": float("nan"),
            },
        ]
    )

    hash_identical = bool(
        (scored["trace_sha256"].astype(str).str.len() == 64).all()
    )
    ready_count = int(scored["joint_final_publication_ready"].sum())
    boundary_count = int(scored["joint_final_boundary_limited"].sum())
    endpoint_decision = json.loads(args.endpoint_decision.read_text())
    decision = {
        "n_units": int(len(scored)),
        "evidence_semantics": "operational_publish_gated",
        "declared_no_accept_cells": sorted(declared),
        "operational_fallback_cell_count": int(
            scored["operational_fallback_cell"].sum()
        ),
        "uses_final_estimates_not_tail_means": True,
        "uses_operational_publish_gated_estimates": True,
        "trace_matrix_complete_and_hash_identical": hash_identical,
        "accepted_snapshot_version": "grit_accepted",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "promotion_criteria_pass": bool(endpoint_decision["accept"]),
        "promotion_endpoint_decision_sha256": _sha256_file(
            args.endpoint_decision
        ),
        "joint_n_mae_improves_parent_with_positive_ci": bool(n_low > 0.0),
        "joint_n_mae_improves_uniform_prior_with_positive_ci": bool(
            u_low > 0.0
        ),
        "joint_phi_mae_improves_parent_with_positive_ci": bool(
            phi_low > 0.0
        ),
        "publication_ready_at_least_85_pct": bool(
            ready_count >= 0.85 * len(scored)
        ),
        "publication_ready_count": ready_count,
        "boundary_limited_count": boundary_count,
        "material_boundary_limited_count": boundary_count,
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "command": " ".join(sys.argv),
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "scored_runs.csv", index=False, float_format="%.17g")
    summary.to_csv(out / "summary.csv", index=False, float_format="%.17g")
    bootstrap.to_csv(
        out / "paired_bootstrap.csv", index=False, float_format="%.17g"
    )
    (out / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"EVIDENCE_SCORED n={len(scored)} "
        f"fallback_cells={sorted(declared)} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
