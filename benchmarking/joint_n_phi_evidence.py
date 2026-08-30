#!/usr/bin/env python3
"""Fail-closed access to the locked 144-case joint ``n``/``phi`` evidence.

The confirmation matrix is generated data held under ``benchmarking/results``
rather than a tracked paper artifact, so consumers must be able to establish
that the generation they read is the one the manuscript reports. This module
is the single selector the orchestrator, the publisher, and the supervision
verifier share. It resolves that generation by name, checks every artifact
against an immutable SHA-256 digest, and checks the structural invariants of
the matrix: case count, readiness and boundary counts, seeds, and the traces
that hold the control fallback.

It reads the frozen analysis and never reruns or rescales it, so no consumer
can regenerate evidence to fit an expectation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_JOINT_N_PHI_RESULT = (
    ROOT / "benchmarking/results/joint_confirmation_seed61504289"
)
EXPECTED_CASES = 144
EXPECTED_RESULT_NAME = "joint_confirmation_seed61504289"
EXPECTED_SIM_SEED_FIRST = 47230000
EXPECTED_SOIL_SEED = "61504289"
EXPECTED_READY_COUNT = 130
EXPECTED_BOUNDARY_COUNT = 9
EXPECTED_FALLBACK_CELLS = ["trace_0052", "trace_0069", "trace_0090"]
EXPECTED_GIT_HEAD = "c9dcf62"  # source-tree HEAD prefix recorded at scoring time

# Digests of the confirmation artifacts, fixed when the matrix was scored.
# The scored quantities are the operational publish-gated estimates produced
# by score_joint_evidence.py, in which the three declared never-accepting cells
# carry the labelled control fallback.
ARTIFACTS: dict[str, dict[str, str]] = {
    "collection_config": {
        "relative_path": "collection/collection_config.json",
        "sha256": "bcec13bc4f5ca9f35d113f93dcd25edfaf09471318066d5acedd4ff33bf42d0a",
        "publication_name": "terrain_joint_collection_config.json",
    },
    "trace_manifest": {
        "relative_path": "collection/trace_manifest.csv",
        "sha256": "6be9a23827495691ad4892c106e8c86655a06244e6e8dca8318a15792538bdac",
        "publication_name": "terrain_joint_trace_manifest.csv",
    },
    "joint_replay_manifest": {
        "relative_path": "replay_candidate/replay_manifest.csv",
        "sha256": "7fb64c5d4d28014f281f8a87b8d7eed96d3b7d043f3d16521c78b7df00ffbaf0",
        "publication_name": "terrain_joint_replay_manifest.csv",
    },
    "parent_replay_manifest": {
        "relative_path": "replay_parent/replay_manifest.csv",
        "sha256": "8f9ac70e7a4ecf5efeb368452501b344260ea0fa9745a4f007b3d9ea207a630d",
        "publication_name": "terrain_joint_parent_replay_manifest.csv",
    },
    "promotion_endpoint_decision": {
        "relative_path": "DECISION.json",
        "sha256": "265b3ef7d43d6806f7da5ae1983de76f42dd8231b3b9032f839d19129b76cabe",
        "publication_name": "terrain_joint_promotion_endpoints.json",
    },
    "scored_runs": {
        "relative_path": "scoring/scored_runs.csv",
        "sha256": "e0ab01404082e38d0288b44fde527081fab9a33e77146257cddb5eceb386aaeb",
        "publication_name": "terrain_joint_scored_runs.csv",
    },
    "summary": {
        "relative_path": "scoring/summary.csv",
        "sha256": "7d0a4d971194d21b77213a1d7659898c788ebdf6960f44c4a8b25ead379ff20e",
        "publication_name": "terrain_joint_summary.csv",
    },
    "paired_bootstrap": {
        "relative_path": "scoring/paired_bootstrap.csv",
        "sha256": "39ac6c91cbd33bc3cb2129f0bde29d880328ff5b57ab4a04efcf52247821fd5d",
        "publication_name": "terrain_joint_paired_bootstrap.csv",
    },
    "decision": {
        "relative_path": "scoring/decision.json",
        "sha256": "112774ca01bb151a4752cac448116e700f891b05424200854d0e00feece612b9",
        "publication_name": "terrain_joint_decision.json",
    },
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def _key_value_manifest(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != ["key", "value"] or frame["key"].duplicated().any():
        raise RuntimeError(f"invalid key/value manifest: {path}")
    return {
        str(key): str(value)
        for key, value in frame.itertuples(index=False, name=None)
    }


def validate_joint_n_phi_evidence(
    result_dir: Path = AUTHORITATIVE_JOINT_N_PHI_RESULT,
) -> dict[str, Any]:
    """Validate hashes, schemas, promotion gates, and frozen run identity."""

    result_dir = Path(result_dir).expanduser().resolve()
    if result_dir.name != EXPECTED_RESULT_NAME or not result_dir.is_dir():
        raise RuntimeError(
            "joint evidence must be the locked result directory "
            f"{EXPECTED_RESULT_NAME}: {result_dir}"
        )

    records = []
    for role, specification in ARTIFACTS.items():
        path = result_dir / specification["relative_path"]
        if not path.is_file():
            raise RuntimeError(f"joint evidence artifact is missing: {path}")
        actual = _digest(path)
        if actual != specification["sha256"]:
            raise RuntimeError(
                f"joint evidence hash mismatch for {role}: "
                f"expected {specification['sha256']}, got {actual}"
            )
        records.append({
            "role": role,
            "source": str(path.relative_to(ROOT)),
            "source_sha256": actual,
            "publication_name": specification["publication_name"],
        })

    scored = pd.read_csv(result_dir / ARTIFACTS["scored_runs"]["relative_path"])
    _require_columns(
        scored,
        {
            "trace_id", "trace_sha256", "n_true", "phi_true_deg",
            "joint_final_n", "joint_final_phi_deg", "parent_final_n",
            "parent_manifold_phi_deg", "uniform_prior_n",
            "uniform_prior_phi_deg", "joint_final_boundary_limited",
            "joint_final_fresh", "joint_final_confident",
            "joint_final_control_envelope_valid",
            "joint_final_publication_ready", "joint_final_update_age_s",
            "joint_final_publication_confidence",
        },
        label="joint scored runs",
    )
    if (
        len(scored) != EXPECTED_CASES
        or scored["trace_id"].duplicated().any()
        or set(scored["trace_id"].astype(str))
        != {f"trace_{index:04d}" for index in range(EXPECTED_CASES)}
        or not scored["trace_sha256"].astype(str).map(
            lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))
        ).all()
    ):
        raise RuntimeError("joint scored-run identity is incomplete or duplicated")
    numeric_columns = [
        "n_true", "phi_true_deg", "joint_final_n", "joint_final_phi_deg",
        "parent_final_n", "parent_manifold_phi_deg", "uniform_prior_n",
        "uniform_prior_phi_deg", "joint_final_update_age_s",
        "joint_final_publication_confidence",
    ]
    numeric = scored[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("joint scored runs contain non-finite endpoints")

    summary = pd.read_csv(result_dir / ARTIFACTS["summary"]["relative_path"])
    if (
        len(summary) != 3
        or set(summary["method"].astype(str))
        != {"joint", "scalar_parent", "uniform_prior"}
        or not (pd.to_numeric(summary["n"], errors="coerce") == EXPECTED_CASES).all()
    ):
        raise RuntimeError("joint summary does not contain the three frozen methods")

    bootstrap = pd.read_csv(
        result_dir / ARTIFACTS["paired_bootstrap"]["relative_path"]
    )
    if (
        len(bootstrap) != 2
        or set(bootstrap["baseline"].astype(str))
        != {"scalar_parent", "uniform_prior"}
        or not (
            pd.to_numeric(bootstrap["n_units"], errors="coerce")
            == EXPECTED_CASES
        ).all()
        or not (
            pd.to_numeric(bootstrap["bootstrap_seed"], errors="coerce")
            == 20260722
        ).all()
        or not (
            pd.to_numeric(bootstrap["bootstrap_resamples"], errors="coerce")
            == 100_000
        ).all()
    ):
        raise RuntimeError("joint paired-bootstrap seed or resample count is not the frozen one")

    decision_path = result_dir / ARTIFACTS["decision"]["relative_path"]
    decision = json.loads(decision_path.read_text())
    required_true = {
        "promotion_criteria_pass",
        "trace_matrix_complete_and_hash_identical",
        "uses_final_estimates_not_tail_means",
        "uses_operational_publish_gated_estimates",
        "joint_n_mae_improves_parent_with_positive_ci",
        "joint_n_mae_improves_uniform_prior_with_positive_ci",
        "joint_phi_mae_improves_parent_with_positive_ci",
        "publication_ready_at_least_85_pct",
    }
    failed = sorted(key for key in required_true if decision.get(key) is not True)
    if failed:
        raise RuntimeError(f"joint promotion decision has failed gates: {failed}")
    if (
        int(decision.get("n_units", -1)) != EXPECTED_CASES
        or int(decision.get("publication_ready_count", -1))
        != EXPECTED_READY_COUNT
        or int(decision.get("material_boundary_limited_count", -1))
        != EXPECTED_BOUNDARY_COUNT
        or decision.get("accepted_snapshot_version")
        != "grit_accepted"
        or decision.get("declared_no_accept_cells")
        != EXPECTED_FALLBACK_CELLS
        or int(decision.get("operational_fallback_cell_count", -1))
        != len(EXPECTED_FALLBACK_CELLS)
        or decision.get("git_dirty") is not False
        or not str(decision.get("git_head", "")).startswith(
            EXPECTED_GIT_HEAD
        )
        or int(decision.get("bootstrap_seed", -1)) != 20260722
        or int(decision.get("bootstrap_resamples", -1)) != 100_000
    ):
        raise RuntimeError("joint promotion decision counts/version changed")

    trace_manifest = pd.read_csv(
        result_dir / ARTIFACTS["trace_manifest"]["relative_path"]
    )
    if (
        len(trace_manifest) != EXPECTED_CASES
        or set(trace_manifest["status"].astype(str)) != {"ok"}
        or trace_manifest["trace_id"].duplicated().any()
        or not (
            pd.to_numeric(trace_manifest["sim_seed"], errors="coerce")
            == np.arange(
                EXPECTED_SIM_SEED_FIRST,
                EXPECTED_SIM_SEED_FIRST + EXPECTED_CASES,
            )
        ).all()
        or pd.Series(trace_manifest["terrain_estimator_enabled"]).astype(bool).any()
    ):
        raise RuntimeError("joint trace manifest is not the frozen clean matrix")

    truth = pd.read_csv(result_dir / "collection/truth.csv")
    if (
        len(truth) != EXPECTED_CASES
        or set(truth["soil_draw_seed"].astype(str)) != {EXPECTED_SOIL_SEED}
    ):
        raise RuntimeError(
            "joint truth sidecar is not the preregistered confirmation draw"
        )

    endpoints = json.loads(
        (
            result_dir
            / ARTIFACTS["promotion_endpoint_decision"]["relative_path"]
        ).read_text()
    )
    if endpoints.get("accept") is not True or any(
        entry.get("pass") is not True
        for entry in endpoints.get("endpoints", {}).values()
    ):
        raise RuntimeError(
            "promotion endpoint decision is not an all-pass accept"
        )

    return {
        "schema_version": 1,
        "evidence_contract": "joint_n_phi_promotion_locked_144_case",
        "result_directory": str(result_dir.relative_to(ROOT)),
        "cases": EXPECTED_CASES,
        "publication_ready_count": EXPECTED_READY_COUNT,
        "boundary_limited_count": EXPECTED_BOUNDARY_COUNT,
        "operational_fallback_cells": list(EXPECTED_FALLBACK_CELLS),
        "promotion_criteria_pass": True,
        "artifacts": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=AUTHORITATIVE_JOINT_N_PHI_RESULT,
    )
    args = parser.parse_args()
    print(json.dumps(validate_joint_n_phi_evidence(args.result_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
