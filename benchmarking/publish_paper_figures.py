#!/usr/bin/env python3
"""Publish the canonical CSV and figure artifacts the manuscript compiles against.

This is the single boundary between generated results and the paper. Nothing
reaches ``my_paper/paper_figures/`` except through this program, which
resolves each study's source directory, validates it against the contract that
study is required to satisfy, and only then stages and atomically replaces the
published artifact.

Validation fails closed on every check, so a study whose provenance cannot be
established is not published in a weakened form: it is not published at all.
The checks cover source-tree cleanliness, expected row and arm counts, the
estimator backend and contract each run used, the digests of the learned
checkpoints loaded at runtime, and, for the terrain-estimator evidence, the
frozen hashes of the confirmation matrix.
"""

from __future__ import annotations

import hashlib
import ast
import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from paper_provenance import DOWNSTREAM_SOURCE_FILES
except ModuleNotFoundError:  # package import in tests/tools
    from benchmarking.paper_provenance import DOWNSTREAM_SOURCE_FILES
try:
    from common import (
        RIG_ACTIVE_ESTIMATOR_BACKEND,
        GRIT_ESTIMATOR_BACKEND,
        GRIT_ESTIMATOR_CONTRACT,
        PARENT_ESTIMATOR_BACKEND,
        PARENT_ESTIMATOR_CONTRACT,
    )
except ModuleNotFoundError:  # package import in tests/tools
    from benchmarking.common import (
        RIG_ACTIVE_ESTIMATOR_BACKEND,
        GRIT_ESTIMATOR_BACKEND,
        GRIT_ESTIMATOR_CONTRACT,
        PARENT_ESTIMATOR_BACKEND,
        PARENT_ESTIMATOR_CONTRACT,
    )
try:
    from joint_n_phi_evidence import (
        AUTHORITATIVE_JOINT_N_PHI_RESULT,
        validate_joint_n_phi_evidence,
    )
except ModuleNotFoundError:  # package import in tests/tools
    from benchmarking.joint_n_phi_evidence import (
        AUTHORITATIVE_JOINT_N_PHI_RESULT,
        validate_joint_n_phi_evidence,
    )
try:
    from profile_live_statistics import (
        build_profile_live_statistics,
        canonical_json as canonical_statistics_json,
        source_input_hashes as statistics_source_input_hashes,
    )
except ModuleNotFoundError:  # package import in tests/tools
    from benchmarking.profile_live_statistics import (
        build_profile_live_statistics,
        canonical_json as canonical_statistics_json,
        source_input_hashes as statistics_source_input_hashes,
    )


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarking" / "results"
DEST = ROOT / "my_paper" / "paper_figures"


SPECS = {
    "mpc_tire_model_sweep": {
        "results.csv": "bench_tire_models_results.csv",
        "summary_by_model.csv": "bench_tire_models_summary.csv",
    },
    "tire_model_with_estimator_ablation": {
        "results.csv": "tire_model_with_estimator_results.csv",
        "summary_by_variant.csv": "tire_model_with_estimator_summary.csv",
        "reference_profile_contract.json": "tire_model_reference_profile_contract.json",
    },
    "speed_profile_ablation": {
        "speed_profile_paired.csv": "speed_profile_ablation.csv",
    },
    "safety_filter_sweep_planner_aware": {
        "results.csv": "safety_filter_planner_aware_results.csv",
        "summary_by_filter.csv": "safety_filter_planner_aware_summary.csv",
    },
    "convoy_counterfactual_eval": {
        "results.csv": "convoy_counterfactual_results.csv",
        "summary.csv": "convoy_counterfactual_summary.csv",
    },
    "latency_awareness_ablation": {
        "results.csv": "latency_awareness_results.csv",
        "summary.csv": "latency_awareness_summary.csv",
        "summary_by_delay.csv": "latency_awareness_by_delay.csv",
    },
    "grit_adaptive_speed_matrix": {
        "results.csv": "grit_adaptive_speed_matrix_results.csv",
        "summary.csv": "grit_adaptive_speed_matrix_summary.csv",
    },
    "safety_filter_sweep_live_grit_mpsf_paper": {
        "results.csv": "live_grit_integration_results.csv",
        "summary_by_filter.csv": "live_grit_integration_summary.csv",
    },
    "mpc_tire_model_sweep_calibrated": {
        "results.csv": "bench_tire_models_calibrated_results.csv",
        "summary_by_model.csv": "bench_tire_models_calibrated_summary.csv",
    },
}

EXPECTED_VARIANTS = {
    "mpc_tire_model_sweep": {"pacejka", "tmeasy", "tire_force_static"},
    "tire_model_with_estimator_ablation": {
        "nn_static", "nn_estimator", "nn_parent_estimator",
        "nn_fixed_fallback",
    },
    "grit_adaptive_speed_matrix": {"oracle", "grit", "conservative", "aggressive"},
    "safety_filter_sweep_live_grit_mpsf_paper": {
        "none_blind", "none_aware", "mpsf_blind", "mpsf_aware",
    },
    "mpc_tire_model_sweep_calibrated": {"pacejka_oracle", "pacejka_rigfit"},
}

PAPER_ROWS = {
    "mpc_tire_model_sweep": 1215,
    "tire_model_with_estimator_ablation": 540,
    "speed_profile_ablation": 48,
    "safety_filter_sweep_planner_aware": 1620,
    "convoy_counterfactual_eval": 90,
    "latency_awareness_ablation": 90,
    "grit_adaptive_speed_matrix": 540,
    "safety_filter_sweep_live_grit_mpsf_paper": 60,
    "mpc_tire_model_sweep_calibrated": 810,
}

CONTACT_PREFIXES = {
    "safety_filter_sweep_planner_aware",
    "convoy_counterfactual_eval",
    "latency_awareness_ablation",
    "safety_filter_sweep_live_grit_mpsf_paper",
}

PROFILE_ARTIFACT_PATHS = {
    # Runs of the scalar-estimator arm record estimator_artifact_hashes(),
    # whose shared "rate_checkpoint" slot holds the joint estimator's
    # rate-format checkpoint, because both arms load the same runtime stack.
    # The scalar arm's own learned artifact is the tire_force_static_parent
    # force map, so both are hashed here.
    "rate_checkpoint_sha256": ROOT / "nn_models/tire_force_rate/best_terrain_nn.pt",
    "rate_scalers_sha256": ROOT / "nn_models/tire_force_rate/scalers.pkl",
    "force_checkpoint_sha256": ROOT / "nn_models/tire_force_static_parent/best_terrain_nn.pt",
    "force_scalers_sha256": ROOT / "nn_models/tire_force_static_parent/scalers.pkl",
}

JOINT_ARTIFACT_PATHS = {
    # These keys name the slots the runtime hash record uses; both resolve to
    # the joint estimator's rate-format force checkpoint, which is the only
    # learned artifact that estimator loads.
    "rate_checkpoint_sha256": ROOT / "nn_models/tire_force_rate/best_terrain_nn.pt",
    "rate_scalers_sha256": ROOT / "nn_models/tire_force_rate/scalers.pkl",
}

_SOURCE_MAP_CACHE: dict[str, Path] | None = None


def _manifest_recorded_source(prefix: str) -> Path | None:
    """Source dir the committed publish manifest already selected, if intact.

    A republish with no explicit orchestrator binding must be idempotent, so
    it reuses the selection the manifest records rather than re-selecting the
    newest matching result directory. Automatic re-selection would let an
    unrelated run, such as a probe or a single ablation arm that happens to
    share the prefix, silently replace a published number. Fresh selection
    still occurs when the manifest holds no entry for the prefix, or when the
    directory it names is absent.
    """
    manifest_path = DEST / "publish_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        entries = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    for entry in entries if isinstance(entries, list) else []:
        if entry.get("prefix") == prefix and entry.get("source_dir"):
            recorded = ROOT / entry["source_dir"]
            if recorded.is_dir():
                return recorded
    return None


def configured_source(prefix: str) -> Path | None:
    """Resolve an orchestrator-bound result directory, if one is active."""
    global _SOURCE_MAP_CACHE
    source_map_path = os.environ.get("ACMD_PUBLISH_SOURCE_MAP", "").strip()
    if not source_map_path:
        return _manifest_recorded_source(prefix)
    if _SOURCE_MAP_CACHE is None:
        path = Path(source_map_path).expanduser().resolve()
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise RuntimeError("ACMD_PUBLISH_SOURCE_MAP must contain a JSON object")
        _SOURCE_MAP_CACHE = {
            str(key): Path(value).expanduser().resolve()
            for key, value in payload.items()
        }
    if prefix not in _SOURCE_MAP_CACHE:
        raise RuntimeError(
            f"orchestrator source map does not bind required prefix {prefix!r}"
        )
    source = _SOURCE_MAP_CACHE[prefix]
    try:
        source.relative_to(RESULTS.resolve())
    except ValueError as exc:
        raise RuntimeError(f"bound result is outside {RESULTS}: {source}") from exc
    if not source.is_dir() or not source.name.startswith(prefix + "_"):
        raise RuntimeError(f"invalid bound result for {prefix}: {source}")
    return source


def is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def profile_contract_matches(values: object) -> bool:
    return isinstance(values, dict) and values == PARENT_ESTIMATOR_CONTRACT


def joint_contract_matches(values: object) -> bool:
    return isinstance(values, dict) and values == GRIT_ESTIMATOR_CONTRACT


PROFILE_DIAGNOSTIC_FIELDS = {
    "diagnostics_applicable",
    "diagnostics_complete",
    "diagnostics_error",
    "required_concurrent_windows",
    "publication_ready",
    "publication_applied",
    "abstained",
    "readiness_rows",
    "update_rows",
    "time_to_first_ready_s",
    "time_to_first_update_s",
    "max_concurrent_windows",
    "lifetime_accepted_windows",
    "lifetime_rejected_windows",
    "profile_force_gain_final",
    "profile_ax_bias_final_mps2",
    "profile_ay_bias_final_mps2",
    "profile_bound_hits_max",
    "feature_envelope_excursions_max",
    "readiness_consistent",
}


def profile_diagnostics_valid(
    frame: pd.DataFrame,
    *,
    prefix: str = "",
    expected_applicable: int | None = None,
    estimator_contract: dict[str, object] = PARENT_ESTIMATOR_CONTRACT,
) -> bool:
    """Validate truth-free live-estimator diagnostics on result rows."""

    columns = {f"{prefix}profile_estimator_{field}" for field in PROFILE_DIAGNOSTIC_FIELDS}
    if not columns.issubset(frame.columns):
        return False

    def _column(field: str) -> pd.Series:
        return frame[f"{prefix}profile_estimator_{field}"]

    boolean_fields = (
        "diagnostics_applicable", "diagnostics_complete",
        "publication_ready", "publication_applied", "abstained",
        "readiness_consistent",
    )
    parsed_booleans = {
        field: _column(field).map(strict_bool) for field in boolean_fields
    }
    if any(values.isna().any() for values in parsed_booleans.values()):
        return False
    applicable = parsed_booleans["diagnostics_applicable"].astype(bool)
    complete = parsed_booleans["diagnostics_complete"].astype(bool)
    consistent = parsed_booleans["readiness_consistent"].astype(bool)
    ready = parsed_booleans["publication_ready"].astype(bool)
    applied = parsed_booleans["publication_applied"].astype(bool)
    abstained = parsed_booleans["abstained"].astype(bool)

    numeric_fields = (
        "required_concurrent_windows", "readiness_rows", "update_rows",
        "time_to_first_ready_s", "time_to_first_update_s",
        "max_concurrent_windows", "lifetime_accepted_windows",
        "lifetime_rejected_windows", "profile_force_gain_final",
        "profile_ax_bias_final_mps2", "profile_ay_bias_final_mps2",
        "profile_bound_hits_max", "feature_envelope_excursions_max",
    )
    numeric = {
        field: pd.to_numeric(_column(field), errors="coerce")
        for field in numeric_fields
    }
    required = numeric["required_concurrent_windows"]
    readiness_rows = numeric["readiness_rows"]
    update_rows = numeric["update_rows"]
    ready_time = numeric["time_to_first_ready_s"]
    update_time = numeric["time_to_first_update_s"]
    windows = numeric["max_concurrent_windows"]
    accepted = numeric["lifetime_accepted_windows"]
    rejected = numeric["lifetime_rejected_windows"]
    gain = numeric["profile_force_gain_final"]
    ax_bias = numeric["profile_ax_bias_final_mps2"]
    ay_bias = numeric["profile_ay_bias_final_mps2"]
    bound_hits = numeric["profile_bound_hits_max"]
    envelope = numeric["feature_envelope_excursions_max"]

    counter_fields = (
        "required_concurrent_windows", "readiness_rows", "update_rows",
        "max_concurrent_windows", "lifetime_accepted_windows",
        "lifetime_rejected_windows", "profile_bound_hits_max",
        "feature_envelope_excursions_max",
    )
    for field in counter_fields:
        values = numeric[field]
        array = values.to_numpy(dtype=float)
        if (
            not np.isfinite(array).all()
            or (array < 0.0).any()
            or not np.allclose(array, np.rint(array), rtol=0.0, atol=1.0e-12)
        ):
            return False
    for values in (ready_time, update_time):
        finite = values.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all() or (finite < 0.0).any():
            return False

    gain_lo, gain_hi = estimator_contract["force_gain_bounds"]
    bias_bound = float(
        estimator_contract["acceleration_bias_bound_mps2"]
    )
    required_windows = int(estimator_contract["min_concurrent_windows"])
    joint_contract = (
        estimator_contract.get("backend") == GRIT_ESTIMATOR_BACKEND
    )
    errors = _column("diagnostics_error").fillna("").astype(str)
    nonapplicable = ~applicable
    if (
        (expected_applicable is not None and int(applicable.sum()) != expected_applicable)
        or not complete.all()
        or not consistent.all()
        or not errors.eq("").all()
        or not (required == required_windows).all()
        or not ready.eq(readiness_rows > 0).all()
        or not applied.eq(update_rows > 0).all()
        or (
            (ready & (windows < required)).any()
            if joint_contract
            else not ready.eq(windows >= required).all()
        )
        or (applied & ~ready).any()
        or (update_rows > readiness_rows).any()
        or not abstained[applicable].eq(~applied[applicable]).all()
        or (ready[nonapplicable] | applied[nonapplicable] | abstained[nonapplicable]).any()
        or not ready.eq(ready_time.notna()).all()
        or not applied.eq(update_time.notna()).all()
        or (applied & (update_time < ready_time)).any()
        or ((accepted + rejected)[applicable] <= 0).any()
        or (windows > accepted).any()
        or (ready & (accepted < required)).any()
        or ((bound_hits > 3).any() and not joint_contract)
        or (envelope > rejected).any()
        or not gain[applicable].between(
            float(gain_lo), float(gain_hi), inclusive="both"
        ).all()
        or not ax_bias[applicable].between(
            -bias_bound, bias_bound, inclusive="both"
        ).all()
        or not ay_bias[applicable].between(
            -bias_bound, bias_bound, inclusive="both"
        ).all()
        or gain[nonapplicable].notna().any()
        or ax_bias[nonapplicable].notna().any()
        or ay_bias[nonapplicable].notna().any()
        or (readiness_rows[nonapplicable] != 0).any()
        or (update_rows[nonapplicable] != 0).any()
        or (windows[nonapplicable] != 0).any()
        or (accepted[nonapplicable] != 0).any()
        or (rejected[nonapplicable] != 0).any()
        or (bound_hits[nonapplicable] != 0).any()
        or (envelope[nonapplicable] != 0).any()
    ):
        return False
    if joint_contract:
        joint_fields = {
            "snapshot_rows",
            "unique_snapshot_count",
            "ready_snapshot_count",
            "applied_snapshot_count",
            "final_snapshot_seq",
            "fallback_rows",
            "max_evidence_age_s",
            "min_snapshot_confidence",
            "max_boundary_mass",
            "min_observability_singular_value",
            "update_wall_ms_median",
            "update_wall_ms_p95",
            "update_wall_ms_max",
        }
        joint_columns = {
            f"{prefix}profile_estimator_{field}" for field in joint_fields
        }
        if not joint_columns.issubset(frame.columns):
            return False
        joint_numeric = {
            field: pd.to_numeric(
                frame[f"{prefix}profile_estimator_{field}"],
                errors="coerce",
            )
            for field in joint_fields
        }
        snapshot_rows = joint_numeric["snapshot_rows"]
        unique_snapshots = joint_numeric["unique_snapshot_count"]
        ready_snapshots = joint_numeric["ready_snapshot_count"]
        applied_snapshots = joint_numeric["applied_snapshot_count"]
        final_sequence = joint_numeric["final_snapshot_seq"]
        integer_fields = (
            snapshot_rows,
            unique_snapshots,
            ready_snapshots,
            applied_snapshots,
            final_sequence,
            joint_numeric["fallback_rows"],
        )
        if any(
            not np.isfinite(values.to_numpy(dtype=float)).all()
            or (values < 0.0).any()
            or not np.allclose(
                values.to_numpy(dtype=float),
                np.rint(values.to_numpy(dtype=float)),
                rtol=0.0,
                atol=1.0e-12,
            )
            for values in integer_fields
        ):
            return False
        has_snapshot = unique_snapshots > 0
        timing_fields = (
            "update_wall_ms_median",
            "update_wall_ms_p95",
            "update_wall_ms_max",
        )
        if (
            (snapshot_rows < unique_snapshots).any()
            or (ready_snapshots > unique_snapshots).any()
            or (applied_snapshots > ready_snapshots).any()
            or (final_sequence < unique_snapshots).any()
            or not ready.eq(ready_snapshots > 0).all()
            or not applied.eq(applied_snapshots > 0).all()
            or any(
                joint_numeric[field].notna().ne(has_snapshot).any()
                for field in timing_fields
            )
            or any(
                (
                    joint_numeric[field].dropna() < 0.0
                ).any()
                for field in timing_fields
            )
            or (
                joint_numeric["update_wall_ms_median"]
                > joint_numeric["update_wall_ms_p95"]
            ).fillna(False).any()
            or (
                joint_numeric["update_wall_ms_p95"]
                > joint_numeric["update_wall_ms_max"]
            ).fillna(False).any()
        ):
            return False
    return True


def strict_bool(value: object) -> bool | None:
    """Parse a serialized boolean without Python's truthy-string ambiguity."""

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0"}:
        return False
    return None


def downstream_provenance_matches(values: dict[str, object]) -> bool:
    """Validate clean committed source hashes recorded by downstream runs.

    Verification compares each run against the commit that run recorded, not
    against the current checkout, because a subsequent committed change to the
    manuscript or to an unrelated benchmark would otherwise make a
    legitimately clean run unverifiable.  Every file in the current contract
    must be covered by the recorded hashes; a run that recorded a larger set
    of contract files is verified over the full set it recorded, so removing a
    file from the contract cannot weaken an existing run's verification.
    """

    source_hashes = values.get("source_sha256")
    git_head = str(values.get("code_git_head", ""))
    if not (
        values.get("paper_evidence_eligible") is True
        and values.get("tracked_worktree_dirty") is False
        and values.get("uncommitted_source_files") == []
        and len(git_head) == 40
        and all(character in "0123456789abcdef" for character in git_head.lower())
        and isinstance(source_hashes, dict)
        and set(DOWNSTREAM_SOURCE_FILES) <= set(source_hashes)
    ):
        return False
    for relative, recorded in sorted(source_hashes.items()):
        result = subprocess.run(
            ["git", "show", f"{git_head}:{relative}"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            return False
        if recorded != hashlib.sha256(result.stdout).hexdigest():
            return False
    return True


def recorded_provenance(directory: Path) -> dict[str, object] | None:
    """Provenance a run recorded for itself, from either manifest format.

    ``write_manifest`` emits ``manifest.csv``; the estimator matrices emit a
    richer ``manifest.json``. Both carry the same keys, so either is accepted.
    """
    def load_json(path: Path):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        # A manifest that parses but is not a mapping is malformed, not
        # absent; treat it the same as unparseable so the caller rejects it
        # with the gate's message rather than an attribute error.
        return payload if isinstance(payload, dict) else None

    json_manifest = directory / "manifest.json"
    csv_manifest = directory / "manifest.csv"
    if json_manifest.is_file():
        payload = load_json(json_manifest)
        if payload is not None and csv_manifest.is_file():
            # A JSON manifest silently shadowing a CSV one is how a
            # transplanted provenance block hides a run's own record. When
            # both exist they must agree on the commit, or neither is
            # trusted.
            csv_payload = _read_csv_manifest(csv_manifest)
            if csv_payload is not None:
                # Agreement on the commit alone is not enough: a forged JSON
                # can copy the CSV's head while flipping the eligibility
                # flags. Every provenance key present in both must agree.
                for key in ("code_git_head", "tracked_worktree_dirty",
                            "paper_evidence_eligible"):
                    if (key in csv_payload and key in payload
                            and str(payload[key]) != str(csv_payload[key])):
                        raise RuntimeError(
                            f"{directory.name}: manifest.json and "
                            f"manifest.csv disagree on {key} "
                            f"({str(payload[key])[:12]} vs "
                            f"{str(csv_payload[key])[:12]}); a transplanted "
                            f"manifest is indistinguishable from a corrupted "
                            f"one, so neither is trusted."
                        )
        return payload
    if not csv_manifest.is_file():
        return None
    return _read_csv_manifest(csv_manifest)


def _read_csv_manifest(csv_manifest: Path) -> dict[str, object] | None:
    values: dict[str, object] = {}
    digests: dict[str, str] = {}
    try:
        text_rows = list(csv.reader(csv_manifest.open(errors="strict")))
    except (UnicodeDecodeError, OSError):
        # An unreadable manifest is absent, not a crash: the caller rejects
        # the directory with the gate's own message.
        return None
    with csv_manifest.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            key, raw = row[0], row[1]
            if key.startswith("source_sha256."):
                digests[key[len("source_sha256."):]] = raw
                continue
            try:
                values[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                values[key] = raw
    if digests:
        values["source_sha256"] = digests
    return values or None


def require_clean_collection(prefix: str, directory: Path) -> None:
    """Fail closed unless the run proved it was collected on a clean tree.

    A result whose provenance cannot be established is not published in a
    weakened form. The check is against the commit the run recorded, not the
    current checkout, so later committed work does not invalidate an already
    clean collection.
    """
    provenance = recorded_provenance(directory)
    if provenance is None:
        raise RuntimeError(
            f"{prefix}: {directory.name} records no provenance manifest, so "
            f"its collection cannot be shown to be reproducible. Recollect it "
            f"through benchmarking/run.py on a committed tree."
        )
    if not downstream_provenance_matches(provenance):
        detail = (
            f"eligible={provenance.get('paper_evidence_eligible')!r} "
            f"dirty={provenance.get('tracked_worktree_dirty')!r} "
            f"head={str(provenance.get('code_git_head'))[:12]!r}"
        )
        raise RuntimeError(
            f"{prefix}: {directory.name} was not collected on a clean, "
            f"committed tree ({detail}). Commit the working tree and "
            f"recollect; a published number must regenerate from the commit "
            f"it names."
        )


def truth_mask(values: pd.Series) -> pd.Series:
    return values.map(lambda value: str(value).strip().lower() in {"true", "1", "1.0"})


def usable_rows(directory: Path) -> int:
    source = directory / "results.csv"
    if not source.exists():
        return 0
    try:
        frame = pd.read_csv(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return 0
    if "status" in frame:
        frame = frame[frame["status"].astype(str) == "ok"]
    elif "ok" in frame:
        frame = frame[truth_mask(frame["ok"])]
    return len(frame)


def select(prefix: str) -> Path:
    bound = configured_source(prefix)
    candidates = (
        [bound]
        if bound is not None
        else [path for path in RESULTS.glob(f"{prefix}_*") if path.is_dir()]
    )
    minimum = PAPER_ROWS[prefix]
    candidates = [path for path in candidates if usable_rows(path) == minimum]
    expected = EXPECTED_VARIANTS.get(prefix)
    if expected is not None:
        matching = []
        for path in candidates:
            try:
                frame = pd.read_csv(path / "results.csv")
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if "variant" in frame:
                observed = set(frame["variant"].astype(str))
            elif {"label", "arm"}.issubset(frame.columns):
                observed = set(frame["label"].astype(str) + "|" + frame["arm"].astype(str))
            else:
                observed = set()
            if observed == expected:
                matching.append(path)
        candidates = matching
    if prefix == "tire_model_with_estimator_ablation":
        verified = []
        for path in candidates:
            manifest = path / "manifest.csv"
            contract_path = path / "reference_profile_contract.json"
            if not manifest.exists() or not contract_path.exists():
                continue
            try:
                metadata = read_key_value_manifest(manifest)
                contract = json.loads(contract_path.read_text())
                frame = pd.read_csv(path / "results.csv")
            except (pd.errors.EmptyDataError, pd.errors.ParserError, json.JSONDecodeError):
                continue
            if metadata is None:
                continue
            backend = str(metadata.get("estimator_backend", "")).strip("'\"")
            design = str(metadata.get("design_version", "")).strip("'\"")
            reference_policy = str(metadata.get("reference_policy", "")).strip("'\"")
            bound_policy = str(metadata.get("lateral_acceleration_bound_policy", "")).strip("'\"")
            prior = str(metadata.get("estimator_initial_prior", "")).strip("'\"")
            launch_contract = str(
                metadata.get("launch_identity_contract", "")
            ).strip("'\"")
            ros_policy = str(
                metadata.get("ros_concurrency_policy", "")
            ).strip("'\"")
            truth_enabled = str(metadata.get("tire_force_truth_enabled", ""))
            learned_hashes_match = all(
                source.is_file()
                and str(metadata.get(key, "")).strip("'\"") == digest(source)
                for key, source in JOINT_ARTIFACT_PATHS.items()
            )
            parent_hashes_match = all(
                source.is_file()
                and str(
                    metadata.get("historical_parent_" + key, "")
                ).strip("'\"") == digest(source)
                for key, source in PROFILE_ARTIFACT_PATHS.items()
            )
            downstream_metadata = {
                "code_git_head": metadata.get("code_git_head"),
                "tracked_worktree_dirty": metadata.get("tracked_worktree_dirty"),
                "uncommitted_source_files": metadata.get("uncommitted_source_files"),
                "paper_evidence_eligible": metadata.get("paper_evidence_eligible"),
                "source_sha256": metadata.get("source_sha256"),
            }
            if (
                backend == RIG_ACTIVE_ESTIMATOR_BACKEND
                and backend == GRIT_ESTIMATOR_BACKEND
                and design
                == "conditioning_joint_parent_fallback_ros_isolated"
                and joint_contract_matches(metadata.get("estimator_contract"))
                and metadata.get("historical_parent_estimator_backend")
                == PARENT_ESTIMATOR_BACKEND
                and profile_contract_matches(
                    metadata.get("historical_parent_estimator_contract")
                )
                and reference_policy == "shared_worst_case_phi13_curvature_v1"
                and bound_policy == "shared_worst_case_phi13_bound_v1"
                and prior == "dirt"
                and launch_contract == "path_speed_seed_ports_domain"
                and ros_policy == "exclusive_process_lease_and_batched_workers"
                and truth_enabled == "False"
                and learned_hashes_match
                and parent_hashes_match
                and downstream_provenance_matches(downstream_metadata)
                and contract.get("schema_version") == 4
                and contract.get("design")
                == "conditioning_joint_parent_fallback_ros_isolated"
                and contract.get("reference_policy") == "shared_worst_case_phi13_curvature_v1"
                and float(contract.get("reference_profile_friction_angle_deg", -1.0)) == 13.0
                and contract.get("lateral_acceleration_bound_policy") == "shared_worst_case_phi13_bound_v1"
                and float(contract.get("lateral_acceleration_bound_friction_angle_deg", -1.0)) == 13.0
                and contract.get("controller_packet_truth") is False
                and contract.get("launch_identity_contract")
                == "path_speed_seed_ports_domain"
                and contract.get("ros_concurrency_policy")
                == "exclusive_process_lease_and_batched_workers"
                and int(contract.get("n_successful_rows", -1)) == 540
                and int(contract.get("profile_estimator_applicable_rows", -1)) == 270
                and int(contract.get("profile_estimator_publication_ready_rows", 0)) > 0
                and contract.get("estimator_backend_counts") == {
                    "disabled": 270,
                    GRIT_ESTIMATOR_BACKEND: 135,
                    PARENT_ESTIMATOR_BACKEND: 135,
                }
                and contract.get("fixed_fallback_contract", {}).get("variant")
                == "nn_fixed_fallback"
                and contract.get("fixed_fallback_contract", {}).get(
                    "controller_prior_terrain"
                ) == "clay"
                and float(contract.get("fixed_fallback_contract", {}).get(
                    "n", float("nan")
                )) == float(GRIT_ESTIMATOR_CONTRACT["fallback_n"])
                and float(contract.get("fixed_fallback_contract", {}).get(
                    "phi_deg", float("nan")
                )) == float(GRIT_ESTIMATOR_CONTRACT["fallback_phi_deg"])
            ):
                ok = frame[frame["status"].astype(str) == "ok"]
                if len(frame) != 540 or len(ok) != 540:
                    continue
                counts = ok.groupby("variant").size().to_dict()
                if counts != {
                    "nn_estimator": 135,
                    "nn_parent_estimator": 135,
                    "nn_static": 135,
                    "nn_fixed_fallback": 135,
                }:
                    continue
                if set(pd.to_numeric(ok["bumpiness"], errors="coerce")) != {0}:
                    continue
                key_columns = [
                    "variant", "terrain", "path", "speed_mps", "bumpiness", "seed"
                ]
                if ok[key_columns].duplicated().any():
                    continue
                expected_levels = {
                    "terrain": {"clay", "dirt", "sand"},
                    "path": {"sinusoidal", "lane_change", "right_left"},
                    "speed_mps": {5.0, 7.0, 9.0},
                    "seed": {400, 401, 402, 403, 404},
                }
                if any(
                    set(ok[column]) != levels
                    for column, levels in expected_levels.items()
                ):
                    continue
                numeric = ok[["rms_cte_m", "mean_speed_mps", "mean_solve_ms"]].apply(
                    pd.to_numeric, errors="coerce"
                )
                if not numeric.notna().all().all():
                    continue
                truth_rows = ok.get("extra_controller_tire_force_truth_rows")
                if truth_rows is None or not (
                    pd.to_numeric(truth_rows, errors="coerce") == 0
                ).all():
                    continue
                identity = ok.get("extra_launch_identity_match")
                if identity is None or not truth_mask(identity).all():
                    continue
                if not {
                    "extra_estimator_backend",
                    "extra_estimator_contract_version",
                    "extra_controller_prior_terrain",
                }.issubset(ok.columns):
                    continue
                joint_rows = ok[ok["variant"].astype(str) == "nn_estimator"]
                parent_rows = ok[
                    ok["variant"].astype(str) == "nn_parent_estimator"
                ]
                static_rows = ok[ok["variant"].astype(str).isin({
                    "nn_static", "nn_fixed_fallback"
                })]
                expected_backends = {
                    "nn_estimator": GRIT_ESTIMATOR_BACKEND,
                    "nn_parent_estimator": PARENT_ESTIMATOR_BACKEND,
                    "nn_static": "disabled",
                    "nn_fixed_fallback": "disabled",
                }
                observed_backends = {
                    str(variant): str(part["extra_estimator_backend"].iloc[0])
                    for variant, part in ok.groupby("variant")
                }
                if (
                    observed_backends != expected_backends
                    or set(joint_rows["extra_estimator_contract_version"].astype(str))
                    != {str(GRIT_ESTIMATOR_CONTRACT["contract_version"])}
                    or set(parent_rows["extra_estimator_contract_version"].astype(str))
                    != {str(PARENT_ESTIMATOR_CONTRACT["contract_version"])}
                    or set(static_rows["extra_estimator_contract_version"].astype(str))
                    != {"backend_compatibility"}
                    or not profile_diagnostics_valid(
                        joint_rows,
                        prefix="extra_",
                        expected_applicable=135,
                        estimator_contract=GRIT_ESTIMATOR_CONTRACT,
                    )
                    or not profile_diagnostics_valid(
                        parent_rows, prefix="extra_", expected_applicable=135
                    )
                    or not profile_diagnostics_valid(
                        static_rows, prefix="extra_", expected_applicable=0
                    )
                ):
                    continue
                estimator_rows = ok[truth_mask(
                    ok["extra_profile_estimator_diagnostics_applicable"]
                )]
                if (
                    int(contract.get("profile_estimator_publication_ready_rows", -1))
                    != int(truth_mask(
                        estimator_rows["extra_profile_estimator_publication_ready"]
                    ).sum())
                    or int(contract.get("profile_estimator_abstained_rows", -1))
                    != int(truth_mask(
                        estimator_rows["extra_profile_estimator_abstained"]
                    ).sum())
                ):
                    continue
                priors = {
                    str(variant): set(
                        part["extra_controller_prior_terrain"].astype(str)
                    )
                    for variant, part in ok.groupby("variant")
                }
                if priors != {
                    "nn_estimator": {"dirt"},
                    "nn_parent_estimator": {"dirt"},
                    "nn_static": {"matched_plant"},
                    "nn_fixed_fallback": {"clay"},
                }:
                    continue
                if (
                    not ok.get("extra_observed_path", pd.Series(dtype=str))
                    .astype(str).eq(ok["path"].astype(str)).all()
                    or not np.isclose(
                        pd.to_numeric(ok.get("extra_observed_speed_mps"), errors="coerce"),
                        pd.to_numeric(ok["speed_mps"], errors="coerce"),
                        rtol=0.0, atol=1.0e-9,
                    ).all()
                    or not (
                        pd.to_numeric(ok.get("extra_observed_sim_seed"), errors="coerce")
                        == pd.to_numeric(ok["seed"], errors="coerce")
                    ).all()
                ):
                    continue
                profile_hashes = ok.get("extra_reference_profile_sha256")
                if profile_hashes is None or not profile_hashes.astype(str).map(is_sha256).all():
                    continue
                if not (
                    ok.groupby(["path", "speed_mps"])["extra_reference_profile_sha256"]
                    .nunique()
                    .eq(1)
                    .all()
                ):
                    continue
                verified.append(path)
        candidates = verified
    if not candidates:
        raise FileNotFoundError(
            f"No paper-complete benchmarking/results/{prefix}_* generation "
            f"with exactly {minimum} usable rows"
        )
    return max(candidates, key=lambda path: (usable_rows(path), path.stat().st_mtime))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_key_value_manifest(path: Path) -> dict[str, object] | None:
    """Read the replayer's two-column manifest without trusting its command."""

    try:
        frame = pd.read_csv(path, dtype=str)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if list(frame.columns) != ["key", "value"] or frame["key"].duplicated().any():
        return None
    values: dict[str, object] = {}
    for row in frame.itertuples(index=False):
        raw = str(row.value)
        try:
            values[str(row.key)] = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            values[str(row.key)] = raw
    return values


def _write_publication(
    publish_dest: Path,
    selected_sources: dict[str, Path],
    joint_evidence: dict[str, object],
) -> None:
    """Build the active joint-estimator publication in an isolated stage."""

    def canonical_destination(staged: Path) -> str:
        relative = staged.relative_to(publish_dest)
        return str((DEST / relative).relative_to(ROOT))

    manifest: list[dict[str, object]] = []
    for prefix, files in SPECS.items():
        source_dir = selected_sources[prefix]
        copied = []
        for relative_source, destination_name in files.items():
            source = source_dir / relative_source
            if not source.exists():
                raise FileNotFoundError(f"{source} is missing")
            destination = publish_dest / destination_name
            shutil.copy2(source, destination)
            copied.append({
                "source": str(source.relative_to(ROOT)),
                "destination": canonical_destination(destination),
                "source_sha256": digest(source),
                "sha256": digest(destination),
                "transformation": "copy",
            })
        manifest.append({
            "prefix": prefix,
            "source_dir": str(source_dir.relative_to(ROOT)),
            "usable_rows": usable_rows(source_dir),
            "files": copied,
        })
        print(
            f"[ok] {prefix}: {source_dir.name} "
            f"({usable_rows(source_dir)} usable rows)"
        )

    statistics_payload = build_profile_live_statistics(
        selected_sources["tire_model_with_estimator_ablation"],
        root=ROOT,
    )
    statistics_destination = publish_dest / "profile_live_statistics.json"
    statistics_destination.write_text(
        canonical_statistics_json(statistics_payload)
    )
    conditioning_source = (
        selected_sources["tire_model_with_estimator_ablation"] / "results.csv"
    )
    statistics_record = {
        "source": str(conditioning_source.relative_to(ROOT)),
        "destination": canonical_destination(statistics_destination),
        "source_sha256": digest(conditioning_source),
        "sha256": digest(statistics_destination),
        "transformation": "profile_live_statistics",
        "input_sha256": statistics_source_input_hashes(statistics_payload),
    }
    conditioning_manifest = next(
        item for item in manifest
        if item["prefix"] == "tire_model_with_estimator_ablation"
    )
    conditioning_manifest["files"].append(statistics_record)
    print("[ok] deterministic paired statistics: four-arm conditioning")

    joint_files = []
    for record in joint_evidence["artifacts"]:
        source = ROOT / str(record["source"])
        destination = publish_dest / str(record["publication_name"])
        shutil.copy2(source, destination)
        if digest(destination) != str(record["source_sha256"]):
            raise RuntimeError(
                "joint evidence changed during publication: " + str(source)
            )
        joint_files.append({
            "source": str(record["source"]),
            "destination": canonical_destination(destination),
            "source_sha256": str(record["source_sha256"]),
            "sha256": digest(destination),
            "transformation": "copy",
            "role": str(record["role"]),
        })

    joint_summary = pd.read_csv(
        AUTHORITATIVE_JOINT_N_PHI_RESULT / "scoring/summary.csv"
    ).set_index("method")
    joint_bootstrap = pd.read_csv(
        AUTHORITATIVE_JOINT_N_PHI_RESULT / "scoring/paired_bootstrap.csv"
    ).set_index("baseline")
    joint_decision = json.loads(
        (
            AUTHORITATIVE_JOINT_N_PHI_RESULT / "scoring/decision.json"
        ).read_text()
    )
    compact_evidence = {
        **joint_evidence,
        "schema_version": 2,
        "accepted_for_paper": True,
        "active_backend": GRIT_ESTIMATOR_BACKEND,
        "historical_matched_parent": PARENT_ESTIMATOR_BACKEND,
        "joint_metrics": {
            "n_mae": float(joint_summary.loc["joint", "n_mae"]),
            "n_rmse": float(joint_summary.loc["joint", "n_rmse"]),
            "n_pct_within_20": float(
                joint_summary.loc["joint", "n_pct_within_20"]
            ),
            "n_spearman": float(
                joint_summary.loc["joint", "n_spearman"]
            ),
            "phi_mae_deg": float(
                joint_summary.loc["joint", "phi_mae_deg"]
            ),
            "phi_rmse_deg": float(
                joint_summary.loc["joint", "phi_rmse_deg"]
            ),
            "phi_pct_within_5_deg": float(
                joint_summary.loc["joint", "phi_pct_within_5_deg"]
            ),
            "phi_spearman": float(
                joint_summary.loc["joint", "phi_spearman"]
            ),
        },
        "paired_mae_improvement": {
            baseline: {
                "n": float(
                    joint_bootstrap.loc[baseline, "n_mae_improvement"]
                ),
                "n_ci95": [
                    float(joint_bootstrap.loc[
                        baseline, "n_mae_improvement_ci_low"
                    ]),
                    float(joint_bootstrap.loc[
                        baseline, "n_mae_improvement_ci_high"
                    ]),
                ],
                "phi_deg": float(
                    joint_bootstrap.loc[
                        baseline, "phi_mae_improvement_deg"
                    ]
                ),
                "phi_ci95_deg": [
                    float(joint_bootstrap.loc[
                        baseline, "phi_mae_improvement_ci_low_deg"
                    ]),
                    float(joint_bootstrap.loc[
                        baseline, "phi_mae_improvement_ci_high_deg"
                    ]),
                ],
            }
            for baseline in ("scalar_parent", "uniform_prior")
        },
        "publication_ready_count": int(
            joint_decision["publication_ready_count"]
        ),
        "boundary_limited_count": int(
            joint_decision["material_boundary_limited_count"]
        ),
        "interpretation": (
            "independent n/phi generalized profile-likelihood estimator; "
            "scalar_parent is the frozen matched scalar parent"
        ),
    }
    joint_evidence_destination = publish_dest / "terrain_joint_evidence.json"
    joint_evidence_destination.write_text(
        json.dumps(compact_evidence, indent=2) + "\n"
    )
    joint_files.append({
        "source": str(
            (
                AUTHORITATIVE_JOINT_N_PHI_RESULT / "scoring/decision.json"
            ).relative_to(ROOT)
        ),
        "destination": canonical_destination(joint_evidence_destination),
        "source_sha256": digest(
            AUTHORITATIVE_JOINT_N_PHI_RESULT / "scoring/decision.json"
        ),
        "sha256": digest(joint_evidence_destination),
        "transformation": "joint_n_phi_compact_evidence",
        "input_sha256": {
            str(record["source"]): str(record["source_sha256"])
            for record in joint_evidence["artifacts"]
        },
    })
    manifest.append({
        "prefix": "joint_n_phi_promotion",
        "source_dir": str(joint_evidence["result_directory"]),
        "usable_rows": int(joint_evidence["cases"]),
        "files": joint_files,
    })
    print(
        "[ok] locked joint n/phi evidence: "
        f"{joint_evidence['cases']} cases, "
        f"{joint_evidence['publication_ready_count']} publication-ready"
    )

    (publish_dest / "publish_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def _publish_atomically(
    destination: Path,
    writer: Callable[[Path], None],
) -> None:
    """Stage a complete publication, then atomically replace each output.

    The manifest is installed last, so it never advertises a partially staged
    generation. If validation or rendering raises inside ``writer``, no file
    below ``destination`` is touched.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".paper_publication_", dir=destination.parent
    ) as temporary:
        stage = Path(temporary)
        writer(stage)
        staged_files = [path for path in stage.iterdir() if path.is_file()]
        if not any(path.name == "publish_manifest.json" for path in staged_files):
            raise RuntimeError("staged publication lacks publish_manifest.json")
        unexpected = [path for path in stage.iterdir() if not path.is_file()]
        if unexpected:
            raise RuntimeError(
                "staged publication contains unexpected directories: "
                + ", ".join(str(path) for path in unexpected)
            )
        for source in sorted(
            staged_files,
            key=lambda path: (path.name == "publish_manifest.json", path.name),
        ):
            os.replace(source, destination / source.name)


def main() -> int:
    # Resolve and validate every source before creating any canonical output.
    joint_evidence = validate_joint_n_phi_evidence(
        AUTHORITATIVE_JOINT_N_PHI_RESULT
    )
    selected_sources = {prefix: select(prefix) for prefix in SPECS}
    for prefix, source_dir in selected_sources.items():
        require_clean_collection(prefix, source_dir)
        for relative_source in SPECS[prefix]:
            source = source_dir / relative_source
            if not source.is_file():
                raise FileNotFoundError(f"{source} is missing")
        if prefix in CONTACT_PREFIXES:
            contact_data = pd.read_csv(source_dir / "results.csv")
            if "collision_source" not in contact_data.columns:
                raise RuntimeError(f"{source_dir} lacks collision_source evidence")
            observed_sources = set(
                contact_data["collision_source"].dropna().astype(str)
            )
            if observed_sources != {"chrono_body_contact"}:
                raise RuntimeError(
                    f"{source_dir} does not use native Chrono body contacts: "
                    f"{sorted(observed_sources)}"
                )

    _publish_atomically(
        DEST,
        lambda stage: _write_publication(
            stage,
            selected_sources,
            joint_evidence,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
