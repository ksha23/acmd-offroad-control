#!/usr/bin/env python3
"""Score final, truth-isolated estimates for the joint ``n``/``phi`` study.

This is the gate behind the terrain-estimator confirmation reported in the
manuscript: it decides whether a replay matrix is admissible evidence and
computes the accuracy, correlation, and publication-rate endpoints quoted
there.

Inference artifacts are validated as a complete, hash-identical matrix before
this process opens the scorer-only truth sidecar, so no estimate can be
produced or revised with knowledge of the answer.  The primary estimates are
the last causal accepted scoring-tail snapshots, including snapshots that the
live confidence, freshness, and boundary publication policy would hold back.
They are deliberately not tail means, because averaging over the tail lets an
early error cancel a late one and reports an accuracy the estimator never had
at any instant.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ConstantInputWarning, spearmanr


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "simulation", ROOT / "benchmarking"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


BOOTSTRAP_SEED = 20260722
BOOTSTRAP_RESAMPLES = 100_000
UNIFORM_N_MEAN = 0.8
UNIFORM_PHI_MEAN_DEG = 21.9
FROZEN_MIN_OBSERVABILITY_SINGULAR_VALUE = 0.10
FROZEN_MAX_FINAL_UPDATE_AGE_S = 3.5
FROZEN_PUBLICATION_CONFIDENCE = 0.20
FROZEN_BOUNDARY_MASS_LIMIT = 0.25
FROZEN_CONTROL_MIN_PHI_DEG = 10.0
ACCEPTED_SNAPSHOT_VERSION = "grit_accepted"
INFERENCE_SEMANTICS_VERSION = "joint_final_snapshot"

_FROZEN_COLLECTION_SETTINGS = {
    "schema_version": 1,
    "mode": "independent_n_phi",
    "jitter": 0.1,
    "path": "sinusoidal",
    "speed_mps": 5.0,
    "sim_time_s": 14.0,
    "lead_in_m": 5.0,
    "terrain_id_probe": False,
    "probe_target_alpha_rad": 0.1,
    "probe_slew_rate_radps": 0.4,
    "probe_signed_dwell_s": 0.15,
    "probe_clearance_m": 35.0,
    "probe_max_latency_s": 0.3,
    "wheel_center_noise_std_m": 0.01,
    "wheel_center_calibration_bias_std_m": 0.003,
    "allow_approx_diag": False,
}

_FROZEN_JOINT_SETTINGS = {
    "candidate_confirmation": True,
    "development_only": False,
    "paper_evidence_eligible": True,
    "scoring_performed": False,
    "inference_semantics_version": INFERENCE_SEMANTICS_VERSION,
    "accepted_snapshot_version": ACCEPTED_SNAPSHOT_VERSION,
    "config.update_interval": 1,
    "config.grid_size": 41,
    "config.student_dof": 4.0,
    "config.smoothing_alpha": 1.0,
    "config.block_dt": 0.5,
    "config.horizon": 8.0,
    "config.min_windows": 8,
    "config.min_window_samples": 4,
    "config.r_ax": 0.35,
    "config.r_ay": 0.45,
    "config.min_information": 0.20,
    "config.min_yaw_rate_rms": 0.015,
    "config.min_model_speed": 2.5,
    "config.max_abs_alpha": 0.35,
    "config.enforce_feature_envelope": True,
    "config.slip_mode": "average",
    "config.fixed_kappa": 0.05,
    "config.rate_mode": "zero",
    "config.force_gain_std": 0.04,
    "config.ax_bias_std": 0.10,
    "config.ay_bias_std": 0.05,
    "config.force_gain_min": 0.70,
    "config.force_gain_max": 1.30,
    "config.acceleration_bias_bound": 0.30,
    "config.profile_iterations": 8,
    "config.phi_grid_size": 17,
    "config.phi_min_deg": 6.0,
    "config.phi_max_deg": 37.8,
    "config.cohesion_multiplier_min": 0.7,
    "config.cohesion_multiplier_max": 1.3,
    "config.cohesion_grid_size": 1,
    "config.cohesion_prior_std": 0.20,
    "config.load_transfer_mode": "static",
    "config.min_joint_information": 0.20,
    "config.min_n_information": 0.0,
    "config.min_phi_information": 0.0,
    "config.min_observability_rank": 2,
    "config.boundary_warning_mass": 0.25,
    "config.posterior_summary": "mean",
    "config.block_alpha_rate": False,
    # Exact JSON emitted by develop_joint_estimator._merged_estimator_kwargs
    # for the accepted grid bounds and sub-clay manifold completion.  It is
    # pinned verbatim rather than parsed, so any drift in the replay's
    # estimator keyword arguments fails scoring instead of passing silently.
    "config.extra_kwargs_json": (
        '{"n_bounds": [0.4, 1.1], "manifold_soft_floor": 0.4, '
        '"manifold_soft_mode": "hold"}'
    ),
    "replay.tail_start_s": 7.0,
    "replay.minimum_confidence": FROZEN_PUBLICATION_CONFIDENCE,
}

_FROZEN_PARENT_SETTINGS = {
    "config.min_confidence": 0.0,
    "config.tail_start": 7.0,
    "config.grid_size": 41,
    "config.student_dof": 4.0,
    "config.dynamics_update_interval": 1,
    "config.dynamics_block_dt": 0.5,
    "config.dynamics_horizon": 8.0,
    "config.dynamics_min_windows": 8,
    "config.dynamics_min_window_samples": 4,
    "config.dynamics_r_ax": 0.35,
    "config.dynamics_r_ay": 0.45,
    "config.dynamics_min_information": 0.20,
    "config.dynamics_min_yaw_rate_rms": 0.015,
    "config.dynamics_min_speed": 2.5,
    "config.dynamics_max_abs_alpha": 0.35,
    "config.dynamics_enforce_feature_envelope": True,
    "config.dynamics_slip_mode": "average",
    "config.dynamics_fixed_kappa": 0.05,
    "config.dynamics_rate_mode": "zero",
    "config.dynamics_force_gain_std": 0.04,
    "config.dynamics_ax_bias_std": 0.10,
    "config.dynamics_ay_bias_std": 0.05,
    "config.dynamics_force_gain_min": 0.70,
    "config.dynamics_force_gain_max": 1.30,
    "config.dynamics_acceleration_bias_bound": 0.30,
    "config.dynamics_profile_iterations": 8,
    "cli.backends": ["scalar_parent"],
}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_trace_ids(count: int) -> list[str]:
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or int(count) < 1
    ):
        raise ValueError("expected count must be a positive integer")
    return [f"trace_{index:04d}" for index in range(int(count))]


def _manifest_value(value: str) -> Any:
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return text


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, (bool, np.bool_)) and bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool):
            return False
        try:
            return math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _require_mapping_values(
    mapping: dict[str, Any],
    expected: dict[str, Any],
    *,
    context: str,
) -> None:
    for key, wanted in expected.items():
        if key not in mapping:
            raise ValueError(f"{context} is missing {key}")
        if not _values_match(mapping[key], wanted):
            raise ValueError(
                f"{context} has {key}={mapping[key]!r}; expected {wanted!r}"
            )


def _load_key_value_manifest(path: Path, *, context: str) -> dict[str, Any]:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"cannot read {context}: {error}") from error
    if list(frame.columns) != ["key", "value"] or frame.empty:
        raise ValueError(f"{context} must be a nonempty key/value CSV")
    if frame["key"].duplicated().any() or frame["key"].eq("").any():
        raise ValueError(f"{context} contains duplicate or empty keys")
    return {
        str(row.key): _manifest_value(str(row.value))
        for row in frame.itertuples(index=False)
    }


def _strict_bool(values: pd.Series, *, name: str) -> np.ndarray:
    normalized = values.astype(str).str.strip().str.lower()
    valid = normalized.isin({"0", "1", "0.0", "1.0", "false", "true"})
    if not valid.all():
        raise ValueError(f"{name} contains non-boolean values")
    return normalized.isin({"1", "1.0", "true"}).to_numpy(dtype=bool)


def _finite(frame: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing: {', '.join(missing)}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{context} contains non-finite numeric values")


def _load_collection_config(
    path: Path,
    *,
    expected_count: int,
    expected_soil_seed: int,
    expected_base_port: int,
    expected_sim_seed: int,
) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read collection config: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("collection config must encode an object")
    _require_mapping_values(
        config, _FROZEN_COLLECTION_SETTINGS, context="collection config"
    )
    _require_mapping_values(
        config,
        {
            "n": expected_count,
            "soil_seed": expected_soil_seed,
            "base_port": expected_base_port,
            "sim_seed_first": expected_sim_seed,
        },
        context="collection config",
    )
    return config


def _load_source_trace_manifest(
    path: Path,
    *,
    expected_count: int,
    collection_config: dict[str, Any],
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            dtype={"trace_id": str, "trace_sha256": str},
            float_precision="round_trip",
        )
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"cannot read source trace manifest: {error}") from error
    forbidden = [
        column
        for column in frame.columns
        if any(
            token in str(column).strip().lower().replace("-", "_").split("_")
            for token in ("truth", "true", "oracle", "soil", "plant")
        )
    ]
    if forbidden:
        raise ValueError(
            "source trace manifest contains forbidden truth fields: "
            + ", ".join(sorted(forbidden))
        )
    required = {
        "trace_id",
        "status",
        "trace_path",
        "trace_sha256",
        "trace_rows",
        "trace_schema_version",
        "trace_quality",
        "controller_prior",
        "controller_prior_n",
        "terrain_estimator_enabled",
        "sim_seed",
        "path",
        "speed_mps",
        "sim_time_s",
        "lead_in_m",
        "maneuver_label",
        "terrain_id_probe",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("source trace manifest is missing: " + ", ".join(missing))
    expected_ids = _expected_trace_ids(expected_count)
    if frame["trace_id"].astype(str).duplicated().any():
        raise ValueError("source trace manifest contains duplicate trace IDs")
    frame = frame.sort_values("trace_id").reset_index(drop=True)
    if frame["trace_id"].astype(str).tolist() != expected_ids:
        raise ValueError(
            "source trace manifest does not contain the expected contiguous "
            "trace_0000.. IDs"
        )
    if set(frame["status"].astype(str)) != {"ok"}:
        raise ValueError("source trace manifest contains failed collection cells")
    if set(frame["trace_quality"].astype(str)) != {
        "exact_runtime_observations"
    }:
        raise ValueError("source traces are not exact runtime observations")
    if set(frame["controller_prior"].astype(str).str.lower()) != {"dirt"}:
        raise ValueError("source traces did not use the fixed dirt controller")
    if set(frame["maneuver_label"].astype(str)) != {"passive"}:
        raise ValueError("source traces did not use the frozen passive maneuver")
    if set(frame["path"].astype(str)) != {"sinusoidal"}:
        raise ValueError("source traces did not use the frozen path")
    if _strict_bool(
        frame["terrain_estimator_enabled"], name="terrain_estimator_enabled"
    ).any():
        raise ValueError("source trace collection enabled terrain estimation")
    if _strict_bool(
        frame["terrain_id_probe"], name="terrain_id_probe"
    ).any():
        raise ValueError("source trace collection enabled the ID probe")

    numeric_expectations = {
        "trace_schema_version": 3.0,
        "controller_prior_n": 0.7,
        "speed_mps": float(collection_config["speed_mps"]),
        "sim_time_s": float(collection_config["sim_time_s"]),
        "lead_in_m": float(collection_config["lead_in_m"]),
    }
    _finite(
        frame,
        list(numeric_expectations) + ["trace_rows", "sim_seed"],
        context="source trace manifest",
    )
    for column, expected in numeric_expectations.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.allclose(values, expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"source trace manifest has non-frozen {column} values"
            )
    rows = pd.to_numeric(frame["trace_rows"], errors="coerce").to_numpy(float)
    if np.any(rows < 2) or np.any(rows != np.floor(rows)):
        raise ValueError("source trace manifest has invalid trace row counts")
    expected_sim_seeds = (
        int(collection_config["sim_seed_first"])
        + np.arange(expected_count, dtype=int)
    )
    sim_seed_values = pd.to_numeric(
        frame["sim_seed"], errors="coerce"
    ).to_numpy(float)
    if np.any(sim_seed_values != np.floor(sim_seed_values)):
        raise ValueError("source trace manifest has non-integer simulator seeds")
    sim_seeds = sim_seed_values.astype(int)
    if not np.array_equal(sim_seeds, expected_sim_seeds):
        raise ValueError("source trace manifest has non-contiguous simulator seeds")

    hashes = frame["trace_sha256"].astype(str).str.lower()
    if (
        not hashes.str.fullmatch(r"[0-9a-f]{64}").all()
        or hashes.duplicated().any()
    ):
        raise ValueError("source trace manifest has invalid or duplicate hashes")
    try:
        from benchmarking.terrain_estimator_trace import load_sensor_trace
    except ModuleNotFoundError:
        from terrain_estimator_trace import load_sensor_trace
    resolved: list[str] = []
    trace_final_times: list[float] = []
    for row in frame.itertuples(index=False):
        raw = Path(str(row.trace_path)).expanduser()
        trace_path = (
            raw.resolve()
            if raw.is_absolute()
            else (path.parent / raw).resolve()
        )
        if not trace_path.is_file():
            raise ValueError(f"source sensor trace is missing: {row.trace_id}")
        actual_hash = _sha256_file(trace_path)
        if actual_hash != str(row.trace_sha256).lower():
            raise ValueError(
                f"source sensor trace hash mismatch: {row.trace_id}"
            )
        try:
            loaded_trace = load_sensor_trace(trace_path)
            actual_rows = len(loaded_trace)
        except ValueError as error:
            raise ValueError(
                f"source sensor trace schema mismatch: {row.trace_id}: {error}"
            ) from error
        if actual_rows != int(row.trace_rows):
            raise ValueError(
                f"source sensor trace row-count mismatch: {row.trace_id}"
            )
        resolved.append(str(trace_path))
        trace_final_times.append(float(loaded_trace["sim_time"].iloc[-1]))
    output = frame.copy()
    output["_resolved_trace_path"] = resolved
    output["_trace_final_time"] = trace_final_times
    output["trace_sha256"] = hashes
    return output


def _validate_model_name(
    manifest: dict[str, Any], key: str, *, context: str
) -> None:
    if key not in manifest:
        raise ValueError(f"{context} is missing {key}")
    raw = Path(str(manifest[key])).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    expected = (ROOT / "nn_models/tire_force_rate").resolve()
    if resolved != expected:
        raise ValueError(
            f"{context} does not use the repository tire_force_rate"
        )


def _validate_replay_manifests(
    *,
    joint_path: Path,
    parent_path: Path,
    trace_manifest_path: Path,
    expected_ids: list[str],
    min_observability_singular_value: float,
    max_final_update_age_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    joint = _load_key_value_manifest(
        joint_path, context="joint replay manifest"
    )
    parent = _load_key_value_manifest(
        parent_path, context="parent replay manifest"
    )
    trace_digest = _sha256_file(trace_manifest_path)
    for name, manifest in (("joint", joint), ("parent", parent)):
        _require_mapping_values(
            manifest,
            {
                "truth_inputs": "none",
                "trace_manifest_sha256": trace_digest,
            },
            context=f"{name} replay manifest",
        )

    joint_expected = dict(_FROZEN_JOINT_SETTINGS)
    joint_expected.update(
        {
            "backend": "grit",
            "selected_trace_count": len(expected_ids),
            "selected_trace_ids": expected_ids,
            "config.min_observability_singular_value": (
                min_observability_singular_value
            ),
            "config.max_final_update_age_s": max_final_update_age_s,
        }
    )
    _require_mapping_values(
        joint, joint_expected, context="joint replay manifest"
    )
    _require_mapping_values(
        parent, _FROZEN_PARENT_SETTINGS, context="parent replay manifest"
    )
    _validate_model_name(
        joint, "config.model_dir", context="joint replay manifest"
    )
    _validate_model_name(
        parent, "config.model_dir", context="parent replay manifest"
    )
    try:
        current_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot validate replay Git provenance: {error}") from error
    _require_mapping_values(
        joint,
        {
            "provenance.git_head": current_head,
            "provenance.git_dirty": False,
        },
        context="joint replay manifest",
    )

    provenance_files = {
        "provenance.driver_source_sha256": (
            ROOT / "benchmarking/develop_joint_estimator.py"
        ),
        "provenance.candidate_source_sha256": (
            ROOT
            / "simulation/estimators/grit_terrain_estimator.py"
        ),
        "provenance.parent_estimator_source_sha256": (
            ROOT
            / "simulation/estimators/scalar_parent_terrain_estimator.py"
        ),
        "provenance.force_checkpoint_sha256": (
            ROOT / "nn_models/tire_force_rate/best_terrain_nn.pt"
        ),
        "provenance.force_scalers_sha256": (
            ROOT / "nn_models/tire_force_rate/scalers.pkl"
        ),
    }
    for key, source in provenance_files.items():
        if not source.is_file():
            raise ValueError(f"cannot validate replay provenance; missing {source}")
        _require_mapping_values(
            joint,
            {key: _sha256_file(source)},
            context="joint replay manifest",
        )
    return joint, parent


def _load_truth_bundle(
    truth_path: Path,
    *,
    soil_dir: Path,
    expected_count: int,
    expected_soil_seed: int,
) -> tuple[pd.DataFrame, list[Path]]:
    try:
        truth = pd.read_csv(truth_path, float_precision="round_trip")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"cannot read scorer truth sidecar: {error}") from error
    required = {
        "trace_id",
        "n_true",
        "Kphi_true",
        "Kc_true",
        "c_true",
        "phi_true_deg",
        "k_true",
        "soil_draw_seed",
        "nuisance_seed",
        "case_order_seed",
        "nuisance_jitter_fraction",
        "soil_mode",
        "phi_draw_seed",
        "cohesion_draw_seed",
        "cohesion_jitter_fraction",
        "cohesion_multiplier_lo",
        "cohesion_multiplier_hi",
        "cohesion_multiplier_true",
    }
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError("truth sidecar is missing: " + ", ".join(missing))
    if truth["trace_id"].astype(str).duplicated().any():
        raise ValueError("truth sidecar contains duplicate trace IDs")
    truth = truth.sort_values("trace_id").reset_index(drop=True)
    expected_ids = _expected_trace_ids(expected_count)
    if truth["trace_id"].astype(str).tolist() != expected_ids:
        raise ValueError(
            "truth sidecar does not contain the expected contiguous trace IDs"
        )
    numeric_columns = sorted(required - {"trace_id", "soil_mode"})
    _finite(truth, numeric_columns, context="truth sidecar")
    if set(truth["soil_mode"].astype(str)) != {"independent_n_phi"}:
        raise ValueError("truth sidecar has the wrong soil mode")

    expected_columns = {
        "soil_draw_seed": expected_soil_seed,
        "nuisance_seed": expected_soil_seed + 1_000_003,
        "case_order_seed": expected_soil_seed + 2_000_003,
        "nuisance_jitter_fraction": 0.0,
        "phi_draw_seed": expected_soil_seed + 1_000_003,
        "cohesion_draw_seed": expected_soil_seed + 1_000_004,
        "cohesion_jitter_fraction": 0.15,
        "cohesion_multiplier_lo": 0.85,
        "cohesion_multiplier_hi": 1.15,
    }
    for column, expected in expected_columns.items():
        values = pd.to_numeric(truth[column], errors="coerce").to_numpy(float)
        if not np.allclose(values, expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"truth sidecar has invalid {column}")

    n_values = truth["n_true"].to_numpy(float)
    phi_values = truth["phi_true_deg"].to_numpy(float)
    cohesion_values = truth["cohesion_multiplier_true"].to_numpy(float)
    if (
        np.any(n_values < 0.55)
        or np.any(n_values > 1.05)
        or np.any(phi_values < 10.0)
        or np.any(phi_values > 35.0)
        or np.any(cohesion_values < 0.85)
        or np.any(cohesion_values > 1.15)
    ):
        raise ValueError("truth sidecar exceeds the frozen terrain bounds")

    def strata(values: np.ndarray, lower: float, upper: float) -> set[int]:
        indices = np.floor(
            (values - lower) * expected_count / (upper - lower)
        ).astype(int)
        indices = np.clip(indices, 0, expected_count - 1)
        return set(int(value) for value in indices)

    expected_strata = set(range(expected_count))
    if (
        strata(n_values, 0.55, 1.05) != expected_strata
        or strata(phi_values, 10.0, 35.0) != expected_strata
    ):
        raise ValueError(
            "truth sidecar does not cover every frozen n/phi stratum exactly once"
        )

    try:
        from benchmarking.collect_terrain_estimator_traces import (
            generate_soils,
            manifold_yaml_from_n,
        )
    except ModuleNotFoundError:
        from collect_terrain_estimator_traces import (
            generate_soils,
            manifold_yaml_from_n,
        )
    generated, generated_nuisance_seed = generate_soils(
        expected_count,
        mode="independent_n_phi",
        seed=expected_soil_seed,
        jitter_fraction=0.1,
    )
    if generated_nuisance_seed != expected_soil_seed + 1_000_003:
        raise ValueError("truth soil generator returned an inconsistent nuisance seed")
    permutation = np.random.default_rng(
        expected_soil_seed + 2_000_003
    ).permutation(expected_count)
    expected_soils = [generated[int(index)] for index in permutation]

    soil_dir = soil_dir.expanduser().resolve()
    if not soil_dir.is_dir():
        raise ValueError(f"truth soil directory does not exist: {soil_dir}")
    soil_paths = sorted(soil_dir.glob("*.yaml"))
    expected_names = [f"case_{index:04d}.yaml" for index in range(expected_count)]
    if [path.name for path in soil_paths] != expected_names:
        raise ValueError(
            "truth soil directory does not contain exactly the expected YAML files"
        )
    truth_to_yaml = {
        "n_true": "n",
        "Kphi_true": "Kphi",
        "Kc_true": "Kc",
        "c_true": "cohesion",
        "phi_true_deg": "friction_angle",
        "k_true": "janosi_shear",
    }
    for index, soil_path in enumerate(soil_paths):
        try:
            soil = yaml.safe_load(soil_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(
                f"cannot read truth soil YAML {soil_path.name}: {error}"
            ) from error
        if not isinstance(soil, dict):
            raise ValueError(f"truth soil YAML is not an object: {soil_path.name}")
        row = truth.iloc[index]
        generated_n, generated_soil = expected_soils[index]
        if not math.isclose(
            float(row["n_true"]),
            float(generated_n),
            rel_tol=2.0e-15,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"truth sidecar does not reproduce soil seed for {row['trace_id']}"
            )
        manifold_soil = manifold_yaml_from_n(float(generated_n))
        generated_multiplier = (
            float(generated_soil["cohesion"])
            / float(manifold_soil["cohesion"])
        )
        if not math.isclose(
            float(row["cohesion_multiplier_true"]),
            generated_multiplier,
            rel_tol=2.0e-15,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "truth sidecar has an inconsistent cohesion multiplier for "
                f"{row['trace_id']}"
            )
        for truth_column, yaml_key in truth_to_yaml.items():
            if yaml_key not in soil:
                raise ValueError(
                    f"truth soil YAML {soil_path.name} is missing {yaml_key}"
                )
            try:
                yaml_value = float(soil[yaml_key])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"truth soil YAML {soil_path.name} has invalid {yaml_key}"
                ) from error
            if not math.isclose(
                float(row[truth_column]),
                yaml_value,
                rel_tol=2.0e-15,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"truth soil YAML mismatch for {row['trace_id']}/{yaml_key}"
                )
            if not math.isclose(
                yaml_value,
                float(generated_soil[yaml_key]),
                rel_tol=2.0e-15,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "truth soil YAML does not reproduce the registered seed for "
                    f"{row['trace_id']}/{yaml_key}"
                )
        expected_yaml_keys = {
            "n",
            "Kphi",
            "Kc",
            "cohesion",
            "friction_angle",
            "janosi_shear",
            "elastic_stiffness",
            "damping",
            "description",
        }
        if set(soil) != expected_yaml_keys:
            raise ValueError(
                f"truth soil YAML has a non-frozen schema: {soil_path.name}"
            )
        if (
            not math.isclose(
                float(soil["elastic_stiffness"]),
                2.0e8,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            or not math.isclose(
                float(soil["damping"]),
                3.0e4,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or str(soil["description"])
            != "fixed-controller terrain-estimator trace case"
        ):
            raise ValueError(
                f"truth soil YAML has non-frozen fixed fields: {soil_path.name}"
            )
    return truth, soil_paths


def _load_inference(
    path: Path,
    *,
    joint: bool,
    min_observability_singular_value: float,
    max_final_update_age_s: float,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, float_precision="round_trip")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"cannot read inference artifact {path}: {error}") from error
    forbidden = [
        column
        for column in frame.columns
        if any(
            token in str(column).lower().split("_")
            for token in ("truth", "true", "oracle", "soil")
        )
    ]
    if forbidden:
        raise ValueError(
            "inference artifact contains forbidden truth fields: "
            + ", ".join(sorted(forbidden))
        )
    required = {
        "trace_id",
        "backend",
        "status",
        "trace_sha256",
        "trace_quality",
        "final_est_n",
        "timeseries_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("inference artifact is missing: " + ", ".join(missing))
    if frame.empty or frame["trace_id"].astype(str).duplicated().any():
        raise ValueError("inference artifact must contain unique trace IDs")
    if set(frame["status"].astype(str)) != {"ok"}:
        raise ValueError("inference artifact contains failed replay cells")
    if set(frame["trace_quality"].astype(str)) != {
        "exact_runtime_observations"
    }:
        raise ValueError("scoring requires exact runtime observations")
    if not frame["trace_sha256"].astype(str).str.fullmatch(
        r"[0-9a-fA-F]{64}"
    ).all():
        raise ValueError("inference artifact contains an invalid trace SHA-256")
    _finite(frame, ["final_est_n"], context="inference artifact")
    diagnostic_columns: list[str] = []
    if joint:
        if set(frame["backend"].astype(str)) != {"grit"}:
            raise ValueError("joint artifact has the wrong backend")
        diagnostic_columns = [
            "final_est_phi_deg",
            "final_snapshot_n",
            "final_snapshot_phi_deg",
            "final_joint_active",
            "final_joint_has_estimate",
            "final_joint_updates",
            "final_joint_projection_failures",
            "final_observability_rank",
            "final_observability_min_singular_value",
            "final_last_likelihood_block_count",
            "final_last_likelihood_residual_count",
            "final_duplicate_likelihood_block_count",
            "final_duplicate_likelihood_update_count",
            "final_last_joint_update_time",
            "final_final_trace_time",
            "final_final_update_age_s",
            "final_final_update_max_age_s",
            "final_accuracy_valid",
            "final_fresh",
            "final_confident",
            "final_control_envelope_valid",
            "final_publication_ready",
            "final_snapshot_was_published",
            "final_publication_confidence",
            "final_n_sigma",
            "final_phi_sigma_deg",
            "final_joint_information_kl",
            "final_cohesion_information_kl",
            "final_n_boundary_mass",
            "final_phi_boundary_mass",
            "final_cohesion_boundary_mass",
            "final_max_boundary_mass",
            "final_n_information_kl",
            "final_phi_information_kl",
            "final_boundary_limited",
            "final_likelihood_evaluations",
            "final_last_effective_front_load",
            "final_last_effective_rear_load",
            "final_last_effective_load_ax",
            "final_last_effective_load_ay",
            "final_projection_wall_time_s",
            "final_profile_wall_time_s",
            "final_observability_wall_time_s",
            "final_posterior_wall_time_s",
            "final_publication_wall_time_s",
            "final_update_wall_time_s",
        ]
        _finite(frame, diagnostic_columns, context="joint diagnostics")
        string_columns = {
            "final_accepted_snapshot_version",
            "final_load_transfer_mode",
        }
        missing_strings = sorted(string_columns - set(frame.columns))
        if missing_strings:
            raise ValueError(
                "joint diagnostics is missing: " + ", ".join(missing_strings)
            )
        if set(
            frame["final_accepted_snapshot_version"].astype(str)
        ) != {ACCEPTED_SNAPSHOT_VERSION}:
            raise ValueError("joint artifact has the wrong snapshot version")
        if not _strict_bool(
            frame["final_joint_active"], name="final_joint_active"
        ).all():
            raise ValueError("joint artifact is not bound to an accepted update")
        if not _strict_bool(
            frame["final_joint_has_estimate"], name="final_joint_has_estimate"
        ).all():
            raise ValueError("joint estimator did not produce every estimate")
        if not _strict_bool(
            frame["final_accuracy_valid"], name="final_accuracy_valid"
        ).all():
            raise ValueError(
                "joint artifact contains a non-accuracy-valid estimate"
            )
        fresh = _strict_bool(frame["final_fresh"], name="final_fresh")
        confident = _strict_bool(
            frame["final_confident"], name="final_confident"
        )
        publication_ready = _strict_bool(
            frame["final_publication_ready"],
            name="final_publication_ready",
        )
        control_envelope_valid = _strict_bool(
            frame["final_control_envelope_valid"],
            name="final_control_envelope_valid",
        )
        snapshot_was_published = _strict_bool(
            frame["final_snapshot_was_published"],
            name="final_snapshot_was_published",
        )
        boundary_limited = _strict_bool(
            frame["final_boundary_limited"],
            name="final_boundary_limited",
        )
        integer = frame[diagnostic_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        integer_columns = [
            "final_joint_updates",
            "final_joint_projection_failures",
            "final_observability_rank",
            "final_last_likelihood_block_count",
            "final_last_likelihood_residual_count",
            "final_duplicate_likelihood_block_count",
            "final_duplicate_likelihood_update_count",
            "final_likelihood_evaluations",
            "final_accuracy_valid",
            "final_fresh",
            "final_confident",
            "final_control_envelope_valid",
            "final_publication_ready",
            "final_snapshot_was_published",
            "final_boundary_limited",
        ]
        integer_values = integer[integer_columns].to_numpy(float)
        if not np.array_equal(integer_values, np.floor(integer_values)):
            raise ValueError("joint artifact contains non-integer diagnostics")
        if (integer["final_joint_updates"] < 1).any():
            raise ValueError("joint artifact contains no-update traces")
        if (integer["final_joint_projection_failures"] != 0).any():
            raise ValueError("joint artifact contains projection failures")
        if (integer["final_observability_rank"] < 2).any():
            raise ValueError("joint artifact contains rank-deficient estimates")
        if (
            integer["final_observability_min_singular_value"]
            + 1.0e-12
            < min_observability_singular_value
        ).any():
            raise ValueError(
                "joint artifact violates the observability singular-value floor"
            )
        if (
            integer["final_duplicate_likelihood_block_count"] != 0
        ).any() or (
            integer["final_duplicate_likelihood_update_count"] != 0
        ).any():
            raise ValueError("joint likelihood consumed duplicate sensor blocks")
        if not np.array_equal(
            integer["final_last_likelihood_residual_count"].to_numpy(int),
            2
            * integer["final_last_likelihood_block_count"].to_numpy(int),
        ):
            raise ValueError("joint likelihood did not use one ax/ay pair per block")
        if set(frame["final_load_transfer_mode"].astype(str)) != {"static"}:
            raise ValueError("joint artifact has the wrong load-transfer mode")
        confidence = integer[
            "final_publication_confidence"
        ].to_numpy(float)
        if np.any(confidence < -1.0e-12) or np.any(
            confidence > 1.0 + 1.0e-12
        ):
            raise ValueError(
                "joint artifact has invalid snapshot confidence"
            )
        expected_confident = (
            confidence >= FROZEN_PUBLICATION_CONFIDENCE
        )
        if not np.array_equal(confident, expected_confident):
            raise ValueError("joint artifact has inconsistent confidence flags")
        if not np.array_equal(snapshot_was_published, expected_confident):
            raise ValueError(
                "joint artifact violates the frozen publication policy"
            )
        expected_control_envelope = (
            frame["final_est_phi_deg"].to_numpy(float)
            >= FROZEN_CONTROL_MIN_PHI_DEG
        )
        if not np.array_equal(
            control_envelope_valid,
            expected_control_envelope,
        ):
            raise ValueError(
                "joint artifact has inconsistent control-envelope flags"
            )
        ages = integer["final_final_update_age_s"].to_numpy(float)
        configured_ages = integer[
            "final_final_update_max_age_s"
        ].to_numpy(float)
        if not np.allclose(
            configured_ages,
            max_final_update_age_s,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("joint artifact has the wrong final-update age limit")
        if np.any(ages < -1.0e-9):
            raise ValueError("joint artifact contains a noncausal estimate")
        last_updates = integer[
            "final_last_joint_update_time"
        ].to_numpy(float)
        if np.any(last_updates < 7.0 - 1.0e-9):
            raise ValueError("joint artifact has no accepted update in the tail")
        recomputed_ages = (
            integer["final_final_trace_time"].to_numpy(float)
            - last_updates
        )
        if not np.allclose(
            ages, recomputed_ages, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("joint artifact has inconsistent final-update ages")
        expected_fresh = ages <= max_final_update_age_s + 1.0e-9
        if not np.array_equal(fresh, expected_fresh):
            raise ValueError("joint artifact has inconsistent freshness flags")
        probability_columns = [
            "final_n_boundary_mass",
            "final_phi_boundary_mass",
            "final_cohesion_boundary_mass",
            "final_max_boundary_mass",
        ]
        probabilities = integer[probability_columns].to_numpy(float)
        if np.any(probabilities < -1.0e-12) or np.any(
            probabilities > 1.0 + 1.0e-12
        ):
            raise ValueError("joint artifact has invalid boundary masses")
        component_boundary = integer[
            [
                "final_n_boundary_mass",
                "final_phi_boundary_mass",
                "final_cohesion_boundary_mass",
            ]
        ].to_numpy(float)
        if not np.allclose(
            probabilities[:, -1],
            np.max(component_boundary, axis=1),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("joint artifact has inconsistent maximum boundary mass")
        expected_boundary_limited = (
            probabilities[:, -1] >= FROZEN_BOUNDARY_MASS_LIMIT
        )
        if not np.array_equal(
            boundary_limited,
            expected_boundary_limited,
        ):
            raise ValueError(
                "joint artifact has inconsistent boundary-limited flags"
            )
        expected_ready = (
            fresh
            & confident
            & control_envelope_valid
            & ~boundary_limited
            & (
                integer["final_observability_rank"].to_numpy(float)
                >= 2.0
            )
            & (
                integer[
                    "final_observability_min_singular_value"
                ].to_numpy(float)
                + 1.0e-12
                >= min_observability_singular_value
            )
        )
        if not np.array_equal(publication_ready, expected_ready):
            raise ValueError(
                "joint artifact has inconsistent publication-ready flags"
            )
        if (
            integer["final_n_sigma"].to_numpy(float) < 0.0
        ).any() or (
            integer["final_phi_sigma_deg"].to_numpy(float) < 0.0
        ).any():
            raise ValueError("joint artifact has invalid snapshot uncertainty")
        timing_columns = [
            "final_projection_wall_time_s",
            "final_profile_wall_time_s",
            "final_observability_wall_time_s",
            "final_posterior_wall_time_s",
            "final_publication_wall_time_s",
            "final_update_wall_time_s",
        ]
        if (integer[timing_columns].to_numpy(float) < 0.0).any():
            raise ValueError("joint artifact has invalid snapshot timings")
        expected_update_time = (
            integer["final_projection_wall_time_s"].to_numpy(float)
            + integer["final_posterior_wall_time_s"].to_numpy(float)
            + integer["final_publication_wall_time_s"].to_numpy(float)
        )
        if not np.allclose(
            integer["final_update_wall_time_s"].to_numpy(float),
            expected_update_time,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("joint artifact has inconsistent total timing")
    else:
        if set(frame["backend"].astype(str)) != {"scalar_parent"}:
            raise ValueError("parent artifact has the wrong backend")

    timeline_paths: list[str] = []
    timeline_hashes: list[str] = []
    timeline_rows: list[int] = []
    timeline_final_times: list[float] = []
    seen_paths: set[Path] = set()
    expected_backend = "grit" if joint else "scalar_parent"
    for row in frame.to_dict(orient="records"):
        raw_value = str(row["timeseries_path"]).strip()
        if not raw_value or raw_value.lower() == "nan":
            raise ValueError(
                f"inference artifact has no time series for {row['trace_id']}"
            )
        raw = Path(raw_value).expanduser()
        timeline_path = (
            raw.resolve()
            if raw.is_absolute()
            else (path.parent / raw).resolve()
        )
        if timeline_path in seen_paths:
            raise ValueError("inference artifact reuses a replay time series")
        seen_paths.add(timeline_path)
        try:
            timeline = pd.read_csv(
                timeline_path, float_precision="round_trip"
            )
        except (
            FileNotFoundError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
        ) as error:
            raise ValueError(
                f"cannot read replay time series for {row['trace_id']}: {error}"
            ) from error
        forbidden_timeline = [
            column
            for column in timeline.columns
            if any(
                token in str(column).lower().split("_")
                for token in ("truth", "true", "oracle", "soil")
            )
        ]
        if forbidden_timeline:
            raise ValueError(
                "replay time series contains forbidden truth fields: "
                + ", ".join(sorted(forbidden_timeline))
            )
        required_timeline = {
            "trace_id",
            "backend",
            "sim_time",
            "n_published",
        }
        if joint:
            required_timeline.update(
                {
                    "phi_published_deg",
                    "confidence",
                    "published_update",
                    "joint_snapshot_advanced",
                    "joint_snapshot_seq",
                    "joint_snapshot_evidence_time_s",
                    "joint_snapshot_n",
                    "joint_snapshot_phi_deg",
                    "joint_snapshot_n_sigma",
                    "joint_snapshot_phi_sigma_deg",
                    "joint_snapshot_confidence",
                    "joint_snapshot_information_kl",
                    "joint_snapshot_observability_rank",
                    "joint_snapshot_observability_min_singular_value",
                    "joint_snapshot_n_boundary_mass",
                    "joint_snapshot_phi_boundary_mass",
                    "joint_snapshot_max_boundary_mass",
                    "joint_snapshot_boundary_limited",
                    "joint_snapshot_projection_wall_time_s",
                    "joint_snapshot_profile_wall_time_s",
                    "joint_snapshot_observability_wall_time_s",
                    "joint_snapshot_posterior_wall_time_s",
                    "joint_snapshot_publication_wall_time_s",
                    "joint_snapshot_update_wall_time_s",
                }
            )
            required_timeline.update(
                f"replay_{column}"
                for column in diagnostic_columns
                if column != "final_est_phi_deg"
            )
            required_timeline.update(
                {
                    "replay_final_accepted_snapshot_version",
                    "replay_final_load_transfer_mode",
                }
            )
        missing_timeline = sorted(
            required_timeline - set(timeline.columns)
        )
        if timeline.empty or missing_timeline:
            detail = (
                ": " + ", ".join(missing_timeline)
                if missing_timeline
                else ""
            )
            raise ValueError(
                f"replay time series is empty or missing fields{detail}"
            )
        trace_id = str(row["trace_id"])
        if set(timeline["trace_id"].astype(str)) != {trace_id}:
            raise ValueError(
                f"replay time-series trace ID mismatch: {trace_id}"
            )
        if set(timeline["backend"].astype(str)) != {expected_backend}:
            raise ValueError(
                f"replay time-series backend mismatch: {trace_id}"
            )
        numeric_columns = ["sim_time", "n_published"]
        if joint:
            numeric_columns.extend(["phi_published_deg", "confidence"])
        _finite(
            timeline,
            numeric_columns,
            context=f"replay time series {trace_id}",
        )
        times = pd.to_numeric(
            timeline["sim_time"], errors="coerce"
        ).to_numpy(float)
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError(
                f"replay time series is not strictly time ordered: {trace_id}"
            )
        final_n = float(
            pd.to_numeric(timeline["n_published"], errors="coerce").iloc[-1]
        )
        if not joint and not math.isclose(
            final_n,
            float(row["final_est_n"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                f"final estimate does not match replay time series: {trace_id}"
            )
        if joint:
            if (
                not math.isclose(
                    float(row["final_snapshot_n"]),
                    float(row["final_est_n"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    float(row["final_snapshot_phi_deg"]),
                    float(row["final_est_phi_deg"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    "final estimate does not match accepted snapshot: "
                    f"{trace_id}"
                )
            for summary_column in diagnostic_columns:
                if summary_column == "final_est_phi_deg":
                    continue
                timeline_column = f"replay_{summary_column}"
                values = pd.to_numeric(
                    timeline[timeline_column], errors="coerce"
                ).to_numpy(float)
                if not np.isfinite(values).all() or not np.allclose(
                    values,
                    float(row[summary_column]),
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        "joint diagnostics do not match replay time series: "
                        f"{trace_id}/{summary_column}"
                    )
            if set(
                timeline[
                    "replay_final_accepted_snapshot_version"
                ].astype(str)
            ) != {str(row["final_accepted_snapshot_version"])}:
                raise ValueError(
                    "joint snapshot version does not match replay time "
                    f"series: {trace_id}"
                )
            if set(
                timeline["replay_final_load_transfer_mode"].astype(str)
            ) != {str(row["final_load_transfer_mode"])}:
                raise ValueError(
                    "joint load-transfer mode does not match replay time "
                    f"series: {trace_id}"
                )
            snapshot_updates = _strict_bool(
                timeline["joint_snapshot_advanced"],
                name=f"joint_snapshot_advanced/{trace_id}",
            )
            if not snapshot_updates.any():
                raise ValueError(
                    f"joint replay has no accepted snapshot row: {trace_id}"
                )
            last_snapshot = timeline.loc[snapshot_updates].iloc[-1]
            snapshot_bindings = {
                "joint_snapshot_seq": "final_joint_updates",
                "joint_snapshot_evidence_time_s": (
                    "final_last_joint_update_time"
                ),
                "joint_snapshot_n": "final_est_n",
                "joint_snapshot_phi_deg": "final_est_phi_deg",
                "joint_snapshot_n_sigma": "final_n_sigma",
                "joint_snapshot_phi_sigma_deg": "final_phi_sigma_deg",
                "joint_snapshot_confidence": (
                    "final_publication_confidence"
                ),
                "joint_snapshot_information_kl": (
                    "final_joint_information_kl"
                ),
                "joint_snapshot_observability_rank": (
                    "final_observability_rank"
                ),
                "joint_snapshot_observability_min_singular_value": (
                    "final_observability_min_singular_value"
                ),
                "joint_snapshot_n_boundary_mass": (
                    "final_n_boundary_mass"
                ),
                "joint_snapshot_phi_boundary_mass": (
                    "final_phi_boundary_mass"
                ),
                "joint_snapshot_max_boundary_mass": (
                    "final_max_boundary_mass"
                ),
                "joint_snapshot_projection_wall_time_s": (
                    "final_projection_wall_time_s"
                ),
                "joint_snapshot_profile_wall_time_s": (
                    "final_profile_wall_time_s"
                ),
                "joint_snapshot_observability_wall_time_s": (
                    "final_observability_wall_time_s"
                ),
                "joint_snapshot_posterior_wall_time_s": (
                    "final_posterior_wall_time_s"
                ),
                "joint_snapshot_publication_wall_time_s": (
                    "final_publication_wall_time_s"
                ),
                "joint_snapshot_update_wall_time_s": (
                    "final_update_wall_time_s"
                ),
            }
            for timeline_column, summary_column in snapshot_bindings.items():
                try:
                    timeline_value = float(last_snapshot[timeline_column])
                    summary_value = float(row[summary_column])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "accepted snapshot contains a nonnumeric binding: "
                        f"{trace_id}/{timeline_column}"
                    ) from error
                if not math.isclose(
                    timeline_value,
                    summary_value,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        "accepted snapshot does not match replay time series: "
                        f"{trace_id}/{timeline_column}"
                    )
            last_snapshot_boundary = _strict_bool(
                pd.Series(
                    [last_snapshot["joint_snapshot_boundary_limited"]]
                ),
                name=f"joint_snapshot_boundary_limited/{trace_id}",
            )[0]
            if last_snapshot_boundary != bool(
                _strict_bool(
                    pd.Series([row["final_boundary_limited"]]),
                    name=f"final_boundary_limited/{trace_id}",
                )[0]
            ):
                raise ValueError(
                    "accepted snapshot boundary flag does not match replay "
                    f"time series: {trace_id}"
                )

            publication_updates = _strict_bool(
                timeline["published_update"],
                name=f"published_update/{trace_id}",
            )
            snapshot_was_published_in_timeline = False
            if publication_updates.any():
                last_publication = timeline.loc[publication_updates].iloc[-1]
                try:
                    publication_values = np.asarray(
                        [
                            last_publication["n_published"],
                            last_publication["phi_published_deg"],
                            last_publication["confidence"],
                            last_publication["sim_time"],
                        ],
                        dtype=float,
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"joint replay has an invalid publication row: {trace_id}"
                    ) from error
                snapshot_was_published_in_timeline = bool(
                    np.isfinite(publication_values).all()
                    and math.isclose(
                        publication_values[0],
                        float(row["final_est_n"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    and math.isclose(
                        publication_values[1],
                        float(row["final_est_phi_deg"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    and math.isclose(
                        publication_values[2],
                        float(row["final_publication_confidence"]),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    and publication_values[3] + 1.0e-9
                    >= float(row["final_last_joint_update_time"])
                )
            declared_published = bool(
                _strict_bool(
                    pd.Series([row["final_snapshot_was_published"]]),
                    name=f"final_snapshot_was_published/{trace_id}",
                )[0]
            )
            if snapshot_was_published_in_timeline != declared_published:
                raise ValueError(
                    "snapshot publication flag does not match replay time "
                    f"series: {trace_id}"
                )
        timeline_paths.append(str(timeline_path))
        timeline_hashes.append(_sha256_file(timeline_path))
        timeline_rows.append(len(timeline))
        timeline_final_times.append(float(times[-1]))

    output = frame.copy()
    output["_resolved_timeseries_path"] = timeline_paths
    output["_timeseries_sha256"] = timeline_hashes
    output["_timeseries_rows"] = timeline_rows
    output["_timeseries_final_time"] = timeline_final_times
    return output.sort_values("trace_id").reset_index(drop=True)


def validate_inference_matrix(
    joint_path: Path,
    parent_path: Path,
    *,
    source_traces: pd.DataFrame,
    expected_count: int,
    min_observability_singular_value: float,
    max_final_update_age_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_ids = _expected_trace_ids(expected_count)
    joint = _load_inference(
        joint_path,
        joint=True,
        min_observability_singular_value=min_observability_singular_value,
        max_final_update_age_s=max_final_update_age_s,
    )
    parent = _load_inference(
        parent_path,
        joint=False,
        min_observability_singular_value=min_observability_singular_value,
        max_final_update_age_s=max_final_update_age_s,
    )
    if joint["trace_id"].astype(str).tolist() != parent[
        "trace_id"
    ].astype(str).tolist():
        raise ValueError("joint and parent trace-ID sets differ")
    if joint["trace_sha256"].str.lower().tolist() != parent[
        "trace_sha256"
    ].str.lower().tolist():
        raise ValueError("joint and parent replays are not trace-hash identical")
    source = source_traces.sort_values("trace_id").reset_index(drop=True)
    if joint["trace_id"].astype(str).tolist() != expected_ids:
        raise ValueError(
            "inference artifacts do not contain the expected contiguous trace IDs"
        )
    if source["trace_id"].astype(str).tolist() != expected_ids:
        raise ValueError(
            "source trace manifest does not contain the expected contiguous trace IDs"
        )
    if joint["trace_sha256"].str.lower().tolist() != source[
        "trace_sha256"
    ].astype(str).str.lower().tolist():
        raise ValueError("inference artifacts do not match the source trace hashes")
    source_rows = pd.to_numeric(
        source["trace_rows"], errors="coerce"
    ).to_numpy(int)
    source_final_times = pd.to_numeric(
        source["_trace_final_time"], errors="coerce"
    ).to_numpy(float)
    for label, inference in (("joint", joint), ("parent", parent)):
        if not np.array_equal(
            inference["_timeseries_rows"].to_numpy(int), source_rows
        ):
            raise ValueError(
                f"{label} replay time-series rows do not match source traces"
            )
        if not np.allclose(
            inference["_timeseries_final_time"].to_numpy(float),
            source_final_times,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"{label} replay time-series times do not match source traces"
            )
    return joint, parent


def _phi_from_n(n_values: np.ndarray) -> np.ndarray:
    try:
        from simulation.estimators.terrain_parameterization import (
            terrain_params_for_n,
        )
    except ModuleNotFoundError:
        from terrain_parameterization import terrain_params_for_n
    return np.asarray(
        [
            np.degrees(float(terrain_params_for_n(float(value))["phi"]))
            if abs(float(terrain_params_for_n(float(value))["phi"])) <= 2 * np.pi
            else float(terrain_params_for_n(float(value))["phi"])
            for value in n_values
        ],
        dtype=float,
    )


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if (
        len(first) < 2
        or len(second) != len(first)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.all(first == first[0])
        or np.all(second == second[0])
    ):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        statistic = float(spearmanr(first, second).statistic)
    return statistic if np.isfinite(statistic) else float("nan")


def _metric_row(
    method: str,
    n_estimate: np.ndarray,
    phi_estimate: np.ndarray,
    n_true: np.ndarray,
    phi_true: np.ndarray,
) -> dict[str, Any]:
    n_error = n_estimate - n_true
    phi_error = phi_estimate - phi_true
    n_percentage = 100.0 * np.abs(n_error) / np.maximum(np.abs(n_true), 1.0e-12)
    return {
        "method": method,
        "n": int(len(n_true)),
        "n_mae": float(np.mean(np.abs(n_error))),
        "n_rmse": float(np.sqrt(np.mean(n_error**2))),
        "n_bias": float(np.mean(n_error)),
        "n_median_pct_error": float(np.median(n_percentage)),
        "n_pct_within_20": float(100.0 * np.mean(n_percentage <= 20.0)),
        "n_spearman": _safe_spearman(n_estimate, n_true),
        "phi_mae_deg": float(np.mean(np.abs(phi_error))),
        "phi_rmse_deg": float(np.sqrt(np.mean(phi_error**2))),
        "phi_bias_deg": float(np.mean(phi_error)),
        "phi_pct_within_5_deg": float(
            100.0 * np.mean(np.abs(phi_error) <= 5.0)
        ),
        "phi_spearman": _safe_spearman(phi_estimate, phi_true),
        "n_estimate_vs_phi_truth_spearman": _safe_spearman(
            n_estimate, phi_true
        ),
        "phi_estimate_vs_n_truth_spearman": _safe_spearman(
            phi_estimate, n_true
        ),
    }


def _bootstrap_comparison(
    *,
    baseline: str,
    baseline_n: np.ndarray,
    baseline_phi: np.ndarray,
    joint_n: np.ndarray,
    joint_phi: np.ndarray,
    n_true: np.ndarray,
    phi_true: np.ndarray,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    n_units = len(n_true)
    baseline_n_abs = np.abs(baseline_n - n_true)
    joint_n_abs = np.abs(joint_n - n_true)
    baseline_phi_abs = np.abs(baseline_phi - phi_true)
    joint_phi_abs = np.abs(joint_phi - phi_true)
    n_distribution = np.empty(resamples, dtype=float)
    phi_distribution = np.empty(resamples, dtype=float)
    n_rmse_distribution = np.empty(resamples, dtype=float)
    phi_rmse_distribution = np.empty(resamples, dtype=float)
    chunk = 10_000
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        indices = rng.integers(0, n_units, size=(stop - start, n_units))
        n_distribution[start:stop] = np.mean(
            baseline_n_abs[indices] - joint_n_abs[indices], axis=1
        )
        phi_distribution[start:stop] = np.mean(
            baseline_phi_abs[indices] - joint_phi_abs[indices], axis=1
        )
        n_rmse_distribution[start:stop] = np.sqrt(
            np.mean((baseline_n[indices] - n_true[indices]) ** 2, axis=1)
        ) - np.sqrt(
            np.mean((joint_n[indices] - n_true[indices]) ** 2, axis=1)
        )
        phi_rmse_distribution[start:stop] = np.sqrt(
            np.mean((baseline_phi[indices] - phi_true[indices]) ** 2, axis=1)
        ) - np.sqrt(
            np.mean((joint_phi[indices] - phi_true[indices]) ** 2, axis=1)
        )

    def endpoints(values: np.ndarray) -> tuple[float, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    n_low, n_high = endpoints(n_distribution)
    phi_low, phi_high = endpoints(phi_distribution)
    n_rmse_low, n_rmse_high = endpoints(n_rmse_distribution)
    phi_rmse_low, phi_rmse_high = endpoints(phi_rmse_distribution)
    return {
        "comparison": f"joint_vs_{baseline}",
        "baseline": baseline,
        "n_units": int(n_units),
        "bootstrap_seed": int(seed),
        "bootstrap_resamples": int(resamples),
        "confidence_level": 0.95,
        "ci_method": "paired_percentile",
        "n_mae_improvement": float(
            np.mean(baseline_n_abs - joint_n_abs)
        ),
        "n_mae_improvement_ci_low": n_low,
        "n_mae_improvement_ci_high": n_high,
        "n_rmse_improvement": float(
            np.sqrt(np.mean((baseline_n - n_true) ** 2))
            - np.sqrt(np.mean((joint_n - n_true) ** 2))
        ),
        "n_rmse_improvement_ci_low": n_rmse_low,
        "n_rmse_improvement_ci_high": n_rmse_high,
        "phi_mae_improvement_deg": float(
            np.mean(baseline_phi_abs - joint_phi_abs)
        ),
        "phi_mae_improvement_ci_low_deg": phi_low,
        "phi_mae_improvement_ci_high_deg": phi_high,
        "phi_rmse_improvement_deg": float(
            np.sqrt(np.mean((baseline_phi - phi_true) ** 2))
            - np.sqrt(np.mean((joint_phi - phi_true) ** 2))
        ),
        "phi_rmse_improvement_ci_low_deg": phi_rmse_low,
        "phi_rmse_improvement_ci_high_deg": phi_rmse_high,
    }


def _count_at_least_fraction(
    passing: int, total: int, required_fraction: float
) -> bool:
    """Apply a population fraction gate without floating count ambiguity."""

    if total < 1 or passing < 0 or passing > total:
        raise ValueError("fraction-gate counts are invalid")
    if not np.isfinite(required_fraction) or not 0.0 <= required_fraction <= 1.0:
        raise ValueError("required fraction must lie in [0,1]")
    required = int(math.ceil(required_fraction * total - 1.0e-12))
    return int(passing) >= required


def _count_at_most_fraction(
    flagged: int, total: int, maximum_fraction: float
) -> bool:
    """Apply a maximum population fraction gate at the exact integer edge."""

    if total < 1 or flagged < 0 or flagged > total:
        raise ValueError("fraction-gate counts are invalid")
    if not np.isfinite(maximum_fraction) or not 0.0 <= maximum_fraction <= 1.0:
        raise ValueError("maximum fraction must lie in [0,1]")
    allowed = int(math.floor(maximum_fraction * total + 1.0e-12))
    return int(flagged) <= allowed


def score(
    *,
    truth_path: Path,
    truth_soil_dir: Path,
    collection_config_path: Path,
    trace_manifest_path: Path,
    joint_path: Path,
    joint_replay_manifest_path: Path,
    parent_path: Path,
    parent_replay_manifest_path: Path,
    expected_count: int,
    expected_soil_seed: int,
    expected_base_port: int,
    expected_sim_seed: int,
    min_observability_singular_value: float = (
        FROZEN_MIN_OBSERVABILITY_SINGULAR_VALUE
    ),
    max_final_update_age_s: float = FROZEN_MAX_FINAL_UPDATE_AGE_S,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    _expected_trace_ids(expected_count)
    if (
        isinstance(expected_soil_seed, bool)
        or not isinstance(expected_soil_seed, (int, np.integer))
        or int(expected_soil_seed) < 0
    ):
        raise ValueError("expected soil seed must be a nonnegative integer")
    if (
        isinstance(expected_base_port, bool)
        or not isinstance(expected_base_port, (int, np.integer))
        or int(expected_base_port) < 1
        or int(expected_base_port) + 2 * expected_count - 1 > 65_535
    ):
        raise ValueError("expected base port cannot address the complete matrix")
    if (
        isinstance(expected_sim_seed, bool)
        or not isinstance(expected_sim_seed, (int, np.integer))
        or int(expected_sim_seed) < 0
    ):
        raise ValueError("expected simulator seed must be a nonnegative integer")
    if (
        not np.isfinite(min_observability_singular_value)
        or not math.isclose(
            float(min_observability_singular_value),
            FROZEN_MIN_OBSERVABILITY_SINGULAR_VALUE,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "minimum observability singular value differs from the frozen value"
        )
    if (
        not np.isfinite(max_final_update_age_s)
        or not math.isclose(
            float(max_final_update_age_s),
            FROZEN_MAX_FINAL_UPDATE_AGE_S,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("maximum final-update age differs from the frozen value")

    collection_config = _load_collection_config(
        collection_config_path,
        expected_count=expected_count,
        expected_soil_seed=expected_soil_seed,
        expected_base_port=expected_base_port,
        expected_sim_seed=expected_sim_seed,
    )
    source_traces = _load_source_trace_manifest(
        trace_manifest_path,
        expected_count=expected_count,
        collection_config=collection_config,
    )
    expected_ids = _expected_trace_ids(expected_count)
    _validate_replay_manifests(
        joint_path=joint_replay_manifest_path,
        parent_path=parent_replay_manifest_path,
        trace_manifest_path=trace_manifest_path,
        expected_ids=expected_ids,
        min_observability_singular_value=(
            min_observability_singular_value
        ),
        max_final_update_age_s=max_final_update_age_s,
    )
    joint, parent = validate_inference_matrix(
        joint_path,
        parent_path,
        source_traces=source_traces,
        expected_count=expected_count,
        min_observability_singular_value=(
            min_observability_singular_value
        ),
        max_final_update_age_s=max_final_update_age_s,
    )
    # Truth remains unopened until the complete inference and replay contracts
    # above have passed.
    truth, _soil_paths = _load_truth_bundle(
        truth_path,
        soil_dir=truth_soil_dir,
        expected_count=expected_count,
        expected_soil_seed=expected_soil_seed,
    )

    merged = joint.merge(
        parent[
            ["trace_id", "final_est_n", "trace_sha256"]
        ].rename(
            columns={
                "final_est_n": "parent_final_est_n",
                "trace_sha256": "parent_trace_sha256",
            }
        ),
        on="trace_id",
        validate="one_to_one",
    ).merge(
        truth[["trace_id", "n_true", "phi_true_deg"]],
        on="trace_id",
        validate="one_to_one",
    )
    merged = merged.sort_values("trace_id").reset_index(drop=True)
    joint_n = merged["final_est_n"].to_numpy(float)
    joint_phi = merged["final_est_phi_deg"].to_numpy(float)
    parent_n = merged["parent_final_est_n"].to_numpy(float)
    parent_phi = _phi_from_n(parent_n)
    n_true = merged["n_true"].to_numpy(float)
    phi_true = merged["phi_true_deg"].to_numpy(float)
    uniform_n = np.full(len(merged), UNIFORM_N_MEAN, dtype=float)
    uniform_phi = np.full(len(merged), UNIFORM_PHI_MEAN_DEG, dtype=float)

    scored = pd.DataFrame(
        {
            "trace_id": merged["trace_id"].astype(str),
            "trace_sha256": merged["trace_sha256"].str.lower(),
            "n_true": n_true,
            "phi_true_deg": phi_true,
            "joint_final_n": joint_n,
            "joint_final_phi_deg": joint_phi,
            "parent_final_n": parent_n,
            "parent_manifold_phi_deg": parent_phi,
            "uniform_prior_n": uniform_n,
            "uniform_prior_phi_deg": uniform_phi,
            "joint_final_n_information_kl": merged[
                "final_n_information_kl"
            ].to_numpy(float),
            "joint_final_phi_information_kl": merged[
                "final_phi_information_kl"
            ].to_numpy(float),
            "joint_final_boundary_limited": _strict_bool(
                merged["final_boundary_limited"],
                name="final_boundary_limited",
            ),
            "joint_final_fresh": _strict_bool(
                merged["final_fresh"], name="final_fresh"
            ),
            "joint_final_confident": _strict_bool(
                merged["final_confident"], name="final_confident"
            ),
            "joint_final_control_envelope_valid": _strict_bool(
                merged["final_control_envelope_valid"],
                name="final_control_envelope_valid",
            ),
            "joint_final_publication_ready": _strict_bool(
                merged["final_publication_ready"],
                name="final_publication_ready",
            ),
            "joint_final_update_age_s": merged[
                "final_final_update_age_s"
            ].to_numpy(float),
            "joint_final_publication_confidence": merged[
                "final_publication_confidence"
            ].to_numpy(float),
        }
    )
    summary = pd.DataFrame(
        [
            _metric_row(
                "joint", joint_n, joint_phi, n_true, phi_true
            ),
            _metric_row(
                "scalar_parent",
                parent_n,
                parent_phi,
                n_true,
                phi_true,
            ),
            _metric_row(
                "uniform_prior",
                uniform_n,
                uniform_phi,
                n_true,
                phi_true,
            ),
        ]
    )
    bootstrap = pd.DataFrame(
        [
            _bootstrap_comparison(
                baseline="scalar_parent",
                baseline_n=parent_n,
                baseline_phi=parent_phi,
                joint_n=joint_n,
                joint_phi=joint_phi,
                n_true=n_true,
                phi_true=phi_true,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
            _bootstrap_comparison(
                baseline="uniform_prior",
                baseline_n=uniform_n,
                baseline_phi=uniform_phi,
                joint_n=joint_n,
                joint_phi=joint_phi,
                n_true=n_true,
                phi_true=phi_true,
                seed=bootstrap_seed,
                resamples=bootstrap_resamples,
            ),
        ]
    )
    joint_summary = summary.set_index("method").loc["joint"]
    contrasts = bootstrap.set_index("baseline")
    fresh_count = int(scored["joint_final_fresh"].sum())
    confident_count = int(scored["joint_final_confident"].sum())
    control_envelope_count = int(
        scored["joint_final_control_envelope_valid"].sum()
    )
    publication_ready_count = int(
        scored["joint_final_publication_ready"].sum()
    )
    boundary_count = int(scored["joint_final_boundary_limited"].sum())
    decision = {
        "n_units": int(len(scored)),
        "uses_final_estimates_not_tail_means": True,
        "uses_last_causal_accuracy_valid_snapshot": True,
        "inference_semantics_version": INFERENCE_SEMANTICS_VERSION,
        "accepted_snapshot_version": ACCEPTED_SNAPSHOT_VERSION,
        "publication_confidence_floor": FROZEN_PUBLICATION_CONFIDENCE,
        "control_min_phi_deg": FROZEN_CONTROL_MIN_PHI_DEG,
        "freshness_max_age_s": FROZEN_MAX_FINAL_UPDATE_AGE_S,
        "trace_matrix_complete_and_hash_identical": True,
        "joint_n_mae_improves_parent_with_positive_ci": bool(
            contrasts.loc[
                "scalar_parent", "n_mae_improvement_ci_low"
            ]
            > 0.0
        ),
        "joint_phi_mae_improves_parent_with_positive_ci": bool(
            contrasts.loc[
                "scalar_parent",
                "phi_mae_improvement_ci_low_deg",
            ]
            > 0.0
        ),
        "joint_n_mae_improves_uniform_prior_with_positive_ci": bool(
            contrasts.loc[
                "uniform_prior", "n_mae_improvement_ci_low"
            ]
            > 0.0
        ),
        "joint_n_spearman_at_least_0_6": bool(
            float(joint_summary["n_spearman"]) >= 0.6
        ),
        "joint_phi_spearman_at_least_0_9": bool(
            float(joint_summary["phi_spearman"]) >= 0.9
        ),
        "joint_n_within_20_at_least_80_pct": bool(
            float(joint_summary["n_pct_within_20"]) >= 80.0
        ),
        "joint_phi_within_5_at_least_90_pct": bool(
            float(joint_summary["phi_pct_within_5_deg"]) >= 90.0
        ),
        "fresh_estimate_count": fresh_count,
        "fresh_estimate_required_count": int(
            math.ceil(0.95 * len(scored) - 1.0e-12)
        ),
        "fresh_estimate_pct": float(100.0 * fresh_count / len(scored)),
        "fresh_estimates_at_least_95_pct": _count_at_least_fraction(
            fresh_count, len(scored), 0.95
        ),
        "confident_estimate_count": confident_count,
        "confident_estimate_pct": float(
            100.0 * confident_count / len(scored)
        ),
        "control_envelope_valid_count": control_envelope_count,
        "control_envelope_valid_pct": float(
            100.0 * control_envelope_count / len(scored)
        ),
        "publication_ready_count": publication_ready_count,
        "publication_ready_required_count": int(
            math.ceil(0.85 * len(scored) - 1.0e-12)
        ),
        "publication_ready_pct": float(
            100.0 * publication_ready_count / len(scored)
        ),
        "publication_ready_at_least_85_pct": _count_at_least_fraction(
            publication_ready_count, len(scored), 0.85
        ),
        "material_boundary_limited_count": boundary_count,
        "material_boundary_limited_allowed_count": int(
            math.floor(0.10 * len(scored) + 1.0e-12)
        ),
        "material_boundary_limited_pct": float(
            100.0 * scored["joint_final_boundary_limited"].mean()
        ),
        "material_boundary_limited_at_most_10_pct": _count_at_most_fraction(
            boundary_count, len(scored), 0.10
        ),
    }
    gates = [
        value
        for key, value in decision.items()
        if key.endswith(
            (
                "_with_positive_ci",
                "_at_least_0_6",
                "_at_least_0_9",
                "_at_least_80_pct",
                "_at_least_85_pct",
                "_at_least_90_pct",
                "_at_least_95_pct",
                "_at_most_10_pct",
            )
        )
    ]
    decision["promotion_criteria_pass"] = bool(gates and all(gates))
    return scored, summary, bootstrap, decision


def _git_state() -> dict[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot capture git state: {error}") from error
    serialized = json.dumps(
        {"head": head, "status_porcelain": status},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "git_head": head,
        "git_dirty": json.dumps(bool(status)),
        "git_status_porcelain": json.dumps(status),
        "git_state_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _write_provenance_manifest(
    path: Path,
    *,
    command: list[str],
    expected_count: int,
    expected_soil_seed: int,
    expected_base_port: int,
    expected_sim_seed: int,
    min_observability_singular_value: float,
    max_final_update_age_s: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    input_paths: dict[str, Path],
    soil_paths: list[Path],
    output_paths: dict[str, Path],
    expected_preregistration_sha256: str,
) -> None:
    rows: list[tuple[str, str]] = [
        ("created_at", datetime.now().astimezone().isoformat(timespec="seconds")),
        ("command", " ".join(command)),
        ("project_root", str(ROOT)),
        ("expected_count", str(int(expected_count))),
        ("expected_soil_seed", str(int(expected_soil_seed))),
        ("expected_base_port", str(int(expected_base_port))),
        ("expected_sim_seed", str(int(expected_sim_seed))),
        (
            "min_observability_singular_value",
            repr(float(min_observability_singular_value)),
        ),
        ("max_final_update_age_s", repr(float(max_final_update_age_s))),
        (
            "publication_confidence_floor",
            repr(float(FROZEN_PUBLICATION_CONFIDENCE)),
        ),
        (
            "control_min_phi_deg",
            repr(float(FROZEN_CONTROL_MIN_PHI_DEG)),
        ),
        ("inference_semantics_version", INFERENCE_SEMANTICS_VERSION),
        ("accepted_snapshot_version", ACCEPTED_SNAPSHOT_VERSION),
        ("bootstrap_seed", str(int(bootstrap_seed))),
        ("bootstrap_resamples", str(int(bootstrap_resamples))),
        ("python_version", sys.version.replace("\n", " ")),
        ("scorer_source_sha256", _sha256_file(Path(__file__).resolve())),
        (
            "expected_preregistration_sha256",
            str(expected_preregistration_sha256).lower(),
        ),
    ]
    rows.extend(_git_state().items())
    for label, source in sorted(input_paths.items()):
        if not source.is_file():
            raise ValueError(f"provenance input is missing: {source}")
        rows.extend(
            [
                (f"input.{label}.path", str(source.resolve())),
                (f"input.{label}.sha256", _sha256_file(source)),
            ]
        )

    bundle = hashlib.sha256()
    for soil_path in soil_paths:
        digest = _sha256_file(soil_path)
        bundle.update(soil_path.name.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(digest.encode("ascii"))
        bundle.update(b"\n")
        rows.extend(
            [
                (
                    f"input.truth_soil.{soil_path.name}.path",
                    str(soil_path.resolve()),
                ),
                (f"input.truth_soil.{soil_path.name}.sha256", digest),
            ]
        )
    rows.extend(
        [
            ("input.truth_soil.count", str(len(soil_paths))),
            ("input.truth_soil.bundle_sha256", bundle.hexdigest()),
        ]
    )
    for label, output in sorted(output_paths.items()):
        if not output.is_file():
            raise ValueError(f"provenance output is missing: {output}")
        rows.extend(
            [
                (f"output.{label}.path", str(output.resolve())),
                (f"output.{label}.sha256", _sha256_file(output)),
            ]
        )

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["key", "value"])
        writer.writerows(rows)
    temporary.replace(path)


def _listed_artifact_paths(
    csv_path: Path,
    *,
    path_column: str,
    label_prefix: str,
) -> dict[str, Path]:
    """Resolve already-validated per-trace artifacts for provenance hashing."""

    try:
        frame = pd.read_csv(
            csv_path,
            dtype={"trace_id": str, path_column: str},
            keep_default_na=False,
        )
    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise ValueError(
            f"cannot enumerate provenance artifacts from {csv_path}: {error}"
        ) from error
    missing = sorted({"trace_id", path_column} - set(frame.columns))
    if missing:
        raise ValueError(
            "cannot enumerate provenance artifacts; missing "
            + ", ".join(missing)
        )
    if frame["trace_id"].duplicated().any():
        raise ValueError("cannot enumerate duplicate provenance trace IDs")
    paths: dict[str, Path] = {}
    for row in frame.itertuples(index=False):
        trace_id = str(row.trace_id)
        raw_value = str(getattr(row, path_column)).strip()
        if not raw_value:
            raise ValueError(
                f"cannot enumerate empty provenance path for {trace_id}"
            )
        raw = Path(raw_value).expanduser()
        resolved = (
            raw.resolve()
            if raw.is_absolute()
            else (csv_path.parent / raw).resolve()
        )
        paths[f"{label_prefix}.{trace_id}"] = resolved
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--truth-soil-dir", type=Path, required=True)
    parser.add_argument("--collection-config", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--joint-estimates", type=Path, required=True)
    parser.add_argument("--joint-replay-manifest", type=Path, required=True)
    parser.add_argument("--parent-estimates", type=Path, required=True)
    parser.add_argument("--parent-replay-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--expected-preregistration-sha256", required=True
    )
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-soil-seed", type=int, required=True)
    parser.add_argument("--expected-base-port", type=int, required=True)
    parser.add_argument("--expected-sim-seed", type=int, required=True)
    parser.add_argument(
        "--min-observability-singular-value",
        type=float,
        default=FROZEN_MIN_OBSERVABILITY_SINGULAR_VALUE,
    )
    parser.add_argument(
        "--max-final-update-age-s",
        type=float,
        default=FROZEN_MAX_FINAL_UPDATE_AGE_S,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    truth = args.truth.expanduser().resolve()
    truth_soil_dir = args.truth_soil_dir.expanduser().resolve()
    collection_config = args.collection_config.expanduser().resolve()
    trace_manifest = args.trace_manifest.expanduser().resolve()
    joint_estimates = args.joint_estimates.expanduser().resolve()
    joint_replay_manifest = args.joint_replay_manifest.expanduser().resolve()
    parent_estimates = args.parent_estimates.expanduser().resolve()
    parent_replay_manifest = args.parent_replay_manifest.expanduser().resolve()
    preregistration = args.preregistration.expanduser().resolve()
    try:
        if (
            not preregistration.is_file()
            or not preregistration.read_text(encoding="utf-8").strip()
        ):
            raise ValueError("preregistration must be a nonempty file")
        expected_preregistration_sha256 = (
            str(args.expected_preregistration_sha256).strip().lower()
        )
        if (
            len(expected_preregistration_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_preregistration_sha256
            )
        ):
            raise ValueError(
                "expected preregistration SHA-256 must be 64 lowercase hex digits"
            )
        if _sha256_file(preregistration) != expected_preregistration_sha256:
            raise ValueError("preregistration SHA-256 does not match the expected hash")
        _expected_trace_ids(args.expected_count)
        scoring_git_state = _git_state()
        if json.loads(scoring_git_state["git_dirty"]):
            raise ValueError(
                "promotion scoring requires a clean current Git worktree"
            )
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"cannot prepare joint holdout scoring: {error}") from error
    try:
        scored, summary, bootstrap, decision = score(
            truth_path=truth,
            truth_soil_dir=truth_soil_dir,
            collection_config_path=collection_config,
            trace_manifest_path=trace_manifest,
            joint_path=joint_estimates,
            joint_replay_manifest_path=joint_replay_manifest,
            parent_path=parent_estimates,
            parent_replay_manifest_path=parent_replay_manifest,
            expected_count=args.expected_count,
            expected_soil_seed=args.expected_soil_seed,
            expected_base_port=args.expected_base_port,
            expected_sim_seed=args.expected_sim_seed,
            min_observability_singular_value=(
                args.min_observability_singular_value
            ),
            max_final_update_age_s=args.max_final_update_age_s,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        per_trace_inputs = _listed_artifact_paths(
            trace_manifest,
            path_column="trace_path",
            label_prefix="source_trace",
        )
        per_trace_inputs.update(
            _listed_artifact_paths(
                joint_estimates,
                path_column="timeseries_path",
                label_prefix="joint_timeseries",
            )
        )
        per_trace_inputs.update(
            _listed_artifact_paths(
                parent_estimates,
                path_column="timeseries_path",
                label_prefix="parent_timeseries",
            )
        )
    except ValueError as error:
        raise SystemExit(f"cannot score joint holdout: {error}") from error
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "scored_runs.csv", index=False, float_format="%.17g")
    summary.to_csv(output / "summary.csv", index=False, float_format="%.17g")
    bootstrap.to_csv(
        output / "paired_bootstrap.csv", index=False, float_format="%.17g"
    )
    decision_path = output / "decision.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_paths = {
        "scored_runs": output / "scored_runs.csv",
        "summary": output / "summary.csv",
        "paired_bootstrap": output / "paired_bootstrap.csv",
        "decision": decision_path,
    }
    input_paths = {
        "truth": truth,
        "collection_config": collection_config,
        "trace_manifest": trace_manifest,
        "joint_estimates": joint_estimates,
        "joint_replay_manifest": joint_replay_manifest,
        "parent_estimates": parent_estimates,
        "parent_replay_manifest": parent_replay_manifest,
        "preregistration": preregistration,
    }
    input_paths.update(per_trace_inputs)
    source_paths = {
        "joint_estimator_source": (
            ROOT
            / "simulation/estimators/grit_terrain_estimator.py"
        ),
        "joint_replay_source": (
            ROOT / "benchmarking/develop_joint_estimator.py"
        ),
        "parent_estimator_source": (
            ROOT
            / "simulation/estimators/scalar_parent_terrain_estimator.py"
        ),
        "parent_replay_source": (
            ROOT / "benchmarking/terrain_estimator_replay.py"
        ),
        "trace_collector_source": (
            ROOT / "benchmarking/collect_terrain_estimator_traces.py"
        ),
        "force_checkpoint": (
            ROOT / "nn_models/tire_force_rate/best_terrain_nn.pt"
        ),
        "force_scalers": ROOT / "nn_models/tire_force_rate/scalers.pkl",
    }
    input_paths.update(source_paths)
    soil_paths = [
        truth_soil_dir / f"case_{index:04d}.yaml"
        for index in range(args.expected_count)
    ]
    try:
        _write_provenance_manifest(
            output / "provenance_manifest.csv",
            command=[
                str(Path(__file__).resolve()),
                *(sys.argv[1:] if argv is None else argv),
            ],
            expected_count=args.expected_count,
            expected_soil_seed=args.expected_soil_seed,
            expected_base_port=args.expected_base_port,
            expected_sim_seed=args.expected_sim_seed,
            min_observability_singular_value=(
                args.min_observability_singular_value
            ),
            max_final_update_age_s=args.max_final_update_age_s,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
            input_paths=input_paths,
            soil_paths=soil_paths,
            output_paths=output_paths,
            expected_preregistration_sha256=(
                expected_preregistration_sha256
            ),
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot bind joint scoring provenance: {error}") from error
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
