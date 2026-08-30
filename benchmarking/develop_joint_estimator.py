#!/usr/bin/env python3
"""Truth-blind replay harness for candidate joint ``n``/``phi`` estimators.

This program explores estimator configurations.  It is deliberately separate
from the registered paper benchmark, so that configuration search cannot reach
a published number without first passing the frozen scorer in
``score_joint_estimator.py``.

It accepts SHA-256-bound, estimator-disabled runtime sensor traces and exposes
no truth or scoring argument, which makes it structurally incapable of tuning
against the answer.  Each trace receives a fresh estimator instance, so no
state carries between cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "simulation", ROOT / "benchmarking"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import simulation  # noqa: E402,F401
from simulation.shared.param_consistency import (  # noqa: E402
    HMMWV_VEHICLE_PARAMS,
    get_terrain_preset,
    terrain_preset_to_internal,
)
from terrain_estimator_replay import (  # noqa: E402
    ReplayConfig,
    replay_trace,
    resolve_manifest_trace_path,
)
from terrain_estimator_trace import (  # noqa: E402
    TRACE_SCHEMA_VERSION,
    reject_oracle_columns,
    sha256_file,
)


BACKEND = "grit"
BACKEND_LABEL = "Joint rig-dynamics profile"
ACCEPTED_SNAPSHOT_VERSION = "grit_accepted"
INFERENCE_SEMANTICS_VERSION = "joint_final_snapshot"
FROZEN_PUBLICATION_CONFIDENCE = 0.20
FROZEN_BOUNDARY_MASS_LIMIT = 0.25
FROZEN_CONTROL_MIN_PHI_DEG = 10.0
_SAFE_TRACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_TOKENS = (
    "truth",
    "oracle",
    "ground",
    "datum",
    "height",
    "sinkage",
    "tire_force",
    "tyre_force",
    "contact_force",
    "soil",
)


@dataclass(frozen=True)
class CandidateConfig:
    """Serializable constructor and audit contract for one candidate."""

    model_dir: str
    update_interval: int = 1
    grid_size: int = 41
    student_dof: float = 4.0
    initial_n_std: float = 0.12
    smoothing_alpha: float = 1.0
    block_dt: float = 0.5
    max_final_update_age_s: float = 3.5
    horizon: float = 8.0
    min_windows: int = 8
    min_window_samples: int = 4
    r_ax: float = 0.35
    r_ay: float = 0.30
    min_information: float = 0.20
    min_yaw_rate_rms: float = 0.015
    min_model_speed: float = 2.5
    max_abs_alpha: float = 0.35
    enforce_feature_envelope: bool = True
    slip_mode: str = "average"
    fixed_kappa: float = 0.05
    rate_mode: str = "zero"
    force_gain_std: float = 0.04
    ax_bias_std: float = 0.10
    ay_bias_std: float = 0.05
    force_gain_min: float = 0.70
    force_gain_max: float = 1.30
    acceleration_bias_bound: float = 0.30
    profile_iterations: int = 8
    phi_grid_size: int = 17
    phi_min_deg: float = 6.0
    phi_max_deg: float = 37.8
    cohesion_multiplier_min: float = 0.7
    cohesion_multiplier_max: float = 1.3
    cohesion_grid_size: int = 1
    cohesion_prior_std: float = 0.20
    load_transfer_mode: str = "static"
    min_joint_information: float = 0.20
    min_n_information: float = 0.0
    min_phi_information: float = 0.0
    min_observability_rank: int = 2
    min_observability_singular_value: float = 0.10
    boundary_warning_mass: float = 0.25
    posterior_summary: str = "mean"
    block_alpha_rate: bool = False
    extra_kwargs_json: str = "{}"

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.max_final_update_age_s)
            or self.max_final_update_age_s <= 0.0
        ):
            raise ValueError(
                "max_final_update_age_s must be finite and positive"
            )

    def constructor_kwargs(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "model_dir": self.model_dir,
            "initial_terrain": terrain_preset_to_internal(
                get_terrain_preset("dirt")
            ),
            "vehicle_params": dict(HMMWV_VEHICLE_PARAMS),
            "update_interval": self.update_interval,
            "grid_size": self.grid_size,
            "student_dof": self.student_dof,
            "initial_n_std": self.initial_n_std,
            "smoothing_alpha": self.smoothing_alpha,
            "block_dt": self.block_dt,
            "horizon": self.horizon,
            "min_windows": self.min_windows,
            "min_window_samples": self.min_window_samples,
            "r_ax": self.r_ax,
            "r_ay": self.r_ay,
            "min_information": self.min_information,
            "min_yaw_rate_rms": self.min_yaw_rate_rms,
            "min_model_speed": self.min_model_speed,
            "max_abs_alpha": self.max_abs_alpha,
            "enforce_feature_envelope": self.enforce_feature_envelope,
            "slip_mode": self.slip_mode,
            "fixed_kappa": self.fixed_kappa,
            "rate_mode": self.rate_mode,
            "force_gain_std": self.force_gain_std,
            "ax_bias_std": self.ax_bias_std,
            "ay_bias_std": self.ay_bias_std,
            "force_gain_bounds": (
                self.force_gain_min,
                self.force_gain_max,
            ),
            "acceleration_bias_bound": self.acceleration_bias_bound,
            "profile_iterations": self.profile_iterations,
            "phi_grid_size": self.phi_grid_size,
            "phi_bounds_deg": (self.phi_min_deg, self.phi_max_deg),
            "cohesion_multiplier_bounds": (
                self.cohesion_multiplier_min,
                self.cohesion_multiplier_max,
            ),
            "cohesion_grid_size": self.cohesion_grid_size,
            "cohesion_prior_std": self.cohesion_prior_std,
            "load_transfer_mode": self.load_transfer_mode,
            "min_joint_information": self.min_joint_information,
            "min_n_information": self.min_n_information,
            "min_phi_information": self.min_phi_information,
            "min_observability_rank": self.min_observability_rank,
            "min_observability_singular_value": (
                self.min_observability_singular_value
            ),
            "boundary_warning_mass": self.boundary_warning_mass,
            "posterior_summary": self.posterior_summary,
            "block_alpha_rate": self.block_alpha_rate,
            "verbose": False,
        }
        try:
            extra = json.loads(self.extra_kwargs_json)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid estimator kwargs JSON: {error}") from error
        if not isinstance(extra, dict):
            raise ValueError("estimator kwargs JSON must encode an object")
        reserved = {"model_dir", "initial_terrain", "vehicle_params"}
        forbidden = sorted(
            str(key)
            for key in extra
            if str(key) in reserved
            or any(token in str(key).lower() for token in _FORBIDDEN_TOKENS)
        )
        if forbidden:
            raise ValueError(
                "development estimator kwargs contain forbidden inputs: "
                + ", ".join(forbidden)
            )
        arguments.update(extra)
        return arguments


@dataclass(frozen=True)
class DevelopmentTask:
    trace_id: str
    trace_path: str
    trace_sha256: str
    trace_quality: str
    output_path: str
    replay_config: ReplayConfig
    candidate_config: CandidateConfig


def _validate_rig_model(model_dir: Path) -> None:
    """Fail closed unless the force model has controlled-rig provenance."""

    manifest_path = model_dir / "repack_manifest.json"
    if not manifest_path.is_file():
        # Checkpoints without a repack manifest carry their provenance fields
        # inline, and the runtime loader validates the same fields.  Expected
        # checkpoint format and input width follow the ``rate_augmented`` flag
        # exactly as that loader derives them
        # (simulation/tire_models/nn_tire_model.py), so the static and the
        # rate-augmented rig checkpoints both bind against one check.
        try:
            import torch

            checkpoint = torch.load(
                model_dir / "best_terrain_nn.pt",
                map_location="cpu",
                weights_only=True,
            )
        except (FileNotFoundError, OSError, RuntimeError, TypeError) as error:
            raise ValueError(
                f"cannot verify rig force-model provenance: {error}"
            ) from error
        rate_augmented = bool(
            isinstance(checkpoint, dict) and checkpoint.get("rate_augmented")
        )
        expected_format = "tire_force_rate_mlp" if rate_augmented else "tire_force_static_mlp"
        expected_inputs = 14 if rate_augmented else 11
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("training_source")
            != "chrono_scm_single_tire_rig"
            or checkpoint.get("checkpoint_format") != expected_format
            or int(checkpoint.get("input_size", -1)) != expected_inputs
            or not str(checkpoint.get("training_csv_sha256", ""))
        ):
            raise ValueError(
                "joint development requires a verified controlled single-tire "
                "rig force map"
            )
        if not (model_dir / "scalers.pkl").is_file():
            raise ValueError(
                "cannot verify rig force-model provenance: missing scalers"
            )
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        raise ValueError(f"cannot verify rig force-model provenance: {error}") from error
    metadata = manifest.get("metadata_injected", {})
    if (
        metadata.get("training_source") != "chrono_scm_single_tire_rig"
        or metadata.get("checkpoint_format") != "tire_force_static_mlp"
        or manifest.get("training_csv_verified") is not True
    ):
        raise ValueError(
            "joint development requires the verified controlled single-tire "
            "static-rig force map"
        )


def build_estimator(config: CandidateConfig):
    from simulation.estimators.grit_terrain_estimator import (
        GritTerrainEstimator,
    )

    return GritTerrainEstimator(
        **config.constructor_kwargs()
    )


def _accepted_snapshot(estimator: Any) -> tuple[dict[str, object], dict[str, float]]:
    """Copy and validate the estimator's immutable accepted-update record."""

    getter = getattr(estimator, "get_last_accepted_snapshot", None)
    if not callable(getter):
        raise ValueError(
            "joint estimator does not expose get_last_accepted_snapshot()"
        )
    raw = getter()
    if raw is None:
        raise ValueError("joint estimator has no accepted snapshot")
    if not isinstance(raw, Mapping):
        raise ValueError("accepted snapshot must be a mapping")
    required = {
        "snapshot_version",
        "update_seq",
        "evidence_time_s",
        "n",
        "phi_deg",
        "terrain_params",
        "confidence",
        "n_sigma",
        "phi_sigma_deg",
        "joint_information_kl",
        "n_information_kl",
        "phi_information_kl",
        "cohesion_information_kl",
        "observability_rank",
        "observability_min_singular_value",
        "n_boundary_mass",
        "phi_boundary_mass",
        "cohesion_boundary_mass",
        "max_boundary_mass",
        "boundary_limited",
        "joint_projection_failures",
        "duplicate_likelihood_block_count",
        "duplicate_likelihood_update_count",
        "likelihood_evaluations",
        "likelihood_block_count",
        "likelihood_residual_count",
        "load_transfer_mode",
        "effective_front_load",
        "effective_rear_load",
        "effective_load_ax",
        "effective_load_ay",
        "projection_wall_time_s",
        "profile_wall_time_s",
        "observability_wall_time_s",
        "posterior_wall_time_s",
        "publication_wall_time_s",
        "update_wall_time_s",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(
            "accepted snapshot is missing: " + ", ".join(missing)
        )
    if str(raw["snapshot_version"]) != ACCEPTED_SNAPSHOT_VERSION:
        raise ValueError("accepted snapshot has the wrong schema version")
    if not isinstance(raw["boundary_limited"], (bool, np.bool_)):
        raise ValueError(
            "accepted snapshot boundary_limited must be boolean"
        )

    integer_names = {
        "update_seq",
        "observability_rank",
        "joint_projection_failures",
        "duplicate_likelihood_block_count",
        "duplicate_likelihood_update_count",
        "likelihood_evaluations",
        "likelihood_block_count",
        "likelihood_residual_count",
    }
    numeric_names = required - {
        "snapshot_version",
        "terrain_params",
        "boundary_limited",
        "load_transfer_mode",
    }
    numeric: dict[str, float] = {}
    for name in numeric_names:
        try:
            numeric[name] = float(raw[name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"accepted snapshot has nonnumeric {name}"
            ) from error
    if not np.isfinite(list(numeric.values())).all():
        raise ValueError("accepted snapshot contains non-finite values")
    timing_names = {
        "projection_wall_time_s",
        "profile_wall_time_s",
        "observability_wall_time_s",
        "posterior_wall_time_s",
        "publication_wall_time_s",
        "update_wall_time_s",
    }
    if any(numeric[name] < 0.0 for name in timing_names):
        raise ValueError("accepted snapshot contains a negative timing")
    if not np.isclose(
        numeric["update_wall_time_s"],
        numeric["projection_wall_time_s"]
        + numeric["posterior_wall_time_s"]
        + numeric["publication_wall_time_s"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("accepted snapshot has inconsistent total timing")
    for name in integer_names:
        if numeric[name] != np.floor(numeric[name]):
            raise ValueError(f"accepted snapshot has non-integer {name}")
    if numeric["update_seq"] < 1:
        raise ValueError("accepted snapshot has no accepted update")
    count_names = {
        "update_seq",
        "observability_rank",
        "joint_projection_failures",
        "duplicate_likelihood_block_count",
        "duplicate_likelihood_update_count",
        "likelihood_evaluations",
        "likelihood_block_count",
        "likelihood_residual_count",
    }
    if any(numeric[name] < 0.0 for name in count_names):
        raise ValueError("accepted snapshot contains a negative count")
    if not 0.0 <= numeric["confidence"] <= 1.0:
        raise ValueError("accepted snapshot confidence lies outside [0,1]")
    boundary_masses = [
        numeric["n_boundary_mass"],
        numeric["phi_boundary_mass"],
        numeric["cohesion_boundary_mass"],
    ]
    if (
        any(value < 0.0 or value > 1.0 for value in boundary_masses)
        or not np.isclose(
            numeric["max_boundary_mass"],
            max(boundary_masses),
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("accepted snapshot has inconsistent boundary mass")
    if bool(raw["boundary_limited"]) != (
        numeric["max_boundary_mass"] >= FROZEN_BOUNDARY_MASS_LIMIT
    ):
        raise ValueError(
            "accepted snapshot has an inconsistent boundary-limited flag"
        )
    if str(raw["load_transfer_mode"]) not in {
        "measured",
        "static",
        "lagged",
    }:
        raise ValueError("accepted snapshot has an invalid load-transfer mode")
    if numeric["n_sigma"] < 0.0 or numeric["phi_sigma_deg"] < 0.0:
        raise ValueError("accepted snapshot contains a negative uncertainty")
    timing_names = {
        "projection_wall_time_s",
        "profile_wall_time_s",
        "observability_wall_time_s",
        "posterior_wall_time_s",
        "publication_wall_time_s",
    }
    if any(numeric[name] < 0.0 for name in timing_names):
        raise ValueError("accepted snapshot contains a negative timing")

    terrain_raw = raw["terrain_params"]
    if not isinstance(terrain_raw, Mapping):
        raise ValueError("accepted snapshot terrain_params must be a mapping")
    terrain_names = {"Kphi", "Kc", "n", "c", "phi", "k"}
    missing_terrain = sorted(terrain_names - set(terrain_raw))
    if missing_terrain:
        raise ValueError(
            "accepted snapshot terrain_params is missing: "
            + ", ".join(missing_terrain)
        )
    try:
        terrain = {
            name: float(terrain_raw[name]) for name in sorted(terrain_names)
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            "accepted snapshot terrain_params contains a nonnumeric value"
        ) from error
    if not np.isfinite(list(terrain.values())).all():
        raise ValueError(
            "accepted snapshot terrain_params contains a non-finite value"
        )
    if not np.isclose(
        terrain["n"], numeric["n"], rtol=0.0, atol=1.0e-12
    ) or not np.isclose(
        terrain["phi"], numeric["phi_deg"], rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "accepted snapshot terrain parameters disagree with n/phi"
        )

    diagnostics: dict[str, object] = {
        "accepted_snapshot_version": ACCEPTED_SNAPSHOT_VERSION,
        "joint_active": 1,
        "joint_has_estimate": 1,
        "joint_updates": int(numeric["update_seq"]),
        "snapshot_n": numeric["n"],
        "snapshot_phi_deg": numeric["phi_deg"],
        "joint_projection_failures": int(
            numeric["joint_projection_failures"]
        ),
        "joint_information_kl": numeric["joint_information_kl"],
        "n_information_kl": numeric["n_information_kl"],
        "phi_information_kl": numeric["phi_information_kl"],
        "cohesion_information_kl": numeric[
            "cohesion_information_kl"
        ],
        "n_boundary_mass": numeric["n_boundary_mass"],
        "phi_boundary_mass": numeric["phi_boundary_mass"],
        "cohesion_boundary_mass": numeric["cohesion_boundary_mass"],
        "max_boundary_mass": numeric["max_boundary_mass"],
        "boundary_limited": int(bool(raw["boundary_limited"])),
        "observability_rank": int(numeric["observability_rank"]),
        "observability_min_singular_value": numeric[
            "observability_min_singular_value"
        ],
        "last_likelihood_block_count": int(
            numeric["likelihood_block_count"]
        ),
        "last_likelihood_residual_count": int(
            numeric["likelihood_residual_count"]
        ),
        # Residual diagnostics are optional.  Estimators that export them
        # populate these fields; their absence is a property of the estimator
        # under replay and must not fail the trace.
        "residual_ax_mad": float(raw.get("residual_ax_mad", float("nan"))),
        "residual_ay_mad": float(raw.get("residual_ay_mad", float("nan"))),
        "residual_sample_count": int(raw.get("residual_sample_count", 0) or 0),
        "duplicate_likelihood_block_count": int(
            numeric["duplicate_likelihood_block_count"]
        ),
        "duplicate_likelihood_update_count": int(
            numeric["duplicate_likelihood_update_count"]
        ),
        "likelihood_evaluations": int(numeric["likelihood_evaluations"]),
        "last_joint_update_time": numeric["evidence_time_s"],
        "publication_confidence": numeric["confidence"],
        "n_sigma": numeric["n_sigma"],
        "phi_sigma_deg": numeric["phi_sigma_deg"],
        "load_transfer_mode": str(raw["load_transfer_mode"]),
        "last_effective_front_load": numeric["effective_front_load"],
        "last_effective_rear_load": numeric["effective_rear_load"],
        "last_effective_load_ax": numeric["effective_load_ax"],
        "last_effective_load_ay": numeric["effective_load_ay"],
        "projection_wall_time_s": numeric["projection_wall_time_s"],
        "profile_wall_time_s": numeric["profile_wall_time_s"],
        "observability_wall_time_s": numeric[
            "observability_wall_time_s"
        ],
        "posterior_wall_time_s": numeric["posterior_wall_time_s"],
        "publication_wall_time_s": numeric["publication_wall_time_s"],
        "update_wall_time_s": numeric["update_wall_time_s"],
    }
    return diagnostics, terrain


def _run_one(task: DevelopmentTask) -> dict[str, object]:
    """Replay one trace, retaining any failure as an explicit matrix cell."""

    try:
        estimator = build_estimator(task.candidate_config)
        summary, time_series = replay_trace(
            task.trace_path,
            BACKEND,
            task.replay_config,
            trace_id=task.trace_id,
            expected_sha256=task.trace_sha256,
            estimator=estimator,
        )
        diagnostics, accepted_terrain = _accepted_snapshot(estimator)
        trace_times = pd.to_numeric(
            time_series.get("sim_time", pd.Series(dtype=float)),
            errors="coerce",
        ).to_numpy(dtype=float)
        final_trace_time = (
            float(np.max(trace_times))
            if trace_times.size and np.isfinite(trace_times).all()
            else float("nan")
        )
        last_update = float(diagnostics["last_joint_update_time"])
        final_update_age = final_trace_time - last_update
        final_update_max_age = float(
            task.candidate_config.max_final_update_age_s
        )
        causal_tail_snapshot = bool(
            np.isfinite([final_trace_time, last_update]).all()
            and last_update + 1.0e-9 >= float(task.replay_config.tail_start)
            and last_update <= final_trace_time + 1.0e-9
        )
        final_fresh = bool(
            causal_tail_snapshot
            and final_update_age >= -1.0e-9
            and final_update_age <= final_update_max_age + 1.0e-9
        )

        required_timeline = {
            "sim_time",
            "n_published",
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
        missing_timeline = sorted(required_timeline - set(time_series.columns))
        if missing_timeline:
            raise ValueError(
                "joint replay time series is missing: "
                + ", ".join(missing_timeline)
            )
        snapshot_flags = pd.to_numeric(
            time_series["joint_snapshot_advanced"], errors="coerce"
        ).to_numpy(float)
        if (
            not np.isfinite(snapshot_flags).all()
            or not np.isin(snapshot_flags, [0.0, 1.0]).all()
        ):
            raise ValueError(
                "joint replay has invalid accepted-snapshot flags"
            )
        snapshot_rows = time_series.loc[snapshot_flags == 1.0]
        if snapshot_rows.empty:
            raise ValueError(
                "joint replay has no accepted-snapshot timeline row"
            )
        last_snapshot = snapshot_rows.iloc[-1]
        snapshot_comparisons = {
            "joint_snapshot_seq": diagnostics["joint_updates"],
            "joint_snapshot_evidence_time_s": diagnostics[
                "last_joint_update_time"
            ],
            "joint_snapshot_n": diagnostics["snapshot_n"],
            "joint_snapshot_phi_deg": diagnostics["snapshot_phi_deg"],
            "joint_snapshot_n_sigma": diagnostics["n_sigma"],
            "joint_snapshot_phi_sigma_deg": diagnostics["phi_sigma_deg"],
            "joint_snapshot_confidence": diagnostics[
                "publication_confidence"
            ],
            "joint_snapshot_information_kl": diagnostics[
                "joint_information_kl"
            ],
            "joint_snapshot_observability_rank": diagnostics[
                "observability_rank"
            ],
            "joint_snapshot_observability_min_singular_value": diagnostics[
                "observability_min_singular_value"
            ],
            "joint_snapshot_n_boundary_mass": diagnostics[
                "n_boundary_mass"
            ],
            "joint_snapshot_phi_boundary_mass": diagnostics[
                "phi_boundary_mass"
            ],
            "joint_snapshot_max_boundary_mass": diagnostics[
                "max_boundary_mass"
            ],
            "joint_snapshot_projection_wall_time_s": diagnostics[
                "projection_wall_time_s"
            ],
            "joint_snapshot_profile_wall_time_s": diagnostics[
                "profile_wall_time_s"
            ],
            "joint_snapshot_observability_wall_time_s": diagnostics[
                "observability_wall_time_s"
            ],
            "joint_snapshot_posterior_wall_time_s": diagnostics[
                "posterior_wall_time_s"
            ],
            "joint_snapshot_publication_wall_time_s": diagnostics[
                "publication_wall_time_s"
            ],
            "joint_snapshot_update_wall_time_s": diagnostics[
                "update_wall_time_s"
            ],
        }
        snapshot_matches_timeline = all(
            np.isclose(
                float(last_snapshot[column]),
                float(value),
                rtol=0.0,
                atol=1.0e-12,
            )
            for column, value in snapshot_comparisons.items()
        ) and bool(last_snapshot["joint_snapshot_boundary_limited"]) == bool(
            diagnostics["boundary_limited"]
        )
        snapshot_integrity = bool(
            diagnostics["joint_projection_failures"] == 0
            and diagnostics["duplicate_likelihood_block_count"] == 0
            and diagnostics["duplicate_likelihood_update_count"] == 0
            and diagnostics["last_likelihood_block_count"] >= 1
            and diagnostics["last_likelihood_residual_count"]
            == 2 * diagnostics["last_likelihood_block_count"]
        )
        accepted_contract = bool(
            diagnostics["joint_has_estimate"]
            and diagnostics["joint_updates"] >= 1
            and diagnostics["observability_rank"]
            >= task.candidate_config.min_observability_rank
            and float(diagnostics["observability_min_singular_value"])
            + 1.0e-12
            >= task.candidate_config.min_observability_singular_value
            and float(diagnostics["joint_information_kl"]) + 1.0e-12
            >= task.candidate_config.min_joint_information
            and diagnostics["load_transfer_mode"]
            == task.candidate_config.load_transfer_mode
        )
        final_confident = bool(
            float(diagnostics["publication_confidence"])
            >= FROZEN_PUBLICATION_CONFIDENCE
        )
        final_control_envelope_valid = bool(
            float(accepted_terrain["phi"]) >= FROZEN_CONTROL_MIN_PHI_DEG
        )
        final_accuracy_valid = bool(
            causal_tail_snapshot
            and snapshot_integrity
            and snapshot_matches_timeline
        )
        publication_flags = pd.to_numeric(
            time_series["published_update"], errors="coerce"
        ).to_numpy(float)
        if (
            not np.isfinite(publication_flags).all()
            or not np.isin(publication_flags, [0.0, 1.0]).all()
        ):
            raise ValueError(
                "joint replay has invalid publication-update flags"
            )
        publication_rows = time_series.loc[publication_flags == 1.0]
        snapshot_was_published = False
        if not publication_rows.empty:
            last_publication = publication_rows.iloc[-1]
            snapshot_was_published = bool(
                float(last_publication["sim_time"]) + 1.0e-9 >= last_update
                and np.isclose(
                    float(last_publication["n_published"]),
                    accepted_terrain["n"],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.isclose(
                    float(last_publication["phi_published_deg"]),
                    accepted_terrain["phi"],
                    rtol=0.0,
                    atol=1.0e-12,
                )
                and np.isclose(
                    float(last_publication["confidence"]),
                    float(diagnostics["publication_confidence"]),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
        publication_policy_valid = bool(
            snapshot_was_published == (
                float(diagnostics["publication_confidence"])
                >= float(task.replay_config.min_confidence)
            )
        )
        final_publication_ready = bool(
            final_accuracy_valid
            and accepted_contract
            and final_fresh
            and final_confident
            and not bool(diagnostics["boundary_limited"])
            and final_control_envelope_valid
        )
        diagnostics.update(
            {
                "final_trace_time": final_trace_time,
                "final_update_age_s": final_update_age,
                "final_update_max_age_s": final_update_max_age,
                "accuracy_valid": int(final_accuracy_valid),
                "fresh": int(final_fresh),
                "confident": int(final_confident),
                "control_envelope_valid": int(
                    final_control_envelope_valid
                ),
                "publication_ready": int(final_publication_ready),
                "snapshot_was_published": int(snapshot_was_published),
            }
        )
        output = Path(task.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        time_series["backend_label"] = BACKEND_LABEL
        for key, value in diagnostics.items():
            time_series[f"replay_final_{key}"] = value
        time_series.to_csv(output, index=False, float_format="%.17g")
        summary["backend_label"] = BACKEND_LABEL
        summary["final_est_n"] = float(accepted_terrain["n"])
        summary["final_est_phi_deg"] = float(accepted_terrain["phi"])
        summary.update({f"final_{key}": value for key, value in diagnostics.items()})
        summary["timeseries_path"] = f"timeseries/{output.name}"
        summary["trace_quality"] = task.trace_quality
        invalid = []
        if not final_accuracy_valid:
            invalid.append(
                "last accepted scoring-tail snapshot is not accuracy-valid"
            )
        if not accepted_contract:
            invalid.append("accepted snapshot violates the frozen estimator contract")
        if not publication_policy_valid:
            invalid.append("accepted snapshot violates the publication policy")
        if invalid:
            summary["status"] = "fail"
            summary["failure"] = "; ".join(invalid)
        return summary
    except Exception as error:
        return {
            "trace_id": task.trace_id,
            "backend": BACKEND,
            "backend_label": BACKEND_LABEL,
            "status": "fail",
            "trace_path": task.trace_path,
            "trace_sha256": task.trace_sha256,
            "trace_quality": task.trace_quality,
            "failure": repr(error),
            "timeseries_path": "",
        }


def load_trace_manifest(
    path: Path, *, allowed_qualities: frozenset[str] = frozenset(
        {"exact_runtime_observations"}
    ),
) -> pd.DataFrame:
    """Validate a complete exact-runtime, oracle-free trace manifest."""

    try:
        traces = pd.read_csv(path, float_precision="round_trip")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(f"cannot read trace manifest: {error}") from error
    reject_oracle_columns(traces.columns, context="joint development manifest")
    extra_oracles = []
    for column in traces.columns:
        normalized = str(column).strip().lower().replace("-", "_")
        tokens = set(normalized.split("_"))
        if tokens.intersection({"true", "truth", "oracle", "plant", "soil"}):
            extra_oracles.append(str(column))
    if extra_oracles:
        raise ValueError(
            "joint development manifest contains forbidden oracle columns: "
            + ", ".join(extra_oracles)
        )
    required = {
        "trace_id", "status", "trace_path", "trace_sha256",
        "trace_quality", "trace_schema_version",
    }
    missing = sorted(required - set(traces.columns))
    if missing:
        raise ValueError("trace manifest is missing: " + ", ".join(missing))
    if traces.empty:
        raise ValueError("trace manifest contains no traces")
    trace_ids = traces["trace_id"].astype(str)
    if trace_ids.duplicated().any():
        raise ValueError("trace manifest contains duplicate trace IDs")
    unsafe = sorted(value for value in trace_ids if not _SAFE_TRACE_ID.fullmatch(value))
    if unsafe:
        raise ValueError("trace manifest contains unsafe trace IDs: " + ", ".join(unsafe))
    if set(traces["status"].astype(str)) != {"ok"}:
        raise ValueError("development replay requires a complete successful matrix")
    qualities = set(traces["trace_quality"].astype(str))
    if not qualities <= set(allowed_qualities):
        raise ValueError(
            "development replay requires exact runtime observations "
            f"(or an explicitly allowed quality); found {sorted(qualities)}"
        )
    versions = pd.to_numeric(
        traces["trace_schema_version"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(versions).all() or not (
        versions == TRACE_SCHEMA_VERSION
    ).all():
        raise ValueError(f"trace manifest requires schema {TRACE_SCHEMA_VERSION}")
    hashes = traces["trace_sha256"].astype(str).str.lower()
    if not hashes.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("trace manifest contains an invalid SHA-256")
    if hashes.duplicated().any():
        raise ValueError("trace manifest contains duplicate trace SHA-256 hashes")

    if "terrain_estimator_enabled" in traces.columns:
        enabled = traces["terrain_estimator_enabled"].map(
            lambda value: str(value).strip().lower()
            in {"1", "1.0", "true", "yes"}
        )
        recognized = traces["terrain_estimator_enabled"].map(
            lambda value: str(value).strip().lower()
            in {"0", "0.0", "1", "1.0", "false", "true", "no", "yes"}
        )
        if not recognized.all() or enabled.any():
            raise ValueError(
                "development replay requires estimator-disabled trace collection"
            )
    if "controller_prior" in traces.columns:
        priors = traces["controller_prior"].astype(str).str.strip().str.lower()
        if not (priors == "dirt").all():
            raise ValueError("development replay requires the fixed dirt prior")
    if "controller_prior_n" in traces.columns:
        prior_n = pd.to_numeric(
            traces["controller_prior_n"], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(prior_n).all() or not np.allclose(
            prior_n, 0.7, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("development replay requires controller prior n=0.7")

    resolved = []
    for row in traces.itertuples(index=False):
        trace_path = resolve_manifest_trace_path(row.trace_path, path)
        if not trace_path.is_file():
            raise ValueError(f"sensor trace is missing: {row.trace_id}")
        if sha256_file(trace_path) != str(row.trace_sha256).lower():
            raise ValueError(f"sensor trace hash mismatch: {row.trace_id}")
        resolved.append(str(trace_path))
    output = traces.copy()
    output["_resolved_trace_path"] = resolved
    return output


def _development_provenance(candidate: CandidateConfig) -> dict[str, str]:
    source_paths = {
        "driver_source_sha256": Path(__file__).resolve(),
        "candidate_source_sha256": ROOT / "simulation/estimators/grit_terrain_estimator.py",
        "parent_estimator_source_sha256": ROOT / "simulation/estimators/scalar_parent_terrain_estimator.py",
        "force_projector_source_sha256": ROOT / "simulation/tire_models/four_wheel_projection.py",
        "terrain_parameterization_source_sha256": ROOT / "simulation/estimators/terrain_parameterization.py",
        "force_model_loader_source_sha256": ROOT / "simulation/tire_models/nn_tire_model.py",
        "force_checkpoint_sha256": Path(candidate.model_dir) / "best_terrain_nn.pt",
        "force_scalers_sha256": Path(candidate.model_dir) / "scalers.pkl",
    }
    model_manifest = Path(candidate.model_dir) / "repack_manifest.json"
    if model_manifest.is_file():
        source_paths["force_repack_manifest_sha256"] = model_manifest
    else:
        source_paths["force_training_metadata_sha256"] = (
            Path(candidate.model_dir) / "TRAINING_METADATA.md"
        )
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise ValueError("cannot bind development provenance; missing " + ", ".join(missing))
    provenance = {key: sha256_file(path) for key, path in source_paths.items()}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot capture git provenance: {error}") from error
    provenance.update(
        {
            "git_head": head,
            "git_dirty": json.dumps(bool(status)),
            "git_status_porcelain": json.dumps(status),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        }
    )
    return provenance


def _write_estimates_atomic(
    path: Path, results: list[dict[str, object]]
) -> pd.DataFrame:
    estimates = pd.DataFrame(results).sort_values("trace_id").reset_index(drop=True)
    temporary = path.with_name(f".{path.name}.tmp")
    estimates.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)
    return estimates


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    candidate: CandidateConfig,
    traces: pd.DataFrame,
    provenance: dict[str, str],
) -> None:
    clean_full_matrix = (
        provenance.get("git_dirty") == "false"
        and args.max_traces is None
    )
    frozen_replay = bool(
        np.isclose(
            float(args.tail_start), 7.0, rtol=0.0, atol=1.0e-12
        )
        and np.isclose(
            float(args.te_min_confidence),
            FROZEN_PUBLICATION_CONFIDENCE,
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    promotion_eligible = bool(
        args.promotion_run and clean_full_matrix and frozen_replay
    )
    rows: list[tuple[str, str]] = [
        ("created_at", datetime.now().isoformat(timespec="seconds")),
        ("command", " ".join(sys.argv)),
        ("project_root", str(ROOT)),
        ("driver", str(Path(__file__).resolve())),
        ("candidate_confirmation", "True"),
        ("development_only", json.dumps(not args.promotion_run)),
        ("paper_evidence_eligible", json.dumps(promotion_eligible)),
        ("truth_inputs", "none"),
        ("scoring_performed", "False"),
        ("inference_semantics_version", INFERENCE_SEMANTICS_VERSION),
        ("accepted_snapshot_version", ACCEPTED_SNAPSHOT_VERSION),
        ("backend", BACKEND),
        ("trace_manifest", str(args.trace_manifest)),
        ("trace_manifest_sha256", sha256_file(args.trace_manifest)),
        ("selected_trace_count", str(len(traces))),
        ("selected_trace_ids", json.dumps(traces["trace_id"].astype(str).tolist())),
    ]
    rows.extend(
        (f"config.{key}", json.dumps(value, sort_keys=True))
        for key, value in asdict(candidate).items()
    )
    rows.extend((f"provenance.{key}", value) for key, value in provenance.items())
    rows.extend(
        [
            ("replay.tail_start_s", repr(float(args.tail_start))),
            ("replay.minimum_confidence", repr(float(args.te_min_confidence))),
            ("workers", str(int(args.workers))),
        ]
    )
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument(
        "--allow-trace-quality", action="append", default=[],
        help="Additionally accepted trace_quality value (development only; "
             "refused with --promotion-run). Used by the noise-robustness "
             "study, whose capsules are exact runtime observations with "
             "seeded noise injected at replay time.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument(
        "--promotion-run",
        action="store_true",
        help=(
            "Mark a complete replay from a clean worktree as eligible for "
            "promotion scoring. Partial or dirty runs remain ineligible."
        ),
    )
    parser.add_argument("--tail-start", type=float, default=7.0)
    parser.add_argument(
        "--te-min-confidence",
        type=float,
        default=FROZEN_PUBLICATION_CONFIDENCE,
    )
    parser.add_argument("--model-dir", default="nn_models/tire_force_rate")
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--student-dof", type=float, default=4.0)
    parser.add_argument(
        "--initial-n-std", type=float, default=0.12,
        help="Width of the initial n prior, centred on the controller's assumed "
             "soil. The 0.12 default is tight enough that the assumption persists "
             "in the estimate; widen it to let the likelihood dominate.")
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--dynamics-block-dt", type=float, default=0.5)
    parser.add_argument(
        "--max-final-update-age-s",
        type=float,
        default=3.5,
        help=(
            "Maximum age of the last accepted joint update at trace end; "
            "must be finite and positive."
        ),
    )
    parser.add_argument("--dynamics-horizon", type=float, default=8.0)
    parser.add_argument("--dynamics-min-windows", type=int, default=8)
    parser.add_argument("--dynamics-min-window-samples", type=int, default=4)
    parser.add_argument("--dynamics-r-ax", type=float, default=0.35)
    parser.add_argument("--dynamics-r-ay", type=float, default=0.30)
    parser.add_argument("--dynamics-min-information", type=float, default=0.20)
    parser.add_argument("--dynamics-min-yaw-rate-rms", type=float, default=0.015)
    parser.add_argument("--dynamics-min-speed", type=float, default=2.5)
    parser.add_argument("--dynamics-max-abs-alpha", type=float, default=0.35)
    parser.add_argument(
        "--dynamics-enforce-feature-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamics-slip-mode", choices=("wheel", "average", "fixed"),
        default="average",
    )
    parser.add_argument("--dynamics-fixed-kappa", type=float, default=0.05)
    parser.add_argument(
        "--dynamics-rate-mode", choices=("signed", "zero", "legacy"),
        default="zero",
    )
    parser.add_argument("--dynamics-force-gain-std", type=float, default=0.04)
    parser.add_argument("--dynamics-ax-bias-std", type=float, default=0.10)
    parser.add_argument("--dynamics-ay-bias-std", type=float, default=0.05)
    parser.add_argument("--dynamics-force-gain-min", type=float, default=0.70)
    parser.add_argument("--dynamics-force-gain-max", type=float, default=1.30)
    parser.add_argument("--dynamics-acceleration-bias-bound", type=float, default=0.30)
    parser.add_argument("--dynamics-profile-iterations", type=int, default=8)
    parser.add_argument("--dynamics-n-lo", type=float, default=None,
                        help="Edge-extension candidate: lower n grid bound "
                             "(requires --dynamics-manifold-floor when below 0.50).")
    parser.add_argument("--dynamics-n-hi", type=float, default=None)
    parser.add_argument("--dynamics-manifold-mode", default="hold",
                        choices=["hold", "linear"],
                        help="Sub-clay parameter completion: hold non-exponent "
                             "parameters at the clay anchor (default), or continue "
                             "them by linear extrapolation (ablation).")
    parser.add_argument("--dynamics-manifold-floor", type=float, default=None,
                        help="Extrapolate the soil manifold's clay--dirt segment "
                             "down to this exponent for sub-clay grid nodes.")
    parser.add_argument("--phi-grid-size", type=int, default=17)
    parser.add_argument("--phi-min-deg", type=float, default=6.0)
    parser.add_argument("--phi-max-deg", type=float, default=37.8)
    parser.add_argument("--cohesion-multiplier-min", type=float, default=0.7)
    parser.add_argument("--cohesion-multiplier-max", type=float, default=1.3)
    parser.add_argument("--cohesion-grid-size", type=int, default=1)
    parser.add_argument("--cohesion-prior-std", type=float, default=0.20)
    parser.add_argument(
        "--load-transfer-mode",
        choices=("measured", "static", "lagged"),
        default="static",
        help=(
            "Load input for the rig projection: current reconstruction, "
            "vehicle static loads, or the previous accepted block."
        ),
    )
    parser.add_argument("--min-joint-information", type=float, default=0.20)
    parser.add_argument("--min-n-information", type=float, default=0.0)
    parser.add_argument("--min-phi-information", type=float, default=0.0)
    parser.add_argument("--min-observability-rank", type=int, default=2)
    parser.add_argument(
        "--min-observability-singular-value", type=float, default=0.10
    )
    parser.add_argument("--boundary-warning-mass", type=float, default=0.25)
    parser.add_argument(
        "--posterior-summary", choices=("map", "mean"), default="mean"
    )
    parser.add_argument(
        "--block-alpha-rate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a causal within-block slip-angle slope for the rate-rig "
            "features instead of the serialized pointwise derivative."
        ),
    )
    parser.add_argument("--estimator-kwargs-json", default="{}")
    return parser.parse_args(argv)


def _candidate_config(args: argparse.Namespace) -> CandidateConfig:
    model = Path(args.model_dir).expanduser()
    if not model.is_absolute():
        model = ROOT / model
    return CandidateConfig(
        model_dir=str(model.resolve()),
        update_interval=args.update_interval,
        grid_size=args.grid_size,
        student_dof=args.student_dof,
        initial_n_std=args.initial_n_std,
        smoothing_alpha=args.smoothing_alpha,
        block_dt=args.dynamics_block_dt,
        max_final_update_age_s=args.max_final_update_age_s,
        horizon=args.dynamics_horizon,
        min_windows=args.dynamics_min_windows,
        min_window_samples=args.dynamics_min_window_samples,
        r_ax=args.dynamics_r_ax,
        r_ay=args.dynamics_r_ay,
        min_information=args.dynamics_min_information,
        min_yaw_rate_rms=args.dynamics_min_yaw_rate_rms,
        min_model_speed=args.dynamics_min_speed,
        max_abs_alpha=args.dynamics_max_abs_alpha,
        enforce_feature_envelope=args.dynamics_enforce_feature_envelope,
        slip_mode=args.dynamics_slip_mode,
        fixed_kappa=args.dynamics_fixed_kappa,
        rate_mode=args.dynamics_rate_mode,
        force_gain_std=args.dynamics_force_gain_std,
        ax_bias_std=args.dynamics_ax_bias_std,
        ay_bias_std=args.dynamics_ay_bias_std,
        force_gain_min=args.dynamics_force_gain_min,
        force_gain_max=args.dynamics_force_gain_max,
        acceleration_bias_bound=args.dynamics_acceleration_bias_bound,
        profile_iterations=args.dynamics_profile_iterations,
        phi_grid_size=args.phi_grid_size,
        phi_min_deg=args.phi_min_deg,
        phi_max_deg=args.phi_max_deg,
        cohesion_multiplier_min=args.cohesion_multiplier_min,
        cohesion_multiplier_max=args.cohesion_multiplier_max,
        cohesion_grid_size=args.cohesion_grid_size,
        cohesion_prior_std=args.cohesion_prior_std,
        load_transfer_mode=args.load_transfer_mode,
        min_joint_information=args.min_joint_information,
        min_n_information=args.min_n_information,
        min_phi_information=args.min_phi_information,
        min_observability_rank=args.min_observability_rank,
        min_observability_singular_value=args.min_observability_singular_value,
        boundary_warning_mass=args.boundary_warning_mass,
        posterior_summary=args.posterior_summary,
        block_alpha_rate=args.block_alpha_rate,
        extra_kwargs_json=_merged_estimator_kwargs(args),
    )


def _merged_estimator_kwargs(args) -> str:
    """Fold the grid-bound flags into the serialized estimator keyword arguments.

    Merging them into one JSON payload means the replay manifest records a
    single, complete statement of the estimator's construction, which the
    scorer compares verbatim against the configuration it expects.
    """

    extra = json.loads(args.estimator_kwargs_json or "{}")
    if args.dynamics_n_lo is not None or args.dynamics_n_hi is not None:
        from simulation.estimators.terrain_parameterization import N_BOUNDS
        extra["n_bounds"] = [
            float(args.dynamics_n_lo if args.dynamics_n_lo is not None else N_BOUNDS[0]),
            float(args.dynamics_n_hi if args.dynamics_n_hi is not None else N_BOUNDS[1]),
        ]
    if args.dynamics_manifold_floor is not None:
        extra["manifold_soft_floor"] = float(args.dynamics_manifold_floor)
        extra["manifold_soft_mode"] = str(args.dynamics_manifold_mode)
    return json.dumps(extra)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or args.update_interval < 1:
        raise SystemExit("worker and update counts must be positive")
    if args.max_traces is not None and args.max_traces < 1:
        raise SystemExit("--max-traces must be positive")
    if not np.isfinite(args.tail_start) or args.tail_start < 0.0:
        raise SystemExit("--tail-start must be finite and non-negative")
    if (
        not np.isfinite(args.max_final_update_age_s)
        or args.max_final_update_age_s <= 0.0
    ):
        raise SystemExit("--max-final-update-age-s must be finite and positive")
    if not 0.0 <= args.te_min_confidence <= 1.0:
        raise SystemExit("--te-min-confidence must lie in [0,1]")
    if not args.phi_min_deg < args.phi_max_deg:
        raise SystemExit("phi bounds must be strictly increasing")
    if args.cohesion_grid_size < 1:
        raise SystemExit("--cohesion-grid-size must be positive")

    args.trace_manifest = args.trace_manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    try:
        if args.allow_trace_quality and args.promotion_run:
            raise SystemExit(
                "--allow-trace-quality is refused on promotion runs"
            )
        traces = load_trace_manifest(
            args.trace_manifest,
            allowed_qualities=frozenset(
                {"exact_runtime_observations", *args.allow_trace_quality}
            ),
        )
        candidate = _candidate_config(args)
        _validate_rig_model(Path(candidate.model_dir))
        build_estimator(candidate)
        provenance = _development_provenance(candidate)
    except (ImportError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"cannot prepare joint development replay: {error}") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "timeseries").mkdir()
    traces = traces.sort_values("trace_id").reset_index(drop=True)
    if args.max_traces is not None:
        traces = traces.head(int(args.max_traces)).copy()
    _write_manifest(
        args.output_dir / "replay_manifest.csv",
        args,
        candidate,
        traces,
        provenance,
    )
    replay_config = ReplayConfig(
        min_confidence=float(args.te_min_confidence),
        tail_start=float(args.tail_start),
    )
    tasks = [
        DevelopmentTask(
            trace_id=str(row["trace_id"]),
            trace_path=str(row["_resolved_trace_path"]),
            trace_sha256=str(row["trace_sha256"]).lower(),
            trace_quality=str(row["trace_quality"]),
            output_path=str(
                args.output_dir / "timeseries" / f"{row['trace_id']}_{BACKEND}.csv"
            ),
            replay_config=replay_config,
            candidate_config=candidate,
        )
        for _index, row in traces.iterrows()
    ]
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_one, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
            estimates = _write_estimates_atomic(
                args.output_dir / "estimates.csv", results
            )
    failures = estimates[estimates["status"].astype(str) != "ok"]
    print(f"wrote inference-only joint replay: {args.output_dir / 'estimates.csv'}")
    if not failures.empty:
        print(f"{len(failures)}/{len(estimates)} replay cells failed")
        for row in failures.itertuples(index=False):
            print(f"  {row.trace_id}: {getattr(row, 'failure', '')}")
        return 2
    print(f"replayed {len(estimates)} exact sensor traces; no truth was read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
