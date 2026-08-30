#!/usr/bin/env python3
"""Replay terrain estimators on shared, sensor-only traces.

Every registered estimator sees the identical recorded sensor stream, so any
difference between their outputs is attributable to the estimator rather than
to the trajectory it was driven along.  The learned force surrogate each
estimator evaluates is supervised by the controlled single-tire Chrono SCM
rig.

This program never accepts a truth manifest and never computes estimation
error.  It produces estimates keyed by opaque ``trace_id``; the separate
``score_joint_estimator.py`` process joins those estimates to plant truth
once all estimator computation is complete, which keeps truth strictly
downstream of inference.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "simulation", ROOT / "benchmarking"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import flatpath  # noqa: E402,F401
from common import (  # noqa: E402
    GRIT_ESTIMATOR_BACKEND,
    GRIT_ESTIMATOR_CONTRACT,
)
from param_consistency import (  # noqa: E402
    HMMWV_VEHICLE_PARAMS,
    get_terrain_preset,
    terrain_preset_to_internal,
)
from terrain_estimator_trace import (  # noqa: E402
    TRACE_SCHEMA_VERSION,
    TraceValidationError,
    load_sensor_trace,
    reject_oracle_columns,
    sha256_file,
)
from tire_input_features import level_specific_force_to_yaw_frame  # noqa: E402


BACKEND_LABELS = {
    "scalar_parent": "Profiled rig-dynamics estimator",
    GRIT_ESTIMATOR_BACKEND: (
        "Joint n/phi rig-dynamics estimator (selectable, not active)"
    ),
    "grit_ungated": "Profiled rig dynamics (no yaw gate)",
}

_SAFE_TRACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RIG_DYNAMICS_JOINT_OBSERVATION_KEYS = frozenset({
    "kappa",
    "alpha_f",
    "alpha_r",
    "u",
    "Fz_f",
    "Fz_r",
    "sr",
    "alpha_rate_r",
    "ay_imu",
    "omega_dot",
    "omega",
    "v_lateral",
    "sim_time",
    "steering_angle",
    "wheel_omegas",
    "ax_imu",
})


def _grit_constructor_kwargs(
    initial_terrain: dict[str, float],
    vehicle_params: dict[str, float],
) -> dict[str, Any]:
    """Translate the immutable candidate contract into estimator arguments."""

    contract = GRIT_ESTIMATOR_CONTRACT
    return {
        "model_dir": str(
            (ROOT / str(contract["force_model_dir"])).resolve()
        ),
        "initial_terrain": dict(initial_terrain),
        "vehicle_params": dict(vehicle_params),
        "update_interval": int(contract["update_interval"]),
        "verbose": False,
        "grid_size": int(contract["n_grid_size"]),
        "student_dof": float(contract["student_dof"]),
        "smoothing_alpha": float(contract["smoothing_alpha"]),
        "block_dt": float(contract["block_dt_s"]),
        "horizon": float(contract["history_horizon_s"]),
        "min_windows": int(contract["min_concurrent_windows"]),
        "min_window_samples": int(contract["min_window_samples"]),
        "r_ax": float(contract["r_ax_mps2"]),
        "r_ay": float(contract["r_ay_mps2"]),
        "min_information": float(contract["min_information"]),
        "min_yaw_rate_rms": float(contract["min_yaw_rate_rms_radps"]),
        "min_model_speed": float(contract["min_speed_mps"]),
        "max_abs_alpha": float(contract["max_abs_slip_angle_rad"]),
        "enforce_feature_envelope": bool(
            contract["enforce_rig_feature_envelope"]
        ),
        "slip_mode": str(contract["slip_mode"]),
        "fixed_kappa": float(contract["fixed_kappa"]),
        "rate_mode": str(contract["rate_mode"]),
        "force_gain_std": float(contract["force_gain_prior_std"]),
        "ax_bias_std": float(contract["ax_bias_prior_std_mps2"]),
        "ay_bias_std": float(contract["ay_bias_prior_std_mps2"]),
        "force_gain_bounds": tuple(contract["force_gain_bounds"]),
        "acceleration_bias_bound": float(
            contract["acceleration_bias_bound_mps2"]
        ),
        "profile_iterations": int(contract["profile_iterations"]),
        "phi_grid_size": int(contract["phi_grid_size"]),
        "phi_bounds_deg": tuple(contract["phi_bounds_deg"]),
        "cohesion_multiplier_bounds": tuple(
            contract["cohesion_multiplier_bounds"]
        ),
        "cohesion_grid_size": int(contract["cohesion_grid_size"]),
        "cohesion_prior_std": float(contract["cohesion_prior_std"]),
        "load_transfer_mode": str(contract["load_transfer_mode"]),
        "min_joint_information": float(contract["min_joint_information"]),
        "min_n_information": float(contract["min_n_information"]),
        "min_phi_information": float(contract["min_phi_information"]),
        "min_observability_rank": int(contract["min_observability_rank"]),
        "min_observability_singular_value": float(
            contract["min_observability_singular_value"]
        ),
        "boundary_warning_mass": float(contract["boundary_warning_mass"]),
        "posterior_summary": str(contract["posterior_summary"]),
        "block_alpha_rate": bool(contract["block_alpha_rate"]),
        "n_bounds": tuple(
            float(value) for value in contract["n_bounds"]
        ),
        "manifold_soft_floor": float(contract["manifold_soft_floor"]),
        "manifold_soft_mode": str(contract["manifold_soft_mode"]),
    }


def _terrain_estimator_observation_for_backend(
    backend: str,
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Restrict the joint estimator's observation to its declared channels.

    The joint estimator's contract admits inertial, wheel-speed, slip, and
    load channels alone.  Filtering the observation here means a trace that
    also carries wheel-centre elevation, which would amount to a ground datum,
    cannot leak that channel into the estimator through a shared dictionary.
    """

    values = dict(observation)
    if str(backend) != GRIT_ESTIMATOR_BACKEND:
        return values
    return {
        key: values[key]
        for key in _RIG_DYNAMICS_JOINT_OBSERVATION_KEYS
        if key in values
    }


def resolve_manifest_trace_path(
    trace_path: str | Path, trace_manifest_path: str | Path
) -> Path:
    """Resolve a trace path using the manifest directory as its portable base.

    Absolute paths are returned unchanged.  Relative paths resolve against the
    directory holding ``trace_manifest.csv``, so a complete collection can be
    moved or restored from a data snapshot without rewriting its provenance
    table and invalidating the hashes recorded there.
    """

    raw = Path(str(trace_path)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    manifest = Path(trace_manifest_path).expanduser().resolve()
    return (manifest.parent / raw).resolve()


@dataclass(frozen=True)
class ReplayConfig:
    update_interval: int = 8
    filter_dt: float = 0.1
    min_confidence: float = 0.0
    tail_start: float = 11.0
    model_dir: str | None = None
    grid_size: int = 41
    student_dof: float = 4.0
    # Block-dynamics estimator settings.  This estimator reads no ground
    # datum: it fits soil parameters to inertial and wheel channels over
    # fixed-length blocks of the trace.  A replay that overrides any of these
    # is marked as such in its manifest, and the scorer refuses to publish it.
    dynamics_update_interval: int = 1
    dynamics_block_dt: float = 0.5
    dynamics_horizon: float = 8.0
    dynamics_min_windows: int = 12
    dynamics_min_window_samples: int = 4
    dynamics_r_ax: float = 0.35
    dynamics_r_ay: float = 0.30
    dynamics_min_information: float = 0.20
    dynamics_min_yaw_rate_rms: float = 0.0
    dynamics_min_speed: float = 2.5
    dynamics_max_abs_alpha: float = 0.35
    dynamics_enforce_feature_envelope: bool = True
    dynamics_slip_mode: str = "average"
    dynamics_fixed_kappa: float = 0.05
    dynamics_rate_mode: str = "zero"
    dynamics_force_gain_std: float = 0.04
    dynamics_ax_bias_std: float = 0.10
    dynamics_ay_bias_std: float = 0.05
    dynamics_force_gain_min: float = 0.70
    dynamics_force_gain_max: float = 1.30
    dynamics_acceleration_bias_bound: float = 0.30
    dynamics_profile_iterations: int = 8


def make_estimator(backend: str, config: ReplayConfig):
    """Instantiate the same estimator classes used by the live controller."""

    if backend not in BACKEND_LABELS:
        raise ValueError(f"unsupported estimator backend: {backend}")
    initial = terrain_preset_to_internal(get_terrain_preset("dirt"))
    vehicle = dict(HMMWV_VEHICLE_PARAMS)
    common: dict[str, Any] = {
        "model_dir": config.model_dir,
        "initial_terrain": initial,
        "vehicle_params": vehicle,
        "update_interval": config.update_interval,
        "filter_dt": config.filter_dt,
        "verbose": False,
    }
    if backend == GRIT_ESTIMATOR_BACKEND:
        from grit_terrain_estimator import (
            GritTerrainEstimator,
        )

        return GritTerrainEstimator(
            **_grit_constructor_kwargs(initial, vehicle)
        )
    if backend in {"scalar_parent", "grit_ungated"}:
        from scalar_parent_terrain_estimator import (
            ScalarParentTerrainEstimator,
        )

        dynamics_common = dict(common)
        dynamics_common["update_interval"] = config.dynamics_update_interval
        return ScalarParentTerrainEstimator(
            **dynamics_common,
            grid_size=config.grid_size,
            student_dof=config.student_dof,
            block_dt=config.dynamics_block_dt,
            horizon=config.dynamics_horizon,
            min_windows=config.dynamics_min_windows,
            min_window_samples=config.dynamics_min_window_samples,
            r_ax=config.dynamics_r_ax,
            r_ay=config.dynamics_r_ay,
            min_information=config.dynamics_min_information,
            min_yaw_rate_rms=(
                0.0 if backend == "grit_ungated"
                else config.dynamics_min_yaw_rate_rms
            ),
            min_model_speed=config.dynamics_min_speed,
            max_abs_alpha=config.dynamics_max_abs_alpha,
            enforce_feature_envelope=(
                config.dynamics_enforce_feature_envelope
            ),
            slip_mode=config.dynamics_slip_mode,
            fixed_kappa=config.dynamics_fixed_kappa,
            rate_mode=config.dynamics_rate_mode,
            force_gain_std=config.dynamics_force_gain_std,
            ax_bias_std=config.dynamics_ax_bias_std,
            ay_bias_std=config.dynamics_ay_bias_std,
            force_gain_bounds=(
                config.dynamics_force_gain_min,
                config.dynamics_force_gain_max,
            ),
            acceleration_bias_bound=(
                config.dynamics_acceleration_bias_bound
            ),
            profile_iterations=config.dynamics_profile_iterations,
        )
    raise ValueError(f"unsupported estimator backend: {backend}")


def replay_trace(
    trace_path: str | Path,
    backend: str,
    config: ReplayConfig,
    *,
    trace_id: str = "trace",
    expected_sha256: str | None = None,
    estimator=None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Replay one estimator without loading or accepting any truth labels."""

    path = Path(trace_path).expanduser().resolve()
    actual_hash = sha256_file(path)
    if expected_sha256 and actual_hash.lower() != str(expected_sha256).lower():
        raise TraceValidationError(
            f"trace hash mismatch for {trace_id}: expected {expected_sha256}, got {actual_hash}"
        )
    trace = load_sensor_trace(path)
    active = estimator if estimator is not None else make_estimator(backend, config)

    published_n = 0.7
    published_phi_deg = (
        float(active.get_friction_angle_deg())
        if hasattr(active, "get_friction_angle_deg")
        else 29.0
    )
    confidence = 0.0
    update_count = 0
    joint_snapshot: Mapping[str, Any] | None = None
    joint_snapshot_sequence = 0
    joint_snapshot_update_count = 0
    time_rows: list[dict[str, object]] = []
    for row in trace.itertuples(index=False):
        ax_horizontal, ay_horizontal = level_specific_force_to_yaw_frame(
            float(row.ax_imu), float(row.ay_imu), float(row.az_imu),
            float(row.quat_e0), float(row.quat_e1),
            float(row.quat_e2), float(row.quat_e3),
        )
        omega_dot = active.estimate_omega_dot(float(row.omega_raw), float(row.sim_time))
        if omega_dot is not None:
            observation = {
                "kappa": float(row.kappa),
                "alpha_f": float(row.alpha_f),
                "alpha_r": float(row.alpha_r),
                "u": float(row.u),
                "Fz_f": float(row.Fz_f),
                "Fz_r": float(row.Fz_r),
                "sr": float(row.alpha_rate_f),
                "alpha_rate_r": float(row.alpha_rate_r),
                "ay_imu": float(ay_horizontal),
                "omega_dot": float(omega_dot),
                "omega": float(row.omega),
                "v_lateral": float(row.v_lateral),
                "x_pos": float(row.x_cg),
                "y_pos": float(row.y_cg),
                "psi": float(row.psi),
                "ax_cmd": float(row.ax_filtered),
                "sim_time": float(row.sim_time),
                "ax_imu": float(ax_horizontal),
                "steering_angle": float(row.steering_angle),
                "wheel_omegas": (
                    float(row.wheel_omega_fl),
                    float(row.wheel_omega_fr),
                    float(row.wheel_omega_rl),
                    float(row.wheel_omega_rr),
                ),
                "wheel_center_heights": (
                    float(row.wheel_center_z_fl),
                    float(row.wheel_center_z_fr),
                    float(row.wheel_center_z_rl),
                    float(row.wheel_center_z_rr),
                ),
                "drive_torques": (
                    float(row.drive_torque_fl),
                    float(row.drive_torque_fr),
                    float(row.drive_torque_rl),
                    float(row.drive_torque_rr),
                ),
                "brake_torques": (
                    float(row.brake_torque_fl),
                    float(row.brake_torque_fr),
                    float(row.brake_torque_rl),
                    float(row.brake_torque_rr),
                ),
                "az_imu": float(row.az_imu),
                "roll_rate": float(row.roll_rate),
                "pitch_rate": float(row.pitch_rate),
            }
            active.observe(
                **_terrain_estimator_observation_for_backend(
                    backend,
                    observation,
                )
            )

        published_update = 0
        joint_snapshot_advanced = 0
        if active.should_update():
            _, confidence = active.estimate()
            update_count += 1
            if backend == GRIT_ESTIMATOR_BACKEND:
                getter = getattr(
                    active, "get_last_accepted_snapshot", None
                )
                candidate_snapshot = (
                    getter() if callable(getter) else None
                )
                if not isinstance(candidate_snapshot, Mapping):
                    raise TraceValidationError(
                        "joint estimator accepted an update without an "
                        "immutable diagnostic snapshot"
                    )
                try:
                    candidate_sequence = int(
                        candidate_snapshot["update_seq"]
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise TraceValidationError(
                        "joint accepted snapshot has no integral update_seq"
                    ) from error
                if candidate_sequence <= joint_snapshot_sequence:
                    raise TraceValidationError(
                        "joint accepted snapshot sequence did not advance"
                    )
                joint_snapshot = candidate_snapshot
                joint_snapshot_sequence = candidate_sequence
                joint_snapshot_advanced = 1
                joint_snapshot_update_count += 1
            if float(confidence) >= config.min_confidence:
                published_n = float(active.get_bekker_n())
                if hasattr(active, "get_friction_angle_deg"):
                    published_phi_deg = float(active.get_friction_angle_deg())
                published_update = 1

        internal_n = float(active.get_bekker_n())
        n_uncertainty = (
            float(active.get_n_uncertainty())
            if hasattr(active, "get_n_uncertainty")
            else float("nan")
        )
        internal_phi_deg = (
            float(active.get_friction_angle_deg())
            if hasattr(active, "get_friction_angle_deg")
            else float("nan")
        )
        phi_uncertainty_deg = (
            float(active.get_phi_uncertainty_deg())
            if hasattr(active, "get_phi_uncertainty_deg")
            else float("nan")
        )
        informative_segments = getattr(active, "informative_segments", float("nan"))
        snapshot = joint_snapshot or {}
        time_rows.append(
            {
                "trace_id": trace_id,
                "backend": backend,
                "backend_label": BACKEND_LABELS.get(backend, backend),
                "sim_time": float(row.sim_time),
                # Observed ego geometry, carried so the separate transition
                # scorer can form axle-aware pre/post masks.  These are
                # runtime state channels the vehicle measures, not terrain
                # labels, so recording them preserves the sensor-only contract.
                "x_pos": float(row.x_cg),
                "y_pos": float(row.y_cg),
                "psi": float(row.psi),
                "n_internal": internal_n,
                "n_published": published_n,
                "n_sigma": n_uncertainty,
                "phi_internal_deg": internal_phi_deg,
                "phi_published_deg": published_phi_deg,
                "phi_sigma_deg": phi_uncertainty_deg,
                "confidence": float(confidence),
                "published_update": published_update,
                "joint_snapshot_advanced": joint_snapshot_advanced,
                "joint_snapshot_seq": snapshot.get("update_seq", ""),
                "joint_snapshot_evidence_time_s": snapshot.get(
                    "evidence_time_s", ""
                ),
                "joint_snapshot_n": snapshot.get("n", ""),
                "joint_snapshot_phi_deg": snapshot.get("phi_deg", ""),
                "joint_snapshot_n_sigma": snapshot.get("n_sigma", ""),
                "joint_snapshot_phi_sigma_deg": snapshot.get(
                    "phi_sigma_deg", ""
                ),
                "joint_snapshot_confidence": snapshot.get("confidence", ""),
                "joint_snapshot_information_kl": snapshot.get(
                    "joint_information_kl", ""
                ),
                "joint_snapshot_observability_rank": snapshot.get(
                    "observability_rank", ""
                ),
                "joint_snapshot_observability_min_singular_value": (
                    snapshot.get(
                        "observability_min_singular_value", ""
                    )
                ),
                "joint_snapshot_n_boundary_mass": snapshot.get(
                    "n_boundary_mass", ""
                ),
                "joint_snapshot_phi_boundary_mass": snapshot.get(
                    "phi_boundary_mass", ""
                ),
                "joint_snapshot_max_boundary_mass": snapshot.get(
                    "max_boundary_mass", ""
                ),
                "joint_snapshot_boundary_limited": snapshot.get(
                    "boundary_limited", ""
                ),
                "joint_snapshot_projection_wall_time_s": snapshot.get(
                    "projection_wall_time_s", ""
                ),
                "joint_snapshot_profile_wall_time_s": snapshot.get(
                    "profile_wall_time_s", ""
                ),
                "joint_snapshot_observability_wall_time_s": snapshot.get(
                    "observability_wall_time_s", ""
                ),
                "joint_snapshot_posterior_wall_time_s": snapshot.get(
                    "posterior_wall_time_s", ""
                ),
                "joint_snapshot_publication_wall_time_s": snapshot.get(
                    "publication_wall_time_s", ""
                ),
                "joint_snapshot_update_wall_time_s": snapshot.get(
                    "update_wall_time_s", ""
                ),
                "informative_segments": informative_segments,
                "dynamics_active": int(bool(getattr(active, "dynamics_active", False))),
                "dynamics_windows": getattr(active, "dynamics_windows", 0),
                "accepted_dynamics_windows": getattr(
                    active, "accepted_dynamics_windows", 0
                ),
                "rejected_dynamics_windows": getattr(
                    active, "rejected_dynamics_windows", 0
                ),
                "feature_clip_count": getattr(active, "feature_clip_count", 0),
                "feature_envelope_excursions": getattr(
                    active, "feature_envelope_excursions", 0
                ),
                "kinematic_excitation_rejections": getattr(
                    active, "kinematic_excitation_rejections", 0
                ),
                "last_informative_time": getattr(
                    active, "last_informative_time", float("nan")
                ),
                "profile_force_gain": getattr(
                    active, "profile_force_gain", float("nan")
                ),
                "profile_ax_bias": getattr(
                    active, "profile_ax_bias", float("nan")
                ),
                "profile_ay_bias": getattr(
                    active, "profile_ay_bias", float("nan")
                ),
                "profile_bound_hits": getattr(active, "profile_bound_hits", 0),
            }
        )

    time_series = pd.DataFrame(time_rows)
    tail = time_series["sim_time"] >= config.tail_start
    if not tail.any():
        raise TraceValidationError(
            f"trace {trace_id} ends before tail_start={config.tail_start:g}s"
        )
    summary = {
        "trace_id": trace_id,
        "backend": backend,
        "backend_label": BACKEND_LABELS.get(backend, backend),
        "status": "ok",
        "trace_path": str(path),
        "trace_sha256": actual_hash,
        "trace_rows": int(len(trace)),
        "tail_start_s": float(config.tail_start),
        "tail_rows": int(tail.sum()),
        "est_n": float(time_series.loc[tail, "n_published"].mean()),
        "internal_est_n": float(time_series.loc[tail, "n_internal"].mean()),
        "final_est_n": float(time_series["n_published"].iloc[-1]),
        "final_internal_n": float(time_series["n_internal"].iloc[-1]),
        "tail_n_sigma": float(time_series.loc[tail, "n_sigma"].mean()),
        "final_n_sigma": float(time_series["n_sigma"].iloc[-1]),
        "est_phi_deg": float(time_series.loc[tail, "phi_published_deg"].mean()),
        "internal_est_phi_deg": float(
            time_series.loc[tail, "phi_internal_deg"].mean()
        ),
        "final_est_phi_deg": float(time_series["phi_published_deg"].iloc[-1]),
        "final_internal_phi_deg": float(
            time_series["phi_internal_deg"].iloc[-1]
        ),
        "tail_phi_sigma_deg": float(
            time_series.loc[tail, "phi_sigma_deg"].mean()
        ),
        "final_phi_sigma_deg": float(time_series["phi_sigma_deg"].iloc[-1]),
        "publish_update_count": int(time_series["published_update"].sum()),
        "joint_snapshot_update_count": int(joint_snapshot_update_count),
        "final_joint_snapshot_seq": int(joint_snapshot_sequence),
        "estimate_call_count": int(update_count),
        "dynamics_active_rows": int(time_series["dynamics_active"].sum()),
        "final_dynamics_windows": int(time_series["dynamics_windows"].iloc[-1]),
        "max_dynamics_windows": int(time_series["dynamics_windows"].max()),
        "accepted_dynamics_windows": int(
            time_series["accepted_dynamics_windows"].iloc[-1]
        ),
        "rejected_dynamics_windows": int(
            time_series["rejected_dynamics_windows"].iloc[-1]
        ),
        "feature_clip_count": int(time_series["feature_clip_count"].iloc[-1]),
        "feature_envelope_excursions": int(
            time_series["feature_envelope_excursions"].iloc[-1]
        ),
        "kinematic_excitation_rejections": int(
            time_series["kinematic_excitation_rejections"].iloc[-1]
        ),
        "last_informative_time": float(
            time_series["last_informative_time"].iloc[-1]
        ),
        "final_profile_force_gain": float(
            time_series["profile_force_gain"].iloc[-1]
        ),
        "final_profile_ax_bias": float(
            time_series["profile_ax_bias"].iloc[-1]
        ),
        "final_profile_ay_bias": float(
            time_series["profile_ay_bias"].iloc[-1]
        ),
        "profile_bound_hit_rows": int(
            (time_series["profile_bound_hits"] > 0).sum()
        ),
    }
    return summary, time_series


@dataclass(frozen=True)
class ReplayTask:
    trace_id: str
    trace_path: str
    trace_sha256: str
    trace_quality: str
    backend: str
    output_path: str
    config: ReplayConfig


def replay_one(task: ReplayTask) -> dict[str, object]:
    """Process-pool entry point for one trace/backend pair."""

    try:
        summary, time_series = replay_trace(
            task.trace_path,
            task.backend,
            task.config,
            trace_id=task.trace_id,
            expected_sha256=task.trace_sha256,
        )
        output = Path(task.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        time_series.to_csv(output, index=False, float_format="%.17g")
        summary["timeseries_path"] = str(output.resolve())
        summary["trace_quality"] = task.trace_quality
        return summary
    except Exception as error:  # worker must report a failed cell, not kill matrix
        return {
            "trace_id": task.trace_id,
            "backend": task.backend,
            "backend_label": BACKEND_LABELS.get(task.backend, task.backend),
            "status": "fail",
            "trace_path": task.trace_path,
            "trace_sha256": task.trace_sha256,
            "trace_quality": task.trace_quality,
            "failure": repr(error),
            "timeseries_path": "",
        }


def _write_manifest(path: Path, args: argparse.Namespace, config: ReplayConfig) -> None:
    rows = [
        ("created_at", datetime.now().isoformat(timespec="seconds")),
        ("command", " ".join(sys.argv)),
        ("project_root", str(ROOT)),
        ("truth_inputs", "none"),
        ("trace_manifest_sha256", sha256_file(args.trace_manifest)),
    ]
    rows.extend((f"config.{key}", repr(value)) for key, value in asdict(config).items())
    rows.extend((f"cli.{key}", repr(value)) for key, value in sorted(vars(args).items()))
    if GRIT_ESTIMATOR_BACKEND in args.backends:
        rows.extend(
            (f"joint_contract.{key}", repr(value))
            for key, value in GRIT_ESTIMATOR_CONTRACT.items()
        )
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(BACKEND_LABELS),
        default=["scalar_parent"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--update-interval", type=int, default=8)
    parser.add_argument("--te-filter-dt", type=float, default=0.1)
    parser.add_argument("--te-min-confidence", type=float, default=0.0)
    parser.add_argument("--tail-start", type=float, default=11.0)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--student-dof", type=float, default=4.0)
    parser.add_argument("--dynamics-update-interval", type=int, default=1)
    parser.add_argument("--dynamics-block-dt", type=float, default=0.5)
    parser.add_argument("--dynamics-horizon", type=float, default=8.0)
    parser.add_argument("--dynamics-min-windows", type=int, default=12)
    parser.add_argument("--dynamics-min-window-samples", type=int, default=4)
    parser.add_argument("--dynamics-r-ax", type=float, default=0.35)
    parser.add_argument("--dynamics-r-ay", type=float, default=0.30)
    parser.add_argument("--dynamics-min-information", type=float, default=0.20)
    parser.add_argument("--dynamics-min-yaw-rate-rms", type=float, default=0.0)
    parser.add_argument("--dynamics-min-speed", type=float, default=2.5)
    parser.add_argument("--dynamics-max-abs-alpha", type=float, default=0.35)
    parser.add_argument(
        "--dynamics-enforce-feature-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dynamics-slip-mode",
        choices=("wheel", "average", "fixed"),
        default="average",
    )
    parser.add_argument("--dynamics-fixed-kappa", type=float, default=0.05)
    parser.add_argument(
        "--dynamics-rate-mode",
        choices=("signed", "zero", "legacy"),
        default="zero",
    )
    parser.add_argument("--dynamics-force-gain-std", type=float, default=0.04)
    parser.add_argument("--dynamics-ax-bias-std", type=float, default=0.10)
    parser.add_argument("--dynamics-ay-bias-std", type=float, default=0.05)
    parser.add_argument("--dynamics-force-gain-min", type=float, default=0.70)
    parser.add_argument("--dynamics-force-gain-max", type=float, default=1.30)
    parser.add_argument(
        "--dynamics-acceleration-bias-bound", type=float, default=0.30
    )
    parser.add_argument("--dynamics-profile-iterations", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1 or args.update_interval < 1:
        raise SystemExit("worker and estimator update intervals must be positive")
    if len(args.backends) != len(set(args.backends)):
        raise SystemExit("--backends must not contain duplicates")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    (output_dir / "timeseries").mkdir(parents=True, exist_ok=True)

    trace_manifest_path = args.trace_manifest.expanduser().resolve()
    args.trace_manifest = trace_manifest_path
    try:
        trace_manifest = pd.read_csv(trace_manifest_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise SystemExit(f"cannot read trace manifest: {error}") from error
    reject_oracle_columns(trace_manifest.columns, context="trace manifest")
    required = {
        "trace_id", "status", "trace_path", "trace_sha256", "trace_quality",
        "trace_schema_version",
    }
    missing = sorted(required - set(trace_manifest.columns))
    if missing:
        raise SystemExit("trace manifest is missing: " + ", ".join(missing))
    trace_manifest = trace_manifest[trace_manifest["status"] == "ok"].copy()
    if trace_manifest.empty:
        raise SystemExit("trace manifest contains no successful traces")
    if trace_manifest["trace_id"].duplicated().any():
        raise SystemExit("trace manifest has duplicate trace_id values")
    trace_ids = trace_manifest["trace_id"].astype(str)
    unsafe = sorted(
        trace_id for trace_id in trace_ids if not _SAFE_TRACE_ID.fullmatch(trace_id)
    )
    if unsafe:
        raise SystemExit(
            "trace manifest has unsafe trace_id values: " + ", ".join(unsafe)
        )
    versions = pd.to_numeric(
        trace_manifest["trace_schema_version"], errors="coerce"
    ).to_numpy(dtype=float)
    if not (versions == TRACE_SCHEMA_VERSION).all():
        raise SystemExit(
            f"trace manifest requires schema version {TRACE_SCHEMA_VERSION}"
        )

    config = ReplayConfig(
        update_interval=args.update_interval,
        filter_dt=args.te_filter_dt,
        min_confidence=args.te_min_confidence,
        tail_start=args.tail_start,
        model_dir=args.model_dir,
        grid_size=args.grid_size,
        student_dof=args.student_dof,
        dynamics_update_interval=args.dynamics_update_interval,
        dynamics_block_dt=args.dynamics_block_dt,
        dynamics_horizon=args.dynamics_horizon,
        dynamics_min_windows=args.dynamics_min_windows,
        dynamics_min_window_samples=args.dynamics_min_window_samples,
        dynamics_r_ax=args.dynamics_r_ax,
        dynamics_r_ay=args.dynamics_r_ay,
        dynamics_min_information=args.dynamics_min_information,
        dynamics_min_yaw_rate_rms=args.dynamics_min_yaw_rate_rms,
        dynamics_min_speed=args.dynamics_min_speed,
        dynamics_max_abs_alpha=args.dynamics_max_abs_alpha,
        dynamics_enforce_feature_envelope=(
            args.dynamics_enforce_feature_envelope
        ),
        dynamics_slip_mode=args.dynamics_slip_mode,
        dynamics_fixed_kappa=args.dynamics_fixed_kappa,
        dynamics_rate_mode=args.dynamics_rate_mode,
        dynamics_force_gain_std=args.dynamics_force_gain_std,
        dynamics_ax_bias_std=args.dynamics_ax_bias_std,
        dynamics_ay_bias_std=args.dynamics_ay_bias_std,
        dynamics_force_gain_min=args.dynamics_force_gain_min,
        dynamics_force_gain_max=args.dynamics_force_gain_max,
        dynamics_acceleration_bias_bound=(
            args.dynamics_acceleration_bias_bound
        ),
        dynamics_profile_iterations=args.dynamics_profile_iterations,
    )
    _write_manifest(output_dir / "replay_manifest.csv", args, config)

    tasks: list[ReplayTask] = []
    for row in trace_manifest.itertuples(index=False):
        for backend in args.backends:
            tasks.append(
                ReplayTask(
                    trace_id=str(row.trace_id),
                    trace_path=str(
                        resolve_manifest_trace_path(row.trace_path, trace_manifest_path)
                    ),
                    trace_sha256=str(row.trace_sha256),
                    trace_quality=str(row.trace_quality),
                    backend=backend,
                    output_path=str(
                        output_dir / "timeseries" / f"{row.trace_id}_{backend}.csv"
                    ),
                    config=config,
                )
            )

    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(replay_one, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    estimates = pd.DataFrame(results).sort_values(["trace_id", "backend"])
    if "timeseries_path" in estimates:
        def portable_timeseries_path(value: object) -> str:
            if not isinstance(value, str) or not value:
                return ""
            path = Path(value).expanduser().resolve()
            try:
                return path.relative_to(output_dir).as_posix()
            except ValueError as error:
                raise SystemExit(
                    f"replay time series escaped output directory: {path}"
                ) from error

        estimates["timeseries_path"] = estimates["timeseries_path"].map(
            portable_timeseries_path
        )
    estimates.to_csv(output_dir / "estimates.csv", index=False)
    failures = estimates[estimates["status"] != "ok"]
    print(f"wrote sensor-only replay estimates: {output_dir / 'estimates.csv'}")
    if not failures.empty:
        print(f"{len(failures)}/{len(estimates)} replay cells failed")
        for row in failures.itertuples(index=False):
            print(f"  {row.trace_id}/{row.backend}: {getattr(row, 'failure', '')}")
        return 2
    print(f"replayed {len(args.backends)} estimator(s) over {len(trace_manifest)} shared traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
