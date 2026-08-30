#!/usr/bin/env python3
"""Sensor-only trace contract for terrain-estimator experiments.

The replayable trace excludes plant soil parameters, Chrono tyre forces, and
truth-state diagnostics, so that an estimator replayed against it cannot read
any quantity the vehicle does not measure.  Ground truth lives in a separate
file that is joined only after replay has produced its estimates, which keeps
scoring downstream of inference.

``sanitize_exact_observations`` produces publishable traces.  It requires an
unrounded ``terrain_observations.csv`` written directly from the live
estimator input object, so that a replayed estimator sees byte-identical
inputs to the online one.  ``sanitize_approximate_diagnostic`` instead accepts
the rounded controller diagnostic log and reconstructs the channels that log
omits; because that reconstruction is inexact, callers must opt into it
explicitly and the resulting trace is labelled approximate.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRACE_SCHEMA_VERSION = 3

# Exact, ordered output schema.  Raw sensor channels are retained alongside
# the shared runtime preprocessing so new methods can be evaluated without
# changing the input trajectory.  Every column is numeric.
TRACE_COLUMNS = (
    "seq",
    "sim_time",
    "x_cg",
    "y_cg",
    "z_cg",
    "quat_e0",
    "quat_e1",
    "quat_e2",
    "quat_e3",
    "psi",
    "u_raw",
    "v_lateral_raw",
    "omega_raw",
    "u",
    "v_lateral",
    "omega",
    "ax_imu",
    "ay_imu",
    "az_imu",
    "ax_filtered",
    "roll_rate",
    "pitch_rate",
    "wheel_omega_fl",
    "wheel_omega_fr",
    "wheel_omega_rl",
    "wheel_omega_rr",
    "wheel_center_z_fl",
    "wheel_center_z_fr",
    "wheel_center_z_rl",
    "wheel_center_z_rr",
    "drive_torque_fl",
    "drive_torque_fr",
    "drive_torque_rl",
    "drive_torque_rr",
    "brake_torque_fl",
    "brake_torque_fr",
    "brake_torque_rl",
    "brake_torque_rr",
    "steering_angle",
    "kappa",
    "alpha_f",
    "alpha_r",
    "alpha_rate_f",
    "alpha_rate_r",
    "Fz_f",
    "Fz_r",
)

_EXACT_SOURCE_METADATA_COLUMNS = {"logger_version", "trace_schema_version"}


class TraceValidationError(ValueError):
    """Raised when a trace violates the sensor-only replay contract."""


def _oracle_columns(columns: Iterable[str]) -> list[str]:
    """Return columns that can expose simulation truth to an estimator."""

    bad: list[str] = []
    for column in columns:
        name = str(column).strip().lower()
        tokens = set(name.replace("-", "_").split("_"))
        if (
            name.startswith("true_")
            or name.endswith("_true")
            or name.startswith("actual_")
            or name.startswith("plant_")
            or name in {
                "n", "n_true", "kphi", "kc", "cohesion", "friction_angle",
                "janosi_shear", "bekker_n", "terrain_n", "terrain_truth",
                "ground_truth",
            }
            or "soil" in tokens
            or "truth" in tokens
            or "tireforce" in tokens
            or "tireforces" in tokens
            or "tire_force" in name
            or (name.startswith(("front_", "rear_")) and name.endswith(("_fx", "_fy", "_fz")))
        ):
            bad.append(str(column))
    return bad


def reject_oracle_columns(columns: Iterable[str], *, context: str) -> None:
    """Raise when a manifest or source exposes labels to replay code."""

    oracle = _oracle_columns(columns)
    if oracle:
        raise TraceValidationError(
            f"{context} contains forbidden oracle columns: " + ", ".join(oracle)
        )


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sensor_trace_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a numeric copy in canonical column order."""

    oracle = _oracle_columns(frame.columns)
    if oracle:
        raise TraceValidationError(
            "sensor trace contains forbidden oracle columns: " + ", ".join(oracle)
        )

    missing = [column for column in TRACE_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in TRACE_COLUMNS]
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise TraceValidationError("invalid sensor trace schema (" + "; ".join(details) + ")")

    if len(frame) < 2:
        raise TraceValidationError("sensor trace must contain at least two rows")

    numeric = frame.loc[:, list(TRACE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise TraceValidationError(
            f"sensor trace has a non-finite value at row {int(row)}, "
            f"column {TRACE_COLUMNS[int(column)]}"
        )

    sequence = numeric["seq"].to_numpy(dtype=float)
    if not np.all(sequence == np.floor(sequence)):
        raise TraceValidationError("seq must contain integers")
    if np.any(np.diff(sequence) <= 0.0):
        raise TraceValidationError("seq must be strictly increasing")

    times = numeric["sim_time"].to_numpy(dtype=float)
    if times[0] < 0.0:
        raise TraceValidationError("sim_time must be non-negative")
    if np.any(np.diff(times) <= 0.0):
        raise TraceValidationError("sim_time must be strictly increasing")

    quaternion = numeric[["quat_e0", "quat_e1", "quat_e2", "quat_e3"]].to_numpy(
        dtype=float
    )
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm < 0.5) or np.any(norm > 1.5):
        raise TraceValidationError("quaternion norm is outside the sanity interval [0.5, 1.5]")

    if np.any(numeric["Fz_f"].to_numpy() <= 0.0) or np.any(
        numeric["Fz_r"].to_numpy() <= 0.0
    ):
        raise TraceValidationError("normal loads must be positive")
    if np.any(numeric["u"].to_numpy() < 0.5 - 1.0e-9):
        raise TraceValidationError("preprocessed u must respect the 0.5 m/s floor")
    if np.any(np.abs(numeric["kappa"].to_numpy()) > 0.8 + 1.0e-9):
        raise TraceValidationError("kappa lies outside the deployed [-0.8, 0.8] range")
    for column in ("alpha_f", "alpha_r"):
        if np.any(np.abs(numeric[column].to_numpy()) > 0.55 + 1.0e-9):
            raise TraceValidationError(
                f"{column} lies outside the deployed [-0.55, 0.55] range"
            )
    if np.any(np.abs(numeric["steering_angle"].to_numpy()) > math.pi):
        raise TraceValidationError("steering_angle exceeds the sanity bound of pi radians")

    return numeric


def load_sensor_trace(path: str | Path) -> pd.DataFrame:
    """Load a strict sensor trace without exposing any extra input columns."""

    source = Path(path)
    if not source.is_file():
        raise TraceValidationError(f"sensor trace does not exist: {source}")
    try:
        frame = pd.read_csv(source, float_precision="round_trip")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise TraceValidationError(f"cannot parse sensor trace {source}: {error}") from error
    return validate_sensor_trace_frame(frame)


def _write_trace(frame: pd.DataFrame, destination: str | Path) -> dict[str, object]:
    """Validate, write at round-trip float precision, and verify the result."""

    clean = validate_sensor_trace_frame(frame)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    clean.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(output)
    verified = load_sensor_trace(output)
    return {
        "trace_path": str(output.resolve()),
        "trace_sha256": sha256_file(output),
        "trace_rows": int(len(verified)),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
    }


def sanitize_exact_observations(
    source: str | Path, destination: str | Path
) -> dict[str, object]:
    """Create a strict trace from an exact, unrounded runtime observation log.

    The source may carry harmless bookkeeping fields, but any truth-looking
    field is rejected rather than silently discarded.  All canonical fields
    must already be present because reconstructing them from a rounded
    diagnostic would not be an exact replay.
    """

    source_path = Path(source)
    try:
        frame = pd.read_csv(source_path, float_precision="round_trip")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise TraceValidationError(
            f"cannot parse exact terrain observation log {source_path}: {error}"
        ) from error
    oracle = _oracle_columns(frame.columns)
    if oracle:
        raise TraceValidationError(
            "exact terrain observation log contains oracle columns: "
            + ", ".join(oracle)
        )
    missing = [column for column in TRACE_COLUMNS if column not in frame.columns]
    if missing:
        raise TraceValidationError(
            "exact terrain observation log is missing required fields: "
            + ", ".join(missing)
        )
    unexpected = sorted(
        set(frame.columns) - set(TRACE_COLUMNS) - _EXACT_SOURCE_METADATA_COLUMNS
    )
    if unexpected:
        raise TraceValidationError(
            "exact terrain observation log has non-schema fields: "
            + ", ".join(unexpected)
        )
    return _write_trace(frame.loc[:, list(TRACE_COLUMNS)], destination)


def sanitize_approximate_diagnostic(
    source: str | Path,
    destination: str | Path,
    *,
    allow_approximate: bool = False,
    mass: float = 2573.0,
    lf: float = 1.593,
    lr: float = 1.709,
    h_cg: float = 0.65,
    gravity: float = 9.81,
) -> dict[str, object]:
    """Convert a rounded controller diagnostic into a labelled approximate trace.

    This mode reconstructs filtered lateral velocity and yaw rate by inverting
    the bicycle slip relations over rounded slip-angle fields.  The rounding
    and the inversion both introduce error, so the result supports estimator
    plumbing checks but is not admissible for published results or for
    online/replay identity checks, which require exact traces.
    """

    if not allow_approximate:
        raise TraceValidationError(
            "rounded diagnostic conversion is disabled; pass allow_approximate=True "
            "only for a labelled development replay"
        )
    source_path = Path(source)
    try:
        diagnostic = pd.read_csv(source_path, float_precision="round_trip")
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise TraceValidationError(f"cannot parse diagnostic {source_path}: {error}") from error

    required = {
        "seq", "sim_time", "x_fa_meas", "y_fa_meas", "psi_meas",
        "u_meas", "v_meas", "omega_meas", "ax_imu", "ay_imu",
        "z_cg", "quat_e0", "quat_e1", "quat_e2", "quat_e3",
        "az_imu", "omega_x_imu", "omega_y_imu",
        "wheel_omega_fl", "wheel_omega_fr", "wheel_omega_rl", "wheel_omega_rr",
        "steering_angle_sensor", "est_alpha_f", "est_alpha_r",
        "est_Fz_f_mean", "est_Fz_r_mean", "est_kappa",
        "est_alpha_rate_f", "est_alpha_rate_r", "est_u_safe",
    }
    missing = sorted(required - set(diagnostic.columns))
    if missing:
        raise TraceValidationError(
            "rounded diagnostic lacks fields required for approximate replay: "
            + ", ".join(missing)
        )

    def number(name: str) -> np.ndarray:
        return pd.to_numeric(diagnostic[name], errors="coerce").to_numpy(dtype=float)

    psi = number("psi_meas")
    u = number("est_u_safe")
    alpha_f = number("est_alpha_f")
    alpha_r = number("est_alpha_r")
    delta = number("steering_angle_sensor")

    # Invert the bicycle slip equations used by compute_bicycle_operating_point:
    #   tan(delta-alpha_f) = (v + Lf*r)/u
    #   tan(-alpha_r)      = (v - Lr*r)/u
    front_ratio = np.tan(delta - alpha_f)
    rear_ratio = np.tan(-alpha_r)
    omega = u * (front_ratio - rear_ratio) / (lf + lr)
    v_lateral = u * front_ratio - lf * omega

    fz_f = number("est_Fz_f_mean")
    fz_r = number("est_Fz_r_mean")
    wheelbase = lf + lr
    if h_cg > 0.0:
        ax_from_front = (mass * gravity * lr - 2.0 * wheelbase * fz_f) / (mass * h_cg)
        ax_from_rear = (2.0 * wheelbase * fz_r - mass * gravity * lf) / (mass * h_cg)
        ax_filtered = 0.5 * (ax_from_front + ax_from_rear)
    else:  # pragma: no cover - defensive parameter guard
        ax_filtered = number("ax_imu")

    frame = pd.DataFrame({
        "seq": number("seq"),
        "sim_time": number("sim_time"),
        "x_cg": number("x_fa_meas") - lf * np.cos(psi),
        "y_cg": number("y_fa_meas") - lf * np.sin(psi),
        "z_cg": number("z_cg"),
        "quat_e0": number("quat_e0"),
        "quat_e1": number("quat_e1"),
        "quat_e2": number("quat_e2"),
        "quat_e3": number("quat_e3"),
        "psi": psi,
        "u_raw": number("u_meas"),
        "v_lateral_raw": number("v_meas"),
        "omega_raw": number("omega_meas"),
        "u": u,
        "v_lateral": v_lateral,
        "omega": omega,
        "ax_imu": number("ax_imu"),
        "ay_imu": number("ay_imu"),
        "az_imu": number("az_imu"),
        "ax_filtered": ax_filtered,
        "roll_rate": number("omega_x_imu"),
        "pitch_rate": number("omega_y_imu"),
        "wheel_omega_fl": number("wheel_omega_fl"),
        "wheel_omega_fr": number("wheel_omega_fr"),
        "wheel_omega_rl": number("wheel_omega_rl"),
        "wheel_omega_rr": number("wheel_omega_rr"),
        # The rounded diagnostic carries no per-wheel centre height, so this
        # approximate path substitutes a fixed offset below the chassis centre
        # of gravity.  Exact traces carry the measured channels directly.
        "wheel_center_z_fl": number("z_cg") - 0.315,
        "wheel_center_z_fr": number("z_cg") - 0.315,
        "wheel_center_z_rl": number("z_cg") - 0.315,
        "wheel_center_z_rr": number("z_cg") - 0.315,
        # The rounded diagnostic carries no wheel-torque channels.  Zeros keep
        # the approximate path schema-compatible without fabricating torque
        # information that the source never measured.
        "drive_torque_fl": np.zeros(len(diagnostic)),
        "drive_torque_fr": np.zeros(len(diagnostic)),
        "drive_torque_rl": np.zeros(len(diagnostic)),
        "drive_torque_rr": np.zeros(len(diagnostic)),
        "brake_torque_fl": np.zeros(len(diagnostic)),
        "brake_torque_fr": np.zeros(len(diagnostic)),
        "brake_torque_rl": np.zeros(len(diagnostic)),
        "brake_torque_rr": np.zeros(len(diagnostic)),
        "steering_angle": delta,
        "kappa": number("est_kappa"),
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
        "alpha_rate_f": number("est_alpha_rate_f"),
        "alpha_rate_r": number("est_alpha_rate_r"),
        "Fz_f": fz_f,
        "Fz_r": fz_r,
    })
    metadata = _write_trace(frame, destination)
    metadata["trace_quality"] = "approximate_rounded_diagnostic"
    metadata["source_diagnostic"] = str(source_path.resolve())
    return metadata


def yaw_from_quaternion(e0: float, e1: float, e2: float, e3: float) -> float:
    """Return the yaw angle encoded by a body orientation quaternion.

    Exact observation loggers use this to record heading in the same
    convention the estimators consume.
    """

    return math.atan2(2.0 * (e0 * e3 + e1 * e2), 1.0 - 2.0 * (e2 * e2 + e3 * e3))
