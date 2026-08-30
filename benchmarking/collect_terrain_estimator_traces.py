#!/usr/bin/env python3
"""Collect fixed-controller sensor traces for fair terrain-estimator replay.

Each plant soil is driven once by the same estimator-disabled NMPC with a
fixed dirt prior.  Estimator backends are intentionally absent from this
program; they are evaluated later by ``terrain_estimator_replay.py`` against
the resulting shared trace.

Publishable collection requires the controller's unrounded
``terrain_observations.csv`` and fails closed if that artifact is missing or
invalid, so a degraded trace can never be mistaken for an exact one.
``--allow-approx-diag`` instead accepts the rounded controller diagnostic and
labels the resulting trace approximate; such traces support plumbing checks
and are inadmissible as evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "simulation", ROOT / "benchmarking"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import flatpath  # noqa: E402,F401
from common import launch_and_collect  # noqa: E402
from param_consistency import (  # noqa: E402
    TERRAIN_PRESETS,
    generate_lhs_terrain_yaml_dicts,
)
from terrain_estimator_trace import (  # noqa: E402
    TraceValidationError,
    sanitize_approximate_diagnostic,
    sanitize_exact_observations,
)


N_LO, N_HI = 0.52, 1.08
INDEPENDENT_N_LO, INDEPENDENT_N_HI = 0.55, 1.05
INDEPENDENT_PHI_LO_DEG, INDEPENDENT_PHI_HI_DEG = 10.0, 35.0
INDEPENDENT_COHESION_MULTIPLIER_LO = 0.85
INDEPENDENT_COHESION_MULTIPLIER_HI = 1.15
_MANIFOLD_KEYS = ("Kphi", "Kc", "cohesion", "friction_angle", "janosi_shear")
_RECOVERABLE_STATUSES = {
    "run_failed", "worker_exception", "diagnostic_io_failure", "trace_io_failure"
}


def manifold_yaml_from_n(n_value: float, jitter_fraction: float = 0.0, rng=None) -> dict:
    """Reproduce the deployed clay--dirt--sand scalar terrain manifold."""

    points = sorted(
        ((float(preset["n"]), preset) for preset in TERRAIN_PRESETS.values()),
        key=lambda item: item[0],
    )
    clipped = min(max(float(n_value), points[0][0]), points[-1][0])
    output = {
        "elastic_stiffness": 2.0e8,
        "damping": 3.0e4,
        "description": "fixed-controller terrain-estimator trace case",
    }
    for index in range(len(points) - 1):
        n0, preset0 = points[index]
        n1, preset1 = points[index + 1]
        if clipped <= n1 or index == len(points) - 2:
            weight = 0.0 if n1 == n0 else (clipped - n0) / (n1 - n0)
            for key in _MANIFOLD_KEYS:
                value = (1.0 - weight) * float(preset0[key]) + weight * float(preset1[key])
                if jitter_fraction and rng is not None:
                    value *= 1.0 + float(jitter_fraction) * float(rng.uniform(-1.0, 1.0))
                output[key] = value
            break
    output["n"] = float(n_value)
    return output


def generate_soils(
    count: int,
    *,
    mode: str,
    seed: int,
    jitter_fraction: float,
    n_lo: float = INDEPENDENT_N_LO,
    n_hi: float = INDEPENDENT_N_HI,
) -> tuple[list[tuple[float, dict]], int]:
    """Generate paired soil cases and return their nuisance RNG seed."""

    if count < 1:
        raise ValueError("count must be positive")
    n_rng = np.random.default_rng(seed)
    nuisance_seed = int(seed) + 1_000_003
    nuisance_rng = np.random.default_rng(nuisance_seed)
    soils: list[tuple[float, dict]] = []
    if mode == "manifold":
        edges = np.linspace(N_LO, N_HI, count + 1)
        for index in range(count):
            n_true = float(n_rng.uniform(edges[index], edges[index + 1]))
            soils.append(
                (
                    n_true,
                    manifold_yaml_from_n(n_true, jitter_fraction, nuisance_rng),
                )
            )
    elif mode == "lhs":
        dictionaries = generate_lhs_terrain_yaml_dicts(count, seed=seed)
        for source in dictionaries:
            soil = dict(source)
            n_true = float(np.clip(soil["n"], N_LO, N_HI))
            if float(source["n"]) < N_LO or float(source["n"]) > N_HI:
                n_true = float(n_rng.uniform(N_LO, N_HI))
            soil["n"] = n_true
            soils.append((n_true, soil))
    elif mode == "independent_n_phi":
        # Decorrelate the Bekker exponent from the friction angle so that a
        # joint estimator must resolve them separately rather than inferring
        # one from the other. Kphi, Kc, and the Janosi length stay on the
        # n-indexed manifold, and cohesion is an independently drawn nuisance.
        # Separate deterministic RNG streams keep every draw reproducible and
        # auditable from the recorded seeds alone.
        n_edges = np.linspace(float(n_lo), float(n_hi), count + 1)
        n_values = np.asarray([
            n_rng.uniform(n_edges[index], n_edges[index + 1])
            for index in range(count)
        ])
        phi_rng = np.random.default_rng(nuisance_seed)
        phi_edges = np.linspace(
            INDEPENDENT_PHI_LO_DEG, INDEPENDENT_PHI_HI_DEG, count + 1
        )
        phi_strata = np.asarray([
            phi_rng.uniform(phi_edges[index], phi_edges[index + 1])
            for index in range(count)
        ])
        phi_values = phi_strata[phi_rng.permutation(count)]
        cohesion_rng = np.random.default_rng(nuisance_seed + 1)
        cohesion_multipliers = cohesion_rng.uniform(
            INDEPENDENT_COHESION_MULTIPLIER_LO,
            INDEPENDENT_COHESION_MULTIPLIER_HI,
            size=count,
        )
        for n_true, phi_true, cohesion_multiplier in zip(
            n_values, phi_values, cohesion_multipliers
        ):
            soil = manifold_yaml_from_n(float(n_true))
            soil["friction_angle"] = float(phi_true)
            soil["cohesion"] = float(soil["cohesion"] * cohesion_multiplier)
            soils.append((float(n_true), soil))
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"unknown soil mode: {mode}")
    return soils, nuisance_seed


@dataclass(frozen=True)
class TraceTask:
    trace_id: str
    yaml_path: str
    run_dir: str
    sim_port: int
    ctrl_port: int
    sim_seed: int
    path: str
    speed: float
    sim_time: float
    lead_in: float
    allow_approx_diag: bool
    maneuver_label: str = "passive"
    excitation_args: tuple = ()
    terrain_id_probe: bool = False
    probe_target_alpha: float = 0.10
    probe_slew_rate: float = 0.40
    probe_signed_dwell: float = 0.15
    probe_clearance: float = 35.0
    probe_max_latency: float = 0.30
    wheel_center_noise_std: float = 0.01
    wheel_center_calibration_bias_std: float = 0.003
    bumpiness: int = 0
    attempt: int = 1
    recovery: bool = False


def _excitation_args(args) -> tuple:
    """Return the controller flags that set how hard the trace excites the plant.

    A constant speed along a fixed sinusoid drives the lateral channel alone
    and leaves the Bekker exponent weakly observable, because sinkage responds
    mainly to longitudinal load transfer. These flags widen the excitation on
    both channels: path shape for the lateral one, speed-reference modulation
    for the longitudinal one. The speed modulation is applied beneath the g--g
    grip cap, so it can never request more than the friction envelope allows.
    """
    out: list[str] = []
    if getattr(args, "sine_amplitude", None) is not None:
        out += ["--sine-amplitude", str(args.sine_amplitude)]
    if getattr(args, "sine_wavelength", None) is not None:
        out += ["--sine-wavelength", str(args.sine_wavelength)]
    if float(getattr(args, "speed_osc_amplitude", 0.0)) > 0.0:
        out += ["--speed-osc-amplitude", str(args.speed_osc_amplitude),
                "--speed-osc-period-s", str(args.speed_osc_period_s)]
    return tuple(out)


def fixed_controller_extra_args(task: TraceTask) -> list[str]:
    """Return auditable fixed-prior arguments with no estimator backend."""

    arguments = [
        "--terrain-config",
        task.yaml_path,
        "--controller-prior-terrain",
        "dirt",
        "--wheel-center-noise-std",
        str(task.wheel_center_noise_std),
        "--wheel-center-calibration-bias-std",
        str(task.wheel_center_calibration_bias_std),
    ]
    arguments.extend(task.excitation_args)
    if task.terrain_id_probe:
        arguments.extend(
            [
                "--terrain-id-probe",
                "--terrain-id-probe-target-alpha",
                str(task.probe_target_alpha),
                "--terrain-id-probe-slew-rate",
                str(task.probe_slew_rate),
                "--terrain-id-probe-signed-dwell",
                str(task.probe_signed_dwell),
                "--terrain-id-probe-clearance",
                str(task.probe_clearance),
                "--terrain-id-probe-max-latency",
                str(task.probe_max_latency),
            ]
        )
    return arguments


def _find_exact_observation_log(run_dir: Path) -> Path | None:
    candidates = sorted(
        path
        for path in run_dir.rglob("terrain_observations*.csv")
        if path.name != "sensor_trace.csv"
    )
    if len(candidates) > 1:
        raise TraceValidationError(
            "multiple exact terrain observation logs found: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0] if candidates else None


def _audit_fixed_prior(diagnostic_path: str | Path) -> None:
    """Confirm the collection controller stayed estimator-disabled at dirt n."""

    try:
        diagnostic = pd.read_csv(diagnostic_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise TraceValidationError(f"cannot audit controller diagnostic: {error}") from error
    required = {"n_terrain_est", "terrain_update_applied"}
    missing = sorted(required - set(diagnostic.columns))
    if missing:
        raise TraceValidationError(
            "cannot audit fixed controller; diagnostic lacks " + ", ".join(missing)
        )
    n_estimate = pd.to_numeric(diagnostic["n_terrain_est"], errors="coerce").to_numpy()
    updates = pd.to_numeric(diagnostic["terrain_update_applied"], errors="coerce").to_numpy()
    if not np.isfinite(n_estimate).all() or not np.isfinite(updates).all():
        raise TraceValidationError("fixed-controller audit contains non-finite values")
    if not np.allclose(n_estimate, 0.7, rtol=0.0, atol=1.0e-9):
        raise TraceValidationError("controller prior changed during trace collection")
    if np.any(updates != 0):
        raise TraceValidationError("terrain estimator updated during fixed trace collection")


def _audit_probe_maneuver(diagnostic_path: str | Path) -> dict[str, object]:
    """Verify that a requested probe completed both signed excitation phases.

    Chrono and the controller may finish normally even when the probe aborts.
    Treating that run as a successful ``terrain_id_probe`` trace silently turns
    a paired experiment into an uncontrolled, often one-sided maneuver.  This
    audit therefore uses controller diagnostics only to certify maneuver
    execution; none of these fields enter estimator replay.
    """

    try:
        diagnostic = pd.read_csv(diagnostic_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise TraceValidationError(f"cannot audit terrain ID probe: {error}") from error
    required = {
        "sim_time",
        "terrain_probe_phase",
        "terrain_probe_reason",
        "est_alpha_f",
        "est_alpha_rate_f",
        "crosstrack_err",
        "ay_imu",
    }
    missing = sorted(required - set(diagnostic.columns))
    if missing:
        raise TraceValidationError(
            "cannot audit terrain ID probe; diagnostic lacks " + ", ".join(missing)
        )
    if diagnostic.empty:
        raise TraceValidationError("terrain ID probe diagnostic is empty")

    phases = diagnostic["terrain_probe_phase"].fillna("").astype(str).str.strip()
    reasons = diagnostic["terrain_probe_reason"].fillna("").astype(str).str.strip()
    numeric = diagnostic[
        ["sim_time", "est_alpha_f", "est_alpha_rate_f", "crosstrack_err", "ay_imu"]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise TraceValidationError("terrain ID probe audit contains non-finite values")

    ordered_phases: list[str] = []
    for phase in phases:
        if not ordered_phases or phase != ordered_phases[-1]:
            ordered_phases.append(phase)
    active_names = {
        "ramp_positive",
        "hold_positive",
        "ramp_negative",
        "hold_negative",
        "recovery",
        "aborting",
    }
    active = phases.isin(active_names)
    positive = phases == "hold_positive"
    negative = phases == "hold_negative"
    complete_indices = np.flatnonzero(phases.to_numpy() == "complete")
    aborted = phases.isin({"aborting", "aborted"})
    abort_reason = ""
    if aborted.any():
        candidate_reasons = reasons[aborted & reasons.ne("")]
        if not candidate_reasons.empty:
            abort_reason = str(candidate_reasons.iloc[0])

    times = numeric["sim_time"].to_numpy(dtype=float)
    sample_dt = np.diff(times, append=times[-1])
    if len(sample_dt) > 1:
        sample_dt[-1] = float(np.median(sample_dt[:-1]))
    sample_dt = np.maximum(sample_dt, 0.0)
    completed = bool(
        len(complete_indices)
        and positive.any()
        and negative.any()
        and not aborted.any()
    )
    active_alpha = numeric.loc[active, "est_alpha_f"].to_numpy(dtype=float)
    active_alpha_rate = numeric.loc[active, "est_alpha_rate_f"].to_numpy(dtype=float)
    active_cte = numeric.loc[active, "crosstrack_err"].to_numpy(dtype=float)
    active_ay = numeric.loc[active, "ay_imu"].to_numpy(dtype=float)
    metadata: dict[str, object] = {
        "probe_audit_ok": completed,
        "probe_final_phase": str(phases.iloc[-1]) if len(phases) else "",
        "probe_abort_reason": abort_reason,
        "probe_phase_sequence": ">".join(phase for phase in ordered_phases if phase),
        "probe_positive_hold_samples": int(positive.sum()),
        "probe_negative_hold_samples": int(negative.sum()),
        "probe_positive_hold_s": float(sample_dt[positive.to_numpy()].sum()),
        "probe_negative_hold_s": float(sample_dt[negative.to_numpy()].sum()),
        "probe_complete_time_s": (
            float(times[int(complete_indices[0])]) if len(complete_indices) else np.nan
        ),
        "probe_peak_abs_alpha_rad": (
            float(np.max(np.abs(active_alpha))) if len(active_alpha) else np.nan
        ),
        "probe_peak_abs_alpha_rate_radps": (
            float(np.max(np.abs(active_alpha_rate)))
            if len(active_alpha_rate) else np.nan
        ),
        "probe_alpha_rate_outside_rig_fraction": (
            float(np.mean(np.abs(active_alpha_rate) > 0.56))
            if len(active_alpha_rate) else np.nan
        ),
        "probe_peak_abs_cte_m": (
            float(np.max(np.abs(active_cte))) if len(active_cte) else np.nan
        ),
        "probe_peak_abs_ay_mps2": (
            float(np.max(np.abs(active_ay))) if len(active_ay) else np.nan
        ),
    }
    if not completed:
        missing_phases = [
            phase
            for phase, present in (
                ("hold_positive", positive.any()),
                ("hold_negative", negative.any()),
                ("complete", len(complete_indices) > 0),
            )
            if not present
        ]
        details = []
        if missing_phases:
            details.append("missing " + ", ".join(missing_phases))
        if aborted.any():
            details.append("aborted" + (f": {abort_reason}" if abort_reason else ""))
        metadata["probe_audit_failure"] = "; ".join(details) or "invalid phase sequence"
    else:
        metadata["probe_audit_failure"] = ""
    return metadata


def _base_result(task: TraceTask) -> dict[str, object]:
    return {
        "trace_id": task.trace_id,
        "status": "pending",
        "run_dir": str(Path(task.run_dir).resolve()),
        "trace_path": "",
        "trace_sha256": "",
        "trace_rows": 0,
        "trace_schema_version": "",
        "trace_quality": "",
        "source_path": "",
        "failure": "",
        "controller_prior": "dirt",
        "controller_prior_n": 0.7,
        "terrain_estimator_enabled": False,
        "sim_seed": task.sim_seed,
        "path": task.path,
        "speed_mps": task.speed,
        "sim_time_s": task.sim_time,
        "lead_in_m": task.lead_in,
        "maneuver_label": task.maneuver_label,
        "terrain_id_probe": task.terrain_id_probe,
        "probe_target_alpha_rad": task.probe_target_alpha,
        "probe_slew_rate_radps": task.probe_slew_rate,
        "probe_signed_dwell_s": task.probe_signed_dwell,
        "probe_clearance_m": task.probe_clearance,
        "probe_max_latency_s": task.probe_max_latency,
        "wheel_center_noise_std_m": task.wheel_center_noise_std,
        "wheel_center_calibration_bias_std_m": (
            task.wheel_center_calibration_bias_std
        ),
        "attempt": int(task.attempt),
    }


def collect_one(task: TraceTask) -> dict[str, object]:
    """Launch one plant case, sanitize its observation stream, and hash it."""

    run_dir = Path(task.run_dir)
    base = _base_result(task)
    result = launch_and_collect(
        experiment="terrain_estimator_fixed_trace",
        variant="Fixed dirt prior",
        controller_mode="standard",
        mpc_model="nn",
        nn_model="tire_force_static",
        terrain="dirt",
        path=task.path,
        speed=task.speed,
        bumpiness=task.bumpiness,
        seed=task.sim_seed,
        run_dir=run_dir,
        sim_port=task.sim_port,
        ctrl_port=task.ctrl_port,
        sim_time=task.sim_time,
        timeout=500.0,
        lead_in=task.lead_in,
        metric_start=min(10.0, 0.5 * task.sim_time),
        extra_args=fixed_controller_extra_args(task),
    )
    if result.status != "ok" or not result.diag_csv:
        base["status"] = "run_failed"
        base["failure"] = (
            f"Chrono collection failed (launch status={result.status!r}, "
            f"diagnostic={result.diag_csv!r})"
        )
        return base

    try:
        _audit_fixed_prior(result.diag_csv)
    except (
        TraceValidationError,
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        message = str(error)
        base["status"] = (
            "diagnostic_io_failure"
            if (
                message.startswith("cannot audit controller diagnostic")
                or isinstance(
                    error,
                    (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError),
                )
            )
            else "protocol_violation"
        )
        base["failure"] = message
        return base

    try:
        if task.terrain_id_probe:
            probe_metadata = _audit_probe_maneuver(result.diag_csv)
            base.update(probe_metadata)
            if not bool(probe_metadata["probe_audit_ok"]):
                base["status"] = "maneuver_failed"
                base["failure"] = str(probe_metadata["probe_audit_failure"])
                return base
        exact = _find_exact_observation_log(run_dir)
        destination = run_dir / "sensor_trace.csv"
        if exact is not None:
            metadata = sanitize_exact_observations(exact, destination)
            metadata["trace_quality"] = "exact_runtime_observations"
            metadata["source_path"] = str(exact.resolve())
        elif task.allow_approx_diag:
            metadata = sanitize_approximate_diagnostic(
                result.diag_csv,
                destination,
                allow_approximate=True,
            )
            metadata["source_path"] = str(Path(result.diag_csv).resolve())
        else:
            raise TraceValidationError(
                "terrain_observations.csv was not produced. Exact replay requires "
                "the controller's unrounded estimator-observation logger. Rerun "
                "with --allow-approx-diag only for explicitly labelled debug work."
            )
        base.update(metadata)
        base["status"] = "ok"
    except (
        TraceValidationError,
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        message = str(error)
        base["status"] = (
            "trace_io_failure"
            if (
                "was not produced" in message
                or message.startswith("cannot parse exact terrain observation log")
                or isinstance(
                    error,
                    (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError),
                )
            )
            else "protocol_violation"
        )
        base["failure"] = message
    return base


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "benchmarking" / "results" / f"terrain_estimator_traces_{stamp}"


def _write_collection_manifest(path: Path, args: argparse.Namespace) -> None:
    rows = [
        ("created_at", datetime.now().isoformat(timespec="seconds")),
        ("command", " ".join(sys.argv)),
        ("project_root", str(ROOT)),
        ("purpose", "fixed-dirt-prior estimator-disabled terrain sensor traces"),
    ]
    rows.extend((key, repr(value)) for key, value in sorted(vars(args).items()))
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def _portable_collection_path(value: object, collection_root: Path) -> object:
    """Store paths inside a collection relative to its relocatable root."""

    if not isinstance(value, str) or not value:
        return value
    path = Path(value).expanduser()
    try:
        return path.resolve().relative_to(collection_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _collection_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the immutable collection configuration used for recovery."""

    return {
        "schema_version": 1,
        "n": int(args.n),
        "base_port": int(args.base_port),
        "soil_seed": int(args.seed),
        "sim_seed_first": int(args.sim_seed),
        "mode": str(args.mode),
        "jitter": float(args.jitter),
        "path": str(args.path),
        "speed_mps": float(args.speed),
        "sim_time_s": float(args.time),
        "lead_in_m": float(args.lead_in),
        "bumpiness": int(args.bumpiness),
        "terrain_id_probe": bool(args.terrain_id_probe),
        "probe_target_alpha_rad": float(args.probe_target_alpha),
        "probe_slew_rate_radps": float(args.probe_slew_rate),
        "probe_signed_dwell_s": float(args.probe_signed_dwell),
        "probe_clearance_m": float(args.probe_clearance),
        "probe_max_latency_s": float(args.probe_max_latency),
        "wheel_center_noise_std_m": float(args.wheel_center_noise_std),
        "wheel_center_calibration_bias_std_m": float(
            args.wheel_center_calibration_bias_std
        ),
        "allow_approx_diag": bool(args.allow_approx_diag),
    }


def _task_for_index(
    output: Path,
    args: argparse.Namespace,
    index: int,
    *,
    attempt: int,
    recovery: bool,
) -> TraceTask:
    trace_id = f"trace_{index:04d}"
    return TraceTask(
        trace_id=trace_id,
        yaml_path=str((output / "truth_soils" / f"case_{index:04d}.yaml").resolve()),
        run_dir=str(
            (output / "raw" / trace_id / f"attempt_{attempt:02d}").resolve()
        ),
        sim_port=int(args.base_port) + 2 * index,
        ctrl_port=int(args.base_port) + 2 * index + 1,
        sim_seed=int(args.sim_seed) + index,
        path=str(args.path),
        speed=float(args.speed),
        sim_time=float(args.time),
        lead_in=float(args.lead_in),
        bumpiness=int(args.bumpiness),
        allow_approx_diag=bool(args.allow_approx_diag),
        excitation_args=_excitation_args(args),
        maneuver_label=(
            "terrain_id_probe" if args.terrain_id_probe else "passive"
        ),
        terrain_id_probe=bool(args.terrain_id_probe),
        probe_target_alpha=float(args.probe_target_alpha),
        probe_slew_rate=float(args.probe_slew_rate),
        probe_signed_dwell=float(args.probe_signed_dwell),
        probe_clearance=float(args.probe_clearance),
        probe_max_latency=float(args.probe_max_latency),
        wheel_center_noise_std=float(args.wheel_center_noise_std),
        wheel_center_calibration_bias_std=float(
            args.wheel_center_calibration_bias_std
        ),
        attempt=int(attempt),
        recovery=bool(recovery),
    )


def _initialize_collection(output: Path, args: argparse.Namespace) -> list[TraceTask]:
    (output / "raw").mkdir(parents=True)
    soil_dir = output / "truth_soils"
    soil_dir.mkdir()
    (output / "collection_config.json").write_text(
        json.dumps(_collection_config(args), indent=2) + "\n", encoding="utf-8"
    )
    _write_collection_manifest(output / "collection_manifest.csv", args)

    soils, nuisance_seed = generate_soils(
        args.n,
        mode=args.mode,
        seed=args.seed,
        jitter_fraction=args.jitter,
        n_lo=float(args.n_lo),
        n_hi=float(args.n_hi),
    )
    case_order_seed = int(args.seed) + 2_000_003
    permutation = np.random.default_rng(case_order_seed).permutation(len(soils))
    soils = [soils[int(index)] for index in permutation]
    truth_rows: list[dict[str, object]] = []
    for index, (n_true, soil) in enumerate(soils):
        trace_id = f"trace_{index:04d}"
        (soil_dir / f"case_{index:04d}.yaml").write_text(
            yaml.safe_dump(soil), encoding="utf-8"
        )
        truth_row: dict[str, object] = {
            "trace_id": trace_id,
            "n_true": float(n_true),
            "Kphi_true": float(soil["Kphi"]),
            "Kc_true": float(soil["Kc"]),
            "c_true": float(soil["cohesion"]),
            "phi_true_deg": float(soil["friction_angle"]),
            "k_true": float(soil["janosi_shear"]),
            "soil_draw_seed": int(args.seed),
            "nuisance_seed": int(nuisance_seed),
            "case_order_seed": case_order_seed,
            "nuisance_jitter_fraction": (
                0.0 if args.mode == "independent_n_phi" else float(args.jitter)
            ),
            "soil_mode": args.mode,
        }
        if args.mode == "independent_n_phi":
            manifold_soil = manifold_yaml_from_n(float(n_true))
            truth_row.update({
                "phi_draw_seed": int(nuisance_seed),
                "cohesion_draw_seed": int(nuisance_seed + 1),
                "cohesion_jitter_fraction": 0.15,
                "cohesion_multiplier_lo": INDEPENDENT_COHESION_MULTIPLIER_LO,
                "cohesion_multiplier_hi": INDEPENDENT_COHESION_MULTIPLIER_HI,
                "cohesion_multiplier_true": float(
                    soil["cohesion"] / manifold_soil["cohesion"]
                ),
            })
        truth_rows.append(truth_row)
    # Truth is a scorer-only sidecar and is never passed into collection/replay.
    truth_frame = pd.DataFrame(truth_rows)
    if args.mode == "independent_n_phi":
        truth_frame.to_csv(
            output / "truth.csv", index=False, float_format="%.17g"
        )
    else:
        # Manifold and LHS truth tables keep the default float formatting, so
        # their bytes stay stable across regenerations of the same collection.
        truth_frame.to_csv(output / "truth.csv", index=False)
    return [
        _task_for_index(output, args, index, attempt=1, recovery=False)
        for index in range(args.n)
    ]


def _write_attempt_tables(
    output: Path,
    latest: dict[str, dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> None:
    pd.DataFrame(latest.values()).sort_values("trace_id").to_csv(
        output / "trace_manifest.csv", index=False
    )
    pd.DataFrame(attempts).sort_values(["trace_id", "attempt"]).to_csv(
        output / "attempt_manifest.csv", index=False
    )


def _portable_result(row: dict[str, Any], output: Path) -> dict[str, Any]:
    portable = dict(row)
    for column in ("run_dir", "trace_path", "source_path"):
        portable[column] = _portable_collection_path(
            portable.get(column, ""), output
        )
    return portable


def _record_attempt(
    output: Path,
    row: dict[str, Any],
    task: TraceTask,
    latest: dict[str, dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> None:
    portable = _portable_result(row, output)
    latest[task.trace_id] = portable
    attempt_row = dict(portable)
    attempt_row["recorded_at"] = datetime.now().isoformat(timespec="seconds")
    attempt_row["recovery"] = bool(task.recovery)
    attempts.append(attempt_row)
    _write_attempt_tables(output, latest, attempts)


def _worker_failure(task: TraceTask, error: BaseException) -> dict[str, Any]:
    row = _base_result(task)
    row["status"] = "worker_exception"
    row["failure"] = repr(error)
    return row


def _run_tasks(
    output: Path,
    tasks: list[TraceTask],
    workers: int,
    latest: dict[str, dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> None:
    if not tasks:
        return
    first = tasks[0]
    try:
        first_row = collect_one(first)
    except Exception as error:  # pragma: no cover - launch boundary guard
        first_row = _worker_failure(first, error)
    _record_attempt(output, first_row, first, latest, attempts)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_tasks = {
            executor.submit(collect_one, task): task for task in tasks[1:]
        }
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            try:
                row = future.result()
            except Exception as error:  # pragma: no cover - process boundary guard
                row = _worker_failure(task, error)
            _record_attempt(output, row, task, latest, attempts)


def _load_recovery(
    output: Path, args: argparse.Namespace
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[TraceTask]]:
    try:
        stored = json.loads(
            (output / "collection_config.json").read_text(encoding="utf-8")
        )
        traces = pd.read_csv(output / "trace_manifest.csv")
        attempt_frame = pd.read_csv(output / "attempt_manifest.csv")
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise SystemExit(f"cannot recover static collection: {error}") from error
    if stored != _collection_config(args):
        raise SystemExit("recovery arguments do not match the stored collection config")
    if traces["trace_id"].duplicated().any():
        raise SystemExit("trace manifest has duplicate trace IDs")
    latest = {
        str(row["trace_id"]): row.to_dict() for _, row in traces.iterrows()
    }
    attempts = attempt_frame.to_dict(orient="records")
    tasks: list[TraceTask] = []
    for index in range(args.n):
        trace_id = f"trace_{index:04d}"
        row = latest.get(trace_id)
        if row is None:
            prior_attempt = 0
            status = "worker_exception"
        else:
            prior_attempt = int(row["attempt"])
            status = str(row["status"])
        if status == "ok":
            continue
        if status not in _RECOVERABLE_STATUSES:
            raise SystemExit(
                f"{trace_id} status {status!r} is a protocol failure, not "
                "recoverable infrastructure"
            )
        tasks.append(
            _task_for_index(
                output,
                args,
                index,
                attempt=prior_attempt + 1,
                recovery=True,
            )
        )
    return latest, attempts, tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--base-port", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=42, help="soil draw seed")
    parser.add_argument(
        "--sim-seed", type=int, default=720,
        help="base sensor-noise seed; trace i uses base+i so soil cases are independent",
    )
    parser.add_argument(
        "--mode",
        choices=["manifold", "lhs", "independent_n_phi"],
        default="manifold",
    )
    parser.add_argument("--jitter", type=float, default=0.10)
    parser.add_argument("--path", default="sinusoidal")
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--bumpiness", type=int, default=0,
                        choices=range(0, 11),
                        help="Terrain bumpiness level passed to the plant "
                             "(0 = flat, the frozen flat-evidence protocol).")
    parser.add_argument("--n-lo", type=float, default=INDEPENDENT_N_LO,
                        help="Lower bound of the stratified exponent draw. The frozen "
                             "paper span starts at 0.55, which leaves the deployed clay "
                             "edge (0.50) untested; an edge-band collection lowers this.")
    parser.add_argument("--n-hi", type=float, default=INDEPENDENT_N_HI)
    parser.add_argument("--sine-amplitude", type=float, default=None,
                        help="Lateral excitation: sinusoidal path amplitude in metres.")
    parser.add_argument("--sine-wavelength", type=float, default=None,
                        help="Lateral excitation: sinusoidal path wavelength in metres.")
    parser.add_argument("--speed-osc-amplitude", type=float, default=0.0,
                        help="Longitudinal excitation: fractional sinusoidal modulation "
                             "of the speed reference, applied under the g--g grip cap.")
    parser.add_argument("--speed-osc-period-s", type=float, default=4.0,
                        help="Period in seconds of the speed-reference oscillation.")
    parser.add_argument(
        "--time", type=float, default=None,
        help="Run duration (default: 30 s with probe, 20 s passive).",
    )
    parser.add_argument(
        "--lead-in", type=float, default=None,
        help="Straight lead-in (default: 30 m with probe, 5 m passive).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--recover",
        action="store_true",
        help=(
            "Resume an existing collection and rerun only infrastructure-failed "
            "cases with the same seeds, ports, and soil files."
        ),
    )
    parser.add_argument(
        "--terrain-id-probe", action="store_true",
        help="Enable the bounded, sensor-gated symmetric identification probe.",
    )
    parser.add_argument("--probe-target-alpha", type=float, default=0.10)
    parser.add_argument("--probe-slew-rate", type=float, default=0.40)
    parser.add_argument("--probe-signed-dwell", type=float, default=0.15)
    parser.add_argument("--probe-clearance", type=float, default=35.0)
    parser.add_argument("--probe-max-latency", type=float, default=0.30)
    parser.add_argument(
        "--wheel-center-noise-std", type=float, default=0.01,
        help="Independent per-wheel elevation noise stdev in metres.",
    )
    parser.add_argument(
        "--wheel-center-calibration-bias-std", type=float, default=0.003,
        help=(
            "Per-run common residual height bias stdev after a known-plane "
            "calibration, in metres."
        ),
    )
    parser.add_argument(
        "--allow-approx-diag",
        action="store_true",
        help="Development only: whitelist and reconstruct the rounded controller "
             "diagnostic when terrain_observations.csv is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.n = 1
        args.workers = 1
    if args.time is None:
        args.time = 30.0 if args.terrain_id_probe else 20.0
    if args.lead_in is None:
        args.lead_in = 30.0 if args.terrain_id_probe else 5.0
    if args.n < 1 or args.workers < 1:
        raise SystemExit("--n and --workers must be positive")
    if args.time <= 0.0 or args.speed <= 0.0:
        raise SystemExit("--time and --speed must be positive")
    if args.lead_in < 0.0:
        raise SystemExit("--lead-in must be non-negative")
    if (
        args.wheel_center_noise_std < 0.0
        or args.wheel_center_calibration_bias_std < 0.0
    ):
        raise SystemExit("wheel-center noise and calibration-bias stdev must be non-negative")
    if args.terrain_id_probe and (
        args.probe_target_alpha <= 0.0
        or args.probe_slew_rate <= 0.0
        or args.probe_signed_dwell <= 0.0
        or args.probe_clearance < 0.0
        or args.probe_max_latency < 0.0
    ):
        raise SystemExit("terrain probe parameters must be non-negative and have positive target/slew")

    if args.recover and args.output_dir is None:
        raise SystemExit("--recover requires --output-dir")
    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    args.output_dir = output_dir
    if args.recover:
        if not output_dir.is_dir():
            raise SystemExit(f"recovery output directory does not exist: {output_dir}")
        latest, attempts, tasks = _load_recovery(output_dir, args)
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise SystemExit(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks = _initialize_collection(output_dir, args)
        latest: dict[str, dict[str, Any]] = {}
        attempts: list[dict[str, Any]] = []

    _run_tasks(output_dir, tasks, args.workers, latest, attempts)
    manifest = pd.read_csv(output_dir / "trace_manifest.csv")
    failures = manifest[manifest["status"] != "ok"]
    print(f"wrote {output_dir / 'trace_manifest.csv'}")
    print(f"wrote {output_dir / 'truth.csv'} (scorer-only ground truth)")
    if not failures.empty:
        print(f"{len(failures)}/{args.n} trace collections were not replayable:")
        for row in failures.itertuples(index=False):
            print(f"  {row.trace_id}: {getattr(row, 'failure', '')}")
        return 2
    quality = manifest["trace_quality"].value_counts().to_dict()
    print(f"collected {len(manifest)} fixed-controller traces: {quality}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
