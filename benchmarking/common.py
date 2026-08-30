#!/usr/bin/env python3
"""Shared machinery for the manuscript's reproducible closed-loop experiments.

Every study in this directory launches its runs, records its provenance, parses
its diagnostics, and draws its figures through this module, so that a result is
produced the same way regardless of which study produced it. The module owns the
frozen estimator contracts the studies select between, the launch and log
isolation that makes parallel runs independent, the defensive parsers that keep
one malformed run from destroying a matrix, and the shared figure helpers.
"""

from __future__ import annotations

import argparse
import csv
import ast
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from paper_provenance import downstream_repository_provenance
except ModuleNotFoundError:  # package import in tests/tools
    from benchmarking.paper_provenance import downstream_repository_provenance


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = PROJECT_ROOT / "simulation"
LAUNCHER = SIM_DIR / "runtime" / "launch_decoupled.py"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
# The deployed tire-force surrogate, used by every study that does not name a
# checkpoint explicitly. It maps the slip and load state, the steering rate, and
# the six Bekker-Mohr soil parameters to tire force, and is supervised solely by
# the controlled single-tire Chrono SCM test stand at the commanded operating
# point, so no vehicle trajectory enters its training.
DEFAULT_NN_MODEL = "tire_force_static"

# Environment variables recorded into every result manifest because they change
# the configuration under test. ACADOS_MPC_BUILD_ROOT selects which compiled
# solver cache a run may reuse; HIL_SIM_EXTRA is appended to every launch and is
# what `run.py` uses to select the plant per command.
ENV_RECORDED_IN_MANIFEST = ("HIL_SIM_EXTRA", "ACADOS_MPC_BUILD_ROOT")

# HIL_SIM_EXTRA carries flags for launch_decoupled, which forwards each one
# to the process it belongs to. A replay driver that spawns chrono_sim_node
# directly must apply only the simulator's share: the controller-side flags
# configure a process a replay run does not start, and the simulator's
# argparser exits on them (which is exactly how the 2026-08-26 convoy and
# latency generations died, 180 runs at rc=2 in two seconds each). Every
# flag run.py can put in HIL_SIM_EXTRA must be classified here, with its
# argument count; an unclassified flag raises rather than being guessed at.
_HIL_SIM_NODE_FLAGS = {"--simple-powertrain": 0}
_HIL_CONTROLLER_ONLY_FLAGS = {"--ff-drag-surrogate": 0, "--dob-ki": 1,
                              "--dob-max": 1}


def sim_node_flags_from_hil_extra() -> list[str]:
    """The chrono_sim_node share of HIL_SIM_EXTRA, for direct spawners."""
    tokens = os.environ.get("HIL_SIM_EXTRA", "").split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        flag = tokens[i]
        if flag in _HIL_SIM_NODE_FLAGS:
            arity = _HIL_SIM_NODE_FLAGS[flag]
            out.extend(tokens[i:i + 1 + arity])
            i += 1 + arity
        elif flag in _HIL_CONTROLLER_ONLY_FLAGS:
            i += 1 + _HIL_CONTROLLER_ONLY_FLAGS[flag]
        else:
            raise ValueError(
                f"unclassified HIL_SIM_EXTRA flag {flag!r}: add it to "
                "_HIL_SIM_NODE_FLAGS or _HIL_CONTROLLER_ONLY_FLAGS in "
                "benchmarking/common.py so replay drivers know whether the "
                "simulator accepts it")
    return out

# Frozen contract of the matched scalar estimator, which infers the single
# manifold coordinate under the same excitation gating as the joint estimator
# and reads the friction angle off the soil manifold. It is the comparison arm
# that isolates what estimating the two soil coordinates independently
# contributes. Its identifier and values are immutable: they name this method in
# every result manifest that has already been collected, so relabelling them
# would silently reinterpret those results, and its static-map dependency must
# not be attached to any other estimator.
PARENT_ESTIMATOR_BACKEND = "scalar_parent"
PARENT_ESTIMATOR_CONTRACT: dict[str, Any] = {
    "contract_version": "matched_parent_profile_live",
    "backend": PARENT_ESTIMATOR_BACKEND,
    "controller_rate_model": "tire_force_static",
    "force_model_dir": "nn_models/tire_force_static_parent",
    "initial_prior": "dirt",
    "controller_min_confidence": 0.0,
    "n_grid_size": 41,
    "student_dof": 4.0,
    "update_interval": 1,
    "block_dt_s": 0.5,
    "history_horizon_s": 8.0,
    "min_concurrent_windows": 12,
    "min_window_samples": 4,
    "r_ax_mps2": 0.35,
    "r_ay_mps2": 0.30,
    "min_information": 0.20,
    "min_yaw_rate_rms_radps": 0.015,
    "min_speed_mps": 2.5,
    "max_abs_slip_angle_rad": 0.35,
    "enforce_rig_feature_envelope": True,
    "slip_mode": "average",
    "fixed_kappa": 0.05,
    "slip_angle_rate_mode": "zero",
    "force_gain_prior_std": 0.04,
    "ax_bias_prior_std_mps2": 0.10,
    "ay_bias_prior_std_mps2": 0.05,
    "force_gain_bounds": [0.70, 1.30],
    "acceleration_bias_bound_mps2": 0.30,
    "profile_iterations": 8,
}

# Frozen contract of GRIT, the estimator that infers the Bekker exponent n and
# the friction angle phi independently and online from vehicle motion. The
# runtime constructor owns these settings rather than exposing them as
# configurable flags, so a study selects a contract instead of assembling one
# and no run can quietly differ from another in a scoring or gating parameter.
GRIT_ESTIMATOR_BACKEND = "grit"
GRIT_ESTIMATOR_CONTRACT: dict[str, Any] = {
    # The gates, grids, and nuisance bounds below were fixed before the
    # confirmation evidence was collected. Selected settings and their reasons:
    # the lateral-acceleration scale r_ay is chosen by split-half selection and
    # confirmed on a fresh seed; the n range extends below the clay anchor by
    # holding the manifold, so that soils softer than any anchor remain
    # representable; and the phi grid step is at most half the estimator's own
    # phi uncertainty, so grid resolution never limits the reported accuracy.
    # "controller_rate_model" identifies the tire model the controller runs
    # while the estimator is under test.
    "contract_version": "independent_n_phi_joint_profile",
    "accepted_snapshot_version": "grit_accepted",
    "promotion_status": "active_paper_backend",
    "backend": GRIT_ESTIMATOR_BACKEND,
    "controller_rate_model": "tire_force_static",
    "force_model_dir": "nn_models/tire_force_rate",
    "initial_prior": "dirt",
    "controller_min_confidence": 0.20,
    "publication_max_evidence_age_s": 3.5,
    "publication_min_observability_rank": 2,
    "publication_min_observability_singular_value": 0.10,
    "publication_boundary_mass_limit": 0.25,
    "fallback_n": 0.50,
    "fallback_phi_deg": 13.0,
    "control_min_phi_deg": 10.0,
    "fallback_policy": "fixed_control_feasible_low_grip_endpoint",
    "output_names": ["n", "phi"],
    "requires_ground_datum": False,
    "truth_inputs": "none",
    "n_grid_size": 41,
    "n_bounds": [0.40, 1.10],
    "manifold_soft_floor": 0.40,
    "manifold_soft_mode": "hold",
    "phi_grid_size": 17,
    "phi_bounds_deg": [6.0, 37.8],
    "cohesion_multiplier_bounds": [0.70, 1.30],
    "cohesion_grid_size": 1,
    "cohesion_prior_std": 0.20,
    "student_dof": 4.0,
    "smoothing_alpha": 1.0,
    "update_interval": 1,
    "block_dt_s": 0.5,
    "history_horizon_s": 8.0,
    "min_concurrent_windows": 8,
    "min_window_samples": 4,
    "r_ax_mps2": 0.35,
    "r_ay_mps2": 0.45,
    "min_information": 0.20,
    "min_joint_information": 0.20,
    "min_n_information": 0.0,
    "min_phi_information": 0.0,
    "min_yaw_rate_rms_radps": 0.015,
    "min_speed_mps": 2.5,
    "max_abs_slip_angle_rad": 0.35,
    "enforce_rig_feature_envelope": True,
    "slip_mode": "average",
    "fixed_kappa": 0.05,
    "rate_mode": "zero",
    "block_alpha_rate": False,
    "load_transfer_mode": "static",
    "force_gain_prior_std": 0.04,
    "ax_bias_prior_std_mps2": 0.10,
    "ay_bias_prior_std_mps2": 0.05,
    "force_gain_bounds": [0.70, 1.30],
    "acceleration_bias_bound_mps2": 0.30,
    "profile_iterations": 8,
    "min_observability_rank": 2,
    "min_observability_singular_value": 0.10,
    "boundary_warning_mass": 0.25,
    "posterior_summary": "mean",
    "max_final_update_age_s": 3.5,
}

# The estimator every published run uses. Runtime and study code select through
# this alias, so the deployed estimator is named in exactly one place. The
# scalar identifier is not reused for it: that identifier names the scalar method
# in manifests already collected and carries the static-map dependency the
# scalar method alone requires.
RIG_ACTIVE_ESTIMATOR_BACKEND = GRIT_ESTIMATOR_BACKEND


def default_ros_workers() -> int:
    """Worker count that keeps a matrix from contending with itself.

    Each run occupies a busy simulator process and a busy controller process, so
    a worker costs roughly two cores. Oversubscription does more than lengthen
    the wall clock: the controller is event-driven, so once solves begin to miss
    the message period the closed-loop trajectory itself changes, and the
    affected matrix has to be discarded rather than merely repeated. Because the
    corruption is silent, appearing only as a plausible shift in the metrics,
    the default deliberately leaves about a third of the machine free.
    """
    cores = os.cpu_count() or 8
    return max(1, min(12, cores // 3))


def bounded_ros_workers(value: object) -> int:
    """Parse a worker count that cannot wrap the launcher's DDS-domain pool."""

    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "workers must be an integer in [1, 101]"
        ) from exc
    if not 1 <= workers <= 101:
        raise argparse.ArgumentTypeError("workers must be in [1, 101]")
    return workers


def require_active_joint_estimator(backend: object) -> str:
    """Fail closed unless a published run selects the deployed estimator.

    A study that silently ran a comparison estimator in the deployed arm's place
    would produce results indistinguishable in shape from a valid matrix, so the
    check is made at launch rather than at scoring time.
    """

    value = str(backend)
    if (
        RIG_ACTIVE_ESTIMATOR_BACKEND != GRIT_ESTIMATOR_BACKEND
        or value != RIG_ACTIVE_ESTIMATOR_BACKEND
    ):
        raise RuntimeError(
            "published benchmarks require the deployed grit "
            "estimator; the other estimators are comparison arms only"
        )
    return value


PATHS = ("sinusoidal", "lane_change", "right_left")
TERRAINS = ("clay", "dirt", "sand")
SPEEDS = (5.0, 7.0, 9.0)
BUMPS = (0, 4, 8)

PATH_ROCK_ZONES: dict[str, dict[str, tuple[float, float]]] = {
    # 'straight' is the human-driven hazard field. Rocks span the full width of
    # the course, beyond the reach of any swerve, so no lateral bypass exists
    # and the operator must thread a route rather than drive around the field.
    "straight": {"x": (10.0, 98.0), "y": (-16.0, 16.0)},
    "sinusoidal": {"x": (12.0, 50.0), "y": (-3.0, 3.0)},
    "lane_change": {"x": (15.0, 50.0), "y": (-1.0, 4.0)},
    "double_lane_change": {"x": (15.0, 60.0), "y": (-1.0, 4.0)},
    "right_left": {"x": (10.0, 22.0), "y": (-3.0, 3.0)},
}

RX_SIM_COMPLETE = re.compile(
    r"Simulation complete:\s*([\d.]+)s in ([\d.]+)s\s*\(RT factor ([\d.]+)x\)"
)
RX_COLLISIONS = re.compile(
    r"(?:Chrono body-contact collisions|Hard collisions):\s*(\d+)\s+"
    r"Near misses:\s*(\d+)"
)


@dataclass
class RunResult:
    experiment: str
    variant: str
    controller_mode: str
    mpc_model: str
    nn_model: str
    terrain: str
    path: str
    speed_mps: float
    bumpiness: int
    seed: int
    run_dir: str
    status: str = "ok"
    rc: int = 0
    wall_s: float = math.nan
    sim_s: float = math.nan
    rt_factor: float = math.nan
    diag_csv: str = ""
    collision_csv: str = ""
    shield_csv: str = ""
    n_samples: int = 0
    rms_cte_m: float = math.nan
    max_abs_cte_m: float = math.nan
    mean_abs_cte_m: float = math.nan
    mean_speed_mps: float = math.nan
    p95_speed_mps: float = math.nan
    speed_ratio: float = math.nan
    mean_solve_ms: float = math.nan
    p99_solve_ms: float = math.nan
    progress_m: float = math.nan
    final_x_m: float = math.nan
    final_y_m: float = math.nan
    collisions: int = 0
    collision_source: str = "chrono_body_contact"
    near_misses: int = 0
    min_clearance_m: float = math.nan
    intervention_rate_pct: float = math.nan
    filter_solve_ms: float = math.nan
    mean_abs_dsteer: float = math.nan
    mean_abs_dthrottle: float = math.nan
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def ensure_runtime_env() -> None:
    os.environ.setdefault(
        "ACADOS_SOURCE_DIR",
        os.path.expanduser("~/Documents/sbel/acados"),
    )


def timestamped_result_dir(prefix: str) -> Path:
    # Second-resolution timestamps collide when two runs start in the same
    # second (e.g. a fast-failing sub-run, or back-to-back launches). Append a
    # numeric suffix on collision instead of raising FileExistsError.
    base = f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    out = RESULTS_ROOT / base
    n = 1
    while out.exists():
        out = RESULTS_ROOT / f"{base}_{n}"
        n += 1
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir()
    (out / "figures").mkdir()
    return out


def write_manifest(out_dir: Path, args: Any, description: str) -> None:
    rows = [
        ("created_at", datetime.now().isoformat(timespec="seconds")),
        ("project_root", str(PROJECT_ROOT)),
        ("description", description),
        ("command", " ".join(sys.argv)),
    ]
    # Environment variables that change what was simulated, not merely where it ran.
    # HIL_SIM_EXTRA is appended to every launch, so a sweep started directly
    # rather than through run.py silently exercises a different plant while its
    # command line and every result column look identical. Unset is itself a
    # configuration, so record it as such rather than omitting the key.
    for name in ENV_RECORDED_IN_MANIFEST:
        value = os.environ.get(name)
        rows.append((f"env.{name}", "<unset>" if value is None else value))
    # Source provenance. Without it a run's "collected on a clean tree" status
    # is an assertion no artifact can support after the fact, and the publish
    # boundary has nothing to verify. Recording it here covers every study that
    # writes a manifest rather than only the estimator matrices that ask for it
    # explicitly. A failure to read git must not lose a completed matrix, so it
    # is recorded as a failure rather than raised.
    # Source provenance. Without it a run's "collected on a clean tree" status
    # is an assertion no artifact can support after the fact, and the publish
    # boundary has nothing to verify. Recording it here covers every study that
    # writes a manifest rather than only those that ask for it explicitly.
    #
    # The measured values win over caller-supplied ones. Letting an argparse
    # namespace override them would make forging eligibility a one-line
    # change, and adversarial review demonstrated exactly that; a study that
    # measures its own provenance earlier (the estimator ablation does) gets
    # the same values re-measured here, so nothing is lost. Duplicate keys are
    # still avoided -- the manifest reader rejects a file with any repeated
    # key outright.
    #
    # A failure to read git must not lose a completed matrix, so it is
    # recorded as ineligible rather than raised; the caller's rows then stand
    # since there is nothing measured to prefer.
    provenance_rows: list[tuple[str, str]] = []
    try:
        provenance = downstream_repository_provenance()
        for key in ("code_git_head", "tracked_worktree_dirty",
                    "uncommitted_source_files", "paper_evidence_eligible"):
            provenance_rows.append((key, repr(provenance[key])))
        for relative, digest in sorted(provenance["source_sha256"].items()):
            provenance_rows.append((f"source_sha256.{relative}", digest))
    except Exception as error:  # noqa: BLE001 - recorded, never fatal
        provenance_rows.append(("paper_evidence_eligible", repr(False)))
        provenance_rows.append(("provenance_error", repr(str(error))))
    reserved = {key for key, _ in provenance_rows}
    for k, v in sorted(vars(args).items()):
        if k not in reserved:
            rows.append((k, repr(v)))
    rows.extend(provenance_rows)
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def base_launch_args(
    *,
    terrain: str,
    path: str,
    speed: float,
    sim_time: float,
    bumpiness: int,
    seed: int,
    run_dir: Path,
    sim_port: int,
    ctrl_port: int,
    lead_in: float = 5.0,
    rocks: int = 0,
    no_plot: bool = True,
) -> list[str]:
    cmd = [
        sys.executable, "-u", str(LAUNCHER),
        "--transport", "ros",
        "--terrain", terrain,
        "--path", path,
        "--speed", str(speed),
        "--time", str(sim_time),
        "--lead-in", str(lead_in),
        "--bumpiness", str(bumpiness),
        "--rocks", str(rocks),
        "--rock-seed", str(seed),
        "--sim-seed", str(seed),
        "--sim-port", str(sim_port),
        "--ctrl-port", str(ctrl_port),
        "--plot-dir", str(run_dir),
        "--no-vis",
    ]
    if no_plot:
        cmd.append("--no-plot")
    if rocks > 0:
        zone = PATH_ROCK_ZONES.get(path, PATH_ROCK_ZONES["sinusoidal"])
        cmd += [
            "--rock-zone-x", str(zone["x"][0]), str(zone["x"][1]),
            "--rock-zone-y", str(zone["y"][0]), str(zone["y"][1]),
            "--rock-size", "0.8", "1.8",
        ]
    return cmd


def estimator_runtime_args(backend: str) -> list[str]:
    """Return the frozen runtime arguments for one of the online estimators.

    ``RIG_ACTIVE_ESTIMATOR_BACKEND`` names the deployed estimator; the matched
    scalar estimator stays selectable so that studies can run it as a comparison
    arm under the same launch path.
    """

    if backend == GRIT_ESTIMATOR_BACKEND:
        # The runtime constructor owns this estimator's frozen settings, so the
        # command line carries no scoring or gating parameter that a run could
        # differ in. The two arguments passed here are the prior it starts from
        # and the confidence threshold at which the controller applies an
        # estimate, both of which the studies vary by design.
        contract = GRIT_ESTIMATOR_CONTRACT
        return [
            "--terrain-estimator-prior", str(contract["initial_prior"]),
            "--te-min-confidence", str(contract["controller_min_confidence"]),
        ]
    if backend != PARENT_ESTIMATOR_BACKEND:
        return []
    contract = PARENT_ESTIMATOR_CONTRACT
    gain_lo, gain_hi = contract["force_gain_bounds"]
    return [
        "--terrain-estimator-prior", str(contract["initial_prior"]),
        "--te-min-confidence", str(contract["controller_min_confidence"]),
        "--parent-grid-size", str(contract["n_grid_size"]),
        "--parent-student-dof", str(contract["student_dof"]),
        "--ukf-model-dir", str(contract["force_model_dir"]),
        "--estimator-update-interval", str(contract["update_interval"]),
        "--estimator-block-dt", str(contract["block_dt_s"]),
        "--estimator-horizon", str(contract["history_horizon_s"]),
        "--estimator-min-windows", str(contract["min_concurrent_windows"]),
        "--estimator-min-window-samples", str(contract["min_window_samples"]),
        "--estimator-r-ax", str(contract["r_ax_mps2"]),
        "--estimator-r-ay", str(contract["r_ay_mps2"]),
        "--estimator-min-information", str(contract["min_information"]),
        "--estimator-min-yaw-rate-rms",
        str(contract["min_yaw_rate_rms_radps"]),
        "--estimator-min-speed", str(contract["min_speed_mps"]),
        "--estimator-max-abs-alpha",
        str(contract["max_abs_slip_angle_rad"]),
        "--estimator-enforce-feature-envelope",
        "--estimator-slip-mode", str(contract["slip_mode"]),
        "--estimator-fixed-kappa", str(contract["fixed_kappa"]),
        "--estimator-rate-mode", str(contract["slip_angle_rate_mode"]),
        "--estimator-force-gain-std",
        str(contract["force_gain_prior_std"]),
        "--estimator-ax-bias-std",
        str(contract["ax_bias_prior_std_mps2"]),
        "--estimator-ay-bias-std",
        str(contract["ay_bias_prior_std_mps2"]),
        "--estimator-force-gain-min", str(gain_lo),
        "--estimator-force-gain-max", str(gain_hi),
        "--estimator-acceleration-bias-bound",
        str(contract["acceleration_bias_bound_mps2"]),
        "--estimator-profile-iterations", str(contract["profile_iterations"]),
    ]


def estimator_contract(backend: str) -> dict[str, Any]:
    """JSON-safe estimator contract for result manifests."""

    if backend == GRIT_ESTIMATOR_BACKEND:
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in GRIT_ESTIMATOR_CONTRACT.items()
        }
    if backend == PARENT_ESTIMATOR_BACKEND:
        # Round-trip through simple containers so callers cannot mutate the
        # process-global contract through the nested bounds list.
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in PARENT_ESTIMATOR_CONTRACT.items()
        }
    return {
        "contract_version": "backend_compatibility",
        "backend": str(backend),
    }


def controller_tire_force_truth_rows(diag: pd.DataFrame) -> int:
    """Count controller rows containing any simulator tire-force payload.

    The deployment contract requires these audit columns to be empty.  Treat
    every non-empty cell as a truth-channel violation, including malformed
    strings that cannot be converted to a number.  Coercing malformed payloads
    to ``NaN`` would incorrectly turn a corrupt/non-auditable channel into proof
    that no truth reached the controller.
    """

    force_columns = [
        column for column in (
            "actual_Fx_front", "actual_Fx_rear",
            "actual_Fy_front", "actual_Fy_rear",
        )
        if column in diag.columns
    ]
    if not force_columns:
        return 0

    def _present(value: object) -> bool:
        try:
            if bool(pd.isna(value)):
                return False
        except (TypeError, ValueError):
            # A non-scalar object is malformed and therefore present.
            return True
        return not (isinstance(value, str) and value == "")

    present = diag[force_columns].map(_present)
    return int(present.any(axis=1).sum())


def parent_estimator_diagnostics(
    diag: pd.DataFrame | None,
    *,
    backend: str,
    enabled: bool,
) -> dict[str, Any]:
    """Extract truth-independent publication/readiness diagnostics.

    The diagnostic is applicable only to an enabled profiled estimator.  A
    static comparison arm is therefore not mislabeled as an estimator
    abstention.  Times are elapsed from the first finite controller sample.
    """

    applicable = bool(enabled and backend == PARENT_ESTIMATOR_BACKEND)
    required_windows = int(
        PARENT_ESTIMATOR_CONTRACT["min_concurrent_windows"]
    )
    output: dict[str, Any] = {
        "profile_estimator_diagnostics_applicable": applicable,
        "profile_estimator_diagnostics_complete": not applicable,
        "profile_estimator_diagnostics_error": "",
        "profile_estimator_required_concurrent_windows": required_windows,
        "profile_estimator_publication_ready": False,
        "profile_estimator_publication_applied": False,
        "profile_estimator_abstained": False,
        "profile_estimator_readiness_rows": 0,
        "profile_estimator_update_rows": 0,
        "profile_estimator_time_to_first_ready_s": None,
        "profile_estimator_time_to_first_update_s": None,
        "profile_estimator_max_concurrent_windows": 0,
        "profile_estimator_lifetime_accepted_windows": 0,
        "profile_estimator_lifetime_rejected_windows": 0,
        "profile_estimator_profile_force_gain_final": None,
        "profile_estimator_profile_ax_bias_final_mps2": None,
        "profile_estimator_profile_ay_bias_final_mps2": None,
        "profile_estimator_profile_bound_hits_max": 0,
        "profile_estimator_feature_envelope_excursions_max": 0,
        "profile_estimator_readiness_consistent": True,
    }
    if not applicable:
        return output
    output["profile_estimator_abstained"] = True
    if diag is None or diag.empty:
        output["profile_estimator_diagnostics_error"] = "missing_or_empty_controller_diag"
        return output

    required = {
        "sim_time",
        "terrain_update_applied",
        "terrain_dynamics_active",
        "terrain_dynamics_windows",
        "terrain_accepted_dynamics_windows",
        "terrain_rejected_dynamics_windows",
        "terrain_profile_force_gain",
        "terrain_profile_ax_bias",
        "terrain_profile_ay_bias",
        "terrain_profile_bound_hits",
        "terrain_feature_envelope_excursions",
        # Empty columns are the auditable proof that the controller packet did
        # not contain simulator tire-force truth.  Missing columns are not
        # treated as equivalent to an audited all-empty channel.
        "actual_Fx_front",
        "actual_Fx_rear",
        "actual_Fy_front",
        "actual_Fy_rear",
    }
    missing = sorted(required - set(diag.columns))
    if missing:
        output["profile_estimator_diagnostics_error"] = (
            "missing_columns:" + ",".join(missing)
        )
        return output

    telemetry_columns = required - {
        "actual_Fx_front", "actual_Fx_rear",
        "actual_Fy_front", "actual_Fy_rear",
    }
    numeric = {
        column: pd.to_numeric(diag[column], errors="coerce")
        for column in telemetry_columns
    }
    times = numeric["sim_time"]
    nonfinite = sorted(
        column for column, values in numeric.items()
        if not np.isfinite(values.to_numpy(dtype=float)).all()
    )
    if nonfinite:
        output["profile_estimator_diagnostics_error"] = (
            "nonfinite_columns:" + ",".join(nonfinite)
        )
        return output
    if times.empty:
        output["profile_estimator_diagnostics_error"] = "no_finite_sim_time"
        return output

    binary_columns = ("terrain_update_applied", "terrain_dynamics_active")
    invalid_binary = sorted(
        column for column in binary_columns
        if not numeric[column].isin((0.0, 1.0)).all()
    )
    if invalid_binary:
        output["profile_estimator_diagnostics_error"] = (
            "invalid_binary_columns:" + ",".join(invalid_binary)
        )
        return output

    counter_columns = (
        "terrain_dynamics_windows",
        "terrain_accepted_dynamics_windows",
        "terrain_rejected_dynamics_windows",
        "terrain_profile_bound_hits",
        "terrain_feature_envelope_excursions",
    )
    invalid_counters = sorted(
        column for column in counter_columns
        if (
            (numeric[column] < 0.0).any()
            or not np.allclose(
                numeric[column].to_numpy(dtype=float),
                np.rint(numeric[column].to_numpy(dtype=float)),
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    )
    if invalid_counters:
        output["profile_estimator_diagnostics_error"] = (
            "invalid_counter_columns:" + ",".join(invalid_counters)
        )
        return output

    gain_lower, gain_upper = PARENT_ESTIMATOR_CONTRACT["force_gain_bounds"]
    bias_bound = float(
        PARENT_ESTIMATOR_CONTRACT["acceleration_bias_bound_mps2"]
    )
    if not numeric["terrain_profile_force_gain"].between(
        float(gain_lower), float(gain_upper), inclusive="both"
    ).all():
        output["profile_estimator_diagnostics_error"] = "force_gain_out_of_bounds"
        return output
    for column in ("terrain_profile_ax_bias", "terrain_profile_ay_bias"):
        if not numeric[column].between(-bias_bound, bias_bound, inclusive="both").all():
            output["profile_estimator_diagnostics_error"] = (
                "acceleration_bias_out_of_bounds:" + column
            )
            return output

    if not times.is_monotonic_increasing:
        output["profile_estimator_diagnostics_error"] = "nonmonotonic_sim_time"
        return output
    for column in (
        "terrain_accepted_dynamics_windows",
        "terrain_rejected_dynamics_windows",
        "terrain_feature_envelope_excursions",
    ):
        if (numeric[column].diff().dropna() < 0.0).any():
            output["profile_estimator_diagnostics_error"] = (
                "nonmonotonic_counter:" + column
            )
            return output

    completed = (
        numeric["terrain_accepted_dynamics_windows"]
        + numeric["terrain_rejected_dynamics_windows"]
    )
    if float(completed.iloc[-1]) <= 0.0:
        output["profile_estimator_diagnostics_error"] = "no_classified_dynamics_windows"
        return output

    finite_times = times
    time_origin = float(finite_times.min())
    ready = numeric["terrain_dynamics_active"].fillna(0.0) > 0.0
    updated = numeric["terrain_update_applied"].fillna(0.0) > 0.0

    def _first_elapsed(mask: pd.Series) -> float | None:
        candidates = times[mask & np.isfinite(times)]
        if candidates.empty:
            return None
        return float(max(0.0, float(candidates.min()) - time_origin))

    def _finite_max_int(column: str) -> int:
        values = numeric[column]
        finite = values[np.isfinite(values)]
        return int(finite.max()) if not finite.empty else 0

    def _last_finite(column: str) -> float | None:
        values = numeric[column]
        finite = values[np.isfinite(values)]
        return float(finite.iloc[-1]) if not finite.empty else None

    max_windows = _finite_max_int("terrain_dynamics_windows")
    publication_ready = bool(ready.any())
    publication_applied = bool(updated.any())
    per_row_ready = numeric["terrain_dynamics_windows"] >= required_windows
    readiness_consistent = bool(
        ready.eq(per_row_ready).all()
        and not (updated & ~ready).any()
        and int(updated.sum()) <= int(ready.sum())
        and max_windows
        <= _finite_max_int("terrain_accepted_dynamics_windows")
    )
    output.update({
        "profile_estimator_diagnostics_complete": readiness_consistent,
        "profile_estimator_diagnostics_error": (
            "" if readiness_consistent else "readiness_or_update_inconsistent"
        ),
        "profile_estimator_publication_ready": publication_ready,
        "profile_estimator_publication_applied": publication_applied,
        "profile_estimator_abstained": not publication_applied,
        "profile_estimator_readiness_rows": int(ready.sum()),
        "profile_estimator_update_rows": int(updated.sum()),
        "profile_estimator_time_to_first_ready_s": _first_elapsed(ready),
        "profile_estimator_time_to_first_update_s": _first_elapsed(updated),
        "profile_estimator_max_concurrent_windows": max_windows,
        "profile_estimator_lifetime_accepted_windows": _finite_max_int(
            "terrain_accepted_dynamics_windows"
        ),
        "profile_estimator_lifetime_rejected_windows": _finite_max_int(
            "terrain_rejected_dynamics_windows"
        ),
        "profile_estimator_profile_force_gain_final": _last_finite(
            "terrain_profile_force_gain"
        ),
        "profile_estimator_profile_ax_bias_final_mps2": _last_finite(
            "terrain_profile_ax_bias"
        ),
        "profile_estimator_profile_ay_bias_final_mps2": _last_finite(
            "terrain_profile_ay_bias"
        ),
        "profile_estimator_profile_bound_hits_max": _finite_max_int(
            "terrain_profile_bound_hits"
        ),
        "profile_estimator_feature_envelope_excursions_max": _finite_max_int(
            "terrain_feature_envelope_excursions"
        ),
        "profile_estimator_readiness_consistent": readiness_consistent,
    })
    return output


def estimator_artifact_hashes(backend: str) -> dict[str, str]:
    """SHA-256 contract for learned artifacts used by a selected live stack."""

    # "rate_checkpoint" hashes the rate-format force checkpoint the joint
    # estimator evaluates. The controller's own tire model is recorded per
    # study in that study's manifest and verified separately, so it is
    # deliberately absent here: this contract identifies the estimator stack,
    # and mixing the two would make an estimator hash change whenever the
    # controller's model changed.
    artifact_paths = {
        "rate_checkpoint_sha256": (
            PROJECT_ROOT / "nn_models/tire_force_rate/best_terrain_nn.pt"
        ),
        "rate_scalers_sha256": (
            PROJECT_ROOT / "nn_models/tire_force_rate/scalers.pkl"
        ),
    }
    if backend in {PARENT_ESTIMATOR_BACKEND, "bekker_ukf"}:
        # These static-map files belong to the scalar estimator alone. The joint
        # estimator evaluates the controller checkpoint directly, so attaching
        # them to it would record a dependency it does not have.
        artifact_paths.update({
            "force_checkpoint_sha256": (
                PROJECT_ROOT / "nn_models/tire_force_static_parent/best_terrain_nn.pt"
            ),
            "force_scalers_sha256": (
                PROJECT_ROOT / "nn_models/tire_force_static_parent/scalers.pkl"
            ),
        })
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing rig learned artifacts: {missing}")
    return {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in artifact_paths.items()
    }


def run_process(cmd: list[str], run_dir: Path, timeout: float) -> tuple[int, float, str]:
    ensure_runtime_env()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    t0 = time.time()
    try:
        # Give each run a private log directory. Sharing the global ``logs/``
        # collision, shield, and CBF logs across concurrent workers races on
        # truncation and attributes one run's contacts to another, which would
        # corrupt the safety results silently. The simulator and the safety
        # filters honour ``HIL_RUN_LOG_DIR`` when it is set.
        _env = os.environ.copy()
        _env["HIL_RUN_LOG_DIR"] = str(run_dir)
        # Pin NumPy's BLAS to one thread per run. The per-run solver work is
        # many small operations -- acados code generation and the filter and
        # tire-surrogate products -- for which a multi-threaded BLAS adds
        # thread-spawn overhead and no parallelism. Across several workers the
        # default thread pool also oversubscribes the machine several times
        # over, inflating each run's wall time and pushing some past their
        # timeout. Single-threaded BLAS is both faster at these sizes and free
        # of contention. The acados backend (BLASFEO, which uses no OpenMP) and
        # Chrono's own OpenMP threads are unaffected by these variables.
        _env.setdefault("OPENBLAS_NUM_THREADS", "1")
        _env.setdefault("MKL_NUM_THREADS", "1")
        _env.setdefault("NUMEXPR_NUM_THREADS", "1")
        with log_path.open("w") as f:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                env=_env,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -9
        with log_path.open("a") as f:
            f.write(f"\nTIMEOUT after {timeout:.1f}s\n")
    return rc, time.time() - t0, log_path.read_text(errors="replace")


def find_diag_csv(run_dir: Path, controller_mode: str, created_after: float) -> Path | None:
    candidates = [
        p for p in run_dir.rglob("diag_*.csv")
        if p.stat().st_mtime >= created_after - 2.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def copy_global_log(name: str, run_dir: Path, created_after: float) -> Path | None:
    src = LOGS_DIR / name
    if not src.exists() or src.stat().st_mtime < created_after - 2.0:
        return None
    dst = run_dir / name
    shutil.copy2(src, dst)
    return dst


def _float_series(df: pd.DataFrame, key: str) -> np.ndarray:
    if key not in df.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)


def _finite(v: np.ndarray) -> np.ndarray:
    return v[np.isfinite(v)]


def parse_diag_csv(path: Path, controller_mode: str, speed: float, metric_start: float = 2.0) -> dict[str, float]:
    # A run whose controller fails during initialization leaves a header-only or
    # zero-byte diagnostic CSV, on which ``pd.read_csv`` raises. The failure is
    # absorbed here so that the run is recorded as producing no diagnostics and
    # the surrounding matrix continues, rather than one failed run ending it.
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return {"n_samples": 0}
    if df.empty:
        return {"n_samples": 0}
    t = _float_series(df, "sim_time")
    mask = np.ones(len(df), dtype=bool)
    if t.size == len(df) and np.any(np.isfinite(t)):
        mask = np.isfinite(t) & (t >= metric_start)
        if not mask.any():
            mask = np.isfinite(t)

    cte = _float_series(df, "crosstrack_err")
    u = _float_series(df, "u_true")
    if u.size == 0:
        u = _float_series(df, "u_meas")
    solve = _float_series(df, "solve_time_ms")
    x = _float_series(df, "x_fa_true")
    if x.size == 0:
        x = _float_series(df, "x_fa_meas")
    y = _float_series(df, "y_fa_true")
    if y.size == 0:
        y = _float_series(df, "y_fa_meas")
    extra = {}

    cte_m = _finite(cte[mask]) if cte.size == len(df) else _finite(cte)
    u_m = _finite(u[mask]) if u.size == len(df) else _finite(u)
    solve_m = _finite(solve)

    progress = math.nan
    final_x = math.nan
    final_y = math.nan
    if x.size == len(df) and y.size == len(df):
        xf = _finite(x)
        yf = _finite(y)
        if len(xf):
            final_x = float(xf[-1])
        if len(yf):
            final_y = float(yf[-1])
        good = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(good) >= 2:
            progress = float(np.sum(np.hypot(np.diff(x[good]), np.diff(y[good]))))
    out = {
        "n_samples": int(len(df)),
        "rms_cte_m": float(np.sqrt(np.mean(cte_m ** 2))) if len(cte_m) else math.nan,
        "max_abs_cte_m": float(np.max(np.abs(cte_m))) if len(cte_m) else math.nan,
        "mean_abs_cte_m": float(np.mean(np.abs(cte_m))) if len(cte_m) else math.nan,
        "mean_speed_mps": float(np.mean(u_m)) if len(u_m) else math.nan,
        "p95_speed_mps": float(np.percentile(u_m, 95)) if len(u_m) else math.nan,
        "speed_ratio": float(np.mean(u_m) / speed) if len(u_m) and speed > 1e-6 else math.nan,
        "mean_solve_ms": float(np.mean(solve_m)) if len(solve_m) else math.nan,
        "p99_solve_ms": float(np.percentile(solve_m, 99)) if len(solve_m) else math.nan,
        "progress_m": progress,
        "final_x_m": final_x,
        "final_y_m": final_y,
    }
    out.update(extra)
    return out


def parse_collision_csv(path: Path | None) -> dict[str, float | int | str]:
    if path is None or not path.exists():
        return {}
    # A run with no collisions leaves a header-only or zero-byte log, and a run
    # the simulator aborts mid-frame can leave a ragged one. An unreadable log
    # is therefore treated as "no collisions logged" rather than raised: an
    # exception here would fail the worker and lose an entire matrix over one
    # run that recorded nothing.
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return {"collisions": 0, "near_misses": 0}
    if df.empty:
        source = ("chrono_body_contact" if "collision_source" in df.columns
                  else "legacy_proximity")
        return {"collisions": 0, "near_misses": 0,
                "collision_source": source}
    hit_ids: set[int] = set()
    near_ids: set[int] = set()
    clearances = []
    # Coerce the integer columns through one guarded helper. Pandas reads a
    # column containing NaN as float64, and int(NaN) raises, which would take
    # down the worker; absent means no event, so zero is the correct value.
    def _i(v, default=0):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(f):
            return default
        return int(f)
    for _, row in df.iterrows():
        rid = _i(row.get("rock_id", -1), -1)
        if _i(row.get("is_collision", 0)) == 1 and rid >= 0:
            hit_ids.add(rid)
        if _i(row.get("is_near_miss", 0)) == 1 and rid >= 0:
            near_ids.add(rid)
        clearance = float(row.get("clearance_m", math.nan))
        if math.isfinite(clearance):
            clearances.append(clearance)
        else:
            # Reconstruct clearance from the separate distance and margin
            # columns when a log does not carry the combined column directly.
            d = float(row.get("dist_2d", math.nan))
            hard = float(row.get("hard_margin", math.nan))
            if math.isfinite(d) and math.isfinite(hard):
                clearances.append(d - hard)
    source = "legacy_proximity"
    if "collision_source" in df.columns:
        values = df["collision_source"].dropna().astype(str)
        if not values.empty:
            source = values.iloc[-1]
    return {
        "collisions": len(hit_ids),
        "near_misses": len(near_ids),
        "min_clearance_m": float(np.min(clearances)) if clearances else math.nan,
        "collision_source": source,
    }


def parse_shield_csv(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}
    if df.empty:
        return {}
    required = {"steer_in", "steer_out", "throttle_in", "throttle_out"}
    if not required.issubset(df.columns):
        return {}
    return {
        "mean_abs_dsteer": float((df["steer_out"] - df["steer_in"]).abs().mean()),
        "mean_abs_dthrottle": float((df["throttle_out"] - df["throttle_in"]).abs().mean()),
    }


def parse_sim_diag_csv(path: Path | None, metric_start: float = 2.0) -> dict[str, float | int | str]:
    if path is None or not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}
    if df.empty:
        return {}
    out: dict[str, float | int | str] = {}
    t = pd.to_numeric(df.get("time", pd.Series(np.zeros(len(df)))), errors="coerce")
    mask = np.isfinite(t) & (t >= metric_start)
    if not mask.any():
        mask = np.isfinite(t)
    if "nearest_clearance_m" in df.columns:
        clearance = pd.to_numeric(df["nearest_clearance_m"], errors="coerce")
        c = clearance[mask] if len(clearance) == len(df) else clearance
        if len(c) and np.isfinite(c).any():
            out["min_clearance_m"] = float(np.nanmin(c))
    for col, key in [("collisions", "collisions"), ("near_misses", "near_misses")]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                # int(NaN) raises; the dropna above protects us, but if the
                # column was all-NaN we already skipped it.
                out[key] = int(vals.iloc[-1])
    if "collision_source" in df.columns:
        values = df["collision_source"].dropna().astype(str)
        if not values.empty:
            out["collision_source"] = values.iloc[-1]
    return out


def parse_log_summary(text: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    if m := RX_SIM_COMPLETE.search(text):
        out["sim_s"] = float(m.group(1))
        out["rt_factor"] = float(m.group(3))
    if m := RX_COLLISIONS.search(text):
        out["collisions"] = int(m.group(1))
        out["near_misses"] = int(m.group(2))
    if "[SAFETY]" in text:
        m = re.search(r"\[SAFETY\]\s*Calls:\s*(\d+),\s*Interventions:\s*(\d+)\s*\(([\d.]+)%\)", text)
        if m:
            out["intervention_rate_pct"] = float(m.group(3))
        m = re.search(r"\[SAFETY\][^\n]*MeanSolveMs:\s*([\d.]+)", text)
        if m:
            out["filter_solve_ms"] = float(m.group(1))
    return out


def launch_and_collect(
    *,
    experiment: str,
    variant: str,
    controller_mode: str,
    mpc_model: str,
    nn_model: str,
    terrain: str,
    path: str,
    speed: float,
    bumpiness: int,
    seed: int,
    run_dir: Path,
    sim_port: int,
    ctrl_port: int,
    sim_time: float,
    timeout: float,
    rocks: int = 0,
    lead_in: float = 5.0,
    extra_args: list[str] | None = None,
    metric_start: float = 2.0,
) -> RunResult:
    created_after = time.time()
    # No shared log needs clearing here: ``run_process`` sets HIL_RUN_LOG_DIR,
    # so each run's collision and filter logs are written into its own
    # ``run_dir`` and parallel workers never contend for one global file.

    cmd = base_launch_args(
        terrain=terrain, path=path, speed=speed, sim_time=sim_time,
        bumpiness=bumpiness, seed=seed, run_dir=run_dir,
        sim_port=sim_port, ctrl_port=ctrl_port, lead_in=lead_in, rocks=rocks,
    )
    # The standard NMPC is the single controller these studies exercise; the
    # argument is retained so every result row records which controller ran.
    if controller_mode != "standard":
        raise ValueError(
            f"controller_mode={controller_mode!r}: 'standard' is the only "
            "supported controller mode."
        )
    cmd += ["--model", mpc_model, "--nn-model", nn_model, "--rms-time-start", str(metric_start)]
    if extra_args:
        cmd += list(extra_args)
    # HIL_SIM_EXTRA is appended to every launch, which is how one plant and
    # controller configuration is applied across a whole suite without editing
    # each study. Because it changes what was simulated while leaving the
    # command line of each study unchanged, it is recorded in every manifest,
    # including when it is unset.
    _sim_extra = os.environ.get("HIL_SIM_EXTRA", "").split()
    if _sim_extra:
        cmd += _sim_extra
    sim_diag = run_dir / "sim_diag.csv"
    if rocks > 0:
        cmd += ["--sim-diag-csv", str(sim_diag)]

    rc, wall_s, text = run_process(cmd, run_dir, timeout)
    diag = find_diag_csv(run_dir, controller_mode, created_after)
    # The sim wrote its collision / shield logs directly into run_dir
    # (HIL_RUN_LOG_DIR), so read them per-run instead of copying a global file.
    _coll = run_dir / "collision_log.csv"
    collision = _coll if (rocks > 0 and _coll.exists()) else None
    shield = None
    for name in ("cbf_filter_log.csv",):
        p = run_dir / name
        if p.exists():
            shield = p

    result = RunResult(
        experiment=experiment,
        variant=variant,
        controller_mode=controller_mode,
        mpc_model=mpc_model,
        nn_model=nn_model,
        terrain=terrain,
        path=path,
        speed_mps=speed,
        bumpiness=bumpiness,
        seed=seed,
        run_dir=str(run_dir),
        rc=rc,
        wall_s=wall_s,
        status="ok" if rc == 0 else f"exit_{rc}",
        diag_csv=str(diag) if diag else "",
        collision_csv=str(collision) if collision else "",
        shield_csv=str(shield) if shield else "",
    )
    for k, v in parse_log_summary(text).items():
        setattr(result, k, v)
    if diag is not None:
        diag_metrics = parse_diag_csv(diag, controller_mode, speed, metric_start)
        for k, v in diag_metrics.items():
            if hasattr(result, k):
                setattr(result, k, v)
            else:
                result.extra[k] = v
    elif rc == 0:
        result.status = "no_diag"
    for k, v in parse_collision_csv(collision).items():
        setattr(result, k, v)
    result.extra["sim_diag_csv"] = str(sim_diag) if sim_diag.exists() else ""
    for k, v in parse_sim_diag_csv(sim_diag if sim_diag.exists() else None, metric_start).items():
        current = getattr(result, k, math.nan)
        if k == "collision_source":
            setattr(result, k, v)
        elif k in ("collisions", "near_misses"):
            if collision is None:
                setattr(result, k, v)
        elif k == "min_clearance_m":
            # Prefer the simulator's continuous 10 Hz clearance trace. The
            # collision event log omits steps during persistent contact by
            # design, so its proximity rows need not contain the run minimum.
            setattr(result, k, v)
        elif not math.isfinite(float(current)):
            setattr(result, k, v)
    for k, v in parse_shield_csv(shield).items():
        setattr(result, k, v)
    return result


def write_results_csv(path: Path, results: list[RunResult]) -> None:
    rows = []
    for r in results:
        d = asdict(r)
        extra = d.pop("extra", {}) or {}
        d.update({f"extra_{k}": v for k, v in extra.items()})
        rows.append(d)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_by_variant(results: list[RunResult], metrics: list[str]) -> pd.DataFrame:
    """Per-variant metric means over the runs that completed.

    ``n_runs``/``n_ok`` count every attempt, so a variant that failed runs
    shows the attrition, but the metric columns aggregate only ``status ==
    "ok"`` rows: a run that exited nonzero can carry partially-written or
    default-valued metrics, and averaging those in would let a crash move a
    headline number instead of merely shrinking its sample.
    """
    df = pd.DataFrame([asdict(r) for r in results])
    if df.empty:
        return df
    counts = df.groupby("variant", sort=False).agg(
        n_runs=("variant", "count"),
        n_ok=("status", lambda s: int((s == "ok").sum())),
    ).reset_index()
    ok = df[df["status"] == "ok"]
    if ok.empty:
        for m in metrics:
            counts[f"{m}_mean"] = math.nan
            counts[f"{m}_std"] = math.nan
        return counts
    agg: dict[str, Any] = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    stats = ok.groupby("variant", sort=False).agg(**agg).reset_index()
    return counts.merge(stats, on="variant", how="left")


def save_summary_markdown(out_dir: Path, title: str, summary: pd.DataFrame, notes: list[str]) -> None:
    lines = [f"# {title}", ""]
    lines.extend(notes)
    lines.append("")
    if not summary.empty:
        lines.append("```csv")
        lines.append(summary.to_csv(index=False).strip())
        lines.append("```")
    (out_dir / "summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Paper figure helpers
# ---------------------------------------------------------------------------

PAPER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _import_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


DISPLAY_LABELS = {
    "pacejka": "Pacejka",
    "pacejka_static": "Pacejka static",
    "tmeasy": "TMeasy",
    "tmeasy_static": "TMeasy static",
    "tire_force_static": "Neural surrogate",
    "tire_force_rate": "Neural surrogate (rate)",
    "tire_force_static_parent": "Neural surrogate (static)",
    "nn_static": "NN static prior",
    "nn_estimator": "NN live terrain estimator",
    "nn_wrong_prior": "NN wrong prior",
}


def _label(text: str) -> str:
    value = str(text)
    return DISPLAY_LABELS.get(value, value.replace("_", " "))


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    s = str(value)
    if not s or s == "nan":
        return None
    p = Path(s)
    return p if p.exists() else None


def _read_csv_if_exists(value: Any, **kwargs) -> pd.DataFrame | None:
    p = _path_or_none(value)
    if p is None:
        return None
    try:
        return pd.read_csv(p, **kwargs)
    except Exception:
        return None


def _manifest_args(out_dir: Path) -> dict[str, Any]:
    manifest = out_dir / "manifest.csv"
    if not manifest.exists():
        return {}
    try:
        rows = pd.read_csv(manifest)
    except Exception:
        return {}
    if not {"key", "value"}.issubset(rows.columns):
        return {}
    out: dict[str, Any] = {}
    for _, row in rows.iterrows():
        key = str(row["key"])
        value = row["value"]
        try:
            out[key] = ast.literal_eval(str(value))
        except Exception:
            out[key] = value
    return out


def _nominal_reference_xy(out_dir: Path, path_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        if str(SIM_DIR) not in sys.path:
            sys.path.insert(0, str(SIM_DIR))
            import flatpath  # noqa: E402,F401
        from reference_path import generate_path_waypoints
    except Exception:
        return None

    args = _manifest_args(out_dir)
    lead_in = float(args.get("lead_in", 5.0))
    sine_amplitude = float(args.get("sine_amplitude", 2.0))
    sine_wavelength = float(args.get("sine_wavelength", 30.0))
    try:
        x, y = generate_path_waypoints(
            path_name,
            lead_in=lead_in,
            sine_amplitude=sine_amplitude,
            sine_wavelength=sine_wavelength,
            ds=0.25,
        )
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except Exception:
        return None


def _variant_colors(variants: list[str]) -> dict[str, str]:
    return {v: PAPER_COLORS[i % len(PAPER_COLORS)] for i, v in enumerate(variants)}


def _numeric_column(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


_NN_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _terrain_internal(terrain: str) -> dict[str, float] | None:
    try:
        if str(SIM_DIR) not in sys.path:
            sys.path.insert(0, str(SIM_DIR))
            import flatpath  # noqa: E402,F401
        from param_consistency import get_terrain_preset, terrain_preset_to_internal
        return terrain_preset_to_internal(get_terrain_preset(str(terrain)))
    except Exception:
        return None


def _load_nn_model_for_row(row: pd.Series):
    model_id = str(row.get("nn_model", "") or "")
    terrain = str(row.get("terrain", "") or "")
    if not model_id or not terrain:
        return None, None
    key = (model_id, terrain)
    if key in _NN_MODEL_CACHE:
        return _NN_MODEL_CACHE[key], _terrain_internal(terrain)
    try:
        if str(SIM_DIR) not in sys.path:
            sys.path.insert(0, str(SIM_DIR))
            import flatpath  # noqa: E402,F401
        from nn_tire_model import load_nn_tire_model
        tp = _terrain_internal(terrain)
        if tp is None:
            return None, None
        model_dir = PROJECT_ROOT / "nn_models" / model_id
        if not model_dir.exists():
            return None, None
        model = load_nn_tire_model(model_dir, tp)
        _NN_MODEL_CACHE[key] = model
        return model, tp
    except Exception:
        return None, None


def _force_arrays_for_row(row: pd.Series, diag: pd.DataFrame, axle: str) -> tuple[np.ndarray, np.ndarray] | None:
    actual_col = f"actual_Fy_{axle}"
    pred_col = f"pred_Fy_{axle}"
    if actual_col not in diag.columns:
        return None
    actual = _numeric_column(diag, actual_col)

    # For neural rows, recompute the prediction from the logged operating-point
    # features rather than reading the logged prediction column, so that a
    # figure always reflects the checkpoint and loader currently in the tree
    # rather than whatever produced the result directory being plotted.
    if str(row.get("mpc_model", "")).lower() == "nn":
        req = ["kappa_diag", "u_safe_diag", "Fz_f_mean", "Fz_r_mean", "alpha_f", "alpha_r"]
        if not set(req).issubset(diag.columns):
            return None
        model, tp = _load_nn_model_for_row(row)
        if model is None or tp is None:
            return None
        alpha = _numeric_column(diag, "alpha_f" if axle == "front" else "alpha_r")
        fz = _numeric_column(diag, "Fz_f_mean" if axle == "front" else "Fz_r_mean")
        u = _numeric_column(diag, "u_safe_diag")
        kappa = _numeric_column(diag, "kappa_diag")
        sr = _numeric_column(diag, "sr_diag") if axle == "front" and "sr_diag" in diag.columns else np.zeros(len(diag))
        pred = np.full(len(diag), np.nan, dtype=float)
        n = min(len(actual), len(alpha), len(fz), len(u), len(kappa), len(sr))
        for i in range(n):
            vals = (alpha[i], fz[i], u[i], kappa[i], sr[i])
            if not all(np.isfinite(v) for v in vals):
                continue
            try:
                _, fy = model.predict_numeric(
                    alpha[i], fz[i], u[i],
                    kappa=kappa[i],
                    n_terrain=tp["n"],
                    steering_rate=sr[i] if axle == "front" else 0.0,
                    terrain_params=tp,
                )
                pred[i] = -2.0 * fy
            except Exception:
                pred[i] = np.nan
        return actual, pred

    if pred_col not in diag.columns:
        return None
    return actual, _numeric_column(diag, pred_col)


def plot_metric_distribution_grid(
    results_csv: Path,
    out_dir: Path,
    specs: list[tuple[str, str, str]],
    filename: str,
    title: str,
) -> None:
    """Plot per-run points plus mean/std for selected metrics.

    ``specs`` entries are ``(column, ylabel, direction_hint)``.
    """
    plt = _import_plotting()
    df = pd.read_csv(results_csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    variants = list(dict.fromkeys(ok["variant"].astype(str)))
    colors = _variant_colors(variants)
    n = len(specs)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False)
    x = np.arange(len(variants), dtype=float)

    rng = np.random.default_rng(7)
    for ax, (metric, ylabel, hint) in zip(axes.flat, specs):
        for i, variant in enumerate(variants):
            vals = pd.to_numeric(ok.loc[ok["variant"] == variant, metric], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if vals.empty:
                continue
            jitter = rng.normal(0.0, 0.035, size=len(vals))
            ax.scatter(
                np.full(len(vals), x[i]) + jitter,
                vals,
                s=22,
                alpha=0.55,
                color=colors[variant],
                edgecolors="none",
            )
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ax.errorbar(
                [x[i]], [mean], yerr=[[std], [std]],
                fmt="o", color="black", ecolor="black", capsize=4,
                markersize=5, zorder=5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_label(v) for v in variants], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(hint, fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=240)
    plt.close(fig)


def _trajectory_columns(diag: pd.DataFrame) -> tuple[str | None, str | None]:
    for x_col, y_col in [
        ("x_fa_true", "y_fa_true"),
        ("x_fa_meas", "y_fa_meas"),
        ("x", "y"),
    ]:
        if x_col in diag.columns and y_col in diag.columns:
            return x_col, y_col
    return None, None


def _time_column(diag: pd.DataFrame) -> str | None:
    for col in ("sim_time", "time", "wall_time"):
        if col in diag.columns:
            return col
    return None


def _thin_xy(x: np.ndarray, y: np.ndarray, max_points: int = 800) -> tuple[np.ndarray, np.ndarray]:
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x = x[idx]
        y = y[idx]
    return x, y


def _unique_rocks(collision_csv: Any) -> pd.DataFrame:
    cdf = _read_csv_if_exists(collision_csv)
    if cdf is None or cdf.empty:
        return pd.DataFrame(columns=["rock_id", "rock_x", "rock_y", "rock_r"])
    cols = ["rock_id", "rock_x", "rock_y", "rock_r"]
    if not set(cols).issubset(cdf.columns):
        return pd.DataFrame(columns=cols)
    return cdf[cols].drop_duplicates("rock_id").sort_values("rock_id")


def plot_trajectory_overlays(
    results_csv: Path,
    out_dir: Path,
    *,
    filename_prefix: str = "trajectory_overlay",
    max_scenarios: int = 4,
    max_variants: int = 8,
) -> None:
    """Create trajectory-vs-reference overlays from existing diag CSVs."""
    plt = _import_plotting()
    df = pd.read_csv(results_csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty or "diag_csv" not in ok.columns:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ok = ok[ok["diag_csv"].map(lambda p: _path_or_none(p) is not None)].copy()
    if ok.empty:
        return

    ok["scenario"] = (
        ok["terrain"].astype(str) + "/" + ok["path"].astype(str)
        + "/v" + ok["speed_mps"].astype(str) + "/b" + ok["bumpiness"].astype(str)
    )
    scenario_score = ok.groupby("scenario")["rms_cte_m"].max().sort_values(ascending=False)
    scenarios = list(scenario_score.index[:max_scenarios])
    variants = list(dict.fromkeys(ok["variant"].astype(str)))[:max_variants]
    colors = _variant_colors(variants)

    for scenario in scenarios:
        sub_all = ok[ok["scenario"] == scenario].copy()
        if sub_all.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.6, 5.4))
        ref_plotted = False
        rock_plotted = False
        extent_x: list[float] = []
        extent_y: list[float] = []
        path_name = str(sub_all.iloc[0]["path"])
        nominal_ref = _nominal_reference_xy(out_dir, path_name)
        if nominal_ref is not None:
            xr, yr = nominal_ref
            ax.plot(xr, yr, "k--", lw=2.0, label="reference", alpha=0.8)
            ref_plotted = True
        for variant in variants:
            rows = sub_all[sub_all["variant"] == variant].sort_values(["seed", "rms_cte_m"])
            if rows.empty:
                continue
            row = rows.iloc[0]
            diag = _read_csv_if_exists(row["diag_csv"])
            if diag is None or diag.empty:
                continue
            x_col, y_col = _trajectory_columns(diag)
            if x_col is None or y_col is None:
                continue
            x, y = _thin_xy(_numeric_column(diag, x_col), _numeric_column(diag, y_col))
            if len(x) < 2:
                continue
            extent_x.extend([float(np.nanmin(x)), float(np.nanmax(x))])
            extent_y.extend([float(np.nanmin(y)), float(np.nanmax(y))])
            ax.plot(x, y, lw=1.8, color=colors[variant], label=_label(variant), alpha=0.9)

            if not ref_plotted and {"x_ref_0", "y_ref_0"}.issubset(diag.columns):
                xr, yr = _thin_xy(_numeric_column(diag, "x_ref_0"), _numeric_column(diag, "y_ref_0"))
                if len(xr) >= 2:
                    # Fallback only: x_ref_0 is a per-solve recovery/blended
                    # reference point, not the nominal path.  Prefer the
                    # reconstructed path above whenever possible.
                    ax.plot(xr, yr, "k--", lw=2.0, label="reference samples", alpha=0.8)
                    ref_plotted = True

            if not rock_plotted and "collision_csv" in row.index:
                rocks = _unique_rocks(row["collision_csv"])
                for _, r in rocks.iterrows():
                    extent_x.extend([
                        float(r["rock_x"]) - float(r["rock_r"]) - 1.5,
                        float(r["rock_x"]) + float(r["rock_r"]) + 1.5,
                    ])
                    extent_y.extend([
                        float(r["rock_y"]) - float(r["rock_r"]) - 1.5,
                        float(r["rock_y"]) + float(r["rock_r"]) + 1.5,
                    ])
                    circ = plt.Circle(
                        (float(r["rock_x"]), float(r["rock_y"])),
                        float(r["rock_r"]) + 1.5,
                        facecolor="none",
                        edgecolor="#444444",
                        lw=1.0,
                        alpha=0.65,
                    )
                    ax.add_patch(circ)
                if not rocks.empty:
                    ax.scatter(rocks["rock_x"], rocks["rock_y"], s=18, c="#444444", marker="x", label="rocks")
                    rock_plotted = True

        ax.set_aspect("equal", adjustable="box")
        if extent_x and extent_y:
            xmin, xmax = min(extent_x), max(extent_x)
            ymin, ymax = min(extent_y), max(extent_y)
            xpad = max(4.0, 0.08 * (xmax - xmin + 1e-6))
            ypad = max(2.0, 0.15 * (ymax - ymin + 1e-6))
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Trajectory vs reference: {scenario}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{filename_prefix}_{_slug(scenario)}.png", dpi=240)
        plt.close(fig)


def collect_force_prediction_metrics(results_csv: Path, out_dir: Path) -> pd.DataFrame:
    """Compute actual-vs-predicted lateral-force metrics from diag CSVs."""
    rows: list[dict[str, Any]] = []
    df = pd.read_csv(results_csv)
    ok = df[df["status"] == "ok"].copy()
    for _, row in ok.iterrows():
        diag = _read_csv_if_exists(row.get("diag_csv"))
        if diag is None or diag.empty:
            continue
        for axle, actual_col, pred_col in [
            ("front", "actual_Fy_front", "pred_Fy_front"),
            ("rear", "actual_Fy_rear", "pred_Fy_rear"),
        ]:
            arrays = _force_arrays_for_row(row, diag, axle)
            if arrays is None:
                continue
            actual, pred = arrays
            good = np.isfinite(actual) & np.isfinite(pred)
            if np.count_nonzero(good) < 10:
                continue
            a = actual[good]
            p = pred[good]
            err = p - a
            denom = np.sum((a - np.mean(a)) ** 2)
            r2 = 1.0 - float(np.sum(err ** 2) / denom) if denom > 1e-9 else math.nan
            rows.append({
                "variant": row.get("variant", ""),
                "terrain": row.get("terrain", ""),
                "path": row.get("path", ""),
                "speed_mps": row.get("speed_mps", math.nan),
                "bumpiness": row.get("bumpiness", math.nan),
                "seed": row.get("seed", math.nan),
                "axle": axle,
                "n": int(len(a)),
                "mae_N": float(np.mean(np.abs(err))),
                "rmse_N": float(np.sqrt(np.mean(err ** 2))),
                "bias_N": float(np.mean(err)),
                "r2": r2,
                "actual_std_N": float(np.std(a)),
                "diag_csv": row.get("diag_csv", ""),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(out_dir / "force_prediction_metrics.csv", index=False)
    return out


def plot_force_prediction_figures(
    results_csv: Path,
    out_dir: Path,
    *,
    max_points_per_variant_axle: int = 2500,
) -> None:
    """Create predicted-vs-actual lateral-force plots from diag CSVs."""
    plt = _import_plotting()
    df = pd.read_csv(results_csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    metrics = collect_force_prediction_metrics(results_csv, out_dir)
    if metrics.empty:
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    variants = list(dict.fromkeys(ok["variant"].astype(str)))
    colors = _variant_colors(variants)
    rng = np.random.default_rng(11)

    samples: list[pd.DataFrame] = []
    for _, row in ok.iterrows():
        diag = _read_csv_if_exists(row.get("diag_csv"))
        if diag is None or diag.empty:
            continue
        for axle in ["front", "rear"]:
            arrays = _force_arrays_for_row(row, diag, axle)
            if arrays is None:
                continue
            actual, pred = arrays
            sub = pd.DataFrame({
                "variant": str(row["variant"]),
                "axle": axle,
                "actual": actual,
                "pred": pred,
            }).dropna()
            if not sub.empty:
                samples.append(sub)
    if not samples:
        return
    force = pd.concat(samples, ignore_index=True)
    sampled = []
    for (variant, axle), sub in force.groupby(["variant", "axle"], sort=False):
        n = min(len(sub), max_points_per_variant_axle)
        idx = rng.choice(sub.index.to_numpy(), size=n, replace=False) if len(sub) > n else sub.index
        sampled.append(sub.loc[idx])
    force_s = pd.concat(sampled, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
    for ax, axle in zip(axes, ["front", "rear"]):
        sub_ax = force_s[force_s["axle"] == axle]
        if sub_ax.empty:
            ax.set_visible(False)
            continue
        for variant, sub in sub_ax.groupby("variant", sort=False):
            ax.scatter(
                sub["actual"], sub["pred"],
                s=7, alpha=0.22, color=colors.get(variant, "#444444"),
                label=_label(variant),
            )
        lim_vals = pd.concat([sub_ax["actual"], sub_ax["pred"]]).to_numpy(dtype=float)
        lim_vals = lim_vals[np.isfinite(lim_vals)]
        if len(lim_vals):
            lo, hi = np.percentile(lim_vals, [1, 99])
            pad = 0.08 * max(1.0, hi - lo)
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_title(f"{axle.capitalize()} axle Fy")
        ax.set_xlabel("Chrono actual Fy (N)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Model predicted Fy (N)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Predicted vs actual lateral tire force")
    fig.tight_layout()
    fig.savefig(fig_dir / "force_predicted_vs_actual_scatter.png", dpi=240)
    plt.close(fig)

    present_variants = [v for v in variants if v in set(force_s["variant"])]
    if present_variants:
        fig, axes = plt.subplots(
            len(present_variants), 2,
            figsize=(10.6, max(3.0, 2.6 * len(present_variants))),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        all_vals = pd.concat([force_s["actual"], force_s["pred"]]).to_numpy(dtype=float)
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals):
            lo, hi = np.percentile(all_vals, [1, 99])
            pad = 0.08 * max(1.0, hi - lo)
            lo -= pad
            hi += pad
        else:
            lo, hi = -1.0, 1.0
        for r, variant in enumerate(present_variants):
            for c, axle in enumerate(["front", "rear"]):
                ax = axes[r, c]
                sub = force_s[(force_s["variant"] == variant) & (force_s["axle"] == axle)]
                if sub.empty:
                    ax.set_visible(False)
                    continue
                ax.scatter(
                    sub["actual"], sub["pred"],
                    s=5, alpha=0.18, color=colors.get(variant, "#444444"),
                    edgecolors="none",
                )
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.7)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                mrow = metrics[(metrics["variant"].astype(str) == variant) & (metrics["axle"] == axle)]
                if not mrow.empty:
                    mae = float(mrow["mae_N"].mean())
                    r2 = float(mrow["r2"].mean())
                    ax.text(
                        0.03, 0.95, f"MAE {mae:.0f} N\nR² {r2:.2f}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
                    )
                if r == 0:
                    ax.set_title(f"{axle.capitalize()} axle")
                if c == 0:
                    ax.set_ylabel(f"{_label(variant)}\npredicted Fy (N)")
                if r == len(present_variants) - 1:
                    ax.set_xlabel("Chrono actual Fy (N)")
                ax.grid(alpha=0.22)
        fig.suptitle("Predicted vs actual Fy by model")
        fig.tight_layout()
        fig.savefig(fig_dir / "force_predicted_vs_actual_by_model.png", dpi=240)
        plt.close(fig)

    summary = metrics.groupby(["variant", "axle"], sort=False).agg(
        mae=("mae_N", "mean"),
        rmse=("rmse_N", "mean"),
        r2=("r2", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, metric, ylabel in [
        (axes[0], "mae", "MAE (N)"),
        (axes[1], "rmse", "RMSE (N)"),
        (axes[2], "r2", "R²"),
    ]:
        labels = []
        values = []
        colors_bar = []
        for variant in variants:
            for axle in ["front", "rear"]:
                sub = summary[(summary["variant"] == variant) & (summary["axle"] == axle)]
                if sub.empty:
                    continue
                labels.append(f"{_label(variant)}\n{axle}")
                values.append(float(sub[metric].iloc[0]))
                colors_bar.append(colors.get(variant, "#444444"))
        ax.bar(np.arange(len(values)), values, color=colors_bar, alpha=0.85)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Force prediction error by tire model")
    fig.tight_layout()
    fig.savefig(fig_dir / "force_prediction_error_summary.png", dpi=240)
    plt.close(fig)

    # Time series for the scenario with the largest crosstrack error, which
    # shows whether the error is a sustained bias or a transient excursion.
    # A summary statistic alone cannot distinguish the two.
    ok2 = ok.copy()
    ok2["scenario"] = (
        ok2["terrain"].astype(str) + "/" + ok2["path"].astype(str)
        + "/v" + ok2["speed_mps"].astype(str) + "/b" + ok2["bumpiness"].astype(str)
    )
    scenario = ok2.groupby("scenario")["rms_cte_m"].max().sort_values(ascending=False).index[0]
    example_rows = []
    for variant in variants[:4]:
        rows = ok2[(ok2["variant"].astype(str) == variant) & (ok2["scenario"] == scenario)].sort_values("seed")
        if not rows.empty:
            example_rows.append(rows.iloc[0])
    if example_rows:
        fig, axes = plt.subplots(len(example_rows), 1, figsize=(10.5, 2.6 * len(example_rows)), sharex=True)
        if len(example_rows) == 1:
            axes = [axes]
        for ax, row in zip(axes, example_rows):
            diag = _read_csv_if_exists(row.get("diag_csv"))
            if diag is None or diag.empty:
                continue
            t_col = _time_column(diag)
            if t_col is None:
                t = np.arange(len(diag), dtype=float)
            else:
                t = _numeric_column(diag, t_col)
            for actual_col, pred_col, color, label_prefix in [
                ("actual_Fy_front", "pred_Fy_front", "#1f77b4", "front"),
                ("actual_Fy_rear", "pred_Fy_rear", "#ff7f0e", "rear"),
            ]:
                axle = "front" if label_prefix == "front" else "rear"
                arrays = _force_arrays_for_row(row, diag, axle)
                if arrays is None:
                    continue
                actual, pred = arrays
                good = np.isfinite(t) & np.isfinite(actual) & np.isfinite(pred)
                if np.count_nonzero(good) < 2:
                    continue
                idx = np.where(good)[0]
                if len(idx) > 700:
                    idx = idx[np.linspace(0, len(idx) - 1, 700).astype(int)]
                ax.plot(t[idx], actual[idx], color=color, lw=1.4, alpha=0.85, label=f"{label_prefix} actual")
                ax.plot(t[idx], pred[idx], color=color, lw=1.2, alpha=0.85, linestyle="--", label=f"{label_prefix} predicted")
            ax.set_ylabel("Fy (N)")
            ax.set_title(f"{_label(row['variant'])}: {scenario}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncols=2)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle("Predicted vs actual Fy over time")
        fig.tight_layout()
        fig.savefig(fig_dir / "force_prediction_timeseries_examples.png", dpi=240)
        plt.close(fig)
