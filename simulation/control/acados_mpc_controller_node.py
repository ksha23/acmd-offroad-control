#!/usr/bin/env python3
"""
ACADOS MPC Controller Node (Decoupled)
========================================

Controller process of the decoupled simulator/controller pair.  It runs the
ACADOS SQP-RTI optimal control problem, the speed-profile generator, the
online terrain estimator, and the run diagnostics, and exchanges messages with
the Chrono simulation node over the configured transport.

Delay estimation and compensation, command integration, rate features, and
tracking analytics live in ``mpc_helpers``; the tire force map is loaded
through ``nn_tire_model``, which admits only checkpoints supervised by the
controlled single-tire Chrono SCM rig.

Subscribes: VehicleState from the simulation node
Publishes:  ControlCommand to the simulation node

Usage:
    python acados_mpc_controller_node.py --nn-model tire_force_static --terrain sand --path sinusoidal
    python acados_mpc_controller_node.py --model pacejka --terrain dirt --path lane_change
"""

import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401
import argparse
import collections
import csv
import json
import math
import os
import sys
from typing import Any, Mapping, Optional, Tuple
import time as wall_time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hil_messages import (
    VehicleState, ControlCommand, SimStatus,
    STEER_MAX_RAD, DRIVE_TORQUE_MAX_NM, BRAKE_TORQUE_MAX_NM,
    make_publisher, make_subscriber,
    ctrl_pub_endpoint, sim_sub_endpoint,
    TOPIC_VEHICLE_STATE, TOPIC_CONTROL_CMD,
)
from param_consistency import (
    get_vehicle_params_for_demo, get_terrain_preset,
    terrain_preset_to_internal,
    STANDARD_GRAVITY_M_S2,
)

# Reuse helper classes from mpc_helpers (shared module)
from mpc_helpers import (
    DelayEstimator,
    StatePredictor,
    quat_to_yaw,
    ControlIntegrator,
    TrackingAnalytics,
    RateTracker,
)
from speed_profile import gg_speed_profile, terrain_grip_limits

# ACADOS solver + unified NN loader
from acados_mpc_solver import (
    AcadosMPC,
    DEFAULT_MPC_DT,
    DEFAULT_MPC_HORIZON_STEPS,
)
from nn_tire_model import load_nn_tire_model
from analytical_tire_models import get_tire_forces as analytical_tire_forces
from terrain_parameterization import terrain_params_for_n as _terrain_params_for_n
from terrain_id_probe import (
    TerrainIDProbe,
    TerrainIDProbeConfig,
    TerrainIDProbeInputs,
)

# Subset of terrain_preset_to_internal keys passed into the OCP each stage
TERRAIN_MPC_PARAM_KEYS = ("Kphi", "Kc", "n", "c", "phi", "k")

# Feedforward motion-resistance (sinkage-drag) term for the longitudinal
# dynamics, u_dot = ax + du_dot_resid, enabled by --ff-drag.
#
# The term is a scalar deceleration indexed by the sinkage exponent, so it
# expresses motion resistance at the level of the dynamics and not the
# actuation. It is therefore distinct in kind from the throttle offsets below:
# the dominant soil effect on this platform is a reduced throttle-to-force
# gain, which a dynamics-level deceleration cannot represent, and sizing this
# coefficient from a converged throttle offset overstates the true drag by
# that same gain factor. The coefficients below are fitted from rollout drift
# on sand (benchmarking/calibrate_motion_resistance.py) and support the
# ablation that compares dynamics-level against actuation-level compensation.
_FF_DRAG_N = np.array([0.50, 0.70, 1.10])
_FF_DRAG_C = np.array([0.0, 0.0, 0.171])


def _c_drag(n_hat: float) -> float:
    """Calibrated feedforward drag deceleration (m/s^2) at sinkage exponent n_hat."""
    return float(np.interp(float(n_hat), _FF_DRAG_N, _FF_DRAG_C))


# Feedforward throttle offset indexed by the sinkage exponent: the per-terrain
# value throttle - a_x/a_x_max that the asymmetric velocity-error observer
# settles at, calibrated from logs with that observer active. Soft, low-n soil
# requires the largest offset. Enabled by --ff-throttle; pairing it with
# --dob-ki 0 yields a purely feedforward actuation map in place of the
# reactive observer.
_FF_THROTTLE_N = np.array([0.50, 0.70, 1.10])
_FF_THROTTLE_D = np.array([0.244, 0.242, 0.067])


def _d_ff_throttle(n_hat: float) -> float:
    """Calibrated feedforward throttle offset at sinkage exponent n_hat."""
    return float(np.interp(float(n_hat), _FF_THROTTLE_N, _FF_THROTTLE_D))


# Two-dimensional feedforward throttle offset d(n_hat, u), calibrated per
# (soil, speed) from the cruise value the reactive observer settles at. The
# throttle required to hold speed on soil depends on the full operating point
# and not on the soil alone: dirt requires about 0.23 at 5 m/s and about 0.37
# at 7 m/s, a dependence a map indexed by n alone cannot express, and precisely
# the information the reactive integrator acquires online. On sand the demand
# is traction-saturated, so a large tabulated value clips at 1.0, matching the
# saturated observer. This is the static counterpart of the reactive observer,
# and it plays the role of the explicit soil-dependent motion-resistance term
# of Dallas et al., expressed here in throttle rather than torque.
_FF2D_N = np.array([0.50, 0.70, 1.10])          # sinkage exponent grid
_FF2D_U = np.array([5.0, 7.0])                   # forward-speed grid (m/s)
_FF2D_D = np.array([
    [0.334, 0.333],   # clay  n=0.50 : flat with speed
    [0.225, 0.366],   # dirt  n=0.70 : strong speed dependence
    [0.400, 0.400],   # sand  n=1.10 : traction-saturated (value clips to 1.0)
])


def _d_ff_throttle_2d(n_hat: float, u: float) -> float:
    """Bilinear feedforward throttle offset at operating point (n_hat, u)."""
    n = float(np.clip(n_hat, _FF2D_N[0], _FF2D_N[-1]))
    uu = float(np.clip(u, _FF2D_U[0], _FF2D_U[-1]))
    per_n = np.array([np.interp(uu, _FF2D_U, _FF2D_D[i])
                      for i in range(len(_FF2D_N))])
    return float(np.interp(n, _FF2D_N, per_n))

# Minimum forward speed represented in MPC state (physical bound).
MPC_STATE_MIN_FORWARD_SPEED_MPS = 0.0
# Speed epsilon used only in slip-angle/rate feature computations.
SLIP_CALC_MIN_SPEED_MPS = 0.5

GRIT_BACKEND = "grit"
RIG_ACTIVE_ESTIMATOR_BACKEND = GRIT_BACKEND
RIG_JOINT_ACCEPTED_SNAPSHOT_VERSION = "grit_accepted"
RIG_JOINT_MIN_PUBLICATION_CONFIDENCE = 0.20
RIG_JOINT_MAX_EVIDENCE_AGE_S = 3.5
RIG_JOINT_MIN_OBSERVABILITY_RANK = 2
RIG_JOINT_MIN_OBSERVABILITY_SINGULAR_VALUE = 0.10
RIG_JOINT_BOUNDARY_MASS_LIMIT = 0.25
RIG_JOINT_FALLBACK_N = 0.50
RIG_JOINT_FALLBACK_PHI_DEG = 13.0
RIG_JOINT_CONTROL_MIN_PHI_DEG = 10.0
# Lower edge of the NMPC's validated operating envelope in the Bekker
# exponent. The estimator's grid extends to n = 0.40 (the sub-clay hold
# extension keeps clay an interior point), but the controller's tire
# conditioning is validated only down to clay; a snapshot below this bound is
# a labelled control-envelope rejection, not data corruption, and control
# falls back to the fixed low-grip point exactly as for sub-envelope phi.
RIG_JOINT_CONTROL_MIN_N = 0.50
TERRAIN_ESTIMATOR_BACKENDS = (
    "scalar_parent",
    GRIT_BACKEND,
    "bekker_ukf",
)
_RIG_DYNAMICS_JOINT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "nn_models" / "tire_force_rate"
)
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


class _JointSnapshotLatch:
    """Keep estimator publication separate from the current control tick.

    Accepted joint snapshots are immutable by contract.  ``begin_control_tick``
    captures the latest accepted generation for all control and diagnostic
    decisions made during that tick.  A snapshot accepted later in the tick is
    therefore first visible on the next call to ``begin_control_tick``.
    """

    def __init__(self) -> None:
        self._accepted_snapshot: Optional[Mapping[str, Any]] = None
        self._control_snapshot: Optional[Mapping[str, Any]] = None

    @property
    def control_snapshot(self) -> Optional[Mapping[str, Any]]:
        return self._control_snapshot

    def accept(self, snapshot: Mapping[str, Any]) -> None:
        if not isinstance(snapshot, Mapping):
            raise TypeError("joint accepted snapshot must be a mapping")
        self._accepted_snapshot = snapshot

    def begin_control_tick(self) -> Optional[Mapping[str, Any]]:
        self._control_snapshot = self._accepted_snapshot
        return self._control_snapshot


def _joint_snapshot_sequence(
    snapshot: Optional[Mapping[str, Any]],
) -> int:
    """Return a finite sequence identifier, including before first acceptance."""

    if not isinstance(snapshot, Mapping):
        return 0
    return int(snapshot.get("update_seq", snapshot.get("seq", 0)))


def _joint_fixed_fallback_params() -> dict[str, float]:
    """Return the fixed low-friction parameters used when no snapshot is ready.

    This is a deterministic low-grip point inside the controller's clay
    operating envelope, and the controller reports whenever it is in use. It
    is a control-feasible conservative default rather than a claimed global
    worst case.
    """

    parameters = _terrain_params_for_n(RIG_JOINT_FALLBACK_N)
    parameters["phi"] = RIG_JOINT_FALLBACK_PHI_DEG
    return {
        key: float(parameters[key])
        for key in TERRAIN_MPC_PARAM_KEYS
    }


def _joint_snapshot_readiness(
    snapshot: Optional[Mapping[str, Any]],
    sim_time: float,
) -> tuple[bool, float, str]:
    """Evaluate the immutable accepted joint snapshot at one controller tick."""

    if not snapshot:
        return False, float("nan"), "no_snapshot"
    try:
        if (
            str(snapshot["snapshot_version"])
            != RIG_JOINT_ACCEPTED_SNAPSHOT_VERSION
        ):
            return False, float("nan"), "invalid_snapshot"
        update_seq_raw = snapshot["update_seq"]
        update_seq = int(update_seq_raw)
        if (
            isinstance(update_seq_raw, bool)
            or float(update_seq_raw) != float(update_seq)
            or update_seq < 1
        ):
            return False, float("nan"), "invalid_snapshot"
        evidence_time = float(snapshot["evidence_time_s"])
        confidence = float(snapshot["confidence"])
        rank_raw = snapshot["observability_rank"]
        rank = int(rank_raw)
        if (
            isinstance(rank_raw, bool)
            or float(rank_raw) != float(rank)
            or rank < 0
        ):
            return False, float("nan"), "invalid_snapshot"
        singular_value = float(
            snapshot["observability_min_singular_value"]
        )
        boundary_mass = float(snapshot["max_boundary_mass"])
        boundary_limited = bool(snapshot["boundary_limited"])
        n_sigma = float(snapshot["n_sigma"])
        phi_sigma = float(snapshot["phi_sigma_deg"])
        projection_wall_time = float(snapshot["projection_wall_time_s"])
        posterior_wall_time = float(snapshot["posterior_wall_time_s"])
        publication_wall_time = float(snapshot["publication_wall_time_s"])
        update_wall_time = float(snapshot["update_wall_time_s"])
        timestamp = float(sim_time)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, float("nan"), "invalid_snapshot"
    numeric = np.asarray(
        [
            timestamp,
            evidence_time,
            confidence,
            singular_value,
            boundary_mass,
            n_sigma,
            phi_sigma,
            projection_wall_time,
            posterior_wall_time,
            publication_wall_time,
            update_wall_time,
        ],
        dtype=float,
    )
    if not np.isfinite(numeric).all():
        return False, float("nan"), "invalid_snapshot"
    if (
        not 0.0 <= confidence <= 1.0
        or not 0.0 <= boundary_mass <= 1.0
        or n_sigma < 0.0
        or phi_sigma < 0.0
        or min(
            projection_wall_time,
            posterior_wall_time,
            publication_wall_time,
            update_wall_time,
        )
        < 0.0
        or not np.isclose(
            update_wall_time,
            projection_wall_time
            + posterior_wall_time
            + publication_wall_time,
            rtol=0.0,
            atol=1.0e-12,
        )
        or boundary_limited
        != (boundary_mass >= RIG_JOINT_BOUNDARY_MASS_LIMIT)
    ):
        return False, float("nan"), "invalid_snapshot"
    if _joint_snapshot_parameters(snapshot) is None:
        return False, float("nan"), "invalid_snapshot"
    age = timestamp - evidence_time
    if age < -1.0e-9:
        return False, age, "future_snapshot"
    if age > RIG_JOINT_MAX_EVIDENCE_AGE_S + 1.0e-9:
        return False, age, "stale"
    if rank < RIG_JOINT_MIN_OBSERVABILITY_RANK:
        return False, age, "rank"
    if (
        singular_value + 1.0e-12
        < RIG_JOINT_MIN_OBSERVABILITY_SINGULAR_VALUE
    ):
        return False, age, "singular_value"
    if boundary_mass >= RIG_JOINT_BOUNDARY_MASS_LIMIT:
        return False, age, "boundary"
    if confidence < RIG_JOINT_MIN_PUBLICATION_CONFIDENCE:
        return False, age, "confidence"
    if float(snapshot["phi_deg"]) < RIG_JOINT_CONTROL_MIN_PHI_DEG:
        return False, age, "control_envelope"
    if float(snapshot["n"]) < RIG_JOINT_CONTROL_MIN_N - 1.0e-12:
        return False, age, "control_envelope"
    return True, age, "ready"


def _joint_snapshot_parameters(
    snapshot: Optional[Mapping[str, Any]],
) -> Optional[dict[str, float]]:
    """Extract a finite, complete MPC parameter bundle from a snapshot."""

    if not snapshot:
        return None
    raw = snapshot.get("terrain_params", snapshot.get("parameters"))
    if not isinstance(raw, Mapping):
        return None
    try:
        parameters = {
            key: float(raw[key])
            for key in TERRAIN_MPC_PARAM_KEYS
        }
        snapshot_n = float(snapshot["n"])
        snapshot_phi = float(snapshot["phi_deg"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    numeric = np.asarray(
        [*parameters.values(), snapshot_n, snapshot_phi],
        dtype=float,
    )
    if not np.isfinite(numeric).all():
        return None
    # Validity here is the estimator's contract grid, including the sub-clay
    # hold extension down to n = 0.40; the controller's own envelope floor
    # (RIG_JOINT_CONTROL_MIN_N) is applied as a labelled gate in
    # _joint_snapshot_control_parameters, so a sub-clay estimate reads as
    # "outside the control envelope", not as a corrupt snapshot.
    if not (
        0.4 - 1.0e-12 <= parameters["n"] <= 1.1 + 1.0e-12
        and 6.0 - 1.0e-12 <= parameters["phi"] <= 37.8 + 1.0e-12
        and all(
            parameters[key] > 0.0
            for key in ("Kphi", "Kc", "c", "k")
        )
    ):
        return None
    if not (
        np.isclose(
            parameters["n"], snapshot_n, rtol=0.0, atol=1.0e-12
        )
        and np.isclose(
            parameters["phi"], snapshot_phi, rtol=0.0, atol=1.0e-12
        )
    ):
        return None
    return parameters


def _joint_snapshot_control_parameters(
    snapshot: Optional[Mapping[str, Any]],
) -> Optional[dict[str, float]]:
    """Return a validated snapshot mapped into the NMPC operating envelope."""

    parameters = _joint_snapshot_parameters(snapshot)
    if parameters is None:
        return None
    if parameters["phi"] < RIG_JOINT_CONTROL_MIN_PHI_DEG:
        return None
    if parameters["n"] < RIG_JOINT_CONTROL_MIN_N - 1.0e-12:
        return None
    return parameters


def _grit_constructor_kwargs(
    initial_terrain: dict,
    vehicle_params: dict,
    *,
    verbose: bool = False,
) -> dict:
    """Return the declared runtime contract of the joint terrain estimator.

    Every field of this configuration is fixed: the estimator is evaluated and
    reported under exactly these settings, so a run may only reproduce them or
    announce a departure from them.
    """

    return {
        "model_dir": str(_RIG_DYNAMICS_JOINT_MODEL_DIR),
        "initial_terrain": dict(initial_terrain),
        "vehicle_params": dict(vehicle_params),
        "update_interval": 1,
        "verbose": bool(verbose),
        "grid_size": 41,
        "student_dof": 4.0,
        "smoothing_alpha": 1.0,
        "block_dt": 0.5,
        "horizon": 8.0,
        "min_windows": 8,
        "min_window_samples": 4,
        "r_ax": 0.35,
        "r_ay": 0.45,
        "min_information": 0.20,
        "min_yaw_rate_rms": 0.015,
        "min_model_speed": 2.5,
        "max_abs_alpha": 0.35,
        "enforce_feature_envelope": True,
        "slip_mode": "average",
        "fixed_kappa": 0.05,
        "rate_mode": "zero",
        "force_gain_std": 0.04,
        "ax_bias_std": 0.10,
        "ay_bias_std": 0.05,
        "force_gain_bounds": (0.70, 1.30),
        "acceleration_bias_bound": 0.30,
        "profile_iterations": 8,
        "phi_grid_size": 17,
        "phi_bounds_deg": (6.0, 37.8),
        "cohesion_multiplier_bounds": (0.70, 1.30),
        "cohesion_grid_size": 1,
        "cohesion_prior_std": 0.20,
        "load_transfer_mode": "static",
        "min_joint_information": 0.20,
        "min_n_information": 0.0,
        "min_phi_information": 0.0,
        "min_observability_rank": 2,
        "min_observability_singular_value": 0.10,
        "boundary_warning_mass": 0.25,
        "posterior_summary": "mean",
        "block_alpha_rate": False,
        "n_bounds": (0.40, 1.10),
        "manifold_soft_floor": 0.40,
        "manifold_soft_mode": "hold",
    }


def _terrain_estimator_observation_for_backend(
    backend: str,
    observation: dict,
) -> dict:
    """Remove channels outside the joint estimator's declared sensor contract.

    The estimator is entitled to the listed observations and to nothing else,
    so restricting the mapping here keeps a richer observation dictionary
    assembled for other consumers from reaching it.
    """

    values = dict(observation)
    if str(backend) != GRIT_BACKEND:
        return values
    return {
        key: values[key]
        for key in _RIG_DYNAMICS_JOINT_OBSERVATION_KEYS
        if key in values
    }

from path_utils import make_path_function
from tire_input_features import (
    VehicleGeometry,
    compute_bicycle_operating_point,
    fz_with_lateral_transfer,
    kappa_from_wheel_speed,
    lateral_load_transfer_dFz,
    level_specific_force_to_yaw_frame,
)


def _resolve_project_path(path_like: str) -> Path:
    """Resolve relative runtime artifact paths from the project root."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(__file__).resolve().parents[2] / path).resolve()


def _estimator_controls_terrain(args) -> bool:
    """Whether terrain must remain private from the controller bootstrap."""
    return bool(
        getattr(args, "terrain_estimator", False)
        and getattr(args, "model", "nn") == "nn"
    )


def _controller_prior_name(args) -> str:
    """Return the controller-side terrain prior without consulting plant truth.

    Runs with the estimator enabled begin from the declared neutral prior. An
    explicit ``--controller-prior-terrain`` takes precedence over it, and runs
    with the estimator disabled use the plant terrain, which is the matched
    condition the component comparisons are measured under.
    """
    explicit = getattr(args, "controller_prior_terrain", None)
    if explicit:
        return str(explicit)
    if _estimator_controls_terrain(args):
        return str(getattr(args, "terrain_estimator_prior", "dirt"))
    return str(args.terrain)


def _controller_visible_sim_config(args, sim_config: dict) -> dict:
    """Remove plant-soil fields when the controller owns a private prior."""
    if getattr(args, "controller_prior_terrain", None) or _estimator_controls_terrain(args):
        return {
            key: value
            for key, value in sim_config.items()
            if key not in ("terrain_params", "terrain_preset")
        }
    return dict(sim_config)


def _reference_profile_friction_angle(args, terrain_params: dict):
    """Select the friction angle that conditions the static speed reference.

    ``--legacy-speed-ref`` selects the controlled-ablation mode, which
    disables the live g-g cap and removes terrain from the precomputed
    curvature profile, so that every terrain-conditioning arm is asked to
    follow the same path at the same speed.
    """
    fixed = getattr(args, "reference_profile_friction_angle_deg", None)
    if fixed is not None:
        return float(fixed)
    if getattr(args, "legacy_speed_ref", False):
        return None
    return terrain_params.get("phi")


def _mpc_build_friction_angle(args, terrain_params: dict):
    """Return the compile-time lateral-bound terrain, if one is requested."""
    fixed = getattr(args, "shared_ay_bound_friction_angle_deg", None)
    if fixed is not None:
        return float(fixed)
    if getattr(args, "terrain_independent_ay_bound", False):
        return None
    return terrain_params.get("phi")


def _config_dict_from_cli(args) -> dict:
    """Authoritative bootstrap for ACADOS (do not block on sim)."""
    vehicle_params = get_vehicle_params_for_demo()
    # ``--controller-prior-terrain`` decouples the terrain the controller
    # assumes from the plant terrain given by ``--terrain``, which is how the
    # mismatched-prior condition is set up: the plant may run on sand while
    # the controller's static prior remains dirt.  Left unset, the prior
    # follows the plant terrain, giving the matched condition.
    prior_name = _controller_prior_name(args)
    tp = get_terrain_preset(prior_name)
    terrain_params = terrain_preset_to_internal(tp)
    return {
        "vehicle_params": vehicle_params,
        "terrain_params": terrain_params,
        "terrain_preset": prior_name,
        "path_type": args.path,
        "v_target": args.speed,
        "sim_time": args.time,
        "sine_amplitude": args.sine_amplitude,
        "sine_wavelength": args.sine_wavelength,
        "lead_in": args.lead_in,
    }


def _unpack_run_config(config: dict, args) -> tuple:
    vehicle_params = config["vehicle_params"]
    terrain_params = config["terrain_params"]
    v_target = float(config.get("v_target", args.speed))
    path_type = config.get("path_type", args.path)
    sine_amp = float(config.get("sine_amplitude", args.sine_amplitude))
    sine_wl = float(config.get("sine_wavelength", args.sine_wavelength))
    lead_in = float(config.get("lead_in", args.lead_in))
    terrain_name = config.get("terrain_preset", args.terrain)
    return (
        vehicle_params,
        terrain_params,
        v_target,
        path_type,
        sine_amp,
        sine_wl,
        lead_in,
        terrain_name,
    )


def _drain_latest_sim_config(state_sub) -> Optional[dict]:
    """Return the most recent buffered SimStatus config payload, without blocking."""
    latest = None
    while True:
        result = state_sub.recv(timeout_ms=0)
        if result is None:
            break
        _, msg = result
        if isinstance(msg, SimStatus) and msg.event == "config" and msg.config:
            latest = msg.config
    return latest


def _send_ready_control_ping(ctrl_pub, integrator: ControlIntegrator) -> None:
    """Publish a neutral command that releases the simulator's controller wait.

    The simulation node holds at ``--wait-for-controller`` until it sees a
    ControlCommand, which would otherwise deadlock against a controller that
    is itself waiting for the first VehicleState.
    """
    cmd = ControlCommand(
        time=0.0,
        wall_time=wall_time.time(),
        seq=0,
        delta=float(integrator.steering_angle),
        drive_torque=0.0,
        acceleration=float(integrator.acceleration),
        delta_dot=0.0,
        jerk=0.0,
        solve_time_ms=0.0,
        mpc_cost=0.0,
    )
    ctrl_pub.send(cmd)


def _acados_build_directory(
    args, tire_model: str, nn_model_id: Optional[str]
) -> Optional[Path]:
    """Return an explicit ACADOS build directory when the CLI requests one.

    The solver otherwise owns the fingerprint-keyed cache layout.  A directory
    supplied here is treated as user-provided and bypasses that keying, so
    concurrent workers with differing solver configurations would share one
    directory; pass it only when a single configuration is in flight.
    """
    if args.acados_build_dir:
        return Path(args.acados_build_dir).expanduser().resolve()
    return None


def _terrain_estimate_bundle(
    terrain_name: str,
    terrain_params: dict,
) -> Tuple[str, float, dict, float]:
    """Return the terrain estimate implied by a known preset.

    Used when no online estimator is running, so the reported class, sinkage
    exponent, and parameter bundle come from the configured preset and carry
    full confidence.
    """
    return (
        terrain_name,
        float(terrain_params.get("n", 1.1)),
        {k: terrain_params[k] for k in TERRAIN_MPC_PARAM_KEYS},
        1.0,
    )


def _wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


# =============================================================================
# Main controller loop
# =============================================================================

def run_controller_node(args):
    print("=" * 60)
    print("ACADOS MPC Controller Node (Decoupled)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Subscribe first; ACADOS builds from CLI without waiting for the simulator.
    # If Chrono already published config, merge it after warmup (queue drain).
    # ------------------------------------------------------------------
    state_sub = make_subscriber(sim_sub_endpoint(args.sim_host, args.sim_port),
                                args.transport, topic=TOPIC_VEHICLE_STATE)
    print(f"  Subscribing to state from {args.sim_host}:{args.sim_port} ({args.transport})")
    print("  ACADOS will compile from CLI now (no wait for sim config).")

    config = _config_dict_from_cli(args)
    (
        vehicle_params,
        terrain_params,
        v_target,
        path_type,
        sine_amp,
        sine_wl,
        lead_in,
        terrain_name,
    ) = _unpack_run_config(config, args)

    # ------------------------------------------------------------------
    # Build MPC (ACADOS)
    # ------------------------------------------------------------------
    dt_mpc = float(args.mpc_dt)
    N_horizon = int(args.mpc_n)
    tire_model = args.model

    # Load NN tire model (only for nn mode)
    nn_tire = None
    nn_model_version: Optional[str] = None
    if tire_model == 'nn':
        base_path = Path(__file__).resolve().parents[2]
        nn_model_version = args.nn_model
        requested_model = Path(nn_model_version).expanduser()
        if requested_model.is_absolute() or len(requested_model.parts) > 1:
            model_dir = requested_model if requested_model.is_absolute() else base_path / requested_model
        else:
            model_dir = base_path / "nn_models" / nn_model_version
        if not model_dir.exists():
            print(f"  ERROR: NN model directory not found: {model_dir}")
            sys.exit(1)

        nn_tire = load_nn_tire_model(str(model_dir), terrain_params)
        print(f"  NN model loaded: {nn_tire.model_type} "
              f"(input_dim={nn_tire.input_dim}, params={nn_tire.n_params})")

    acados_build_dir = _acados_build_directory(args, tire_model, nn_model_version)

    # Solver uses symbolic ax-based kappa for the prediction horizon;
    # 'measured' mode only affects the controller's operating point.
    _solver_kappa = 'approx' if args.kappa == 'measured' else args.kappa

    mpc = AcadosMPC(
        nn_tire_model=nn_tire,
        dt=dt_mpc,
        N=N_horizon,
        lateral_load_transfer=not args.no_lat_transfer,
        kappa_mode=_solver_kappa,
        tire_model=tire_model,
        build_dir=acados_build_dir,
        symbolic_rates=args.symbolic_rates,
        friction_angle_deg=_mpc_build_friction_angle(args, terrain_params),
        rate_feature_dt=float(args.nn_rate_sample_dt),
        oracle_terrain=(terrain_name if tire_model in ('pacejka-oracle', 'pacejka-rigfit') else None),
        speed_weight=float(args.speed_weight),
        speed_cost_mode=args.speed_cost_mode,
        obstacle_weight=float(args.obstacle_weight),
        longitudinal_force_balance=bool(getattr(args, "longitudinal_force_balance", False)),
    )
    _force_balance = bool(getattr(args, "longitudinal_force_balance", False))
    if tire_model == 'nn':
        model_label = f"ACADOS-NN ({nn_tire.model_type})"
    else:
        model_label = f"ACADOS-{tire_model}"
    print(f"  MPC built: {model_label}, N={N_horizon}, dt={dt_mpc}s")
    if getattr(args, "shared_ay_bound_friction_angle_deg", None) is not None:
        print(
            "  NMPC lateral bound policy: shared design envelope "
            f"phi={float(args.shared_ay_bound_friction_angle_deg):g} deg "
            f"(ay_max={mpc.ay_max:.3f} m/s^2)"
        )

    # Rate tracker for rate-augmented NN (skip if symbolic rates)
    rate_tracker = None
    if mpc.rate_mode and not mpc._symbolic_rate_mode:
        rate_tracker = RateTracker(sample_dt=args.nn_rate_sample_dt)
        print(
            f"  Rate-augmented surrogate: dkappa/dt, dalpha/dt, du/dt over "
            f">={args.nn_rate_sample_dt:g}s sim intervals "
            f"(matching the checkpoint's training sample interval)"
        )
    elif mpc._symbolic_rate_mode:
        print("  Symbolic rate mode: rates computed inside MPC dynamics (no RateTracker needed)")

    # Warmup ACADOS solver (first solves trigger JIT)
    print("  Warming up ACADOS solver...", end="", flush=True)
    v_warm = float(
        np.clip(
            max(MPC_STATE_MIN_FORWARD_SPEED_MPS, v_target),
            mpc.u_min,
            mpc.u_max,
        )
    )
    z0_warm = np.zeros(mpc.nx)
    z0_warm[3] = v_warm
    x_ref_w = np.linspace(0, v_warm * dt_mpc * N_horizon, N_horizon + 1)
    y_ref_w = np.zeros(N_horizon + 1)
    psi_ref_w = np.zeros(N_horizon + 1)
    v_ref_w = v_warm * np.ones(N_horizon + 1)
    for _ in range(int(args.warmup_iters)):
        warm_kwargs = dict(terrain_params=terrain_params)
        mpc.solve(z0_warm, x_ref_w, y_ref_w, psi_ref_w, v_ref_w, x_ref_w[-1], 0, 0,
                  **warm_kwargs)
        print(".", end="", flush=True)
    mpc.reset_warmstart()  # force kinematic rollout from real z0
    print(" done!")

    # Merge sim-published config if it is already in the transport queue.
    sim_cfg = _drain_latest_sim_config(state_sub)
    if sim_cfg:
        # Never let a plant-soil broadcast overwrite an explicit controller
        # prior or an estimator-enabled controller's blind bootstrap prior.
        sim_cfg = _controller_visible_sim_config(args, sim_cfg)
        config.update(sim_cfg)
        (
            vehicle_params,
            terrain_params,
            v_target,
            path_type,
            sine_amp,
            sine_wl,
            lead_in,
            terrain_name,
        ) = _unpack_run_config(config, args)
        print("  Merged buffered sim config (Chrono published before/during ACADOS init).")
    else:
        print("  No sim config in queue yet — using CLI. Match --terrain/--path/--speed/--lead-in to chrono, "
              "or start sim before controller so config is buffered.")

    print(f"  Terrain: {terrain_name}")
    print(f"  Path: {path_type}, v_target: {v_target} m/s")
    if lead_in > 0:
        print(f"  Lead-in: {lead_in:.0f}m straight before path")

    # ------------------------------------------------------------------
    # Timestamped run directory
    # ------------------------------------------------------------------
    model_tag = f"acados_{nn_tire.model_type}" if nn_tire is not None else f"acados_{tire_model}"
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(args.plot_dir) / f"{run_ts}_{terrain_name}_{path_type}_{model_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Optional logging of the predicted horizon for rollout validation. It is
    # gated on an environment variable and therefore inert on ordinary runs;
    # when enabled, each solve's trajectory is buffered and written to
    # mpc_predictions.npz in the run directory at shutdown.
    _log_mpc_pred = bool(os.environ.get("LOG_MPC_PREDICTIONS"))
    _pred_times, _pred_Z = [], []

    # ------------------------------------------------------------------
    # Reference path
    # ------------------------------------------------------------------
    ref_path = make_path_function(
        path_type=path_type,
        v_target=v_target,
        sine_amplitude=sine_amp,
        sine_wavelength=sine_wl,
        use_closest_point=not args.no_path_reindex,
        lead_in=lead_in,
        csv_dir=str(run_dir),
        friction_angle_deg=_reference_profile_friction_angle(args, terrain_params),
        ay_safety=float(getattr(args, 'ay_safety', 0.65)),
    )
    path_func = ref_path.get_reference
    _fixed_reference_phi = getattr(
        args, "reference_profile_friction_angle_deg", None
    )
    if _fixed_reference_phi is not None:
        _reference_policy = (
            "shared design-envelope curvature-only "
            f"(phi={float(_fixed_reference_phi):g} deg)"
        )
    elif getattr(args, "legacy_speed_ref", False):
        _reference_policy = "terrain-independent curvature-only"
    else:
        _reference_policy = "prior-conditioned curvature + live g-g cap"
    print(f"  Reference speed policy: {_reference_policy}")

    # ------------------------------------------------------------------
    # Transport delay compensation
    # ------------------------------------------------------------------
    delay_est = DelayEstimator(initial_delay=args.initial_delay)
    state_predictor = StatePredictor(vehicle_params, dt_prop=args.state_predict_dt)
    control_buffer = collections.deque(maxlen=int(args.control_buffer_len))

    # ------------------------------------------------------------------
    # Control integrator
    # ------------------------------------------------------------------
    integrator = ControlIntegrator(
        mpc, v_target=v_target,
        dob_ki=float(getattr(args, "dob_ki", 0.15)),
        dob_max=float(getattr(args, "dob_max", 0.35)),
        dob_bleed=float(getattr(args, "dob_bleed", 0.5)),
    )

    # ------------------------------------------------------------------
    # Tracking analytics
    # ------------------------------------------------------------------
    analytics = TrackingAnalytics(
        ref_path=ref_path,
        v_target=v_target,
        rms_time_start=args.rms_time_start,
        path_type=path_type,
    )

    # ------------------------------------------------------------------
    # Publisher for control commands
    # ------------------------------------------------------------------
    ctrl_pub = make_publisher(ctrl_pub_endpoint(args.ctrl_port), args.transport,
                              topic=TOPIC_CONTROL_CMD)
    print(f"  Publishing controls on port {args.ctrl_port} ({args.transport})")

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    seq = 0
    last_state: VehicleState = None
    solve_times = []
    last_sim_time = None
    # Pending prediction targets: (sim_time_target, predicted 9-state at target).
    pred_targets = collections.deque(maxlen=max(64, 4 * int(mpc.N)))
    pred_pos_err_hist = []
    pred_psi_err_hist = []
    pred_u_err_hist = []
    pred_v_err_hist = []
    pred_omega_err_hist = []
    pred_age_hist = []
    # Previous applied road-wheel angle used to estimate realized steering-rate
    # at the current state sample.
    prev_applied_delta = float(integrator.steering_angle)
    # Effective δ̇ for delay predictor buffer and NN steering_rate feature.
    last_delta_dot_cmd = 0.0
    last_Jx_cmd = 0.0
    # Measured-ax state (from IMU, complementary-filtered)
    _measured_ax = 0.0
    _ax_filter_tau = args.ax_filter_tau
    # Exponential-moving-average filter on the velocity state, applied before
    # the MPC sees it. Lateral velocity is the limiting channel: its noise
    # standard deviation of 0.05 m/s sits against a signal of roughly 0.1 to
    # 0.5 m/s, and any residual formed as (z_meas - z_pred)/dt amplifies that
    # noise by 1/dt.
    _vel_filter_tau = float(getattr(args, 'vel_filter_tau', 0.0))
    _vel_filt_u: Optional[float] = None
    _vel_filt_v: Optional[float] = None
    _vel_filt_omega: Optional[float] = None
    terrain_class_est, n_terrain_est, terrain_params_est, terrain_confidence = (
        _terrain_estimate_bundle(
            terrain_name,
            terrain_params,
        )
    )

    # ------------------------------------------------------------------
    # Online terrain parameter estimator
    # ------------------------------------------------------------------
    terrain_estimator = None
    _te_omega_prev = None
    _te_time_prev = None
    _te_alpha_f_prev = None
    _te_alpha_r_prev = None
    # Cache of the most recent live terrain estimate. Every ControlCommand
    # carries this snapshot so the sim-side safety shield can re-condition
    # its NN surrogate and tighten the friction-cone gate by sigma_phi.
    latest_terrain_update = {
        "seq": 0, "n": None, "n_sigma": None, "phi_deg": None, "phi_sigma_deg": None,
        "Kphi": None, "Kc": None, "c": None, "k": None,
        "terrain_class": None, "confidence": None,
    }
    _joint_snapshot_latch = _JointSnapshotLatch()
    _joint_control_snapshot: Optional[Mapping[str, Any]] = None
    _joint_estimator_fault = False
    _joint_policy_state = None
    _joint_publication_ready = 0
    _joint_snapshot_age_s = float("nan")
    _joint_fallback_reason = "not_joint_backend"
    _joint_fallback = _joint_fixed_fallback_params()
    if args.terrain_estimator and args.model == "nn":
        # Initialise the estimator from a neutral prior near the middle of the
        # admissible sinkage-exponent range: the dirt preset, n = 0.7. This
        # follows Dallas et al., who start from an incorrect but not extreme
        # initial guess, so that the reported convergence is not an artifact
        # of a prior placed on the answer.
        _te_prior_name = _controller_prior_name(args)
        _te_default = terrain_preset_to_internal(get_terrain_preset(_te_prior_name))
        if args.terrain_estimator_backend == "bekker_ukf":
            from bekker_ukf_terrain_estimator import BekkerUKFTerrainEstimator
            terrain_estimator = BekkerUKFTerrainEstimator(
                initial_terrain=_te_default,
                update_interval=args.te_update_interval,
                verbose=bool(getattr(args, "te_verbose", False)),
                q_n=float(getattr(args, "nn_ukf_q_n", 0.01)))
            _te_src_desc = "backend=bekker_ukf [online analytical-Bekker UKF]"
        elif args.terrain_estimator_backend == GRIT_BACKEND:
            from grit_terrain_estimator import (
                GritTerrainEstimator,
            )
            _joint_kwargs = _grit_constructor_kwargs(
                _te_default,
                vehicle_params,
                verbose=bool(getattr(args, "te_verbose", False)),
            )
            # Candidate-evaluation overrides. The declared contract above is
            # unchanged, so a run either reproduces it exactly or prints its
            # departure at launch, and logs from different configurations can
            # never be mistaken for one another.
            if getattr(args, "te_joint_model_dir", None):
                _joint_kwargs["model_dir"] = str(args.te_joint_model_dir)
                print(f"[CANDIDATE-OVERRIDE] joint estimator model_dir = "
                      f"{args.te_joint_model_dir}")
            if getattr(args, "te_joint_r_ay", None) is not None:
                _joint_kwargs["r_ay"] = float(args.te_joint_r_ay)
                print(f"[CANDIDATE-OVERRIDE] joint estimator r_ay = "
                      f"{args.te_joint_r_ay}")
            terrain_estimator = GritTerrainEstimator(
                **_joint_kwargs
            )
            _te_src_desc = (
                "backend=grit [joint sinkage-exponent and "
                "friction-angle profile over the rate force surrogate, "
                "requiring no ground datum]"
            )
        elif args.terrain_estimator_backend == "scalar_parent":
            from scalar_parent_terrain_estimator import (
                ScalarParentTerrainEstimator,
            )
            terrain_estimator = ScalarParentTerrainEstimator(
                model_dir=(
                    str(Path(args.ukf_model_dir).resolve())
                    if args.ukf_model_dir else None
                ),
                initial_terrain=_te_default,
                vehicle_params=vehicle_params,
                update_interval=int(args.estimator_update_interval),
                verbose=bool(getattr(args, "te_verbose", False)),
                grid_size=int(args.parent_grid_size),
                student_dof=float(args.parent_student_dof),
                block_dt=float(args.estimator_block_dt),
                horizon=float(args.estimator_horizon),
                min_windows=int(args.estimator_min_windows),
                min_window_samples=int(args.estimator_min_window_samples),
                r_ax=float(args.estimator_r_ax),
                r_ay=float(args.estimator_r_ay),
                min_information=float(args.estimator_min_information),
                min_yaw_rate_rms=float(args.estimator_min_yaw_rate_rms),
                min_model_speed=float(args.estimator_min_speed),
                max_abs_alpha=float(args.estimator_max_abs_alpha),
                enforce_feature_envelope=bool(
                    args.estimator_enforce_feature_envelope
                ),
                smoothing_alpha=1.0,
                slip_mode=str(args.estimator_slip_mode),
                fixed_kappa=float(args.estimator_fixed_kappa),
                rate_mode=str(args.estimator_rate_mode),
                force_gain_std=float(args.estimator_force_gain_std),
                ax_bias_std=float(args.estimator_ax_bias_std),
                ay_bias_std=float(args.estimator_ay_bias_std),
                force_gain_bounds=(
                    float(args.estimator_force_gain_min),
                    float(args.estimator_force_gain_max),
                ),
                acceleration_bias_bound=float(
                    args.estimator_acceleration_bias_bound
                ),
                profile_iterations=int(args.estimator_profile_iterations),
            )
            _te_src_desc = (
                "backend=scalar_parent [profiled longitudinal/lateral "
                "dynamics over the single-tire-rig force map, requiring no "
                "ground datum]"
            )
        else:
            raise SystemExit(
                "unsupported terrain estimator backend: "
                f"{args.terrain_estimator_backend}"
            )
        # Start from the configured prior, which carries no plant information.
        # The joint backend replaces it below with its labelled fallback.
        terrain_params_est = dict(_te_default)
        n_terrain_est = _te_default['n']
        terrain_class_est = "estimating"
        terrain_confidence = 0.0
        estimator_outputs = ",".join(terrain_estimator.output_names)
        if args.terrain_estimator_backend == GRIT_BACKEND:
            _te_runtime_desc = (
                "publish_every=1, block_dt=0.5s, horizon=8s, "
                "grid=41x17, confidence>=0.20, evidence_age<=3.5s, "
                "boundary_mass<0.25"
            )
        elif args.terrain_estimator_backend == "scalar_parent":
            _te_runtime_desc = (
                f"publish_every={args.estimator_update_interval}, "
                f"block_dt={args.estimator_block_dt:g}s, "
                f"horizon={args.estimator_horizon:g}s"
            )
        else:
            _te_runtime_desc = (
                f"publish_every={args.te_update_interval}, "
                f"filter_dt={args.te_filter_dt:g}s"
            )
        print(
            f"  Terrain estimator: ON  (mode={args.terrain_estimator_mode}, "
            f"outputs={estimator_outputs}, init={_te_prior_name}/n={_te_default['n']:.2f}, "
            f"{_te_runtime_desc}, {_te_src_desc})"
        )
        if args.terrain_estimator_backend == GRIT_BACKEND:
            # The estimator starts from a labelled, control-feasible low-grip
            # fallback and departs from it once an accepted snapshot satisfies
            # every publication gate.
            terrain_params_est = dict(_joint_fallback)
            n_terrain_est = float(_joint_fallback["n"])
            terrain_confidence = 0.0
            terrain_class_est = "joint_fallback:no_snapshot"
            latest_terrain_update.update({
                "seq": 1,
                "n": float(_joint_fallback["n"]),
                "n_sigma": None,
                "phi_deg": float(_joint_fallback["phi"]),
                "phi_sigma_deg": None,
                "Kphi": float(_joint_fallback["Kphi"]),
                "Kc": float(_joint_fallback["Kc"]),
                "c": float(_joint_fallback["c"]),
                "k": float(_joint_fallback["k"]),
                "terrain_class": terrain_class_est,
                "confidence": 0.0,
            })
            _joint_policy_state = (0, False, "no_snapshot")
            _joint_fallback_reason = "no_snapshot"
    elif args.terrain_estimator and args.model != "nn":
        print("  WARNING: --terrain-estimator requires --model nn (disabled)")

    terrain_id_probe = None
    if bool(getattr(args, "terrain_id_probe", False)):
        terrain_id_probe = TerrainIDProbe(TerrainIDProbeConfig(
            target_abs_alpha_rad=float(args.terrain_id_probe_target_alpha),
            target_slew_rad_s=float(args.terrain_id_probe_slew_rate),
            signed_dwell_s=float(args.terrain_id_probe_signed_dwell),
            max_latency_s=float(args.terrain_id_probe_max_latency),
        ))
        print(
            "  Terrain ID probe: ON "
            f"(achieved |alpha_f|={args.terrain_id_probe_target_alpha:.3f} rad, "
            f"slew={args.terrain_id_probe_slew_rate:.3f} rad/s)"
        )

    # ------------------------------------------------------------------
    # Online lateral-force bias diagnostic
    # ------------------------------------------------------------------
    # A signed exponential moving average of the lateral-force error
    # (Chrono truth minus surrogate prediction) per axle. The sign carries the
    # direction of the net force correction the model would need, and the
    # averaging lag damps the oscillation seen on sinusoidal paths. This is a
    # diagnostic quantity: it is recorded but not fed back into the control
    # law.
    _fy_bias_signed_f = 0.0
    _fy_bias_signed_r = 0.0
    _FY_BIAS_ALPHA = 0.10   # EMA smoothing (signed)
    _FY_BIAS_CLIP = 3000.0  # max correction [N]
    _FY_BIAS_MIN_SPEED = 2.0  # only update above this speed

    # ------------------------------------------------------------------
    # Diagnostic CSV logger
    # ------------------------------------------------------------------
    csv_file = None
    csv_writer = None
    csv_path = None
    if not args.no_csv:
        csv_path = run_dir / f"diag_{terrain_name}_{path_type}_{model_tag}.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_header = [
            "sim_time", "wall_time", "seq",
            "x_fa_meas", "y_fa_meas", "psi_meas", "u_meas", "v_meas", "omega_meas",
            "x_fa_true", "y_fa_true", "psi_true", "u_true",
            "x_fa_comp", "y_fa_comp", "psi_comp", "u_comp", "v_comp", "omega_comp",
            "ax_state", "delta_prev_state", "ax_imu", "ay_imu",
            "wheel_omega_fl", "wheel_omega_fr", "wheel_omega_rl", "wheel_omega_rr",
            "steering_angle_sensor",
            "z_cg", "quat_e0", "quat_e1", "quat_e2", "quat_e3",
            "az_imu", "omega_x_imu", "omega_y_imu",
            "x_ref_0", "y_ref_0", "psi_ref_0", "v_ref_0",
            "delta_dot", "Jx", "mpc_cost", "solver_status", "solver_iters",
            "terrain_class_est", "terrain_confidence", "n_terrain_est",
            "n_terrain_estimator",
            "phi_terrain_est_deg", "phi_terrain_estimator_deg",
            "phi_terrain_sigma_deg", "n_terrain_sigma", "terrain_update_applied",
            "terrain_dynamics_active", "terrain_dynamics_windows",
            "terrain_accepted_dynamics_windows",
            "terrain_rejected_dynamics_windows", "terrain_profile_force_gain",
            "terrain_profile_ax_bias", "terrain_profile_ay_bias",
            "terrain_profile_bound_hits", "terrain_feature_envelope_excursions",
            "terrain_joint_snapshot_seq", "terrain_joint_evidence_time",
            "terrain_joint_evidence_age_s", "terrain_joint_publication_ready",
            "terrain_joint_fallback_reason", "terrain_joint_snapshot_confidence",
            "terrain_joint_n_boundary_mass", "terrain_joint_phi_boundary_mass",
            "terrain_joint_max_boundary_mass", "terrain_joint_boundary_limited",
            "terrain_joint_observability_rank",
            "terrain_joint_observability_min_singular_value",
            "terrain_joint_projection_wall_ms",
            "terrain_joint_profile_wall_ms",
            "terrain_joint_observability_wall_ms",
            "terrain_joint_posterior_wall_ms",
            "terrain_joint_publication_wall_ms",
            "terrain_joint_update_wall_ms",
            "steering", "throttle", "braking", "steering_angle", "acceleration",
            "tau_one_way_ms", "tau_solve_ms", "tau_comp_ms", "solve_time_ms",
            "crosstrack_err", "heading_err_deg", "speed_err",
            "pred1_age_s", "pred1_pos_err_m", "pred1_psi_err_deg",
            "pred1_u_err_mps", "pred1_v_err_mps", "pred1_omega_err_radps",
            "actual_Fx_front", "actual_Fx_rear",
            "actual_Fy_front", "actual_Fy_rear", "pred_Fy_front", "pred_Fy_rear",
            "alpha_f", "alpha_r", "Fz_f_mean", "Fz_r_mean",
            "kappa_diag", "kappa_meas_diag", "sr_diag", "u_safe_diag", "speed_fade_diag",
            "est_alpha_f", "est_alpha_r", "est_Fz_f_mean", "est_Fz_r_mean",
            "est_kappa", "est_alpha_rate_f", "est_alpha_rate_r", "est_u_safe",
            "terrain_probe_phase", "terrain_probe_target_alpha",
            "terrain_probe_override", "terrain_probe_reason",
            "terrain_probe_nominal_delta", "terrain_probe_requested_delta",
        ]
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        print(f"  Diagnostic CSV: {csv_path}")

    # Strict, sensor-only replay artifact.  This intentionally excludes
    # Chrono tire-force diagnostics, true pose, plant soil parameters, and the
    # live terrain estimate.  ``.17g`` writes round-trip-safe Python floats so
    # replay takes the same filter-time branches as the online estimator.
    terrain_obs_file = None
    terrain_obs_writer = None
    terrain_obs_path = None
    if not args.no_csv:
        terrain_obs_path = run_dir / "terrain_observations.csv"
        terrain_obs_file = open(terrain_obs_path, "w", newline="")
        terrain_obs_writer = csv.writer(terrain_obs_file)
        terrain_obs_writer.writerow([
            "seq", "sim_time", "x_cg", "y_cg", "z_cg",
            "quat_e0", "quat_e1", "quat_e2", "quat_e3", "psi",
            "u_raw", "v_lateral_raw", "omega_raw",
            "u", "v_lateral", "omega",
            "ax_imu", "ay_imu", "az_imu", "ax_filtered",
            "roll_rate", "pitch_rate",
            "wheel_omega_fl", "wheel_omega_fr",
            "wheel_omega_rl", "wheel_omega_rr",
            "wheel_center_z_fl", "wheel_center_z_fr",
            "wheel_center_z_rl", "wheel_center_z_rr",
            "drive_torque_fl", "drive_torque_fr",
            "drive_torque_rl", "drive_torque_rr",
            "brake_torque_fl", "brake_torque_fr",
            "brake_torque_rl", "brake_torque_rr",
            "steering_angle", "kappa", "alpha_f", "alpha_r",
            "alpha_rate_f", "alpha_rate_r", "Fz_f", "Fz_r",
        ])

    tire_geom = VehicleGeometry(
        Lf=float(mpc.Lf),
        Lr=float(mpc.Lr),
        M=float(mpc.M),
        h_cg=float(mpc.h_cg),
        T=float(mpc.T),
    )

    print(f"  Delay compensation: {'ON' if not args.no_delay_comp else 'OFF'} "
          f"(initial τ={args.initial_delay * 1000:.0f}ms)")
    print(f"  Running ACADOS controller loop...")
    print("  ACADOS init complete — sending ready signal(s) to sim (then waiting for VehicleState)...")

    # Unblock chrono_sim_node --wait-for-controller before any VehicleState is published.
    for _ in range(3):
        _send_ready_control_ping(ctrl_pub, integrator)
        wall_time.sleep(0.05)

    running = True
    last_ready_ping = wall_time.time()
    ready_ping_interval_s = float(args.ready_ping_interval_s)
    while running:
        now = wall_time.time()
        if last_state is None and now - last_ready_ping >= ready_ping_interval_s:
            _send_ready_control_ping(ctrl_pub, integrator)
            last_ready_ping = now

        # --- Receive state ---
        result = state_sub.recv(timeout_ms=int(args.zmq_recv_timeout_ms))
        if result is None:
            continue

        topic, msg = result

        if isinstance(msg, SimStatus):
            if msg.event == "stop":
                print("  Received stop signal from sim node.")
                running = False
                break
            if (
                msg.event == "config"
                and msg.config
                and last_state is None
            ):
                # Config arrived after the post-warmup queue drain, which
                # happens when a cached solver makes initialisation very fast.
                sim_payload = msg.config
                sim_payload = _controller_visible_sim_config(args, sim_payload)
                config.update(sim_payload)
                (
                    vehicle_params,
                    terrain_params,
                    v_target,
                    path_type,
                    sine_amp,
                    sine_wl,
                    lead_in,
                    terrain_name,
                ) = _unpack_run_config(config, args)
                state_predictor = StatePredictor(vehicle_params, dt_prop=args.state_predict_dt)
                ref_path = make_path_function(
                    path_type=path_type,
                    v_target=v_target,
                    sine_amplitude=sine_amp,
                    sine_wavelength=sine_wl,
                    use_closest_point=not args.no_path_reindex,
                    lead_in=lead_in,
                    csv_dir=str(run_dir),
                    friction_angle_deg=_reference_profile_friction_angle(
                        args, terrain_params
                    ),
                    ay_safety=float(getattr(args, 'ay_safety', 0.65)),
                )
                path_func = ref_path.get_reference
                analytics.ref_path = ref_path
                analytics.v_target = v_target
                analytics.path_type = path_type
                integrator.v_target = v_target
                if not args.terrain_estimator:
                    # The plant terrain conditions the controller only when
                    # no estimator is running. With one running, the declared
                    # prior is retained and the terrain is identified online.
                    terrain_class_est, n_terrain_est, terrain_params_est, terrain_confidence = (
                        _terrain_estimate_bundle(
                            terrain_name,
                            terrain_params,
                        )
                    )
                print("  Applied sim config from stream (received after ACADOS init).")
            continue

        if not isinstance(msg, VehicleState):
            continue

        last_state = msg
        recv_time = wall_time.time()
        if last_sim_time is None:
            dt_ctrl = dt_mpc
        else:
            dt_ctrl = float(np.clip(msg.time - last_sim_time, 1e-3, 0.2))
        last_sim_time = msg.time
        terrain_update_applied = 0

        # Update delay estimate
        delay_est.update_transport(msg.wall_time, recv_time)

        # --- Build state vector ---
        psi = quat_to_yaw(msg.quat_e0, msg.quat_e1, msg.quat_e2, msg.quat_e3)
        Lf = mpc.Lf

        # Transform CG → front axle
        x_fa = msg.x_cg + Lf * np.cos(psi)
        y_fa = msg.y_cg + Lf * np.sin(psi)

        # Complementary filter fuses the model prediction (ax + Jx*dt, smooth
        # but drifts) with the IMU reading (noisy but accurate in the mean).
        # High-frequency content comes from the model; DC correction from IMU.
        ax_raw = float(np.clip(msg.ax, mpc.ax_min, mpc.ax_max))
        if _ax_filter_tau > 0 and dt_ctrl > 1e-6:
            alpha = min(dt_ctrl / (_ax_filter_tau + dt_ctrl), 1.0)
            ax_pred = float(np.clip(
                _measured_ax + last_Jx_cmd * dt_ctrl,
                mpc.ax_min, mpc.ax_max))
            _measured_ax = (1.0 - alpha) * ax_pred + alpha * ax_raw
        else:
            _measured_ax = ax_raw

        # MPC state [x,y,ψ,u,v,ω, ax, δ_prev, Jx_prev] (direct δ control)
        ax_for_state = _measured_ax
        z0_measured = np.array([
            x_fa, y_fa, psi,
            max(msg.u, MPC_STATE_MIN_FORWARD_SPEED_MPS),
            msg.v, msg.omega,
            ax_for_state,
            integrator.steering_angle,
            last_Jx_cmd,
        ])

        # Smooth u, v, and omega before the MPC consumes them, so that sensor
        # noise reaches neither the initial condition nor any derived residual.
        if _vel_filter_tau > 0 and dt_ctrl > 1e-6:
            _alpha_vel = min(dt_ctrl / (_vel_filter_tau + dt_ctrl), 1.0)
            if _vel_filt_u is None:
                # First measurement: initialise to raw values
                _vel_filt_u = float(z0_measured[3])
                _vel_filt_v = float(z0_measured[4])
                _vel_filt_omega = float(z0_measured[5])
            else:
                _vel_filt_u = (1.0 - _alpha_vel) * _vel_filt_u + _alpha_vel * float(z0_measured[3])
                _vel_filt_v = (1.0 - _alpha_vel) * _vel_filt_v + _alpha_vel * float(z0_measured[4])
                _vel_filt_omega = (1.0 - _alpha_vel) * _vel_filt_omega + _alpha_vel * float(z0_measured[5])
            z0_measured[3] = _vel_filt_u
            z0_measured[4] = _vel_filt_v
            z0_measured[5] = _vel_filt_omega

        # Append α_f_prev, α_r_prev, δ_sr_prev for symbolic rate mode (nx=12)
        # Use z0_measured[3,4,5] (post-filter) for consistency with the filtered MPC state.
        if mpc._symbolic_rate_mode:
            u_s = max(float(z0_measured[3]), SLIP_CALC_MIN_SPEED_MPS)
            Lr = mpc.Lr
            af = integrator.steering_angle - np.arctan2(float(z0_measured[4]) + Lf * float(z0_measured[5]), u_s)
            ar = -np.arctan2(float(z0_measured[4]) - Lr * float(z0_measured[5]), u_s)
            delta_sr0 = integrator.steering_angle  # sr starts at 0
            z0_measured = np.append(z0_measured, [af, ar, delta_sr0])
        elif mpc._symbolic_sr:
            # δ_sr_prev: set to current δ so that sr ≈ 0 at measurement time
            z0_measured = np.append(z0_measured, [integrator.steering_angle])

        # Evaluate matured 1-step prediction targets from previous solves.
        pred_age = float("nan")
        pred_pos_err = float("nan")
        pred_psi_err_deg = float("nan")
        pred_u_err = float("nan")
        pred_v_err = float("nan")
        pred_omega_err = float("nan")
        while pred_targets and float(msg.time) >= float(pred_targets[0][0]):
            t_pred, z_pred = pred_targets.popleft()
            z_pred = np.asarray(z_pred, dtype=float).reshape(-1)
            if z_pred.size < 6:
                continue
            pred_age = float(msg.time - t_pred)
            pred_pos_err = float(np.hypot(
                float(z0_measured[0]) - float(z_pred[0]),
                float(z0_measured[1]) - float(z_pred[1]),
            ))
            psi_err = _wrap_to_pi(float(z0_measured[2]) - float(z_pred[2]))
            pred_psi_err_deg = float(np.degrees(psi_err))
            pred_u_err = float(z0_measured[3] - z_pred[3])
            pred_v_err = float(z0_measured[4] - z_pred[4])
            pred_omega_err = float(z0_measured[5] - z_pred[5])

            pred_age_hist.append(pred_age)
            pred_pos_err_hist.append(pred_pos_err)
            pred_psi_err_hist.append(abs(psi_err))
            pred_u_err_hist.append(abs(pred_u_err))
            pred_v_err_hist.append(abs(pred_v_err))
            pred_omega_err_hist.append(abs(pred_omega_err))

        # --- Delay compensation ---
        # StatePredictor: 8-state [x,y,ψ,u,v,ω, δ, ax]; carry Jx_prev through.
        if not args.no_delay_comp:
            tau = delay_est.compensation_delay
            z8 = np.array([
                z0_measured[0], z0_measured[1], z0_measured[2],
                z0_measured[3], z0_measured[4], z0_measured[5],
                z0_measured[7],  # δ
                z0_measured[6],  # ax
            ])
            z8_pred, _, jx_prev_pred = state_predictor.propagate(
                z8,
                control_buffer,
                tau,
                sim_time_s=float(msg.time),
                command_lag_s=float(delay_est.one_way_delay),
                return_last_cmd=True,
            )
            z0 = np.array([
                z8_pred[0], z8_pred[1], z8_pred[2],
                z8_pred[3], z8_pred[4], z8_pred[5],
                z8_pred[7],  # ax
                z8_pred[6],  # δ_prev
                jx_prev_pred,  # Jx_prev at compensated time
            ])
            if mpc._symbolic_rate_mode:
                # Recompute α from predicted state for consistency
                u_s_pred = max(z8_pred[3], SLIP_CALC_MIN_SPEED_MPS)
                af_pred = z8_pred[6] - np.arctan2(z8_pred[4] + Lf * z8_pred[5], u_s_pred)
                ar_pred = -np.arctan2(z8_pred[4] - mpc.Lr * z8_pred[5], u_s_pred)
                delta_sr0_pred = z8_pred[6]  # predicted δ → sr starts at 0
                z0 = np.append(z0, [af_pred, ar_pred, delta_sr0_pred])
            elif mpc._symbolic_sr:
                # δ_sr_prev: set to predicted δ so sr ≈ 0 at compensated time
                z0 = np.append(z0, [z8_pred[6]])
        else:
            z0 = z0_measured
            tau = 0.0

        if (
            terrain_estimator is not None
            and args.terrain_estimator_backend == GRIT_BACKEND
        ):
            # Freeze one accepted generation for the entire control tick.  The
            # estimator update below may accept a newer snapshot, but that
            # generation is intentionally not consumed until the next tick.
            _joint_control_snapshot = (
                _joint_snapshot_latch.begin_control_tick()
            )
            _joint_ready, _joint_snapshot_age_s, _joint_reason = (
                _joint_snapshot_readiness(
                    _joint_control_snapshot,
                    float(msg.time),
                )
            )
            if _joint_estimator_fault:
                _joint_ready = False
                _joint_reason = "estimator_exception"
            _joint_parameters = (
                _joint_snapshot_control_parameters(_joint_control_snapshot)
                if _joint_ready
                else None
            )
            if _joint_ready and _joint_parameters is None:
                _joint_ready = False
                _joint_reason = "invalid_parameters"
            _joint_publication_ready = int(_joint_ready)
            _joint_fallback_reason = (
                "none" if _joint_ready else _joint_reason
            )
            if _joint_ready:
                _applied_parameters = dict(_joint_parameters)
                _applied_confidence = float(
                    _joint_control_snapshot["confidence"]
                )
                _applied_n_sigma = float(
                    _joint_control_snapshot["n_sigma"]
                )
                _applied_phi_sigma = float(
                    _joint_control_snapshot["phi_sigma_deg"]
                )
                _applied_class = "joint_estimated"
            else:
                _applied_parameters = dict(_joint_fallback)
                _applied_confidence = 0.0
                _applied_n_sigma = None
                _applied_phi_sigma = None
                _applied_class = f"joint_fallback:{_joint_reason}"
            _snapshot_sequence = _joint_snapshot_sequence(
                _joint_control_snapshot
            )
            _policy_state = (
                _snapshot_sequence,
                bool(_joint_ready),
                str(_joint_reason),
            )
            _policy_changed = _policy_state != _joint_policy_state
            if _policy_changed:
                latest_terrain_update["seq"] += 1
                _joint_policy_state = _policy_state
            terrain_update_applied = int(_policy_changed and _joint_ready)
            terrain_params_est = _applied_parameters
            n_terrain_est = float(_applied_parameters["n"])
            terrain_confidence = _applied_confidence
            terrain_class_est = _applied_class
            latest_terrain_update.update({
                "n": float(_applied_parameters["n"]),
                "n_sigma": _applied_n_sigma,
                "phi_deg": float(_applied_parameters["phi"]),
                "phi_sigma_deg": _applied_phi_sigma,
                "Kphi": float(_applied_parameters["Kphi"]),
                "Kc": float(_applied_parameters["Kc"]),
                "c": float(_applied_parameters["c"]),
                "k": float(_applied_parameters["k"]),
                "terrain_class": _applied_class,
                "confidence": _applied_confidence,
            })

        # --- Parse obstacle positions early (needed for reference modification) ---
        _obs_raw_early = getattr(msg, 'obstacles', None)
        _obs_list_mpc = []
        if _obs_raw_early and len(_obs_raw_early) >= 3:
            for _oi in range(0, len(_obs_raw_early) - 2, 3):
                _ox = float(_obs_raw_early[_oi])
                _oy = float(_obs_raw_early[_oi + 1])
                _or = float(_obs_raw_early[_oi + 2]) + 3.5  # 3.5 m margin (vehicle half-width + clearance + speed buffer)
                _obs_list_mpc.append((_ox, _oy, _or))

        # --- Generate reference trajectory ---
        x_ref, y_ref, psi_ref, v_ref, x_goal, y_goal, psi_goal = path_func(
            msg.time, z0, mpc.N, mpc.dt
        )

        _path_done = ref_path.is_complete(threshold=2.0)
        if _path_done:
            v_ref[:] = 0.0

        # --- Terrain-, dynamics-, and state-aware speed profile -----------
        # A live friction-circle (g-g) profile whose grip budget comes from
        # the surrogate at the current soil estimate. It enters as a cap, so
        # it can only reduce the commanded speed where the terrain and the
        # vehicle dynamics cannot support it. This is the default reference;
        # --legacy-speed-ref selects the static curvature-only alternative.
        if not getattr(args, "legacy_speed_ref", False) and not _path_done:
            _L = mpc.Lf + mpc.Lr
            _Fz_f_axle = mpc.M * STANDARD_GRAVITY_M_S2 * mpc.Lr / _L
            _Fz_r_axle = mpc.M * STANDARD_GRAVITY_M_S2 * mpc.Lf / _L
            # Uncertainty hedge: evaluate the grip envelope at the pessimistic
            # soil quantile n_hat - k*sigma_n, softer soil offering less grip,
            # reconstructing the matching Bekker-Mohr parameters along the
            # preset manifold. The MPC prediction model retains the mean
            # estimate, so the mean conditions the model and the quantile
            # conditions the envelope.
            _n_grip, _tp_grip = n_terrain_est, terrain_params_est
            _n_sig = latest_terrain_update.get("n_sigma")
            if float(args.hedge_k) > 0.0 and _n_sig is not None:
                _n_grip = float(np.clip(n_terrain_est - float(args.hedge_k) * _n_sig,
                                        0.4, 1.3))
                _tp_grip = _terrain_params_for_n(_n_grip)
                # The hedge perturbs the coupled pressure/sinkage coordinate
                # alone. The independently estimated friction angle is
                # preserved rather than projected back onto the
                # one-dimensional manifold that the reconstruction implies.
                _tp_grip["phi"] = float(terrain_params_est["phi"])
            _ay_max, _ax_acc, _ax_brk = terrain_grip_limits(
                nn_tire, n_terrain=_n_grip, terrain_params=_tp_grip,
                Fz_front_axle=_Fz_f_axle, Fz_rear_axle=_Fz_r_axle,
                u=msg.u, mass=mpc.M, grip_safety=args.grip_safety,
                ax_actuator_max=mpc.ax_max, ax_actuator_min=mpc.ax_min)
            # Analytic, speed-robust curvature. Finite-differencing the MPC
            # horizon produces curvature spikes whenever the node spacing
            # varies with speed, which collapses v_ref on a straight section.
            # The path spline's curvature at the horizon nodes' arc lengths is
            # independent of that spacing.
            _ds_seg = np.hypot(np.diff(x_ref), np.diff(y_ref))
            _s0 = float(ref_path.s[ref_path._last_idx])
            _cum = _s0 + np.concatenate([[0.0], np.cumsum(_ds_seg)])
            _kappa_an = ref_path.curvature_at(0.5 * (_cum[:-1] + _cum[1:]))
            _v_gg = gg_speed_profile(
                x_ref, y_ref, psi_ref, msg.u,
                ay_max=_ay_max, ax_accel=_ax_acc, ax_brake=_ax_brk,
                v_cap=float(v_target), kappa_override=_kappa_an)
            # Optional longitudinal excitation for terrain-identification
            # traces. It is applied before the g-g cap so the grip budget
            # still binds and the modulation can only ask for less speed than
            # the envelope allows. A sinusoidal path traversed at constant
            # speed excites the lateral channel alone, which leaves the
            # sinkage exponent weakly observable; modulating the speed adds an
            # independent longitudinal excitation without departing the path.
            if float(getattr(args, "speed_osc_amplitude", 0.0)) > 0.0:
                _osc = 1.0 + float(args.speed_osc_amplitude) * float(
                    np.sin(2.0 * np.pi * float(msg.time)
                           / max(float(args.speed_osc_period_s), 1e-3)))
                v_ref = v_ref * max(_osc, 0.05)
            v_ref = np.minimum(v_ref, _v_gg)


        # --- Per-tire operating conditions (shared tire_input_features.py) ---
        delta_meas_now = float(integrator.steering_angle)
        # Estimation reads the realized road-wheel sensor rather than the
        # controller's internal steering integrator. The two diverge whenever
        # a probe, a safety filter, or a human command alters the steering
        # between the NMPC output and the actuator.
        delta_sensor_now = float(getattr(msg, "steering_angle", delta_meas_now))
        _terrain_mu = math.tan(math.radians(float(terrain_params_est['phi'])))
        # Measured kappa from wheel speed sensors (available when kappa_mode='measured')
        _meas_kappa = kappa_from_wheel_speed(
            msg.wheel_omega_fl, msg.wheel_omega_fr,
            msg.wheel_omega_rl, msg.wheel_omega_rr,
            msg.u,
        )
        # Derive the operating point from the filtered velocities, so that the
        # features share the noise-smoothed state the MPC itself receives.
        _u_obs = float(z0_measured[3])   # filtered (or raw if filter disabled)
        _v_obs = float(z0_measured[4])
        _omega_obs = float(z0_measured[5])
        kappa_h, alpha_f_h, alpha_r_h, u_safe_h, Fz_f_h, Fz_r_h = (
            compute_bicycle_operating_point(
                delta_meas_now,
                _u_obs,
                _v_obs,
                _omega_obs,
                _measured_ax,
                geom=tire_geom,
                kappa_mode=args.kappa,
                terrain_mu=_terrain_mu,
                measured_kappa=_meas_kappa,
            )
        )
        (
            te_kappa, te_alpha_f, te_alpha_r, te_u_safe, te_Fz_f, te_Fz_r,
        ) = compute_bicycle_operating_point(
            delta_sensor_now,
            _u_obs,
            _v_obs,
            _omega_obs,
            _measured_ax,
            geom=tire_geom,
            kappa_mode="measured",
            measured_kappa=_meas_kappa,
        )
        te_ax_horizontal, te_ay_horizontal = level_specific_force_to_yaw_frame(
            float(msg.ax), float(msg.ay), float(getattr(msg, "az", 0.0)),
            float(msg.quat_e0), float(msg.quat_e1),
            float(msg.quat_e2), float(msg.quat_e3),
        )
        te_alpha_rate_f = (
            0.0 if _te_alpha_f_prev is None
            else (te_alpha_f - _te_alpha_f_prev) / max(dt_ctrl, 1.0e-4)
        )
        te_alpha_rate_r = (
            0.0 if _te_alpha_r_prev is None
            else (te_alpha_r - _te_alpha_r_prev) / max(dt_ctrl, 1.0e-4)
        )
        _te_alpha_f_prev = te_alpha_f
        _te_alpha_r_prev = te_alpha_r

        if terrain_obs_writer is not None:
            _terrain_obs_values = [
                float(msg.time), float(msg.x_cg), float(msg.y_cg), float(msg.z_cg),
                float(msg.quat_e0), float(msg.quat_e1),
                float(msg.quat_e2), float(msg.quat_e3), float(psi),
                float(msg.u), float(msg.v), float(msg.omega),
                float(te_u_safe), float(_v_obs), float(_omega_obs),
                float(msg.ax), float(msg.ay), float(getattr(msg, "az", 0.0)),
                float(_measured_ax),
                float(getattr(msg, "omega_x", 0.0)),
                float(getattr(msg, "omega_y", 0.0)),
                float(msg.wheel_omega_fl), float(msg.wheel_omega_fr),
                float(msg.wheel_omega_rl), float(msg.wheel_omega_rr),
                float(msg.wheel_center_z_fl), float(msg.wheel_center_z_fr),
                float(msg.wheel_center_z_rl), float(msg.wheel_center_z_rr),
                float(msg.drive_torque_fl), float(msg.drive_torque_fr),
                float(msg.drive_torque_rl), float(msg.drive_torque_rr),
                float(msg.brake_torque_fl), float(msg.brake_torque_fr),
                float(msg.brake_torque_rl), float(msg.brake_torque_rr),
                float(delta_sensor_now), float(te_kappa),
                float(te_alpha_f), float(te_alpha_r),
                float(te_alpha_rate_f), float(te_alpha_rate_r),
                float(te_Fz_f), float(te_Fz_r),
            ]
            terrain_obs_writer.writerow(
                [int(seq), *[format(value, ".17g") for value in _terrain_obs_values]]
            )
        # Realized road-wheel rate over the last control interval.
        sr_h = (delta_meas_now - prev_applied_delta) / max(dt_ctrl, 1e-4)

        # --- Solve MPC ---
        t0_solve = wall_time.time()
        solve_kwargs = dict(
            n_terrain=n_terrain_est,
            sr_meas=sr_h,
            terrain_params=terrain_params_est,
        )
        # Feedforward sinkage drag: anticipate soft-soil motion resistance in
        # the NMPC longitudinal prediction, u_dot = ax + du_dot_resid, indexed
        # by the live soil estimate. The solver fades the residual in above
        # about 0.5 m/s, where the model is well conditioned.
        if getattr(args, "ff_drag", False) and float(args.ff_drag_scale) != 0.0:
            _du_dot = -float(args.ff_drag_scale) * _c_drag(n_terrain_est)
            solve_kwargs["dynamics_residuals"] = np.tile(
                np.array([_du_dot, 0.0, 0.0], dtype=float), (mpc.N + 1, 1))
        if rate_tracker is not None:
            solve_kwargs['rates_front'] = rate_tracker.front
            solve_kwargs['rates_rear'] = rate_tracker.rear

        # Obstacle avoidance: hand the parsed obstacle list to the OCP.
        # ``--mpc-blind-obstacles`` withholds it, leaving the safety filter as
        # the sole obstacle-avoiding layer. That is the condition of interest
        # for a teleoperator with no advance knowledge of the obstacle field,
        # who depends entirely on the shield.
        if _obs_list_mpc and not getattr(args, 'mpc_blind_obstacles', False):
            solve_kwargs['obstacles'] = _obs_list_mpc

        delta_cmd, Jx, Z_opt, U_opt = mpc.solve(
            z0, x_ref, y_ref, psi_ref, v_ref,
            x_goal, y_goal, psi_goal,
            **solve_kwargs,
        )
        t_solve = wall_time.time() - t0_solve
        solve_times.append(t_solve)
        delay_est.update_solve(t_solve)

        if _log_mpc_pred and Z_opt is not None:
            # predicted [x, y, psi, u, v, omega] over the horizon (stage dt = mpc.dt)
            _pred_times.append(float(msg.time))
            _pred_Z.append(np.asarray(Z_opt[:6, :], dtype=np.float32).copy())

        if Z_opt is None:
            # Solver fallback path: solver may return a hold command.
            if not np.isfinite(delta_cmd):
                delta_cmd = float(z0[7])
            if not np.isfinite(Jx):
                Jx = float(z0[8])
        elif Z_opt.shape[1] > 1:
            # Queue one-step-ahead state prediction for measurement residuals.
            pred_targets.append((
                float(msg.time + mpc.dt),
                np.array(Z_opt[:9, 1], dtype=float, copy=True),
            ))

        if not np.isfinite(delta_cmd):
            delta_cmd = integrator.steering_angle
        if not np.isfinite(Jx):
            Jx = 0.0

        # Suppress steering during lead-in acceleration phase
        if lead_in > 0 and z0[0] < lead_in and msg.u < args.lead_in_speed_fraction * v_target:
            delta_cmd = 0.0

        # ---- End-of-path override: brake to a stop ----
        # The reference path already ramps v_ref to zero over its final 5 m,
        # so the MPC commands braking of its own accord. Integrator state and
        # transport latency can nonetheless leave residual forward thrust once
        # the path is exhausted, so the integrator acceleration is driven to
        # maximum deceleration and the vehicle stops within its braking
        # distance rather than coasting beyond the path.
        if _path_done:
            if seq % 50 == 0 and msg.u > 0.1:
                print(f"  [PATH DONE] t={msg.time:.1f}s  u={msg.u:.2f} m/s"
                      f"  — braking to stop")
            if msg.u < 0.1:
                delta_cmd = 0.0

        # Optional single achieved-slip terrain-identification maneuver. Its
        # road-wheel target enters ahead of the common actuator limiter, so
        # the command, the integrator state, the delay buffer, and the next
        # NMPC initial state all describe the same physical steering input.
        _probe_nominal_delta = float(delta_cmd)
        _probe_requested_delta = float(delta_cmd)
        _probe_phase = "off"
        _probe_target_alpha = 0.0
        _probe_override = 0
        _probe_reason = ""
        if terrain_id_probe is not None:
            _probe_cp = ref_path.closest_point_on_path(
                float(z0_measured[0]), float(z0_measured[1])
            )
            _probe_curvature = float(
                np.asarray(ref_path.curvature_at([_probe_cp["s"]])).reshape(-1)[0]
            )
            _probe_obstacles_raw = getattr(msg, "obstacles", None)
            _probe_feed_valid = _probe_obstacles_raw is not None
            _probe_clearance = float("inf")
            for _ox, _oy, _or in _obs_list_mpc:
                _probe_clearance = min(
                    _probe_clearance,
                    math.hypot(float(msg.x_cg) - _ox, float(msg.y_cg) - _oy) - _or,
                )
            _probe_safety_intervened = any((
                abs(float(getattr(msg, "steering_app", 0.0))
                    - float(getattr(msg, "steering_op", 0.0))) > 1.0e-3,
                abs(float(getattr(msg, "throttle_app", 0.0))
                    - float(getattr(msg, "throttle_op", 0.0))) > 1.0e-3,
                abs(float(getattr(msg, "braking_app", 0.0))
                    - float(getattr(msg, "braking_op", 0.0))) > 1.0e-3,
            ))
            _probe_command = terrain_id_probe.update(
                dt_ctrl,
                TerrainIDProbeInputs(
                    requested=True,
                    speed_mps=float(_u_obs),
                    measured_front_alpha_rad=float(te_alpha_f),
                    lateral_accel_mps2=float(msg.ay),
                    cross_track_error_m=float(_probe_cp["e_lat"]),
                    nominal_steering_rad=float(_probe_nominal_delta),
                    reference_curvature_inv_m=_probe_curvature,
                    obstacle_feed_valid=_probe_feed_valid,
                    clear_road=(
                        _probe_feed_valid
                        and _probe_clearance >= float(args.terrain_id_probe_clearance)
                    ),
                    # Small NMPC brake trims are ignored; a substantial brake
                    # command still blocks or aborts the excitation.
                    braking=float(getattr(msg, "braking_app", 0.0)) > 0.05,
                    solver_ok=str(getattr(mpc, "last_solver_status", "0")) == "0",
                    safety_intervened=_probe_safety_intervened,
                    path_complete=bool(_path_done),
                    latency_s=float(delay_est.compensation_delay),
                ),
            )
            _probe_phase = _probe_command.phase.value
            _probe_target_alpha = float(_probe_command.target_alpha_rad)
            _probe_override = int(_probe_command.steering_override)
            _probe_reason = str(_probe_command.reason)
            if _probe_command.steering_override:
                _probe_requested_delta = float(np.clip(
                    math.atan2(
                        _v_obs + mpc.Lf * _omega_obs,
                        max(abs(_u_obs), SLIP_CALC_MIN_SPEED_MPS),
                    ) + _probe_target_alpha,
                    -min(float(mpc.delta_max), 0.40),
                    min(float(mpc.delta_max), 0.40),
                ))
                delta_cmd = _probe_requested_delta

        # Rate limiter on the realized control interval. The MPC discretisation
        # step of 0.1 s is far longer than the control interval of roughly
        # 0.012 s, so its polytopic steering-rate constraint permits a step
        # that would exceed the physical rate limit within one control period.
        max_delta_change = mpc.max_steer_rate * max(dt_ctrl, 1e-4)
        delta_cmd = float(np.clip(
            delta_cmd,
            integrator.steering_angle - max_delta_change,
            integrator.steering_angle + max_delta_change))

        # Effective δ̇ for delay buffer, diagnostics, and NN steering_rate feature
        delta_dot = (delta_cmd - integrator.steering_angle) / max(dt_ctrl, 1e-4)
        last_delta_dot_cmd = float(delta_dot)
        last_Jx_cmd = float(Jx)

        # --- Update the rate tracker on its own sampling interval ---
        if rate_tracker is not None:
            rate_tracker.update(
                float(msg.time),
                kappa_h, alpha_f_h, u_safe_h,
                kappa_h, alpha_r_h, u_safe_h,
            )

        # --- Apply controls: set δ directly, integrate ax from Jx ---
        integrator.steering_angle = float(np.clip(
            delta_cmd, -mpc.delta_max, mpc.delta_max))
        _saved_delta = integrator.steering_angle

        # The MPC's ax state comes from the IMU, so the plan already reflects
        # terrain drag and the speed cost acts against it. The integrator then
        # accumulates ax as acceleration += Jx·dt, supplying integral action
        # on the throttle channel.
        v_ref_now = float(v_ref[0]) if len(v_ref) else float(v_target)

        # Feedforward terrain-aware throttle offset: index the per-terrain
        # offset by the live soil estimate and hand it to the integrator. The
        # branches below are inert unless the corresponding flag is set.
        if getattr(args, "ff_drag_surrogate", False) and nn_tire is not None:
            # Compaction-resistance term of Dallas et al., expressed in
            # throttle units:
            #   throttle_ff = N * F_comp / (M * ax_max),  F_comp = -Fx(kappa=0)
            # where the drag is queried from the surrogate at the current
            # (u, Fz, n_hat) rather than read from a table, and the OCP is
            # unchanged. It represents the motion resistance alone; the
            # reduced throttle-to-force gain on soil is not expressed by it.
            # Vehicle constants come from the shared source of truth, not
            # literals: a hard-coded mass here is exactly the drift class the
            # shared-vehicle-constants hardening removed everywhere else.
            _M_veh = float(get_vehicle_params_for_demo()["M"])
            _Fz_w = _M_veh * STANDARD_GRAVITY_M_S2 / 4.0  # static per-wheel load (N)
            _fx0, _ = nn_tire.predict_numeric(
                0.0, _Fz_w, max(float(msg.u), 0.5), kappa=0.0,
                n_terrain=n_terrain_est, steering_rate=0.0,
                terrain_params=terrain_params_est, rates=(0.0, 0.0, 0.0))
            _F_comp = max(-float(_fx0), 0.0)              # per-wheel drag (N)
            _ax_max = float(getattr(mpc, "ax_max", 1.9))
            integrator._d_ff = float(np.clip(
                4.0 * _F_comp / (_M_veh * _ax_max), 0.0, 0.6))
        elif getattr(args, "ff_throttle_2d", False) and float(args.ff_throttle_scale) != 0.0:
            integrator._d_ff = float(args.ff_throttle_scale) * _d_ff_throttle_2d(
                n_terrain_est, float(msg.u))
        elif getattr(args, "ff_throttle", False) and float(args.ff_throttle_scale) != 0.0:
            integrator._d_ff = float(args.ff_throttle_scale) * _d_ff_throttle(n_terrain_est)
        # In force-balance mode the solver's longitudinal control is the slip
        # rate κ̇ rather than jerk, so the integrator's ax += Jx·dt path does
        # not apply. The integrator instead receives the planned acceleration
        # read across stages 1 to k of the OCP's own predicted speed
        # trajectory. Taking it as (u_pred1 - u0_measured)/dt would difference
        # the noisy measurement against the plan and excite an under-damped
        # throttle limit cycle, whereas the internal plan is smooth because
        # the slip is rate-limited. The throttle map then realises it as
        # ax/ax_max plus any residual observer bias.
        _fb_desired_ax = None
        if _force_balance and Z_opt is not None and Z_opt.shape[1] > 2:
            _k = min(4, Z_opt.shape[1] - 1)
            _fb_desired_ax = float((Z_opt[3, _k] - Z_opt[3, 1]) / ((_k - 1) * mpc.dt))
        _, throttle, braking = integrator.update(
            0.0, Jx, dt_ctrl, msg.u,
            v_ref_now=v_ref_now,
            desired_ax=_fb_desired_ax,
        )

        # Force stopping if path is done
        if _path_done:
            integrator.acceleration = 0.0
            throttle = 0.0
            if msg.u > 0.1:
                braking = 1.0
            else:
                braking = 1.0

        integrator.steering_angle = _saved_delta
        # Physical command bus: road-wheel angle (rad) + drive torque (N m,
        # negative = brake). The integrator's internal throttle/brake fractions
        # are mapped onto the actuator torque range once, here.
        delta_cmd_phys = float(integrator.steering_angle)
        drive_torque_cmd = float(
            throttle * DRIVE_TORQUE_MAX_NM - braking * BRAKE_TORQUE_MAX_NM)
        # Normalised steering fraction, recorded in the diagnostic CSV for the
        # analysis scripts. It is not part of the command bus.
        steering = float(np.clip(
            delta_cmd_phys * integrator.steering_gain, -1.0, 1.0))

        # For next cycle's realized-rate estimate, keep the delta from this
        # cycle's measured state sample (before applying the new command).
        prev_applied_delta = delta_meas_now

        # --- Record in control buffer for delay compensation ---
        control_buffer.append((msg.time, delta_dot, Jx))

        # --- Record tracking analytics ---
        tf = msg.tire_forces or {}
        true_x = tf.get('true_x_cg')
        if true_x is not None:
            true_psi = tf['true_psi']
            true_x_fa = true_x + Lf * np.cos(true_psi)
            true_y_fa = tf['true_y_cg'] + Lf * np.sin(true_psi)
            true_u = tf['true_u']
            analytics.record(
                msg.time, true_x_fa, true_y_fa, true_psi, true_u,
                v_ref_now=v_ref_now,
            )
        else:
            analytics.record(
                msg.time, z0_measured[0], z0_measured[1], psi, msg.u,
                v_ref_now=v_ref_now,
            )
        analytics.record_control(
            msg.time, steering, throttle, braking,
            integrator.steering_angle, integrator.acceleration,
            t_solve * 1000.0, delay_est.compensation_delay * 1000.0,
        )


        # --- Record lateral force: Chrono truth against model prediction ---
        # Evaluated at the same pre-command operating point as the state
        # sample. Both the measured slip and the slip the OCP itself assumes
        # are logged, so that the force comparison can be read against the
        # model that was actually optimized.
        kappa_meas_diag = float(_meas_kappa)
        if mpc.kappa_mode == 'approx':
            mu_diag = max(_terrain_mu, 1e-3)
            kappa_diag = float(np.clip(_measured_ax / (mu_diag * 9.81), -0.8, 0.8))
        elif mpc.kappa_mode == 'zero':
            kappa_diag = 0.0
        else:
            kappa_diag = kappa_meas_diag
        alpha_f, alpha_r = alpha_f_h, alpha_r_h
        u_safe = u_safe_h
        Fz_f_mean, Fz_r_mean = Fz_f_h, Fz_r_h
        sr_diag = sr_h

        # Clamp slip angles to training-data range (matches MPC solver).
        _alpha_max = 0.55
        alpha_f = float(max(-_alpha_max, min(_alpha_max, alpha_f)))
        alpha_r = float(max(-_alpha_max, min(_alpha_max, alpha_r)))

        # No low-speed force fade: diagnostics should reflect direct model output.
        _speed_fade = 1.0

        # Predict axle forces whenever the truth diagnostics need them.
        pred_Fy_f = pred_Fy_r = float("nan")
        _need_predicted_fy = msg.tire_forces is not None
        if _need_predicted_fy:
            if nn_tire is not None:
                rates_f = rate_tracker.front if rate_tracker is not None else None
                rates_r = rate_tracker.rear if rate_tracker is not None else None
                if mpc.lateral_load_transfer:
                    dFz = lateral_load_transfer_dFz(msg.u, msg.omega, geom=tire_geom)
                    Fz_fo, Fz_fi, Fz_ro, Fz_ri = fz_with_lateral_transfer(
                        Fz_f_mean, Fz_r_mean, dFz
                    )
                    _, Fy_fo = nn_tire.predict_numeric(
                        alpha_f, Fz_fo, u_safe,
                        kappa=kappa_diag, n_terrain=n_terrain_est, steering_rate=sr_diag,
                        terrain_params=terrain_params_est, rates=rates_f)
                    _, Fy_fi = nn_tire.predict_numeric(
                        alpha_f, Fz_fi, u_safe,
                        kappa=kappa_diag, n_terrain=n_terrain_est, steering_rate=sr_diag,
                        terrain_params=terrain_params_est, rates=rates_f)
                    _, Fy_ro = nn_tire.predict_numeric(
                        alpha_r, Fz_ro, u_safe,
                        kappa=kappa_diag, n_terrain=n_terrain_est, steering_rate=0.0,
                        terrain_params=terrain_params_est, rates=rates_r)
                    _, Fy_ri = nn_tire.predict_numeric(
                        alpha_r, Fz_ri, u_safe,
                        kappa=kappa_diag, n_terrain=n_terrain_est, steering_rate=0.0,
                        terrain_params=terrain_params_est, rates=rates_r)
                    pred_Fy_f = -(Fy_fo + Fy_fi)
                    pred_Fy_r = -(Fy_ro + Fy_ri)
                else:
                    _, Fy_fw = nn_tire.predict_numeric(
                        alpha_f, Fz_f_mean, u_safe,
                        kappa=kappa_diag, n_terrain=n_terrain_est, steering_rate=sr_diag,
                        terrain_params=terrain_params_est, rates=rates_f)
                    _, Fy_rw = nn_tire.predict_numeric(
                        alpha_r, Fz_r_mean, u_safe,
                        kappa=kappa_diag, n_terrain=n_terrain_est, steering_rate=0.0,
                        terrain_params=terrain_params_est, rates=rates_r)
                    pred_Fy_f = -2.0 * Fy_fw
                    pred_Fy_r = -2.0 * Fy_rw
            else:
                # Analytical tire model (pacejka, pacejka-oracle, tmeasy)
                _anal_model = 'pacejka' if tire_model in ('pacejka-oracle', 'pacejka-rigfit') else tire_model
                _anal_kwargs = mpc._oracle_pacejka_params if tire_model in ('pacejka-oracle', 'pacejka-rigfit') else {}
                Fyf, Fyr, _ = analytical_tire_forces(
                    _anal_model, alpha_f, alpha_r,
                    2.0 * Fz_f_mean, 2.0 * Fz_r_mean, kappa_diag,
                    **_anal_kwargs,
                )
                pred_Fy_f = float(Fyf)
                pred_Fy_r = float(Fyr)

        if msg.tire_forces is not None:
            tf = msg.tire_forces
            actual_Fy_f = tf.get('front_left_Fy', 0) + tf.get('front_right_Fy', 0)
            actual_Fy_r = tf.get('rear_left_Fy', 0) + tf.get('rear_right_Fy', 0)
            analytics.record_tire_forces(
                msg.time, actual_Fy_f, actual_Fy_r, pred_Fy_f, pred_Fy_r)

            # Update the signed lateral-force bias diagnostic
            if msg.u > _FY_BIAS_MIN_SPEED:
                af, ar, pf, pr = float(actual_Fy_f), float(actual_Fy_r), float(pred_Fy_f), float(pred_Fy_r)
                signed_err_f = af - pf
                signed_err_r = ar - pr
                _fy_bias_signed_f = (1 - _FY_BIAS_ALPHA) * _fy_bias_signed_f + _FY_BIAS_ALPHA * signed_err_f
                _fy_bias_signed_r = (1 - _FY_BIAS_ALPHA) * _fy_bias_signed_r + _FY_BIAS_ALPHA * signed_err_r
                _fy_bias_signed_f = float(np.clip(_fy_bias_signed_f, -_FY_BIAS_CLIP, _FY_BIAS_CLIP))
                _fy_bias_signed_r = float(np.clip(_fy_bias_signed_r, -_FY_BIAS_CLIP, _FY_BIAS_CLIP))

        # --- Online terrain parameter estimation ---
        # This path takes no input from Chrono's optional tire-force truth
        # diagnostics. Every observation below is obtainable from vehicle
        # state, steering, wheel encoders, fixed geometry, or the IMU, which
        # is what makes the estimate deployable rather than an oracle.
        if terrain_estimator is not None:
            try:
                _te_omega_dot = terrain_estimator.estimate_omega_dot(
                    msg.omega, msg.time
                )
                if _te_omega_dot is not None:
                    _te_observation = {
                        "kappa": float(te_kappa),
                        "alpha_f": float(te_alpha_f),
                        "alpha_r": float(te_alpha_r),
                        "u": float(te_u_safe),
                        "Fz_f": float(te_Fz_f),
                        "Fz_r": float(te_Fz_r),
                        "sr": float(te_alpha_rate_f),
                        "alpha_rate_r": float(te_alpha_rate_r),
                        "ay_imu": float(te_ay_horizontal),
                        "omega_dot": _te_omega_dot,
                        "omega": float(_omega_obs),
                        "v_ref": float(v_target),
                        "v_lateral": float(_v_obs),
                        "x_pos": float(msg.x_cg),
                        "y_pos": float(msg.y_cg),
                        "psi": float(np.arctan2(
                            2*(msg.quat_e0*msg.quat_e3 + msg.quat_e1*msg.quat_e2),
                            1 - 2*(msg.quat_e2**2 + msg.quat_e3**2))),
                        "ax_cmd": float(z0_measured[6]),
                        "sim_time": float(msg.time),
                        "steering_angle": delta_sensor_now,
                        "wheel_omegas": (
                            float(msg.wheel_omega_fl),
                            float(msg.wheel_omega_fr),
                            float(msg.wheel_omega_rl),
                            float(msg.wheel_omega_rr),
                        ),
                        "wheel_center_heights": (
                            float(msg.wheel_center_z_fl),
                            float(msg.wheel_center_z_fr),
                            float(msg.wheel_center_z_rl),
                            float(msg.wheel_center_z_rr),
                        ),
                        "drive_torques": (
                            float(msg.drive_torque_fl),
                            float(msg.drive_torque_fr),
                            float(msg.drive_torque_rl),
                            float(msg.drive_torque_rr),
                        ),
                        "brake_torques": (
                            float(msg.brake_torque_fl),
                            float(msg.brake_torque_fr),
                            float(msg.brake_torque_rl),
                            float(msg.brake_torque_rr),
                        ),
                        "ax_imu": float(te_ax_horizontal),
                        "az_imu": float(getattr(msg, "az", 0.0)),
                        "roll_rate": float(getattr(msg, "omega_x", 0.0)),
                        "pitch_rate": float(getattr(msg, "omega_y", 0.0)),
                        "throttle_cmd": float(throttle),
                    }
                    terrain_estimator.observe(
                        **_terrain_estimator_observation_for_backend(
                            args.terrain_estimator_backend,
                            _te_observation,
                        )
                    )

                if terrain_estimator.should_update():
                    _te_params, _te_conf = terrain_estimator.estimate()
                    if (
                        args.terrain_estimator_backend
                        == GRIT_BACKEND
                    ):
                        _snapshot_getter = getattr(
                            terrain_estimator,
                            "get_last_accepted_snapshot",
                            None,
                        )
                        _snapshot = (
                            _snapshot_getter()
                            if callable(_snapshot_getter)
                            else None
                        )
                        if not isinstance(_snapshot, Mapping):
                            raise RuntimeError(
                                "joint estimator did not publish an immutable "
                                "accepted snapshot"
                            )
                        _joint_snapshot_latch.accept(_snapshot)
                        _joint_estimator_fault = False
                    elif _te_conf >= args.te_min_confidence:
                        # The controller consumes the reconstructed parameter
                        # bundle, while the diagnostics retain the estimator's
                        # own smoothed sinkage exponent.
                        _te_mpc = terrain_estimator.get_terrain_mpc_params()
                        terrain_params_est = _te_mpc
                        n_terrain_est = float(terrain_estimator.get_bekker_n())
                        terrain_confidence = _te_conf
                        terrain_update_applied = 1
                        terrain_class_est = getattr(
                            terrain_estimator, '_terrain_name', 'estimated'
                        )
                        latest_terrain_update.update({
                            "seq": latest_terrain_update["seq"] + 1,
                            "n": float(_te_mpc['n']),
                            "n_sigma": (
                                float(terrain_estimator.get_n_uncertainty())
                                if hasattr(terrain_estimator, "get_n_uncertainty")
                                else None),
                            "phi_deg": float(_te_mpc['phi']),
                            "phi_sigma_deg": float(
                                terrain_estimator.get_phi_uncertainty_deg()
                            ),
                            "Kphi": float(_te_mpc['Kphi']),
                            "Kc": float(_te_mpc['Kc']),
                            "c": float(_te_mpc['c']),
                            "k": float(_te_mpc['k']),
                            "terrain_class": str(terrain_class_est),
                            "confidence": float(_te_conf),
                        })
            except Exception as _te_exc:
                if (
                    args.terrain_estimator_backend
                    == GRIT_BACKEND
                ):
                    _joint_estimator_fault = True
                import traceback
                print(f"[TERRAIN-EST] Error: {_te_exc}", flush=True)
                traceback.print_exc()

        # --- Publish command ---
        cmd = ControlCommand(
            time=msg.time,
            wall_time=wall_time.time(),
            seq=seq,
            delta=delta_cmd_phys,
            drive_torque=drive_torque_cmd,
            acceleration=integrator.acceleration,
            delta_dot=delta_dot,
            jerk=Jx,
            solve_time_ms=t_solve * 1000.0,
            terrain_n=latest_terrain_update["n"],
            terrain_n_sigma=latest_terrain_update["n_sigma"],
            terrain_grip_scale=None,
            terrain_phi_deg=latest_terrain_update["phi_deg"],
            terrain_phi_sigma_deg=latest_terrain_update["phi_sigma_deg"],
            terrain_Kphi=latest_terrain_update["Kphi"],
            terrain_Kc=latest_terrain_update["Kc"],
            terrain_c=latest_terrain_update["c"],
            terrain_k=latest_terrain_update["k"],
            terrain_class=latest_terrain_update["terrain_class"],
            terrain_confidence=latest_terrain_update["confidence"],
            terrain_update_seq=latest_terrain_update["seq"],
        )
        ctrl_pub.send(cmd)

        # --- Write diagnostic CSV row ---
        if csv_writer is not None:
            tf = msg.tire_forces or {}
            true_x = tf.get('true_x_cg')
            if true_x is not None:
                true_psi_v = tf['true_psi']
                true_x_fa = true_x + Lf * np.cos(true_psi_v)
                true_y_fa = tf['true_y_cg'] + Lf * np.sin(true_psi_v)
                true_u_v = tf['true_u']
            else:
                true_x_fa = z0_measured[0]
                true_y_fa = z0_measured[1]
                true_psi_v = psi
                true_u_v = msg.u

            ct_err = analytics.crosstrack_errors[-1] if analytics.crosstrack_errors else 0
            hd_err = np.degrees(analytics.heading_errors[-1]) if analytics.heading_errors else 0
            sp_err = analytics.speed_errors[-1] if analytics.speed_errors else 0

            fy_af = analytics.actual_Fy_front[-1] if analytics.actual_Fy_front else ''
            fy_ar = analytics.actual_Fy_rear[-1] if analytics.actual_Fy_rear else ''
            fy_nf = analytics.pred_Fy_front[-1] if analytics.pred_Fy_front else ''
            fy_nr = analytics.pred_Fy_rear[-1] if analytics.pred_Fy_rear else ''
            fx_af = (
                float(tf.get("front_left_Fx", 0.0))
                + float(tf.get("front_right_Fx", 0.0))
                if msg.tire_forces is not None else ""
            )
            fx_ar = (
                float(tf.get("rear_left_Fx", 0.0))
                + float(tf.get("rear_right_Fx", 0.0))
                if msg.tire_forces is not None else ""
            )

            mpc_cost = getattr(mpc, 'last_cost', float('nan'))
            solver_status = getattr(mpc, 'last_solver_status', '')
            solver_iters = getattr(mpc, 'last_iter_count', -1)
            phi_applied_deg = float(terrain_params_est.get('phi', float('nan')))
            phi_estimator_deg = phi_applied_deg
            if terrain_estimator is not None:
                try:
                    phi_estimator_deg = float(terrain_estimator.get_friction_angle_deg())
                except Exception:
                    phi_estimator_deg = phi_applied_deg
            dynamics_active = int(bool(getattr(
                terrain_estimator, "dynamics_active", False
            )))
            dynamics_windows = int(getattr(
                terrain_estimator, "dynamics_windows", 0
            ))
            accepted_dynamics_windows = int(getattr(
                terrain_estimator, "accepted_dynamics_windows", 0
            ))
            rejected_dynamics_windows = int(getattr(
                terrain_estimator, "rejected_dynamics_windows", 0
            ))
            profile_force_gain = getattr(
                terrain_estimator, "profile_force_gain", ""
            )
            profile_ax_bias = getattr(
                terrain_estimator, "profile_ax_bias", ""
            )
            profile_ay_bias = getattr(
                terrain_estimator, "profile_ay_bias", ""
            )
            profile_bound_hits = int(getattr(
                terrain_estimator, "profile_bound_hits", 0
            ))
            feature_envelope_excursions = int(getattr(
                terrain_estimator, "feature_envelope_excursions", 0
            ))
            _joint_diag = (
                _joint_control_snapshot
                if isinstance(_joint_control_snapshot, Mapping)
                else {}
            )
            _joint_snapshot_seq = _joint_snapshot_sequence(
                _joint_control_snapshot
            )
            _joint_evidence_time = _joint_diag.get("evidence_time_s", "")
            _joint_snapshot_confidence = _joint_diag.get("confidence", "")
            _joint_n_boundary_mass = _joint_diag.get("n_boundary_mass", "")
            _joint_phi_boundary_mass = _joint_diag.get(
                "phi_boundary_mass", ""
            )
            _joint_max_boundary_mass = _joint_diag.get(
                "max_boundary_mass", ""
            )
            _joint_boundary_limited = _joint_diag.get(
                "boundary_limited", ""
            )
            _joint_observability_rank = _joint_diag.get(
                "observability_rank", ""
            )
            _joint_observability_min_singular = _joint_diag.get(
                "observability_min_singular_value", ""
            )
            _joint_projection_wall_s = _joint_diag.get(
                "projection_wall_time_s", ""
            )
            _joint_profile_wall_s = _joint_diag.get(
                "profile_wall_time_s", ""
            )
            _joint_observability_wall_s = _joint_diag.get(
                "observability_wall_time_s", ""
            )
            _joint_posterior_wall_s = _joint_diag.get(
                "posterior_wall_time_s", ""
            )
            _joint_publication_wall_s = _joint_diag.get(
                "publication_wall_time_s", ""
            )
            _joint_update_wall_s = _joint_diag.get(
                "update_wall_time_s", ""
            )
            n_estimator_value = float(n_terrain_est)
            if terrain_estimator is not None:
                try:
                    n_estimator_value = float(
                        terrain_estimator.get_bekker_n()
                    )
                except Exception:
                    n_estimator_value = float(n_terrain_est)

            csv_writer.writerow([
                f"{msg.time:.4f}", f"{recv_time:.6f}", seq,
                f"{z0_measured[0]:.6f}", f"{z0_measured[1]:.6f}",
                f"{psi:.6f}", f"{msg.u:.4f}", f"{msg.v:.4f}", f"{msg.omega:.6f}",
                f"{true_x_fa:.6f}", f"{true_y_fa:.6f}",
                f"{true_psi_v:.6f}", f"{true_u_v:.4f}",
                f"{z0[0]:.6f}", f"{z0[1]:.6f}", f"{z0[2]:.6f}",
                f"{z0[3]:.4f}", f"{z0[4]:.4f}", f"{z0[5]:.6f}",
                f"{z0[6]:.6f}", f"{z0[7]:.4f}",
                f"{msg.ax:.6f}", f"{msg.ay:.6f}",
                f"{msg.wheel_omega_fl:.6f}", f"{msg.wheel_omega_fr:.6f}",
                f"{msg.wheel_omega_rl:.6f}", f"{msg.wheel_omega_rr:.6f}",
                f"{delta_sensor_now:.6f}",
                f"{msg.z_cg:.6f}", f"{msg.quat_e0:.9f}", f"{msg.quat_e1:.9f}",
                f"{msg.quat_e2:.9f}", f"{msg.quat_e3:.9f}",
                f"{msg.az:.6f}", f"{msg.omega_x:.6f}", f"{msg.omega_y:.6f}",
                f"{x_ref[0]:.6f}", f"{y_ref[0]:.6f}",
                f"{psi_ref[0]:.6f}", f"{v_ref[0]:.4f}",
                f"{delta_dot:.6f}", f"{Jx:.6f}",
                f"{mpc_cost:.4f}", solver_status, solver_iters,
                terrain_class_est, f"{terrain_confidence:.4f}", f"{n_terrain_est:.6f}",
                f"{n_estimator_value:.6f}",
                f"{phi_applied_deg:.6f}",
                f"{phi_estimator_deg:.6f}",
                f"{latest_terrain_update['phi_sigma_deg']:.6f}" if latest_terrain_update["phi_sigma_deg"] is not None else "",
                # Posterior std of the Bekker-n belief (sqrt P[n,n]); the raw
                # signal for uncertainty-aware (hedge/probe) behaviour.
                (f"{terrain_estimator.get_n_uncertainty():.6f}"
                 if terrain_estimator is not None
                 and hasattr(terrain_estimator, "get_n_uncertainty") else ""),
                terrain_update_applied,
                dynamics_active, dynamics_windows, accepted_dynamics_windows,
                rejected_dynamics_windows,
                (f"{profile_force_gain:.6f}"
                 if profile_force_gain != "" else ""),
                (f"{profile_ax_bias:.6f}" if profile_ax_bias != "" else ""),
                (f"{profile_ay_bias:.6f}" if profile_ay_bias != "" else ""),
                profile_bound_hits, feature_envelope_excursions,
                _joint_snapshot_seq,
                (f"{float(_joint_evidence_time):.6f}"
                 if _joint_evidence_time != "" else ""),
                (f"{_joint_snapshot_age_s:.6f}"
                 if np.isfinite(_joint_snapshot_age_s) else ""),
                _joint_publication_ready,
                _joint_fallback_reason,
                (f"{float(_joint_snapshot_confidence):.6f}"
                 if _joint_snapshot_confidence != "" else ""),
                (f"{float(_joint_n_boundary_mass):.6f}"
                 if _joint_n_boundary_mass != "" else ""),
                (f"{float(_joint_phi_boundary_mass):.6f}"
                 if _joint_phi_boundary_mass != "" else ""),
                (f"{float(_joint_max_boundary_mass):.6f}"
                 if _joint_max_boundary_mass != "" else ""),
                _joint_boundary_limited,
                _joint_observability_rank,
                (f"{float(_joint_observability_min_singular):.6f}"
                 if _joint_observability_min_singular != "" else ""),
                (f"{1000.0 * float(_joint_projection_wall_s):.6f}"
                 if _joint_projection_wall_s != "" else ""),
                (f"{1000.0 * float(_joint_profile_wall_s):.6f}"
                 if _joint_profile_wall_s != "" else ""),
                (f"{1000.0 * float(_joint_observability_wall_s):.6f}"
                 if _joint_observability_wall_s != "" else ""),
                (f"{1000.0 * float(_joint_posterior_wall_s):.6f}"
                 if _joint_posterior_wall_s != "" else ""),
                (f"{1000.0 * float(_joint_publication_wall_s):.6f}"
                 if _joint_publication_wall_s != "" else ""),
                (f"{1000.0 * float(_joint_update_wall_s):.6f}"
                 if _joint_update_wall_s != "" else ""),
                f"{steering:.6f}", f"{throttle:.4f}", f"{braking:.4f}",
                f"{integrator.steering_angle:.6f}", f"{integrator.acceleration:.4f}",
                f"{delay_est.one_way_delay*1000:.2f}",
                f"{delay_est.solve_time*1000:.2f}",
                f"{delay_est.compensation_delay*1000:.2f}",
                f"{t_solve*1000:.2f}",
                f"{ct_err:.6f}", f"{hd_err:.4f}", f"{sp_err:.4f}",
                f"{pred_age:.6f}", f"{pred_pos_err:.6f}", f"{pred_psi_err_deg:.4f}",
                f"{pred_u_err:.6f}", f"{pred_v_err:.6f}", f"{pred_omega_err:.6f}",
                fx_af, fx_ar, fy_af, fy_ar, fy_nf, fy_nr,
                f"{alpha_f:.6f}", f"{alpha_r:.6f}",
                f"{Fz_f_mean:.1f}", f"{Fz_r_mean:.1f}",
                f"{kappa_diag:.6f}", f"{kappa_meas_diag:.6f}", f"{sr_diag:.6f}",
                f"{u_safe:.4f}", f"{_speed_fade:.4f}",
                f"{te_alpha_f:.6f}", f"{te_alpha_r:.6f}",
                f"{te_Fz_f:.1f}", f"{te_Fz_r:.1f}",
                f"{te_kappa:.6f}", f"{te_alpha_rate_f:.6f}",
                f"{te_alpha_rate_r:.6f}", f"{te_u_safe:.4f}",
                _probe_phase, f"{_probe_target_alpha:.6f}", _probe_override,
                _probe_reason, f"{_probe_nominal_delta:.6f}",
                f"{_probe_requested_delta:.6f}",
            ])

        seq += 1

        # --- Periodic report ---
        if int(args.status_every_n) > 0 and seq % int(args.status_every_n) == 0:
            mean_ms = np.mean(solve_times[-20:]) * 1000
            tau_ms = delay_est.compensation_delay * 1000
            trk = analytics.periodic_summary()
            te_str = ""
            if terrain_estimator is not None:
                te_str = (f"  TE={terrain_class_est}({terrain_confidence:.0%})"
                          f"[μ_fric={terrain_estimator.mu_estimate:.3f}]")
            print(f"  t={msg.time:.1f}s  solve={mean_ms:.1f}ms  "
                  f"τ_comp={tau_ms:.1f}ms  {trk}  "
                  f"u={msg.u:.2f}m/s{te_str}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if solve_times:
        st = np.array(solve_times)
        ct_arr = np.array(analytics.crosstrack_errors) if analytics.crosstrack_errors else None
        avg_cte = np.mean(np.abs(ct_arr)) if ct_arr is not None else float("nan")
        print(f"\n  ACADOS Controller Summary ({model_label}):")
        print(f"    Total solves:   {len(st)}")
        print(f"    Mean solve:     {np.mean(st)*1000:.2f} ms")
        print(f"    Max solve:      {np.max(st)*1000:.2f} ms")
        print(f"    Effective rate: {1.0/np.mean(st):.1f} Hz")
        print(f"    Avg |CTE|:      {avg_cte:.4f} m")
        print(f"    Final τ_comp:   {delay_est.compensation_delay*1000:.1f} ms")
        if pred_pos_err_hist:
            print("    1-step prediction residuals:")
            print(f"      Mean pos:     {np.mean(pred_pos_err_hist):.4f} m")
            print(f"      RMS pos:      {np.sqrt(np.mean(np.square(pred_pos_err_hist))):.4f} m")
            print(f"      Mean |ψ|:      {np.degrees(np.mean(pred_psi_err_hist)):.2f}°")
            print(f"      Mean |u|:      {np.mean(pred_u_err_hist):.3f} m/s")
            print(f"      Mean |v|:      {np.mean(pred_v_err_hist):.3f} m/s")
            print(f"      Mean |ω|:      {np.mean(pred_omega_err_hist):.4f} rad/s")

    print(analytics.final_summary())

    # Close CSV
    if csv_file is not None:
        csv_file.close()
        if csv_path is not None:
            print(f"  Diagnostic CSV written: {csv_path} ({seq} rows)")
    if terrain_obs_file is not None:
        terrain_obs_file.close()
        print(f"  Terrain observations written: {terrain_obs_path} ({seq} rows)")
    if _log_mpc_pred and _pred_times:
        pred_path = run_dir / "mpc_predictions.npz"
        np.savez_compressed(pred_path,
                            times=np.asarray(_pred_times, dtype=np.float64),
                            Z=np.stack(_pred_Z), dt=float(mpc.dt))
        print(f"  MPC predictions written: {pred_path} ({len(_pred_times)} solves)")
    if not args.no_plot:
        analytics.plot_results(
            plot_dir=str(run_dir),
            terrain_name=terrain_name,
            model_label=model_label,
        )


    ctrl_pub.close()
    state_sub.close()


# =============================================================================
# Entry point
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="ACADOS MPC Controller Node (decoupled)")
    p.add_argument("--mpc-blind-obstacles", action="store_true",
                   help="Withhold obstacle data from the controller so the NMPC acts "
                        "as a pure path tracker and the downstream safety filter is "
                        "the sole collision-avoidance layer. This is the condition "
                        "for a teleoperator with no advance knowledge of the "
                        "obstacle field.")

    # Model (NN or analytical tire model)
    p.add_argument("--model", default="nn",
                   choices=["nn", "pacejka", "pacejka-oracle", "pacejka-rigfit", "tmeasy"],
                   help="Tire model: nn (learned surrogate), pacejka (one global "
                        "SCM-calibrated parameter set), pacejka-oracle "
                        "(terrain-specific mu and B from ground truth, an "
                        "information-advantaged reference), pacejka-rigfit "
                        "(parameters fitted to the single-tire rig corpus), or "
                        "tmeasy")
    p.add_argument("--nn-model", default="tire_force_static",
                   help="Tire-surrogate checkpoint directory, used when --model nn")
    p.add_argument("--kappa", default="measured", choices=["zero", "approx", "measured"])
    p.add_argument("--no-lat-transfer", action="store_true",
                   help="Evaluate the tire model at the axle mean load instead of "
                        "at outer and inner wheel loads")
    p.add_argument(
        "--nn-rate-sample-dt",
        type=float,
        default=0.05,
        help="Simulation seconds between finite-difference anchors for the "
             "rate features; set this to the sampling interval the rate "
             "checkpoint was trained on (default 0.05)",
    )
    p.add_argument(
        "--symbolic-rates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute rate features (dκ, dα, du, sr) symbolically inside the "
             "MPC dynamics instead of freezing them as parameters (default: on). "
             "Adds 3 extra states (α_f_prev, α_r_prev, δ_sr_prev) for finite-"
             "difference/rate channels. Use --no-symbolic-rates to disable.",
    )
    # Path
    p.add_argument("--path", default="lane_change",
                   choices=["lane_change", "double_lane_change", "right_left", "sinusoidal", "straight"])
    p.add_argument("--speed", type=float, default=5.0, help="Target speed (m/s)")
    p.add_argument(
        "--speed-weight",
        type=float,
        default=70.0,
        help="Stage-cost weight on (u - v_ref)^2. Lower values keep the MPC "
             "from chasing reference speed as aggressively in turns.",
    )
    p.add_argument(
        "--speed-cost-mode",
        choices=["symmetric", "overspeed"],
        default="symmetric",
        help="'symmetric' tracks v_ref from both sides; 'overspeed' treats "
             "v_ref as a cap and does not reward accelerating up to it.",
    )
    p.add_argument(
        "--obstacle-weight",
        type=float,
        default=5e3,
        help="Stage/terminal soft obstacle-barrier weight for autonomous MPC obstacle avoidance.",
    )
    p.add_argument("--sine-amplitude", type=float, default=2.0)
    p.add_argument("--sine-wavelength", type=float, default=30.0)
    p.add_argument("--lead-in", type=float, default=0.0,
                   help="Straight lead-in distance (m) before path starts")
    p.add_argument("--no-path-reindex", action="store_true")

    # Terrain
    p.add_argument("--terrain", default="sand", choices=["sand", "clay", "dirt"])
    p.add_argument("--grip-safety", type=float, default=0.72,
                   help="Fraction of the surrogate's predicted lateral capacity "
                        "the speed profile is willing to use. Below 1 the "
                        "controller keeps grip margin; above 1 it over-drives "
                        "the terrain. Used by the grip-margin crossover study.")
    p.add_argument("--controller-prior-terrain", default=None,
                   choices=["sand", "clay", "dirt"],
                   help="Override what terrain the controller's static prior assumes. "
                        "Without estimation it defaults to --terrain. With estimation "
                        "it defaults to --terrain-estimator-prior (dirt).")
    p.add_argument(
        "--terrain-estimator-prior",
        default="dirt",
        choices=["sand", "clay", "dirt"],
        help="Blind controller/estimator bootstrap prior used before the first "
             "accepted online terrain update (default: dirt). Plant terrain "
             "broadcasts cannot overwrite this prior while estimation is on.",
    )
    p.add_argument("--ay-safety", type=float, default=0.65,
                   help="Fraction of the Coulomb lateral-accel limit the "
                        "curvature-limited speed profile may use in turns. "
                        "Higher = faster cornering reference (speed sweeps).")
    p.add_argument(
        "--reference-profile-friction-angle-deg",
        type=float,
        default=None,
        help="Optional fixed design-envelope friction angle for the static "
             "curvature speed profile. It is independent of plant terrain.",
    )
    p.add_argument(
        "--shared-ay-bound-friction-angle-deg",
        type=float,
        default=None,
        help="Optional fixed design-envelope friction angle used to compile "
             "one common NMPC lateral-acceleration bound across ablation arms.",
    )
    p.add_argument(
        "--terrain-independent-ay-bound",
        action="store_true",
        help="Use the common 6 m/s^2 NMPC lateral-acceleration ceiling instead "
             "of compiling a terrain-prior-specific ceiling. Intended for "
             "controlled terrain-conditioning ablations.",
    )
    p.add_argument("--time", type=float, default=15.0, help="Expected sim duration")

    # Delay compensation
    p.add_argument("--no-delay-comp", action="store_true",
                   help="Disable transport delay compensation in MPC")
    p.add_argument("--initial-delay", type=float, default=0.02,
                   help="Initial one-way delay estimate (s)")
    p.add_argument(
        "--state-predict-dt",
        type=float,
        default=0.005,
        help="Delay-compensation predictor RK substep (s); smaller = finer τ forward roll",
    )
    p.add_argument(
        "--control-buffer-len",
        type=int,
        default=50,
        help="Max past (δ̇, Jx) samples kept for delay compensation",
    )

    # MPC / solver build (defaults match acados_mpc_solver.DEFAULT_*)
    p.add_argument(
        "--mpc-dt",
        type=float,
        default=DEFAULT_MPC_DT,
        help="MPC discretisation step [s]; must match precompiled solver if you use one",
    )
    p.add_argument(
        "--mpc-n",
        type=int,
        default=DEFAULT_MPC_HORIZON_STEPS,
        help="MPC horizon length (stages); must match precompiled solver if you use one",
    )
    p.add_argument(
        "--acados-build-dir",
        default=None,
        metavar="DIR",
        help="Exact directory for ACADOS codegen (overrides ACADOS_MPC_BUILD_ROOT / tmp)",
    )
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Dummy solves before connecting to sim (JIT / first-factor warm-up)",
    )
    p.add_argument(
        "--zmq-recv-timeout-ms",
        type=int,
        default=200,
        help="State subscriber poll timeout (ms)",
    )
    p.add_argument(
        "--ready-ping-interval-s",
        type=float,
        default=0.25,
        help="While waiting for first VehicleState, re-send neutral ControlCommand period (s)",
    )
    p.add_argument(
        "--status-every-n",
        type=int,
        default=20,
        help="Print timing/tracking line every N control steps (0 = disable)",
    )
    p.add_argument(
        "--lead-in-speed-fraction",
        type=float,
        default=0.8,
        help="During lead-in, zero steering while u below this fraction of v_target",
    )

    # Throttle disturbance observer (asymmetric velocity-error DOB)
    p.add_argument(
        "--dob-ki",
        type=float,
        default=0.15,
        help="Throttle DOB integrator gain [throttle/(m/s)/s]; 0 disables the DOB",
    )
    p.add_argument(
        "--dob-max",
        type=float,
        default=0.35,
        help="Asymmetric upper clip on the DOB throttle bias (0 = no compensation)",
    )
    p.add_argument(
        "--dob-bleed",
        type=float,
        default=0.5,
        help="Exponential bleed rate of the DOB during MPC braking [1/s]",
    )
    # Feedforward sinkage-drag term: injects du_dot_resid = -c_drag(n_hat) into
    # the NMPC longitudinal prediction, u_dot = ax + du_dot_resid, so the
    # planner anticipates soft-soil motion resistance rather than correcting
    # for it reactively through the throttle observer. c_drag(n) is calibrated
    # from rollout drift with the observer disabled
    # (benchmarking/calibrate_motion_resistance.py).
    p.add_argument(
        "--ff-drag",
        action="store_true",
        help="Enable the feedforward sinkage-drag term in the NMPC longitudinal "
             "prediction (du_dot_resid = -c_drag(n_hat)).",
    )
    p.add_argument(
        "--ff-drag-scale",
        type=float,
        default=1.0,
        help="Scale on the calibrated feedforward drag deceleration (1.0 = as "
             "calibrated; 0 disables).",
    )
    p.add_argument(
        "--ff-throttle",
        action="store_true",
        help="Apply a calibrated feedforward terrain throttle offset "
             "d_ff(n_hat) in place of the integral throttle observer. Pair "
             "with --dob-ki 0 for a purely feedforward actuation map.",
    )
    p.add_argument(
        "--ff-drag-surrogate",
        action="store_true",
        help="Feedforward drag queried live from the surrogate: "
             "throttle_ff = N*(-Fx(kappa=0))/(M*ax_max), the compaction-drag "
             "term of Dallas et al. evaluated at the current operating point. "
             "Pair with --dob-ki 0.",
    )
    p.add_argument(
        "--ff-throttle-2d",
        action="store_true",
        help="Feedforward throttle offset d_ff(n_hat, u) indexed by soil and "
             "forward speed, capturing the operating-point dependence that a "
             "soil-only map cannot express. Pair with --dob-ki 0.",
    )
    p.add_argument(
        "--ff-throttle-scale",
        type=float,
        default=1.0,
        help="Scale on the calibrated feedforward throttle offset (0 disables).",
    )
    p.add_argument(
        "--terrain-speed-profile",
        action="store_true",
        help="Accepted with no effect: the terrain-, dynamics-, and "
             "state-aware g-g speed profile is the default reference.",
    )
    p.add_argument(
        "--legacy-speed-ref",
        action="store_true",
        help="Use the static curvature-only speed reference in place of the "
             "terrain-, dynamics-, and state-aware g-g profile, which gives "
             "every arm the same requested speed for controlled ablations.",
    )
    p.add_argument(
        "--longitudinal-force-balance",
        action="store_true",
        help="Close the longitudinal channel through an explicit force "
             "balance: the NMPC state becomes the slip ratio kappa with "
             "control kappa-dot, and u_dot = SumFx(kappa)/M comes from the "
             "surrogate rather than from the kinematic u_dot = ax. Throttle "
             "is realised from the planned acceleration.",
    )

    # Analytics
    p.add_argument("--rms-time-start", type=float, default=2.0,
                   help="Start time for RMS calculation, skips startup (s)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip generating end-of-run plots")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip diagnostic CSV output")
    p.add_argument("--plot-dir", default="plots",
                   help="Directory for output plots (default: plots/)")
    p.add_argument(
        "--ax-filter-tau", type=float, default=0.5,
        help="Complementary filter time constant (s) for fusing IMU ax with "
             "model prediction.  Suppresses terrain-induced noise.  0 = no filter.",
    )
    p.add_argument(
        "--vel-filter-tau", type=float, default=0.05,
        help="Exponential-moving-average time constant (s) for smoothing "
             "[u, v, omega] before the MPC. Sized against the 0.05 m/s lateral "
             "velocity noise; 0 disables the filter. The 0.05 s default gives "
             "a filter coefficient of about 0.67 at 10 Hz.",
    )

    # Network
    p.add_argument("--sim-host", default="localhost", help="Sim node host")
    p.add_argument("--sim-port", type=int, default=5555, help="Sim state port")
    p.add_argument("--ctrl-port", type=int, default=5556, help="Control command port")
    p.add_argument("--transport", choices=["zmq", "ros"], default=os.environ.get("HIL_TRANSPORT", "ros"),
                   help="IPC transport for the sim<->controller link: ros "
                        "(default, direct rclpy/DDS; needs ROS 2 sourced) or zmq.")

    # Online terrain parameter estimation
    p.add_argument("--terrain-estimator", action="store_true",
                   help="Enable online terrain estimation from deployable vehicle-state, "
                        "wheel, steering, and sensor channels (no oracle data).")
    p.add_argument("--terrain-estimator-mode", choices=["n"], default="n",
                   help="Compatibility flag; the selected backend declares its "
                        "live output fields.")
    p.add_argument("--terrain-estimator-backend",
                   choices=TERRAIN_ESTIMATOR_BACKENDS,
                   default=RIG_ACTIVE_ESTIMATOR_BACKEND,
                   help="Runtime terrain estimator. 'grit' is the "
                        "default: it estimates the sinkage exponent and the "
                        "friction angle jointly over the rate force surrogate "
                        "without a ground datum, applies only snapshots that are "
                        "fresh, observable, and away from the grid boundary, and "
                        "otherwise holds the labelled control-feasible low-grip "
                        "fallback. 'scalar_parent' estimates the sinkage "
                        "exponent alone over the same force map. 'bekker_ukf' is "
                        "the analytical comparison backend.")
    p.add_argument("--nn-ukf-q-n", type=float, default=0.04,
                   help="Process-noise std on n per 0.1 s for the bekker_ukf backend.")
    p.add_argument("--parent-grid-size", type=int, default=41,
                   help="Number of sinkage-exponent grid points for the "
                        "single-parameter profile (even values are rounded up).")
    p.add_argument("--parent-student-dof", type=float, default=4.0,
                   help="Student-t likelihood degrees of freedom for the "
                        "single-parameter profile.")
    p.add_argument("--estimator-update-interval", type=int, default=1)
    p.add_argument("--estimator-block-dt", type=float, default=0.5)
    p.add_argument("--estimator-horizon", type=float, default=8.0)
    p.add_argument("--estimator-min-windows", type=int, default=12)
    p.add_argument("--estimator-min-window-samples", type=int, default=4)
    p.add_argument("--estimator-r-ax", type=float, default=0.35)
    p.add_argument("--estimator-r-ay", type=float, default=0.30)
    p.add_argument("--estimator-min-information", type=float, default=0.20)
    p.add_argument("--estimator-min-yaw-rate-rms", type=float, default=0.015)
    p.add_argument("--estimator-min-speed", type=float, default=2.5)
    p.add_argument("--estimator-max-abs-alpha", type=float, default=0.35)
    p.add_argument(
        "--estimator-enforce-feature-envelope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--estimator-slip-mode",
        choices=("wheel", "average", "fixed"),
        default="average",
    )
    p.add_argument("--estimator-fixed-kappa", type=float, default=0.05)
    p.add_argument(
        "--estimator-rate-mode",
        choices=("signed", "zero", "legacy"),
        default="zero",
    )
    p.add_argument("--estimator-force-gain-std", type=float, default=0.04)
    p.add_argument("--estimator-ax-bias-std", type=float, default=0.10)
    p.add_argument("--estimator-ay-bias-std", type=float, default=0.05)
    p.add_argument("--estimator-force-gain-min", type=float, default=0.70)
    p.add_argument("--estimator-force-gain-max", type=float, default=1.30)
    p.add_argument(
        "--estimator-acceleration-bias-bound", type=float, default=0.30
    )
    p.add_argument("--estimator-profile-iterations", type=int, default=8)
    p.add_argument("--te-update-interval", type=int, default=10,
                   help="Run terrain estimation every N accepted 10 Hz-equivalent samples "
                        "(default 10 -> 1 s)")
    p.add_argument("--te-filter-dt", type=float, default=0.1,
                   help="Terrain-filter update period in simulation seconds "
                        "(default 0.1 = 10 Hz).")
    p.add_argument("--te-min-confidence", type=float, default=0.3,
                   help="Minimum confidence to apply estimated terrain params to MPC")
    p.add_argument("--speed-osc-amplitude", type=float, default=0.0,
                   help="Fractional sinusoidal modulation of the speed reference, "
                        "for example 0.25 for plus or minus 25%. Applied before the "
                        "g-g cap, so it can only ask for less speed than the grip "
                        "envelope allows. A sinusoidal path traversed at constant "
                        "speed excites the lateral channel alone and leaves the "
                        "Bekker exponent weakly observable; this adds an independent "
                        "longitudinal excitation without departing the path.")
    p.add_argument("--speed-osc-period-s", type=float, default=4.0,
                   help="Period in seconds of the speed-reference oscillation.")
    p.add_argument("--te-joint-model-dir", default=None,
                   help="Candidate evaluation only: override the joint estimator's "
                        "force-model directory. Left unset, the declared contract "
                        "applies (nn_models/tire_force_rate). Any override is "
                        "printed at launch so a run can never silently mix "
                        "configurations.")
    p.add_argument("--te-joint-r-ay", type=float, default=None,
                   help="Candidate evaluation only: override the joint estimator's "
                        "lateral residual scale. Left unset, the declared contract "
                        "value of 0.45 applies.")
    p.add_argument("--hedge-k", type=float, default=0.0,
                   help="Uncertainty-aware envelope: evaluate the g-g grip limits "
                        "at the pessimistic soil quantile n_hat - k*sigma_n rather "
                        "than at the mean estimate. The MPC prediction model still "
                        "uses the mean, so the mean conditions the model and the "
                        "quantile conditions the envelope. 0 uses the mean for "
                        "both.")
    p.add_argument("--ukf-model-dir", default=None,
                   help="Optional override for the force checkpoint used by the "
                        "single-parameter estimators. Defaults to "
                        "nn_models/tire_force_static_parent.")
    p.add_argument("--te-verbose", action="store_true",
                   help="Print verbose terrain-estimator predictions (every "
                        "10 observations) for offline parsing/validation.")

    p.add_argument("--terrain-id-probe", action="store_true",
                   help="Run one clear-road achieved-slip doublet for terrain "
                        "identification. The target enters before the common "
                        "road-wheel angle/rate limiter.")
    p.add_argument("--terrain-id-probe-target-alpha", type=float, default=0.10)
    p.add_argument("--terrain-id-probe-slew-rate", type=float, default=0.40)
    p.add_argument("--terrain-id-probe-signed-dwell", type=float, default=0.15)
    p.add_argument("--terrain-id-probe-clearance", type=float, default=35.0)
    p.add_argument("--terrain-id-probe-max-latency", type=float, default=0.30)

    # Accepted so that a request for open-loop steering excitation fails with
    # a clear message rather than silently doing nothing. Excitation must
    # enter ahead of the actuator limiter, in road-wheel coordinates, so that
    # the controller and estimator observe the same command history;
    # --terrain-id-probe provides that.
    p.add_argument("--excitation-steer-amp", type=float, default=0.0,
                   help=argparse.SUPPRESS)
    p.add_argument("--excitation-steer-period", type=float, default=1.0,
                   help=argparse.SUPPRESS)

    args = p.parse_args()
    if args.excitation_steer_amp > 0.0:
        p.error("--excitation-steer-* is not supported; use --terrain-id-probe")
    run_controller_node(args)


if __name__ == "__main__":
    main()
