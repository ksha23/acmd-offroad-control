#!/usr/bin/env python3
"""Vehicle operating-point reconstruction for the tire force surrogate.

The surrogate is defined on per-wheel slip, load, and speed, none of which is
measured directly on the vehicle.  These functions reconstruct those
quantities at runtime from IMU, wheel-speed, and steering measurements plus
fixed vehicle geometry, so that the controller, the diagnostics, and the
terrain estimator all query the surrogate at the same operating point.
Training-data collection lives in ``data_collection/``.
"""

from __future__ import annotations
import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401

from dataclasses import dataclass
import math
from typing import Literal, Tuple

import numpy as np

from param_consistency import HMMWV_VEHICLE_PARAMS, HMMWV_TIRE_RADIUS_M

KappaMode = Literal["zero", "approx", "measured"]


@dataclass(frozen=True)
class VehicleGeometry:
    """Bicycle and load-transfer geometry, matching the AcadosMPC vehicle."""

    Lf: float
    Lr: float
    M: float
    h_cg: float
    T: float

    @classmethod
    def from_hmmwv_defaults(cls) -> "VehicleGeometry":
        vp = HMMWV_VEHICLE_PARAMS
        return cls(
            Lf=float(vp["Lf"]),
            Lr=float(vp["Lr"]),
            M=float(vp["M"]),
            h_cg=float(vp["h_cg"]),
            T=float(vp["T"]),
        )


@dataclass
class BicycleOperatingPoint:
    """One timestep of tire-surrogate inputs at the per-axle mean wheel."""

    kappa: float
    alpha_f: float
    alpha_r: float
    u_safe: float
    Fz_f: float  # mean front wheel vertical load (N), from the load-transfer model
    Fz_r: float
    steering_rate_cmd: float  # δ̇ supplied as the steering_rate feature (rad/s)


def level_specific_force_to_yaw_frame(
    ax_body: float,
    ay_body: float,
    az_body: float,
    quat_e0: float,
    quat_e1: float,
    quat_e2: float,
    quat_e3: float,
) -> Tuple[float, float]:
    """Return horizontal specific force in the vehicle's yaw-only frame.

    A body-frame accelerometer mixes gravity/specific force into its lateral
    channel when the chassis rolls or pitches.  The four-wheel rig projection
    predicts horizontal longitudinal/lateral acceleration for a planar body,
    so compare it with the IMU vector rotated through the measured attitude
    and then back by yaw only.  Gravity is vertical in the global frame and
    therefore drops out of the returned horizontal components.
    """

    q = np.asarray([quat_e0, quat_e1, quat_e2, quat_e3], dtype=float)
    force_body = np.asarray([ax_body, ay_body, az_body], dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(force_body).all():
        raise ValueError("specific force and quaternion must be finite")
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-9:
        raise ValueError("specific-force leveling requires a nonzero quaternion")
    w, x, y, z = q / norm

    # Body -> global rotation, followed by global -> yaw-only vehicle frame.
    force_global_x = (
        (1.0 - 2.0 * (y * y + z * z)) * force_body[0]
        + 2.0 * (x * y - w * z) * force_body[1]
        + 2.0 * (x * z + w * y) * force_body[2]
    )
    force_global_y = (
        2.0 * (x * y + w * z) * force_body[0]
        + (1.0 - 2.0 * (x * x + z * z)) * force_body[1]
        + 2.0 * (y * z - w * x) * force_body[2]
    )
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        float(cos_yaw * force_global_x + sin_yaw * force_global_y),
        float(-sin_yaw * force_global_x + cos_yaw * force_global_y),
    )


def kappa_from_wheel_speed(
    wheel_omega_fl: float,
    wheel_omega_fr: float,
    wheel_omega_rl: float,
    wheel_omega_rr: float,
    u_body: float,
    tire_radius: float = HMMWV_TIRE_RADIUS_M,
) -> float:
    """Measured longitudinal slip ratio from wheel speed sensors.

    κ = (R·ω_avg − |u|) / |u|

    This follows Chrono's ``GetLongitudinalSlip()`` convention, the same
    convention that labels the single-tire rig training rows.  The 0.5 m/s
    denominator floor guards the singularity as the vehicle approaches rest,
    and the result is clipped to the surrogate's training envelope.
    """
    omega_avg = (wheel_omega_fl + wheel_omega_fr
                 + wheel_omega_rl + wheel_omega_rr) / 4.0
    Vw = tire_radius * abs(omega_avg)
    u_abs = abs(u_body)
    if max(u_abs, Vw) < 0.5:
        return 0.0
    denom = max(u_abs, 0.5)
    kappa = float((Vw - u_abs) / denom)
    return float(np.clip(kappa, -0.8, 0.8))


def compute_bicycle_operating_point(
    steering_angle_rad: float,
    u_body: float,
    v_body: float,
    omega: float,
    ax_body: float,
    *,
    geom: VehicleGeometry,
    kappa_mode: KappaMode = "zero",
    terrain_mu: float = 0.4,
    measured_kappa: float = 0.0,
    g: float = 9.81,
) -> Tuple[float, float, float, float, float, float]:
    """Reconstruct the per-axle tire operating point from vehicle state.

    The convention matches the pre-solve operating point assembled in
    ``acados_mpc_controller_node``.  No delay compensation is applied here, so
    a caller that needs parity with the solver's initial state ``z0`` must
    pass a state that has already been propagated through the delay predictor.

    Args:
        steering_angle_rad: Road-wheel angle δ [rad].
        u_body, v_body: Longitudinal and lateral body velocities [m/s].
        omega: Yaw rate [rad/s].
        ax_body: Longitudinal acceleration [m/s^2], used for load transfer.
        geom: Vehicle geometry.
        kappa_mode: Source of the longitudinal slip ratio.
        terrain_mu: Effective friction coefficient for the ``'approx'`` slip
            model; tan(phi) of the terrain friction angle is the intended
            value.
        measured_kappa: Slip ratio from wheel-speed sensors, consumed when
            ``kappa_mode`` is ``'measured'``.
        g: Gravitational acceleration [m/s^2].

    Returns:
        ``(kappa, alpha_f, alpha_r, u_safe, Fz_f, Fz_r)``, with the vertical
        loads given per wheel.
    """
    u_safe = float(max(abs(u_body), 0.5))
    alpha_f = float(
        steering_angle_rad - math.atan2(v_body + geom.Lf * omega, u_safe)
    )
    alpha_r = float(-math.atan2(v_body - geom.Lr * omega, u_safe))
    # Clamp slip angles to the training range so the surrogate is never
    # queried outside the envelope it was supervised on.
    _alpha_max = 0.55
    alpha_f = float(max(-_alpha_max, min(_alpha_max, alpha_f)))
    alpha_r = float(max(-_alpha_max, min(_alpha_max, alpha_r)))
    L = geom.Lf + geom.Lr
    Fz_f = float((geom.M * g * geom.Lr - geom.M * ax_body * geom.h_cg) / L / 2.0)
    Fz_r = float((geom.M * g * geom.Lf + geom.M * ax_body * geom.h_cg) / L / 2.0)
    if kappa_mode == "measured":
        kappa = float(np.clip(measured_kappa, -0.8, 0.8))
    elif kappa_mode == "approx":
        mu_eff = max(terrain_mu, 0.1)
        kappa = float(np.clip(ax_body / (mu_eff * 9.81), -0.8, 0.8))
    else:
        kappa = 0.0
    return kappa, alpha_f, alpha_r, u_safe, Fz_f, Fz_r


def lateral_load_transfer_dFz(
    u_body: float,
    omega: float,
    *,
    geom: VehicleGeometry,
) -> float:
    """Lateral load transfer half-amplitude per side (N), with ay ≈ u·omega."""
    ay = u_body * omega
    return float(geom.M * ay * geom.h_cg / geom.T / 2.0)


def fz_with_lateral_transfer(
    Fz_f_mean: float,
    Fz_r_mean: float,
    dFz: float,
) -> Tuple[float, float, float, float]:
    """Outer and inner wheel loads, clamped as in ``acados_mpc_solver``.

    The clamps hold each wheel between 10% and 190% of its axle mean so that a
    transient lateral acceleration can neither unload a wheel to zero nor
    drive the surrogate past the vertical loads it was supervised on.
    """
    Fz_fo = min(Fz_f_mean + dFz, 1.9 * Fz_f_mean)
    Fz_fi = max(Fz_f_mean - dFz, 0.1 * Fz_f_mean)
    Fz_ro = min(Fz_r_mean + dFz, 1.9 * Fz_r_mean)
    Fz_ri = max(Fz_r_mean - dFz, 0.1 * Fz_r_mean)
    return Fz_fo, Fz_fi, Fz_ro, Fz_ri
