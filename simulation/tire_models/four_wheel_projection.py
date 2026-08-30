"""Four-wheel vehicle projection of the single-tire-rig force surrogate.

The terrain estimator observes a body-frame wrench, whereas the surrogate is
defined per wheel.  This module resolves the four per-wheel force queries into
the planar wrench ``(Fx, Fy, Mz)``, and is the runtime path for that
conversion: it depends on the surrogate alone, so runtime code never has to
import an offline evaluation module to obtain a body wrench.

Individual left and right wheel velocities and longitudinal forces are carried
separately all the way through the yaw-moment sum, because a track-width
moment arm acts on the left/right longitudinal force difference and would be
lost by averaging the two sides first.
"""

from __future__ import annotations

import math
import os as _os
import sys as _sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401

from nn_tire_model import load_nn_tire_model  # noqa: E402
from param_consistency import HMMWV_VEHICLE_PARAMS  # noqa: E402


_MODEL_CACHE: dict[str, Any] = {}


def _load_force_model(
    model_dir: str, initial_terrain: Mapping[str, float]
) -> Any:
    """Reuse one immutable force checkpoint per worker process."""

    if model_dir not in _MODEL_CACHE:
        _MODEL_CACHE[model_dir] = load_nn_tire_model(model_dir, initial_terrain)
    return _MODEL_CACHE[model_dir]


# Required keys of the controller's vehicle schema.  ``g`` is a physical
# constant rather than a vehicle property and may be omitted.
_REQUIRED_VEHICLE_KEYS = ("M", "Izz", "Lf", "Lr", "T", "h_cg")


@dataclass(frozen=True)
class ProjectionVehicle:
    """Planar vehicle properties required by the four-wheel projection.

    The defaults are taken from the shared vehicle struct rather than restated,
    so they cannot drift away from the plant the controller and estimators
    assume.
    """

    m: float = float(HMMWV_VEHICLE_PARAMS["M"])
    Iz: float = float(HMMWV_VEHICLE_PARAMS["Izz"])
    Lf: float = float(HMMWV_VEHICLE_PARAMS["Lf"])
    Lr: float = float(HMMWV_VEHICLE_PARAMS["Lr"])
    track: float = float(HMMWV_VEHICLE_PARAMS["T"])
    h_cg: float = float(HMMWV_VEHICLE_PARAMS["h_cg"])
    g: float = 9.81

    @classmethod
    def from_mapping(
        cls, values: Optional[Mapping[str, float]] = None
    ) -> "ProjectionVehicle":
        """Build from the controller's ``M/Izz/Lf/Lr/T/h_cg`` schema.

        A partial mapping is refused.  Silently substituting a default would
        let a caller run the projection against a vehicle other than the one
        being driven, and the resulting loads would look entirely ordinary.
        Passing no mapping at all returns the shared-struct vehicle.
        """

        if values is None:
            return cls()
        missing = [key for key in _REQUIRED_VEHICLE_KEYS if key not in values]
        if missing:
            raise KeyError(
                "vehicle mapping is missing required keys: "
                + ", ".join(missing)
            )
        return cls(
            m=float(values["M"]),
            Iz=float(values["Izz"]),
            Lf=float(values["Lf"]),
            Lr=float(values["Lr"]),
            track=float(values["T"]),
            h_cg=float(values["h_cg"]),
            g=float(values.get("g", 9.81)),
        )


def _per_wheel_force(
    model: Any,
    vertical_load: float,
    longitudinal_velocity: float,
    lateral_velocity: float,
    terrain: Mapping[str, float],
    slip_ratio: float,
    steering_rate: float,
) -> Tuple[float, float]:
    """Evaluate the force surrogate at one wheel operating point."""

    speed = max(abs(float(longitudinal_velocity)), 0.5)
    slip_angle = math.atan2(float(lateral_velocity), speed)
    force_x, force_y = model.predict_numeric(
        alpha=slip_angle,
        Fz=float(vertical_load),
        u=speed,
        kappa=float(np.clip(slip_ratio, -0.8, 0.8)),
        n_terrain=float(terrain["n"]),
        steering_rate=float(steering_rate),
        terrain_params=dict(terrain),
        rates=np.zeros(3, dtype=float),
    )
    return float(force_x), float(force_y)


def four_wheel_body_wrench(
    z_aug: Sequence[float],
    delta: float,
    ax_in: float,
    terrain_params: Mapping[str, float],
    vehicle: ProjectionVehicle,
    force_model: Any,
    *,
    ay_in: Optional[float] = None,
    kappa_in: float = 0.05,
    steering_rate_in: float = 0.0,
    rear_steering_rate_in: float = 0.0,
    wheel_omegas: Optional[Sequence[float]] = None,
    tire_radius: float = 0.47,
    Fz_front_mean: Optional[float] = None,
    Fz_rear_mean: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Return the body-frame ``(Fx, Fy, Mz)`` wrench of the four wheels.

    When ``ay_in`` is omitted, the lateral acceleration required for lateral
    load transfer is obtained from a first projection at longitudinally
    transferred loads, and the wheels are then re-evaluated at the resulting
    left/right loads.  Supplying a measured ``ay_in`` skips that first pass.
    """

    state = np.asarray(z_aug, dtype=float).reshape(-1)
    if state.size != 7 or not np.isfinite(state).all():
        raise ValueError("z_aug must contain seven finite states")
    _, _, _, u, v, omega, n_value = state
    n_value = float(np.clip(n_value, 0.2, 1.4))
    terrain = dict(terrain_params)
    terrain["n"] = n_value

    wheelbase = vehicle.Lf + vehicle.Lr
    track = vehicle.track
    if Fz_front_mean is None or Fz_rear_mean is None:
        front_mean = vehicle.m * vehicle.g * vehicle.Lr / (2.0 * wheelbase)
        rear_mean = vehicle.m * vehicle.g * vehicle.Lf / (2.0 * wheelbase)
        longitudinal_transfer = vehicle.m * ax_in * vehicle.h_cg / wheelbase
    else:
        front_mean = max(float(Fz_front_mean), 100.0)
        rear_mean = max(float(Fz_rear_mean), 100.0)
        longitudinal_transfer = 0.0

    u_left = u - 0.5 * track * omega
    u_right = u + 0.5 * track * omega
    v_front = v + vehicle.Lf * omega
    v_rear = v - vehicle.Lr * omega
    cosine = math.cos(delta)
    sine = math.sin(delta)
    wheel_velocities = (
        (u_left * cosine + v_front * sine,
         -u_left * sine + v_front * cosine),
        (u_right * cosine + v_front * sine,
         -u_right * sine + v_front * cosine),
        (u_left, v_rear),
        (u_right, v_rear),
    )

    if wheel_omegas is not None and len(wheel_omegas) == 4:
        kappas = tuple(
            float(np.clip(
                (
                    abs(float(wheel_rate)) * float(tire_radius)
                    - abs(float(longitudinal))
                ) / max(abs(float(longitudinal)), 0.5),
                -0.8,
                0.8,
            ))
            for wheel_rate, (longitudinal, _lateral) in zip(
                wheel_omegas, wheel_velocities
            )
        )
    else:
        kappas = (float(kappa_in),) * 4

    def project_wheels(
        load_lf: float,
        load_rf: float,
        load_lr: float,
        load_rr: float,
    ) -> Tuple[float, ...]:
        loads = (load_lf, load_rf, load_lr, load_rr)
        rates = (
            steering_rate_in,
            steering_rate_in,
            rear_steering_rate_in,
            rear_steering_rate_in,
        )
        forces = tuple(
            _per_wheel_force(
                force_model,
                load,
                longitudinal,
                lateral,
                terrain,
                kappa,
                rate,
            )
            for load, (longitudinal, lateral), kappa, rate in zip(
                loads, wheel_velocities, kappas, rates
            )
        )
        return tuple(component for force in forces for component in force)

    load_lf_0 = max(front_mean - 0.5 * longitudinal_transfer, 100.0)
    load_rf_0 = max(front_mean - 0.5 * longitudinal_transfer, 100.0)
    load_lr_0 = max(rear_mean + 0.5 * longitudinal_transfer, 100.0)
    load_rr_0 = max(rear_mean + 0.5 * longitudinal_transfer, 100.0)
    if ay_in is None:
        initial_forces = project_wheels(
            load_lf_0, load_rf_0, load_lr_0, load_rr_0
        )
        (
            force_x_lf,
            force_y_lf,
            force_x_rf,
            force_y_rf,
            _force_x_lr,
            force_y_lr,
            _force_x_rr,
            force_y_rr,
        ) = initial_forces
        front_y_body = (
            (force_x_lf + force_x_rf) * sine
            + (force_y_lf + force_y_rf) * cosine
        )
        ay_use = (front_y_body + force_y_lr + force_y_rr) / vehicle.m - u * omega
    else:
        ay_use = float(ay_in)

    front_mass = vehicle.m * vehicle.Lr / wheelbase
    rear_mass = vehicle.m * vehicle.Lf / wheelbase
    lateral_transfer_front = front_mass * ay_use * vehicle.h_cg / track
    lateral_transfer_rear = rear_mass * ay_use * vehicle.h_cg / track
    load_lf = max(load_lf_0 - lateral_transfer_front, 100.0)
    load_rf = max(load_rf_0 + lateral_transfer_front, 100.0)
    load_lr = max(load_lr_0 - lateral_transfer_rear, 100.0)
    load_rr = max(load_rr_0 + lateral_transfer_rear, 100.0)
    (
        force_x_lf,
        force_y_lf,
        force_x_rf,
        force_y_rf,
        force_x_lr,
        force_y_lr,
        force_x_rr,
        force_y_rr,
    ) = project_wheels(load_lf, load_rf, load_lr, load_rr)

    force_x_lf_body = force_x_lf * cosine - force_y_lf * sine
    force_y_lf_body = force_x_lf * sine + force_y_lf * cosine
    force_x_rf_body = force_x_rf * cosine - force_y_rf * sine
    force_y_rf_body = force_x_rf * sine + force_y_rf * cosine
    force_x_total = force_x_lf_body + force_x_rf_body + force_x_lr + force_x_rr
    force_y_total = force_y_lf_body + force_y_rf_body + force_y_lr + force_y_rr
    yaw_moment = (
        vehicle.Lf * (force_y_lf_body + force_y_rf_body)
        - vehicle.Lr * (force_y_lr + force_y_rr)
        - 0.5 * track * force_x_lf_body
        + 0.5 * track * force_x_rf_body
        - 0.5 * track * force_x_lr
        + 0.5 * track * force_x_rr
    )
    return float(force_x_total), float(force_y_total), float(yaw_moment)


class FourWheelProjector:
    """Owns the force checkpoint and exposes the four-wheel projection.

    The checkpoint is loaded on first use so that constructing a projector is
    inexpensive in processes that may never evaluate one.
    """

    def __init__(
        self,
        model_dir: str | Path,
        initial_terrain: Mapping[str, float],
    ) -> None:
        self.model_dir = str(Path(model_dir).expanduser().resolve())
        self._initial_terrain = dict(initial_terrain)
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = _load_force_model(
                self.model_dir, self._initial_terrain
            )
        return self._model

    def body_wrench(self, *args: Any, **kwargs: Any) -> Tuple[float, float, float]:
        return four_wheel_body_wrench(
            *args, force_model=self.model, **kwargs
        )
