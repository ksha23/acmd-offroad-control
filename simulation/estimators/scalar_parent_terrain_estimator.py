"""Ground-datum-free terrain inference from measured vehicle dynamics.

The estimator keeps scalar grid weights over the same clay--dirt--sand ``n``
manifold used by the controller.  Each hypothesis is projected through
the controlled single-tire rig force surrogate at the measured four-wheel
operating point.  Non-overlapping causal blocks compare the resulting body
accelerations with levelled longitudinal and lateral IMU acceleration.

The single-tire rig and full vehicle do not have identical force scale or
zero-offset errors.  Treating that transfer error as terrain evidence biases
the estimate.  The rolling likelihood therefore profiles three bounded
nuisance parameters for every terrain hypothesis: one shared force gain and
one bias per IMU axis.  Tight zero-centred/identity priors retain terrain
identifiability while absorbing small rig-to-vehicle discrepancy.  A robust
Student-t profile cost supplies the terrain MAP and normalized profile weights
for uncertainty reporting.  This is not a recursive Bayesian filter: each
update replaces the weights using the current causal horizon.

No wheel-centre height, terrain elevation, Chrono contact force, or plant soil
field is accepted as evidence.  The only learned component is the existing
controlled-rig tire-force map.
"""

from __future__ import annotations

import math
import os as _os
import sys as _sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401

from four_wheel_projection import (  # noqa: E402
    FourWheelProjector,
    ProjectionVehicle,
)
from terrain_parameterization import N_BOUNDS, terrain_params_for_n  # noqa: E402


DynamicsPrediction = Callable[
    [Dict[str, Any], np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]


class ScalarParentTerrainEstimator:
    """Rolling robust profile-likelihood terrain-coordinate estimator."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        initial_terrain: Optional[Dict[str, float]] = None,
        vehicle_params: Optional[Dict[str, float]] = None,
        *,
        block_dt: float = 0.5,
        horizon: float = 8.0,
        min_windows: int = 12,
        min_window_samples: int = 4,
        r_ax: float = 0.35,
        r_ay: float = 0.30,
        min_information: float = 0.20,
        min_yaw_rate_rms: float = 0.0,
        min_model_speed: float = 2.5,
        max_abs_alpha: float = 0.35,
        enforce_feature_envelope: bool = True,
        smoothing_alpha: float = 1.0,
        slip_mode: str = "average",
        fixed_kappa: float = 0.05,
        rate_mode: str = "zero",
        force_gain_std: float = 0.04,
        ax_bias_std: float = 0.10,
        ay_bias_std: float = 0.05,
        force_gain_bounds: Tuple[float, float] = (0.7, 1.3),
        acceleration_bias_bound: float = 0.30,
        profile_iterations: int = 8,
        update_interval: int = 10,
        verbose: bool = False,
        grid_size: int = 41,
        student_dof: float = 4.0,
        initial_n_std: float = 0.12,
        dynamics_prediction: Optional[DynamicsPrediction] = None,
        **_ignored,
    ):
        self._verbose = bool(verbose)
        self._update_interval = max(1, int(update_interval))
        self._student_dof = max(1.0, float(student_dof))
        self._n_lo, self._n_hi = N_BOUNDS
        count = max(11, int(grid_size))
        if count % 2 == 0:
            count += 1
        self._grid = np.linspace(self._n_lo, self._n_hi, count)
        n_initial = float(np.clip(
            (initial_terrain or {}).get("n", 0.7), self._n_lo, self._n_hi
        ))
        prior_std = max(
            float(initial_n_std), 0.5 * (self._grid[1] - self._grid[0])
        )
        self._posterior = self._normalize_log_weights(
            -0.5 * ((self._grid - n_initial) / prior_std) ** 2
        )
        self._n_posterior = float(np.sum(self._grid * self._posterior))
        self._n_smooth = n_initial
        self._smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self._estimated_params = terrain_params_for_n(n_initial)
        self._confidence = 0.0
        self._observation_count = 0
        self.output_names = ("n",)

        self._vehicle = ProjectionVehicle.from_mapping(vehicle_params)
        default_model = (
            Path(__file__).resolve().parents[2]
            / "nn_models"
            / "tire_force_static_parent"
        )
        self._model_dir = str(
            Path(model_dir).expanduser().resolve() if model_dir else default_model
        )
        if not Path(self._model_dir).is_dir():
            raise FileNotFoundError(
                f"Tire-rig dynamics model not found: {self._model_dir}"
            )
        self._soil_from_n = terrain_params_for_n
        self._force_projector = FourWheelProjector(
            self._model_dir, self._estimated_params
        )
        self._fourwheel_wrench = self._force_projector.body_wrench
        self._block_dt = max(0.25, float(block_dt))
        self._horizon = max(self._block_dt, float(horizon))
        self._min_windows = max(1, int(min_windows))
        self._min_window_samples = max(2, int(min_window_samples))
        self._r_ax = max(1.0e-4, float(r_ax))
        self._r_ay = max(1.0e-4, float(r_ay))
        self._min_information = max(0.0, float(min_information))
        self._min_yaw_rate_rms = max(0.0, float(min_yaw_rate_rms))
        self._min_model_speed = max(0.5, float(min_model_speed))
        self._max_abs_alpha = max(0.01, float(max_abs_alpha))
        self._enforce_feature_envelope = bool(enforce_feature_envelope)
        self._slip_mode = str(slip_mode).strip().lower()
        if self._slip_mode not in {"wheel", "average", "fixed"}:
            raise ValueError("slip_mode must be wheel, average, or fixed")
        self._fixed_kappa = float(np.clip(fixed_kappa, -0.8, 0.8))
        self._rate_mode = str(rate_mode).strip().lower()
        if self._rate_mode not in {"signed", "zero", "legacy"}:
            raise ValueError("rate_mode must be signed, zero, or legacy")
        gain_lower, gain_upper = map(float, force_gain_bounds)
        if not 0.0 < gain_lower < 1.0 < gain_upper:
            raise ValueError("force_gain_bounds must straddle one and be positive")
        self._profile_prior_std = np.asarray(
            [force_gain_std, ax_bias_std, ay_bias_std], dtype=float
        )
        if (
            not np.isfinite(self._profile_prior_std).all()
            or np.any(self._profile_prior_std <= 0.0)
        ):
            raise ValueError("profile prior standard deviations must be positive")
        bias_bound = float(acceleration_bias_bound)
        if not np.isfinite(bias_bound) or bias_bound <= 0.0:
            raise ValueError("acceleration_bias_bound must be positive")
        self._profile_lower = np.asarray(
            [gain_lower, -bias_bound, -bias_bound], dtype=float
        )
        self._profile_upper = np.asarray(
            [gain_upper, bias_bound, bias_bound], dtype=float
        )
        self._profile_iterations = max(2, int(profile_iterations))
        self._dynamics_prediction = (
            dynamics_prediction
            if dynamics_prediction is not None
            else self._predict_dynamics_grid
        )

        maximum_windows = int(math.ceil(self._horizon / self._block_dt)) + 2
        self._history: deque[Dict[str, Any]] = deque(maxlen=maximum_windows)
        self._raw_block: deque[Dict[str, Any]] = deque(maxlen=4096)
        self._next_block_time: Optional[float] = None
        self._accepted_windows = 0
        self._rejected_windows = 0
        self._informative_windows = 0
        self._posterior_updates = 0
        self._dynamics_active = False
        self._profile_parameters = np.asarray([1.0, 0.0, 0.0], dtype=float)
        self._profile_bound_hits = 0
        self._feature_envelope_excursions = 0
        self._kinematic_excitation_rejections = 0
        self._last_informative_time: Optional[float] = None

    @staticmethod
    def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
        shifted = np.asarray(log_weights, dtype=float) - float(np.max(log_weights))
        weights = np.exp(shifted)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(len(weights), 1.0 / len(weights))
        return weights / total

    @staticmethod
    def _finite_median(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=float)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("median inputs must be non-empty and finite")
        return float(np.median(array))

    @classmethod
    def _median_sample(cls, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        scalar_keys = (
            "kappa", "alpha_f", "alpha_r", "u", "Fz_f", "Fz_r",
            "sr", "sr_r", "ax", "ay", "omega", "v", "delta",
        )
        output: Dict[str, Any] = {
            key: cls._finite_median([sample[key] for sample in samples])
            for key in scalar_keys
        }
        output["wheel_omegas"] = tuple(
            cls._finite_median([
                sample["wheel_omegas"][index] for sample in samples
            ])
            for index in range(4)
        )
        return output

    def _prediction_inputs(
        self, sample: Dict[str, Any]
    ) -> Tuple[Optional[Tuple[float, float, float, float]], float, float, float]:
        if self._slip_mode == "wheel":
            wheel_omegas = tuple(float(value) for value in sample["wheel_omegas"])
            kappa = float(sample["kappa"])
        elif self._slip_mode == "average":
            wheel_omegas = None
            kappa = float(np.clip(sample["kappa"], -0.8, 0.8))
        else:
            wheel_omegas = None
            kappa = self._fixed_kappa

        if self._rate_mode == "signed":
            # _4wheel_body_wrench forms tire-frame alpha = atan(v/u)-delta,
            # the negative of the controller's positive-left slip convention.
            rate_f = -float(sample["sr"])
            rate_r = -float(sample["sr_r"])
        elif self._rate_mode == "legacy":
            rate_f = float(sample["sr"])
            rate_r = float(sample["sr_r"])
        else:
            rate_f = 0.0
            rate_r = 0.0
        return wheel_omegas, kappa, rate_f, rate_r

    def _inside_rig_feature_envelope(self, sample: Dict[str, Any]) -> bool:
        """Audit the exact projected tire inputs against rig collection bounds."""

        u = float(sample["u"])
        v = float(sample["v"])
        omega = float(sample["omega"])
        delta = float(sample["delta"])
        half_track = 0.5 * self._vehicle.track
        u_left = u - half_track * omega
        u_right = u + half_track * omega
        v_front = v + self._vehicle.Lf * omega
        v_rear = v - self._vehicle.Lr * omega
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
        speeds = [max(abs(longitudinal), 0.5)
                  for longitudinal, _lateral in wheel_velocities]
        alphas = [math.atan2(lateral, speed)
                  for speed, (_longitudinal, lateral)
                  in zip(speeds, wheel_velocities)]

        wheel_omegas, shared_kappa, rate_f, rate_r = self._prediction_inputs(sample)
        if wheel_omegas is None:
            kappas = (shared_kappa,) * 4
        else:
            kappas = tuple(
                float(np.clip(
                    (abs(wheel_rate) * 0.47 - abs(longitudinal))
                    / max(abs(longitudinal), 0.5),
                    -0.8,
                    0.8,
                ))
                for wheel_rate, (longitudinal, _lateral)
                in zip(wheel_omegas, wheel_velocities)
            )

        wheelbase = self._vehicle.Lf + self._vehicle.Lr
        front_mass = self._vehicle.m * self._vehicle.Lr / wheelbase
        rear_mass = self._vehicle.m * self._vehicle.Lf / wheelbase
        lateral_front = (
            front_mass * float(sample["ay"]) * self._vehicle.h_cg
            / self._vehicle.track
        )
        lateral_rear = (
            rear_mass * float(sample["ay"]) * self._vehicle.h_cg
            / self._vehicle.track
        )
        loads = (
            float(sample["Fz_f"]) - lateral_front,
            float(sample["Fz_f"]) + lateral_front,
            float(sample["Fz_r"]) - lateral_rear,
            float(sample["Fz_r"]) + lateral_rear,
        )
        rates = (rate_f, rate_f, rate_r, rate_r)

        # These limits are the controlled static-rig LHS bounds in
        # data_collection/collect_static_data.cpp.  Do not silently clip an
        # observation and then call the extrapolated force terrain evidence.
        return bool(
            all(2.0 <= value <= 10.0 for value in speeds)
            and all(abs(value) <= 0.6 for value in alphas)
            and all(2_500.0 <= value <= 7_500.0 for value in loads)
            and all(abs(value) <= 1.0 for value in kappas)
            and all(abs(value) <= 0.56 for value in rates)
        )

    def _predict_dynamics_grid(
        self, sample: Dict[str, Any], n_grid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predicted body specific force and yaw acceleration."""

        state = np.asarray([
            0.0, 0.0, 0.0, sample["u"], sample["v"], sample["omega"], 0.0,
        ], dtype=float)
        wheel_omegas, kappa, rate_f, rate_r = self._prediction_inputs(sample)
        ax = np.empty(len(n_grid), dtype=float)
        ay = np.empty(len(n_grid), dtype=float)
        yaw_accel = np.empty(len(n_grid), dtype=float)
        for index, n_value in enumerate(n_grid):
            state[6] = float(n_value)
            force_x, force_y, moment = self._fourwheel_wrench(
                state,
                float(sample["delta"]),
                float(sample["ax"]),
                self._soil_from_n(float(n_value)),
                self._vehicle,
                ay_in=float(sample["ay"]),
                kappa_in=kappa,
                steering_rate_in=rate_f,
                rear_steering_rate_in=rate_r,
                wheel_omegas=wheel_omegas,
                Fz_front_mean=float(sample["Fz_f"]),
                Fz_rear_mean=float(sample["Fz_r"]),
            )
            ax[index] = force_x / self._vehicle.m
            ay[index] = force_y / self._vehicle.m
            yaw_accel[index] = moment / self._vehicle.Iz
        return ax, ay, yaw_accel

    def _finish_block(self) -> bool:
        samples = list(self._raw_block)
        self._raw_block.clear()
        if len(samples) < self._min_window_samples:
            self._rejected_windows += 1
            return False
        times = np.asarray([sample["time"] for sample in samples], dtype=float)
        minimum_span = 0.60 * self._block_dt
        if (
            not np.isfinite(times).all()
            or float(np.ptp(times)) + 1.0e-9 < minimum_span
        ):
            self._rejected_windows += 1
            return False
        median = self._median_sample(samples)
        if (
            median["u"] < self._min_model_speed
            or max(abs(median["alpha_f"]), abs(median["alpha_r"]))
            > self._max_abs_alpha
        ):
            self._rejected_windows += 1
            return False
        yaw_rate_rms = float(math.sqrt(np.mean([
            float(sample["omega"]) ** 2 for sample in samples
        ])))
        if yaw_rate_rms + 1.0e-12 < self._min_yaw_rate_rms:
            # Across-terrain force separation alone is not proof that the
            # vehicle is exciting the lateral dynamics.  In near-straight
            # motion the rig-to-vehicle transfer discrepancy can still make
            # the profile likelihood look sharp and pull n back toward its
            # prior.  A conventional yaw-rate gyro supplies an independent,
            # causal observability gate; rejected blocks leave the last
            # terrain estimate unchanged.
            self._kinematic_excitation_rejections += 1
            self._rejected_windows += 1
            return False
        if not self._inside_rig_feature_envelope(median):
            self._feature_envelope_excursions += 1
            if self._enforce_feature_envelope:
                self._rejected_windows += 1
                return False

        # The gyro slope is deliberately not an observation.  The current
        # four-wheel projection overstates yaw moment by roughly a factor of
        # two on development traces; allowing that structural mismatch to
        # vote on terrain made the filter worse than the static prior.
        try:
            predicted_ax, predicted_ay, predicted_yaw = self._dynamics_prediction(
                median, self._grid
            )
        except Exception as error:  # pragma: no cover - runtime guard
            self._rejected_windows += 1
            if self._verbose:
                print(f"  [scalar_parent] projection failed: {error!r}")
            return False
        predictions = tuple(
            np.asarray(values, dtype=float).reshape(-1)
            for values in (predicted_ax, predicted_ay, predicted_yaw)
        )
        if any(
            len(values) != len(self._grid) or not np.isfinite(values).all()
            for values in predictions
        ):
            self._rejected_windows += 1
            return False

        normalized_spread = max(
            float(np.ptp(predictions[0])) / self._r_ax,
            float(np.ptp(predictions[1])) / self._r_ay,
        )
        if normalized_spread < self._min_information:
            self._rejected_windows += 1
            return False
        self._history.append({
            "time": float(np.median(times)),
            "measurement": np.asarray([median["ax"], median["ay"]]),
            "prediction": np.column_stack(predictions[:2]),
        })
        self._accepted_windows += 1
        self._last_informative_time = float(np.median(times))
        return True

    def _expire_history(self, current_time: float) -> None:
        cutoff = float(current_time) - self._horizon
        while self._history and self._history[0]["time"] <= cutoff + 1.0e-9:
            self._history.popleft()
        if len(self._history) < self._min_windows:
            self._dynamics_active = False
            self._informative_windows = len(self._history)

    @staticmethod
    def _bounded_quadratic_solution(
        hessian: np.ndarray,
        rhs: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        """Solve a positive-definite three-variable box quadratic exactly.

        The unconstrained solution is the overwhelmingly common path.  When a
        bound is active, enumerating the 3^3 active sets is deterministic,
        dependency-free, and cheaper here than a general optimizer.
        """

        candidate = np.linalg.solve(hessian, rhs)
        if np.all(candidate >= lower) and np.all(candidate <= upper):
            return candidate

        best = None
        best_value = math.inf
        # -1: lower bound, 0: free, +1: upper bound.
        for code in np.ndindex(3, 3, 3):
            state = np.asarray(code, dtype=int) - 1
            fixed = np.flatnonzero(state != 0)
            free = np.flatnonzero(state == 0)
            trial = np.empty(3, dtype=float)
            if fixed.size:
                trial[fixed] = np.where(
                    state[fixed] < 0, lower[fixed], upper[fixed]
                )
            if free.size:
                conditional = rhs[free].copy()
                if fixed.size:
                    conditional -= hessian[np.ix_(free, fixed)] @ trial[fixed]
                try:
                    trial[free] = np.linalg.solve(
                        hessian[np.ix_(free, free)], conditional
                    )
                except np.linalg.LinAlgError:
                    continue
                if np.any(trial[free] < lower[free]) or np.any(
                    trial[free] > upper[free]
                ):
                    continue
            value = 0.5 * float(trial @ hessian @ trial) - float(rhs @ trial)
            if value < best_value:
                best_value = value
                best = trial.copy()
        if best is None:  # pragma: no cover - positive priors make this impossible
            return np.clip(candidate, lower, upper)
        return best

    def _profile_cost(
        self,
        measurements: np.ndarray,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return robust profile costs and nuisance MAPs for every n grid point."""

        scales = np.asarray([self._r_ax, self._r_ay], dtype=float)
        prior_mean = np.asarray([1.0, 0.0, 0.0], dtype=float)
        prior_precision = 1.0 / self._profile_prior_std**2
        costs = np.empty(len(self._grid), dtype=float)
        parameters = np.empty((len(self._grid), 3), dtype=float)
        for index in range(len(self._grid)):
            prediction = predictions[:, index, :]
            design = np.empty((2 * len(measurements), 3), dtype=float)
            target = measurements.reshape(-1)
            design[0::2] = np.column_stack((
                prediction[:, 0], np.ones(len(measurements)),
                np.zeros(len(measurements)),
            ))
            design[1::2] = np.column_stack((
                prediction[:, 1], np.zeros(len(measurements)),
                np.ones(len(measurements)),
            ))
            row_scale = np.tile(scales, len(measurements))
            normalized_design = design / row_scale[:, None]
            normalized_target = target / row_scale
            weights = np.ones(len(target), dtype=float)
            beta = prior_mean.copy()
            for _iteration in range(self._profile_iterations):
                hessian = (
                    normalized_design.T
                    @ (weights[:, None] * normalized_design)
                    + np.diag(prior_precision)
                )
                rhs = (
                    normalized_design.T @ (weights * normalized_target)
                    + prior_precision * prior_mean
                )
                beta = self._bounded_quadratic_solution(
                    hessian, rhs, self._profile_lower, self._profile_upper
                )
                residual = normalized_target - normalized_design @ beta
                weights = (self._student_dof + 1.0) / (
                    self._student_dof + residual**2
                )
            residual = normalized_target - normalized_design @ beta
            costs[index] = (
                (self._student_dof + 1.0)
                * float(np.sum(np.log1p(residual**2 / self._student_dof)))
                + float(np.sum(
                    ((beta - prior_mean) / self._profile_prior_std) ** 2
                ))
            )
            parameters[index] = beta
        return costs, parameters

    def _quadratic_profile_map(self, costs: np.ndarray) -> Tuple[float, int]:
        index = int(np.argmin(costs))
        estimate = float(self._grid[index])
        if 0 < index < len(self._grid) - 1:
            denominator = float(
                costs[index - 1] - 2.0 * costs[index] + costs[index + 1]
            )
            if denominator > 1.0e-12:
                offset = 0.5 * float(
                    (costs[index - 1] - costs[index + 1]) / denominator
                )
                offset = float(np.clip(offset, -1.0, 1.0))
                estimate += offset * float(self._grid[index + 1] - self._grid[index])
        return float(np.clip(estimate, self._n_lo, self._n_hi)), index

    def _update_posterior(self) -> bool:
        if (
            len(self._history) < self._min_windows
        ):
            self._dynamics_active = False
            return False
        measurements = np.stack([
            entry["measurement"] for entry in self._history
        ])
        predictions = np.stack([
            entry["prediction"] for entry in self._history
        ])
        costs, parameters = self._profile_cost(measurements, predictions)
        if not np.isfinite(costs).all() or not np.isfinite(parameters).all():
            self._dynamics_active = False
            return False
        target, map_index = self._quadratic_profile_map(costs)
        self._posterior = self._normalize_log_weights(
            -0.5 * (costs - float(np.min(costs)))
        )
        self._n_posterior = float(np.sum(self._grid * self._posterior))
        self._profile_parameters = parameters[map_index].copy()
        tolerance = 1.0e-8
        self._profile_bound_hits = int(np.count_nonzero(
            (self._profile_parameters <= self._profile_lower + tolerance)
            | (self._profile_parameters >= self._profile_upper - tolerance)
        ))
        self._n_smooth += self._smoothing_alpha * (target - self._n_smooth)
        self._informative_windows = len(self._history)
        self._posterior_updates += 1
        self._observation_count += 1
        self._dynamics_active = True
        return True

    def estimate_omega_dot(self, omega: float, timestamp: float) -> float:
        del omega, timestamp
        return 0.0

    def observe(
        self,
        kappa: float,
        alpha_f: float,
        alpha_r: float,
        u: float,
        Fz_f: float,
        Fz_r: float,
        sr: float,
        ay_imu: float,
        omega_dot: float,
        *,
        sim_time: float = 0.0,
        omega: float = 0.0,
        v_lateral: float = 0.0,
        ax_imu: float = 0.0,
        steering_angle: Optional[float] = None,
        wheel_omegas: Optional[Sequence[float]] = None,
        alpha_rate_r: float = 0.0,
        **_ignored,
    ) -> bool:
        """Accept conventional state/sensor channels; ignore all extra fields."""

        del omega_dot
        timestamp = float(sim_time)
        speed_values: Sequence[float] = () if wheel_omegas is None else wheel_omegas
        speeds = tuple(float(value) for value in speed_values)
        if len(speeds) != 4:
            return False
        u_safe = max(abs(float(u)), 0.5)
        delta = (
            float(steering_angle)
            if steering_angle is not None
            else float(alpha_f) + math.atan2(
                float(v_lateral) + self._vehicle.Lf * float(omega), u_safe
            )
        )
        sample: Dict[str, Any] = {
            "time": timestamp,
            "kappa": float(kappa),
            "alpha_f": float(alpha_f),
            "alpha_r": float(alpha_r),
            "u": u_safe,
            "Fz_f": float(Fz_f),
            "Fz_r": float(Fz_r),
            "sr": float(sr),
            "sr_r": float(alpha_rate_r),
            "ax": float(ax_imu),
            "ay": float(ay_imu),
            "omega": float(omega),
            "v": float(v_lateral),
            "delta": delta,
            "wheel_omegas": speeds,
        }
        scalars = [value for value in sample.values() if not isinstance(value, tuple)]
        if (
            not np.isfinite(np.asarray(scalars, dtype=float)).all()
            or not np.isfinite(np.asarray(speeds, dtype=float)).all()
        ):
            return False

        if self._next_block_time is None:
            self._next_block_time = (
                math.floor(timestamp / self._block_dt) + 1
            ) * self._block_dt
        evidence_advanced = False
        while timestamp + 1.0e-9 >= self._next_block_time:
            evidence_advanced = self._finish_block() or evidence_advanced
            self._expire_history(self._next_block_time)
            self._next_block_time += self._block_dt
        self._raw_block.append(sample)
        if evidence_advanced:
            self._update_posterior()
        return True

    def should_update(self) -> bool:
        return self._posterior_updates >= self._update_interval

    def estimate(self) -> Tuple[Dict[str, float], float]:
        self._posterior_updates = 0
        self._observation_count = 0
        self._estimated_params = terrain_params_for_n(self._n_smooth)
        variance = self.get_n_uncertainty() ** 2
        self._confidence = float(1.0 / (1.0 + 50.0 * variance))
        return dict(self._estimated_params), self._confidence

    def get_n_uncertainty(self) -> float:
        """Return profile-weight standard deviation about the weighted mean."""

        variance = float(np.sum(
            self._posterior * (self._grid - self._n_posterior) ** 2
        ))
        return float(math.sqrt(max(variance, 0.0)))

    def get_bekker_n(self) -> float:
        return float(self._n_smooth)

    def get_terrain_mpc_params(self) -> Dict[str, float]:
        return dict(self._estimated_params)

    def get_friction_angle_deg(self) -> float:
        return float(self._estimated_params["phi"])

    @property
    def mu_estimate(self) -> float:
        return float(math.tan(math.radians(self.get_friction_angle_deg())))

    def get_phi_uncertainty_deg(self) -> float:
        """Push the discrete n weights through the piecewise soil manifold."""

        phi_grid = np.asarray([
            terrain_params_for_n(float(n_value))["phi"]
            for n_value in self._grid
        ], dtype=float)
        phi_mean = float(np.sum(self._posterior * phi_grid))
        variance = float(np.sum(self._posterior * (phi_grid - phi_mean) ** 2))
        return float(math.sqrt(max(variance, 0.0)))

    @property
    def informative_segments(self) -> int:
        return int(self._informative_windows)

    @property
    def dynamics_active(self) -> bool:
        return bool(self._dynamics_active)

    @property
    def dynamics_windows(self) -> int:
        return int(len(self._history))

    @property
    def accepted_dynamics_windows(self) -> int:
        return int(self._accepted_windows)

    @property
    def rejected_dynamics_windows(self) -> int:
        return int(self._rejected_windows)

    @property
    def profile_force_gain(self) -> float:
        return float(self._profile_parameters[0])

    @property
    def profile_ax_bias(self) -> float:
        return float(self._profile_parameters[1])

    @property
    def profile_ay_bias(self) -> float:
        return float(self._profile_parameters[2])

    @property
    def profile_bound_hits(self) -> int:
        return int(self._profile_bound_hits)

    @property
    def feature_envelope_excursions(self) -> int:
        return int(self._feature_envelope_excursions)

    @property
    def kinematic_excitation_rejections(self) -> int:
        """Blocks rejected because measured yaw excitation was too small."""

        return int(self._kinematic_excitation_rejections)

    @property
    def last_informative_time(self) -> float:
        """Timestamp of the latest accepted evidence block, or NaN."""

        if self._last_informative_time is None:
            return float("nan")
        return float(self._last_informative_time)


# Compatibility alias for the already-exposed CLI backend key.  New code and
# paper prose should use the profile-likelihood name above.
EstimatorBayesTerrainEstimator = ScalarParentTerrainEstimator
