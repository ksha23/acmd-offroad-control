"""Joint ground-datum-free terrain profile estimator.

The estimator evaluates one robust likelihood on a rolling causal horizon over
an independently gridded Bekker exponent ``n`` and friction angle ``phi``.
Every accepted IMU block contributes exactly one longitudinal and one lateral
specific-force residual to that likelihood.  A single bounded force-transfer
gain and one bias per IMU axis are shared by all blocks in the horizon and are
profiled as nuisance variables for each terrain hypothesis.

This class neither runs a manifold ``n`` estimator first nor uses its
posterior as a prior.  The default ``n`` and
``phi`` priors are independent uniforms over the controlled-rig envelope.  An
optional bounded cohesion multiplier is integrated into the same joint
profile; it is not a second likelihood or a terrain label.

Only conventional runtime state, command, wheel-speed, and levelled IMU
channels accepted by :class:`ScalarParentTerrainEstimator` are used.
Wheel height, a ground datum, plant terrain truth, and tire/contact-force truth
are not constructor inputs or observations. The only learned component is the
controlled single-tire rig force surrogate; the promoted configuration uses
the rate-augmented rig checkpoint.
"""

from __future__ import annotations

import math
import time
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__:
    from .scalar_parent_terrain_estimator import (
        ScalarParentTerrainEstimator,
    )
    from .terrain_parameterization import terrain_params_for_n
else:
    # The decoupled controller is launched as a script and imports estimator
    # modules from ``simulation/estimators`` as flat modules.
    from scalar_parent_terrain_estimator import (  # type: ignore
        ScalarParentTerrainEstimator,
    )
    from terrain_parameterization import terrain_params_for_n  # type: ignore


JointPrediction = Callable[
    [Dict[str, Any], np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]


class GritTerrainEstimator(
    ScalarParentTerrainEstimator
):
    """Single-likelihood rolling profile over independent ``(n, phi)``."""

    _FORBIDDEN_CONSTRUCTOR_TOKENS = (
        "truth",
        "oracle",
        "ground",
        "datum",
        "height",
        "sinkage",
        "tire_force",
        "tyre_force",
        "contact_force",
        "soil_label",
    )

    def __init__(
        self,
        *args: Any,
        phi_grid_size: int = 17,
        phi_bounds_deg: Tuple[float, float] = (6.0, 37.8),
        cohesion_multiplier_bounds: Tuple[float, float] = (0.7, 1.3),
        cohesion_grid_size: int = 1,
        cohesion_prior_std: float = 0.20,
        load_transfer_mode: str = "static",
        min_joint_information: float = 0.20,
        min_n_information: float = 0.0,
        min_phi_information: float = 0.0,
        min_observability_rank: int = 2,
        min_observability_singular_value: float = 0.10,
        boundary_warning_mass: float = 0.25,
        posterior_summary: str = "mean",
        block_alpha_rate: bool = False,
        n_bounds: Optional[Tuple[float, float]] = None,
        manifold_soft_floor: Optional[float] = None,
        manifold_soft_mode: str = "hold",
        joint_prediction: Optional[JointPrediction] = None,
        **kwargs: Any,
    ) -> None:
        forbidden = sorted(
            str(key)
            for key in kwargs
            if any(
                token in str(key).strip().lower()
                for token in self._FORBIDDEN_CONSTRUCTOR_TOKENS
            )
        )
        if forbidden:
            raise ValueError(
                "joint estimator constructor contains forbidden inputs: "
                + ", ".join(forbidden)
            )

        initial_terrain = kwargs.get("initial_terrain")
        if initial_terrain is None and len(args) >= 2:
            initial_terrain = args[1]
        initial_mapping = dict(initial_terrain or {})
        super().__init__(*args, **kwargs)

        # Optional edge extension. The default grid spans the preset anchors,
        # which puts the deployed clay preset exactly on the lower boundary; a
        # bounded posterior mean can never report its own edge, so the softest
        # soil is structurally the least estimable. Widening the hypothesis
        # grid below clay (with the manifold's soft extrapolation) makes clay
        # an interior point. Defaults preserve the published behaviour.
        self._manifold_soft_floor = (
            float(manifold_soft_floor) if manifold_soft_floor is not None else None
        )
        self._manifold_soft_mode = str(manifold_soft_mode)
        if n_bounds is not None:
            lo, hi = map(float, n_bounds)
            if not (np.isfinite([lo, hi]).all() and lo < hi):
                raise ValueError("n_bounds must be finite with lo < hi")
            if lo < float(self._n_lo) and self._manifold_soft_floor is None:
                raise ValueError(
                    "extending the grid below the preset anchors requires "
                    "manifold_soft_floor, else every sub-clay node clips to "
                    "identical clay parameters and the likelihood degenerates"
                )
            self._n_lo, self._n_hi = lo, hi
            self._grid = np.linspace(lo, hi, len(self._grid))

        phi_count = max(5, int(phi_grid_size))
        if phi_count % 2 == 0:
            phi_count += 1
        phi_lower, phi_upper = map(float, phi_bounds_deg)
        if (
            not np.isfinite([phi_lower, phi_upper]).all()
            or phi_lower < 6.0 - 1.0e-12
            or phi_upper > 37.8 + 1.0e-12
            or phi_lower >= phi_upper
        ):
            raise ValueError(
                "phi_bounds_deg must lie inside the controlled-rig envelope "
                "[6, 37.8]"
            )
        self._phi_grid = np.linspace(phi_lower, phi_upper, phi_count)

        cohesion_lower, cohesion_upper = map(
            float, cohesion_multiplier_bounds
        )
        if (
            not np.isfinite([cohesion_lower, cohesion_upper]).all()
            or cohesion_lower <= 0.0
            or cohesion_lower > 1.0
            or cohesion_upper < 1.0
            or cohesion_lower >= cohesion_upper
        ):
            raise ValueError(
                "cohesion multiplier bounds must be positive, ordered, and "
                "contain one"
            )
        cohesion_count = int(cohesion_grid_size)
        if cohesion_count < 1:
            raise ValueError("cohesion_grid_size must be positive")
        if cohesion_count == 1:
            self._cohesion_grid = np.asarray([1.0], dtype=float)
        else:
            cohesion_count = max(3, cohesion_count)
            if cohesion_count % 2 == 0:
                cohesion_count += 1
            half = cohesion_count // 2
            self._cohesion_grid = np.concatenate(
                (
                    np.linspace(cohesion_lower, 1.0, half + 1)[:-1],
                    np.linspace(1.0, cohesion_upper, half + 1),
                )
            )
        if not np.isfinite(cohesion_prior_std) or cohesion_prior_std <= 0.0:
            raise ValueError("cohesion_prior_std must be positive")
        cohesion_log_prior = -0.5 * (
            (self._cohesion_grid - 1.0) / float(cohesion_prior_std)
        ) ** 2
        cohesion_prior = self._normalize_log_weights(cohesion_log_prior)

        load_mode = str(load_transfer_mode).strip().lower()
        if load_mode not in {"measured", "static", "lagged"}:
            raise ValueError(
                "load_transfer_mode must be measured, static, or lagged"
            )
        information_gates = {
            "min_joint_information": min_joint_information,
            "min_n_information": min_n_information,
            "min_phi_information": min_phi_information,
        }
        for name, value in information_gates.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if int(min_observability_rank) not in {0, 1, 2, 3}:
            raise ValueError("min_observability_rank must lie in [0, 3]")
        if (
            not np.isfinite(min_observability_singular_value)
            or min_observability_singular_value < 0.0
        ):
            raise ValueError(
                "min_observability_singular_value must be non-negative"
            )
        if (
            not np.isfinite(boundary_warning_mass)
            or not 0.0 <= boundary_warning_mass <= 1.0
        ):
            raise ValueError("boundary_warning_mass must lie in [0, 1]")
        summary = str(posterior_summary).strip().lower()
        if summary not in {"map", "mean"}:
            raise ValueError("posterior_summary must be map or mean")

        # Independent broad priors are the defining design choice.  In
        # particular, phi is never centered on terrain_params_for_n(n)["phi"].
        n_prior = np.full(len(self._grid), 1.0 / len(self._grid), dtype=float)
        phi_prior = np.full(
            len(self._phi_grid), 1.0 / len(self._phi_grid), dtype=float
        )
        self._joint_prior = (
            n_prior[:, None, None]
            * phi_prior[None, :, None]
            * cohesion_prior[None, None, :]
        )
        self._joint_prior = self._normalize(self._joint_prior)
        self._joint_posterior = self._joint_prior.copy()
        self._posterior = np.sum(self._joint_posterior, axis=(1, 2))
        self._n_posterior = float(np.sum(self._grid * self._posterior))
        self._phi_posterior = np.sum(self._joint_posterior, axis=(0, 2))
        self._cohesion_posterior = np.sum(
            self._joint_posterior, axis=(0, 1)
        )
        self._n_prior = n_prior
        self._phi_prior = phi_prior
        self._cohesion_prior = cohesion_prior

        self._load_transfer_mode = load_mode
        self._min_joint_information = float(min_joint_information)
        self._min_n_information = float(min_n_information)
        self._min_phi_information = float(min_phi_information)
        self._min_observability_rank = int(min_observability_rank)
        self._min_observability_singular_value = float(
            min_observability_singular_value
        )
        self._boundary_warning_mass = float(boundary_warning_mass)
        self._posterior_summary = summary
        self._block_alpha_rate = bool(block_alpha_rate)
        self._joint_prediction = (
            joint_prediction
            if joint_prediction is not None
            else self._predict_joint_grid
        )

        initial_phi = float(
            initial_mapping.get(
                "phi", terrain_params_for_n(
                    self._n_smooth, soft_floor=self._manifold_soft_floor,
                    soft_mode=self._manifold_soft_mode,
                )["phi"]
            )
        )
        self._phi_smooth = float(
            np.clip(initial_phi, self._phi_grid[0], self._phi_grid[-1])
        )
        self._cohesion_output = 1.0
        self._joint_active = False
        self._joint_has_estimate = False
        self._joint_updates = 0
        self._joint_projection_failures = 0
        self._joint_information_kl = 0.0
        self._n_information_kl = 0.0
        self._phi_information_kl = 0.0
        self._cohesion_information_kl = 0.0
        self._profile_cost_span = 0.0
        self._n_boundary_mass = float(n_prior[0] + n_prior[-1])
        self._phi_boundary_mass = float(phi_prior[0] + phi_prior[-1])
        self._cohesion_boundary_mass = (
            0.0
            if len(cohesion_prior) == 1
            else float(cohesion_prior[0] + cohesion_prior[-1])
        )
        self._n_map_at_boundary = False
        self._phi_map_at_boundary = False
        self._cohesion_map_at_boundary = False
        self._boundary_limited = False
        self._observability_rank = 0
        self._observability_min_singular_value = 0.0
        self._observability_max_singular_value = 0.0
        self._observability_condition = math.inf
        self._nuisance_projection_rank = 0
        self._nuisance_projection_condition = math.inf
        self._last_likelihood_block_count = 0
        self._last_likelihood_residual_count = 0
        # Raw-unit likelihood residuals at the cost-minimizing hypothesis,
        # accumulated so the measurement-noise scales can be derived from data
        # instead of asserted. Truth-free: sensor-vs-prediction only.
        self._residual_ax_abs: list = []
        self._residual_ay_abs: list = []
        self._duplicate_likelihood_block_count = 0
        self._duplicate_likelihood_update_count = 0
        self._likelihood_evaluations = 0
        self._joint_information_rejections = 0
        self._n_information_rejections = 0
        self._phi_information_rejections = 0
        self._observability_rejections = 0
        self._gate_rejected_updates = 0
        self._last_profile_block_times: Tuple[float, ...] = ()
        self._last_joint_update_time = float("nan")
        self._lagged_load_state: Optional[Tuple[float, float, float, float]] = None
        self._measured_load_blocks = 0
        self._static_load_blocks = 0
        self._lagged_load_blocks = 0
        self._lagged_load_fallback_blocks = 0
        self._last_effective_front_load = float("nan")
        self._last_effective_rear_load = float("nan")
        self._last_effective_load_ax = float("nan")
        self._last_effective_load_ay = float("nan")
        self._last_projection_wall_time_s = 0.0
        self._last_profile_wall_time_s = 0.0
        self._last_observability_wall_time_s = 0.0
        self._last_posterior_wall_time_s = 0.0
        self._snapshot_update_seq = 0
        self._last_accepted_snapshot: Optional[Mapping[str, Any]] = None
        self.output_names = (
            ("n", "phi", "cohesion_multiplier")
            if len(self._cohesion_grid) > 1
            else ("n", "phi")
        )

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        weights = np.asarray(values, dtype=float)
        total = float(np.sum(weights))
        if (
            weights.size == 0
            or not np.isfinite(weights).all()
            or not np.isfinite(total)
            or total <= 0.0
        ):
            return np.full(weights.shape, 1.0 / weights.size)
        return weights / total

    @classmethod
    def _discrete_kl(cls, posterior: np.ndarray, prior: np.ndarray) -> float:
        p = cls._normalize(posterior)
        q = cls._normalize(prior)
        positive = p > 0.0
        tiny = np.finfo(float).tiny
        value = float(
            np.sum(p[positive] * (np.log(p[positive]) - np.log(
                np.maximum(q[positive], tiny)
            )))
        )
        return float(max(value, 0.0))

    @classmethod
    def _weighted_std(cls, grid: np.ndarray, weights: np.ndarray) -> float:
        probability = cls._normalize(weights)
        values = np.asarray(grid, dtype=float)
        mean = float(np.sum(probability * values))
        variance = float(np.sum(probability * (values - mean) ** 2))
        return float(math.sqrt(max(variance, 0.0)))

    def _static_per_wheel_loads(self) -> Tuple[float, float]:
        """Return level static front/rear loads for one wheel per axle."""

        wheelbase = self._vehicle.Lf + self._vehicle.Lr
        front = (
            self._vehicle.m * self._vehicle.g * self._vehicle.Lr
            / (2.0 * wheelbase)
        )
        rear = (
            self._vehicle.m * self._vehicle.g * self._vehicle.Lf
            / (2.0 * wheelbase)
        )
        return float(front), float(rear)

    def _prepare_projection_sample(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Freeze the causal load inputs used by one tire-grid projection.

        ``measured`` preserves the original reconstructed axle means and
        current lateral transfer.  ``static`` removes both longitudinal and
        lateral measured-acceleration feedback.  ``lagged`` uses the previous
        accepted block's reconstruction and falls back to static loads for its
        first accepted block.  The returned copy is the only sample seen by
        the force projection and feature-envelope audit.
        """

        prepared = dict(sample)
        if self._load_transfer_mode == "measured":
            front = float(sample["Fz_f"])
            rear = float(sample["Fz_r"])
            ax_load = float(sample["ax"])
            ay_load = float(sample["ay"])
            source = "measured"
        elif self._load_transfer_mode == "static":
            front, rear = self._static_per_wheel_loads()
            ax_load = 0.0
            ay_load = 0.0
            source = "static"
        elif self._lagged_load_state is None:
            front, rear = self._static_per_wheel_loads()
            ax_load = 0.0
            ay_load = 0.0
            source = "lagged_static_fallback"
        else:
            front, rear, ax_load, ay_load = self._lagged_load_state
            source = "lagged"
        prepared["Fz_f"] = front
        prepared["Fz_r"] = rear
        prepared["ax"] = ax_load
        prepared["ay"] = ay_load
        prepared["_joint_load_source"] = source
        prepared["_joint_load_front_mean"] = front
        prepared["_joint_load_rear_mean"] = rear
        prepared["_joint_load_ax"] = ax_load
        prepared["_joint_load_ay"] = ay_load
        return prepared

    def _commit_projection_loads(
        self,
        prepared: Dict[str, Any],
        measured: Dict[str, Any],
    ) -> None:
        """Record one accepted block and advance the causal lag state."""

        source = str(prepared["_joint_load_source"])
        if source == "measured":
            self._measured_load_blocks += 1
        elif source == "static":
            self._static_load_blocks += 1
        elif source == "lagged":
            self._lagged_load_blocks += 1
        elif source == "lagged_static_fallback":
            self._lagged_load_fallback_blocks += 1
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"unknown joint load source {source!r}")
        self._last_effective_front_load = float(prepared["Fz_f"])
        self._last_effective_rear_load = float(prepared["Fz_r"])
        self._last_effective_load_ax = float(prepared["ax"])
        self._last_effective_load_ay = float(prepared["ay"])
        if self._load_transfer_mode == "lagged":
            self._lagged_load_state = (
                float(measured["Fz_f"]),
                float(measured["Fz_r"]),
                float(measured["ax"]),
                float(measured["ay"]),
            )

    def _predict_joint_grid(
        self,
        sample: Dict[str, Any],
        n_grid: np.ndarray,
        phi_grid_deg: np.ndarray,
        cohesion_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project one conventional sensor block over the joint rig grid."""

        model = self._force_projector.model
        rate_augmented = bool(getattr(model, "rate_augmented", False))
        expected_width = 14 if rate_augmented else 11
        if int(getattr(model, "input_dim", expected_width)) != expected_width:
            raise ValueError(
                "joint inference requires a validated static or rate-augmented "
                "rig force map"
            )

        u = float(sample["u"])
        v = float(sample["v"])
        omega = float(sample["omega"])
        delta = float(sample["delta"])
        cosine = math.cos(delta)
        sine = math.sin(delta)
        half_track = 0.5 * self._vehicle.track
        u_left = u - half_track * omega
        u_right = u + half_track * omega
        v_front = v + self._vehicle.Lf * omega
        v_rear = v - self._vehicle.Lr * omega
        longitudinal = np.asarray(
            [
                u_left * cosine + v_front * sine,
                u_right * cosine + v_front * sine,
                u_left,
                u_right,
            ],
            dtype=float,
        )
        lateral = np.asarray(
            [
                -u_left * sine + v_front * cosine,
                -u_right * sine + v_front * cosine,
                v_rear,
                v_rear,
            ],
            dtype=float,
        )
        speeds = np.maximum(np.abs(longitudinal), 0.5)
        alphas = np.arctan2(lateral, speeds)

        wheel_omegas, shared_kappa, rate_f, rate_r = self._prediction_inputs(
            sample
        )
        if wheel_omegas is None:
            kappas = np.full(4, shared_kappa, dtype=float)
        else:
            kappas = np.asarray(
                [
                    np.clip(
                        (abs(float(wheel_rate)) * 0.47 - abs(speed))
                        / max(abs(speed), 0.5),
                        -0.8,
                        0.8,
                    )
                    for wheel_rate, speed in zip(wheel_omegas, longitudinal)
                ],
                dtype=float,
            )
        rates = np.asarray([rate_f, rate_f, rate_r, rate_r], dtype=float)

        wheelbase = self._vehicle.Lf + self._vehicle.Lr
        front_mass = self._vehicle.m * self._vehicle.Lr / wheelbase
        rear_mass = self._vehicle.m * self._vehicle.Lf / wheelbase
        front_transfer = (
            front_mass * float(sample["ay"]) * self._vehicle.h_cg
            / self._vehicle.track
        )
        rear_transfer = (
            rear_mass * float(sample["ay"]) * self._vehicle.h_cg
            / self._vehicle.track
        )
        loads = np.maximum(
            np.asarray(
                [
                    float(sample["Fz_f"]) - front_transfer,
                    float(sample["Fz_f"]) + front_transfer,
                    float(sample["Fz_r"]) - rear_transfer,
                    float(sample["Fz_r"]) + rear_transfer,
                ],
                dtype=float,
            ),
            100.0,
        )

        soil_rows = []
        for n_value in np.asarray(n_grid, dtype=float):
            terrain = terrain_params_for_n(
                float(n_value), soft_floor=self._manifold_soft_floor,
                soft_mode=self._manifold_soft_mode,
            )
            for phi_deg in np.asarray(phi_grid_deg, dtype=float):
                phi_feature = model.phi_feature_value(
                    math.radians(float(phi_deg))
                )
                for multiplier in np.asarray(cohesion_grid, dtype=float):
                    soil_rows.append(
                        (
                            terrain["Kphi"],
                            terrain["Kc"],
                            float(n_value),
                            terrain["c"] * float(multiplier),
                            phi_feature,
                            terrain["k"],
                        )
                    )
        terrain_rows = np.asarray(soil_rows, dtype=float)
        hypothesis_count = len(terrain_rows)
        features = np.empty(
            (hypothesis_count, 4, expected_width), dtype=float
        )
        features[..., 0] = kappas[None, :]
        features[..., 1] = alphas[None, :]
        features[..., 2] = speeds[None, :]
        features[..., 3] = loads[None, :]
        features[..., 4] = rates[None, :]
        if rate_augmented:
            # The additional features are ordered
            # (d_kappa/dt, d_alpha/dt, du/dt).  The trace contains a directly
            # reconstructed slip-angle rate, but differentiating wheel slip
            # or speed sample-by-sample would amplify encoder noise.  Use one
            # causal least-squares slope per accepted 0.5 s block instead.
            # ``rate_mode=signed`` puts the measured slip-angle rate in both
            # its command and actual-rate positions; ``rate_mode=zero`` keeps
            # only the longitudinal block slopes active.
            features[..., 5] = float(sample.get("dkappa", 0.0))
            features[..., 6] = rates[None, :]
            features[..., 7] = float(sample.get("du", 0.0))
            features[..., 8:] = terrain_rows[:, None, :]
        else:
            features[..., 5:] = terrain_rows[:, None, :]
        force = np.asarray(
            model.predict_feature_rows(
                features.reshape(-1, expected_width)
            ),
            dtype=float,
        ).reshape(hypothesis_count, 4, 2)
        if not np.isfinite(force).all():
            raise ValueError("rig force map returned non-finite predictions")

        force_x = force[..., 0]
        force_y = force[..., 1]
        front_x = force_x[:, :2] * cosine - force_y[:, :2] * sine
        front_y = force_x[:, :2] * sine + force_y[:, :2] * cosine
        total_x = np.sum(front_x, axis=1) + np.sum(force_x[:, 2:], axis=1)
        total_y = np.sum(front_y, axis=1) + np.sum(force_y[:, 2:], axis=1)
        shape = (len(n_grid), len(phi_grid_deg), len(cohesion_grid))
        return (
            (total_x / self._vehicle.m).reshape(shape),
            (total_y / self._vehicle.m).reshape(shape),
        )

    def _finish_block(self) -> bool:
        """Accept a block and calculate exactly one joint prediction tensor."""

        samples = list(self._raw_block)
        self._raw_block.clear()
        if len(samples) < self._min_window_samples:
            self._rejected_windows += 1
            return False
        times = np.asarray([sample["time"] for sample in samples], dtype=float)
        if (
            not np.isfinite(times).all()
            or float(np.ptp(times)) + 1.0e-9 < 0.60 * self._block_dt
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
            self._kinematic_excitation_rejections += 1
            self._rejected_windows += 1
            return False
        centered_time = times - float(np.mean(times))
        time_energy = float(centered_time @ centered_time)
        if time_energy <= 1.0e-12:
            self._rejected_windows += 1
            return False

        def block_slope(key: str) -> float:
            values = np.asarray(
                [float(sample[key]) for sample in samples], dtype=float
            )
            return float(
                centered_time @ (values - float(np.mean(values)))
                / time_energy
            )

        median["dkappa"] = block_slope("kappa")
        median["du"] = block_slope("u")
        if self._block_alpha_rate:
            median["sr"] = block_slope("alpha_f")
            median["sr_r"] = block_slope("alpha_r")
        model = self._force_projector.model
        if bool(getattr(model, "rate_augmented", False)) and (
            abs(float(median["dkappa"])) > 0.4 + 1.0e-9
            or abs(float(median["du"])) > 1.5 + 1.0e-9
        ):
            self._feature_envelope_excursions += 1
            if self._enforce_feature_envelope:
                self._rejected_windows += 1
                return False
        projection_sample = self._prepare_projection_sample(median)
        if not self._inside_rig_feature_envelope(projection_sample):
            self._feature_envelope_excursions += 1
            if self._enforce_feature_envelope:
                self._rejected_windows += 1
                return False

        projection_started = time.perf_counter()
        try:
            ax, ay = self._joint_prediction(
                projection_sample,
                self._grid,
                self._phi_grid,
                self._cohesion_grid,
            )
        except Exception as error:  # pragma: no cover - runtime guard
            self._last_projection_wall_time_s = (
                time.perf_counter() - projection_started
            )
            self._joint_projection_failures += 1
            self._rejected_windows += 1
            if self._verbose:
                print(f"  [grit] projection failed: {error!r}")
            return False
        self._last_projection_wall_time_s = (
            time.perf_counter() - projection_started
        )
        expected = (
            len(self._grid),
            len(self._phi_grid),
            len(self._cohesion_grid),
        )
        predictions = tuple(np.asarray(value, dtype=float) for value in (ax, ay))
        if any(value.shape != expected or not np.isfinite(value).all()
               for value in predictions):
            self._joint_projection_failures += 1
            self._rejected_windows += 1
            return False
        normalized_spread = max(
            float(np.ptp(predictions[0])) / self._r_ax,
            float(np.ptp(predictions[1])) / self._r_ay,
        )
        if normalized_spread < self._min_information:
            self._rejected_windows += 1
            return False

        block_time = float(np.median(times))
        self._history.append(
            {
                "time": block_time,
                "measurement": np.asarray(
                    [median["ax"], median["ay"]], dtype=float
                ),
                "prediction": np.stack(predictions, axis=-1),
            }
        )
        self._commit_projection_loads(projection_sample, median)
        self._accepted_windows += 1
        self._last_informative_time = block_time
        return True

    def _expire_history(self, current_time: float) -> None:
        cutoff = float(current_time) - self._horizon
        while self._history and self._history[0]["time"] <= cutoff + 1.0e-9:
            self._history.popleft()
        if len(self._history) < self._min_windows:
            self._joint_active = False
            self._dynamics_active = False
            self._informative_windows = len(self._history)

    def _batched_bounded_quadratic_solution(
        self,
        hessians: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """Solve every three-variable box quadratic with scalar semantics."""

        matrices = np.asarray(hessians, dtype=float)
        vectors = np.asarray(rhs, dtype=float)
        unconstrained = np.linalg.solve(
            matrices, vectors[:, :, None]
        )[:, :, 0]
        bound_guard = (
            64.0
            * np.finfo(float).eps
            * np.maximum(
                1.0,
                np.maximum(
                    np.abs(self._profile_lower),
                    np.abs(self._profile_upper),
                ),
            )
        )
        active = np.any(
            (unconstrained <= self._profile_lower + bound_guard)
            | (unconstrained >= self._profile_upper - bound_guard),
            axis=1,
        )
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            return unconstrained

        active_hessians = matrices[active_indices]
        active_rhs = vectors[active_indices]
        active_count = len(active_indices)
        best_parameters = np.empty((active_count, 3), dtype=float)
        best_values = np.full(active_count, math.inf, dtype=float)

        # Match _bounded_quadratic_solution's active-set order and strict
        # objective comparison, but evaluate each state for all affected
        # hypotheses at once.  The all-fixed states guarantee a feasible
        # candidate even in a numerically pathological system.
        for code in np.ndindex(3, 3, 3):
            state = np.asarray(code, dtype=int) - 1
            fixed = np.flatnonzero(state != 0)
            free = np.flatnonzero(state == 0)
            trial = np.empty((active_count, 3), dtype=float)
            if fixed.size:
                trial[:, fixed] = np.where(
                    state[fixed] < 0,
                    self._profile_lower[fixed],
                    self._profile_upper[fixed],
                )
            valid = np.ones(active_count, dtype=bool)
            if free.size:
                conditional = active_rhs[:, free].copy()
                if fixed.size:
                    conditional -= np.einsum(
                        "mif,mf->mi",
                        active_hessians[:, free][:, :, fixed],
                        trial[:, fixed],
                        optimize=False,
                    )
                solved = np.linalg.solve(
                    active_hessians[:, free][:, :, free],
                    conditional[:, :, None],
                )[:, :, 0]
                trial[:, free] = solved
                valid &= np.all(
                    (solved >= self._profile_lower[free])
                    & (solved <= self._profile_upper[free]),
                    axis=1,
                )
            values = (
                0.5
                * np.einsum(
                    "mi,mij,mj->m",
                    trial,
                    active_hessians,
                    trial,
                    optimize=False,
                )
                - np.einsum(
                    "mi,mi->m",
                    active_rhs,
                    trial,
                    optimize=False,
                )
            )
            better = valid & (values < best_values)
            best_parameters[better] = trial[better]
            best_values[better] = values[better]

        missing = ~np.isfinite(best_values)
        if np.any(missing):  # pragma: no cover - all-fixed states are feasible
            best_parameters[missing] = np.clip(
                unconstrained[active_indices[missing]],
                self._profile_lower,
                self._profile_upper,
            )
        parameters = unconstrained
        parameters[active_indices] = best_parameters
        return parameters

    def _profile_joint_cost(
        self, measurements: np.ndarray, predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Profile shared force gain and IMU biases for every hypothesis."""

        values = np.asarray(predictions, dtype=float)
        if (
            values.ndim != 3
            or values.shape[0] != len(measurements)
            or values.shape[2] != 2
        ):
            raise ValueError(
                "predictions must have shape (windows, hypotheses, 2)"
            )
        scales = np.asarray([self._r_ax, self._r_ay], dtype=float)
        prior_mean = np.asarray([1.0, 0.0, 0.0], dtype=float)
        prior_precision = 1.0 / self._profile_prior_std**2
        target = np.asarray(measurements, dtype=float).reshape(-1)
        row_scale = np.tile(scales, len(measurements))
        normalized_target = target / row_scale
        hypothesis_count = values.shape[1]
        residual_count = len(target)

        # The same three-variable robust regression applies to every
        # terrain hypothesis. Stack those independent systems so NumPy
        # performs each IRLS
        # iteration as one batched operation.  The design, priors, Student-t
        # reweighting, iteration count, and final cost are unchanged.
        design = np.empty(
            (hypothesis_count, residual_count, 3), dtype=float
        )
        hypothesis_values = np.transpose(values, (1, 0, 2))
        design[:, 0::2, 0] = hypothesis_values[:, :, 0]
        design[:, 1::2, 0] = hypothesis_values[:, :, 1]
        design[:, 0::2, 1] = 1.0
        design[:, 1::2, 1] = 0.0
        design[:, 0::2, 2] = 0.0
        design[:, 1::2, 2] = 1.0
        normalized_design = design / row_scale[None, :, None]
        transposed_design = np.swapaxes(normalized_design, 1, 2)
        weights = np.ones(
            (hypothesis_count, residual_count), dtype=float
        )
        regularization = np.diag(prior_precision)
        prior_rhs = prior_precision * prior_mean
        parameters = np.broadcast_to(
            prior_mean, (hypothesis_count, 3)
        ).copy()

        for _iteration in range(self._profile_iterations):
            hessians = (
                np.matmul(
                    transposed_design,
                    weights[:, :, None] * normalized_design,
                )
                + regularization[None, :, :]
            )
            rhs = (
                np.matmul(
                    transposed_design,
                    (weights * normalized_target[None, :])[:, :, None],
                )[:, :, 0]
                + prior_rhs
            )
            parameters = self._batched_bounded_quadratic_solution(
                hessians, rhs
            )
            residual = normalized_target[None, :] - np.einsum(
                "hri,hi->hr",
                normalized_design,
                parameters,
                optimize=False,
            )
            weights = (self._student_dof + 1.0) / (
                self._student_dof + residual**2
            )

        residual = normalized_target[None, :] - np.einsum(
            "hri,hi->hr",
            normalized_design,
            parameters,
            optimize=False,
        )
        costs = (
            (self._student_dof + 1.0)
            * np.sum(
                np.log1p(residual**2 / self._student_dof), axis=1
            )
            + np.sum(
                (
                    (parameters - prior_mean)
                    / self._profile_prior_std
                )
                ** 2,
                axis=1,
            )
        )
        best = int(np.argmin(costs))
        self._residual_ax_abs.extend(
            np.abs(residual[best, 0::2] * scales[0]).tolist()
        )
        self._residual_ay_abs.extend(
            np.abs(residual[best, 1::2] * scales[1]).tolist()
        )
        cap = 4096
        if len(self._residual_ax_abs) > cap:
            del self._residual_ax_abs[:-cap]
        if len(self._residual_ay_abs) > cap:
            del self._residual_ay_abs[:-cap]
        return costs, parameters

    @staticmethod
    def _axis_derivative(
        predictions: np.ndarray,
        axis: int,
        index: int,
        coordinates: np.ndarray,
        map_index: Tuple[int, int, int],
    ) -> np.ndarray:
        lower = max(0, index - 1)
        upper = min(len(coordinates) - 1, index + 1)
        if lower == upper:
            raise ValueError("cannot differentiate a singleton grid")
        low_index = list(map_index)
        high_index = list(map_index)
        low_index[axis] = lower
        high_index[axis] = upper
        coordinate_range = float(coordinates[-1] - coordinates[0])
        normalized_step = float(
            (coordinates[upper] - coordinates[lower]) / coordinate_range
        )
        return (
            predictions[(slice(None), *high_index, slice(None))]
            - predictions[(slice(None), *low_index, slice(None))]
        ) / normalized_step

    def _update_observability(
        self,
        predictions: np.ndarray,
        map_index: Tuple[int, int, int],
    ) -> None:
        """Measure local terrain sensitivity after removing nuisance tangents."""

        grids: Sequence[np.ndarray] = (
            self._grid,
            self._phi_grid,
            self._cohesion_grid,
        )
        derivatives = []
        for axis, grid in enumerate(grids):
            if len(grid) > 1:
                derivatives.append(
                    self._axis_derivative(
                        predictions, axis, map_index[axis], grid, map_index
                    )
                )
        if not derivatives:
            self._observability_rank = 0
            self._observability_min_singular_value = 0.0
            self._observability_max_singular_value = 0.0
            self._observability_condition = math.inf
            self._nuisance_projection_rank = 0
            self._nuisance_projection_condition = math.inf
            return

        scales = np.tile(
            np.asarray([self._r_ax, self._r_ay], dtype=float),
            predictions.shape[0],
        )
        jacobian = np.column_stack(
            [derivative.reshape(-1) / scales for derivative in derivatives]
        )
        map_prediction = predictions[
            (slice(None), *map_index, slice(None))
        ]
        nuisance = np.empty((2 * predictions.shape[0], 3), dtype=float)
        nuisance[:, 0] = map_prediction.reshape(-1)
        nuisance[0::2, 1] = 1.0
        nuisance[1::2, 1] = 0.0
        nuisance[0::2, 2] = 0.0
        nuisance[1::2, 2] = 1.0
        nuisance /= scales[:, None]
        # Use only the numerical column space of the nuisance design.  A plain
        # reduced QR fabricates orthogonal directions when force gain becomes
        # collinear with the two bias columns (for example under a constant
        # prediction), and can therefore erase genuine terrain sensitivity.
        nuisance_u, nuisance_singular, _ = np.linalg.svd(
            nuisance, full_matrices=False
        )
        nuisance_maximum = (
            float(np.max(nuisance_singular))
            if nuisance_singular.size
            else 0.0
        )
        nuisance_tolerance = max(
            np.finfo(float).eps
            * max(nuisance.shape)
            * nuisance_maximum,
            1.0e-10 * nuisance_maximum,
        )
        nuisance_rank = int(
            np.count_nonzero(nuisance_singular > nuisance_tolerance)
        )
        basis = nuisance_u[:, :nuisance_rank]
        residual_jacobian = (
            jacobian - basis @ (basis.T @ jacobian)
            if nuisance_rank
            else jacobian.copy()
        )
        self._nuisance_projection_rank = nuisance_rank
        if nuisance_rank > 0:
            retained = nuisance_singular[:nuisance_rank]
            self._nuisance_projection_condition = float(
                retained[0] / retained[-1]
            )
        else:
            self._nuisance_projection_condition = math.inf
        singular = np.linalg.svd(residual_jacobian, compute_uv=False)
        if singular.size == 0 or not np.isfinite(singular).all():
            singular = np.asarray([0.0], dtype=float)
        maximum = float(np.max(singular))
        minimum = float(np.min(singular))
        tolerance = max(1.0e-9, 1.0e-3 * maximum)
        rank = int(np.count_nonzero(singular > tolerance))
        self._observability_rank = rank
        self._observability_min_singular_value = minimum
        self._observability_max_singular_value = maximum
        self._observability_condition = (
            float(maximum / minimum) if minimum > tolerance else math.inf
        )

    def _grid_summary(self, grid: np.ndarray, weights: np.ndarray) -> float:
        probability = self._normalize(weights)
        if self._posterior_summary == "mean":
            return float(np.sum(grid * probability))
        index = int(np.argmax(probability))
        estimate = float(grid[index])
        if 0 < index < len(grid) - 1:
            costs = -2.0 * np.log(np.maximum(probability, np.finfo(float).tiny))
            denominator = float(
                costs[index - 1] - 2.0 * costs[index] + costs[index + 1]
            )
            if denominator > 1.0e-12:
                offset = 0.5 * float(
                    (costs[index - 1] - costs[index + 1]) / denominator
                )
                estimate += float(np.clip(offset, -1.0, 1.0)) * float(
                    grid[index + 1] - grid[index]
                )
        return float(np.clip(estimate, grid[0], grid[-1]))

    def _update_posterior(self) -> bool:
        posterior_started = time.perf_counter()
        if len(self._history) < self._min_windows:
            self._joint_active = False
            self._dynamics_active = False
            return False
        block_times = tuple(float(entry["time"]) for entry in self._history)
        duplicate_count = len(block_times) - len(set(block_times))
        self._duplicate_likelihood_block_count += int(duplicate_count)
        if duplicate_count:
            self._duplicate_likelihood_update_count += 1
            self._joint_active = False
            self._dynamics_active = False
            return False

        measurements = np.stack(
            [entry["measurement"] for entry in self._history]
        )
        tensor = np.stack([entry["prediction"] for entry in self._history])
        hypothesis_count = int(np.prod(tensor.shape[1:-1]))
        flattened = tensor.reshape(len(tensor), hypothesis_count, 2)
        self._likelihood_evaluations += 1
        self._last_likelihood_block_count = len(block_times)
        self._last_likelihood_residual_count = 2 * len(block_times)
        self._last_profile_block_times = block_times
        profile_started = time.perf_counter()
        costs, parameters = self._profile_joint_cost(measurements, flattened)
        self._last_profile_wall_time_s = (
            time.perf_counter() - profile_started
        )
        if not np.isfinite(costs).all() or not np.isfinite(parameters).all():
            self._joint_active = False
            self._dynamics_active = False
            return False

        cost_tensor = costs.reshape(self._joint_prior.shape)
        log_posterior = (
            np.log(np.maximum(self._joint_prior, np.finfo(float).tiny))
            - 0.5 * (cost_tensor - float(np.min(cost_tensor)))
        )
        log_posterior -= float(np.max(log_posterior))
        candidate = self._normalize(np.exp(log_posterior))
        n_marginal = np.sum(candidate, axis=(1, 2))
        phi_marginal = np.sum(candidate, axis=(0, 2))
        cohesion_marginal = np.sum(candidate, axis=(0, 1))
        self._joint_information_kl = self._discrete_kl(
            candidate, self._joint_prior
        )
        self._n_information_kl = self._discrete_kl(n_marginal, self._n_prior)
        self._phi_information_kl = self._discrete_kl(
            phi_marginal, self._phi_prior
        )
        self._cohesion_information_kl = self._discrete_kl(
            cohesion_marginal, self._cohesion_prior
        )
        self._profile_cost_span = float(np.ptp(costs))
        self._n_boundary_mass = float(n_marginal[0] + n_marginal[-1])
        self._phi_boundary_mass = float(
            phi_marginal[0] + phi_marginal[-1]
        )
        self._cohesion_boundary_mass = (
            0.0
            if len(cohesion_marginal) == 1
            else float(cohesion_marginal[0] + cohesion_marginal[-1])
        )
        n_map_index = int(np.argmax(n_marginal))
        phi_map_index = int(np.argmax(phi_marginal))
        cohesion_map_index = int(np.argmax(cohesion_marginal))
        self._n_map_at_boundary = n_map_index in {0, len(n_marginal) - 1}
        self._phi_map_at_boundary = phi_map_index in {
            0, len(phi_marginal) - 1
        }
        self._cohesion_map_at_boundary = bool(
            len(cohesion_marginal) > 1
            and cohesion_map_index in {0, len(cohesion_marginal) - 1}
        )
        material_boundary_mass = (
            max(
                self._n_boundary_mass,
                self._phi_boundary_mass,
                self._cohesion_boundary_mass,
            )
            >= self._boundary_warning_mass
        )
        # An edge MAP is limiting for a MAP output.  For a reported marginal
        # mean, however, a single edge bin can be the tallest point of a broad
        # or mildly bimodal distribution while carrying little total mass.
        # Preserve the edge-MAP diagnostics, but call the mean boundary-limited
        # only when a material fraction of its weight actually reaches an
        # envelope boundary.
        self._boundary_limited = bool(
            material_boundary_mass
            or (
                self._posterior_summary == "map"
                and (
                    self._n_map_at_boundary
                    or self._phi_map_at_boundary
                    or self._cohesion_map_at_boundary
                )
            )
        )

        map_index = tuple(
            int(value)
            for value in np.unravel_index(
                int(np.argmax(candidate)), candidate.shape
            )
        )
        observability_started = time.perf_counter()
        self._update_observability(tensor, map_index)  # type: ignore[arg-type]
        self._last_observability_wall_time_s = (
            time.perf_counter() - observability_started
        )
        required_dimensions = 2 + int(len(self._cohesion_grid) > 1)
        required_rank = min(self._min_observability_rank, required_dimensions)
        joint_rejected = (
            self._joint_information_kl < self._min_joint_information
        )
        n_rejected = self._n_information_kl < self._min_n_information
        phi_rejected = self._phi_information_kl < self._min_phi_information
        observability_rejected = (
            self._observability_rank < required_rank
            or self._observability_min_singular_value
            < self._min_observability_singular_value
        )
        self._joint_information_rejections += int(joint_rejected)
        self._n_information_rejections += int(n_rejected)
        self._phi_information_rejections += int(phi_rejected)
        self._observability_rejections += int(observability_rejected)
        if (
            joint_rejected
            or n_rejected
            or phi_rejected
            or observability_rejected
        ):
            self._gate_rejected_updates += 1
            self._joint_active = False
            self._dynamics_active = False
            return False

        self._joint_posterior = candidate
        self._posterior = n_marginal
        self._n_posterior = float(np.sum(self._grid * n_marginal))
        self._phi_posterior = phi_marginal
        self._cohesion_posterior = cohesion_marginal
        n_target = self._grid_summary(self._grid, n_marginal)
        phi_target = self._grid_summary(self._phi_grid, phi_marginal)
        cohesion_target = self._grid_summary(
            self._cohesion_grid, cohesion_marginal
        )
        self._n_smooth += self._smoothing_alpha * (n_target - self._n_smooth)
        self._phi_smooth += self._smoothing_alpha * (
            phi_target - self._phi_smooth
        )
        self._cohesion_output += self._smoothing_alpha * (
            cohesion_target - self._cohesion_output
        )
        flat_index = int(np.ravel_multi_index(map_index, candidate.shape))
        self._profile_parameters = parameters[flat_index].copy()
        tolerance = 1.0e-8
        self._profile_bound_hits = int(np.count_nonzero(
            (self._profile_parameters <= self._profile_lower + tolerance)
            | (self._profile_parameters >= self._profile_upper - tolerance)
        ))
        self._informative_windows = len(self._history)
        self._posterior_updates += 1
        self._observation_count += 1
        self._joint_updates += 1
        self._joint_active = True
        self._joint_has_estimate = True
        self._dynamics_active = True
        self._last_joint_update_time = block_times[-1]
        self._last_posterior_wall_time_s = (
            time.perf_counter() - posterior_started
        )
        return True

    def _capture_accepted_snapshot(
        self,
        parameters: Mapping[str, float],
        confidence: float,
        publication_wall_time_s: float,
    ) -> None:
        """Freeze the diagnostics belonging to one published accepted update."""

        terrain_parameters = MappingProxyType({
            str(key): float(value) for key, value in parameters.items()
        })
        snapshot = {
            "snapshot_version": "grit_accepted",
            "update_seq": int(self._joint_updates),
            "evidence_time_s": float(self._last_joint_update_time),
            "n": float(self._n_smooth),
            "phi_deg": float(self._phi_smooth),
            "terrain_params": terrain_parameters,
            "confidence": float(confidence),
            "n_sigma": float(self.get_n_uncertainty()),
            "phi_sigma_deg": float(self.get_phi_uncertainty_deg()),
            "joint_information_kl": float(self._joint_information_kl),
            "n_information_kl": float(self._n_information_kl),
            "phi_information_kl": float(self._phi_information_kl),
            "cohesion_information_kl": float(
                self._cohesion_information_kl
            ),
            "observability_rank": int(self._observability_rank),
            "observability_min_singular_value": float(
                self._observability_min_singular_value
            ),
            "n_boundary_mass": float(self._n_boundary_mass),
            "phi_boundary_mass": float(self._phi_boundary_mass),
            "cohesion_boundary_mass": float(self._cohesion_boundary_mass),
            "max_boundary_mass": float(max(
                self._n_boundary_mass,
                self._phi_boundary_mass,
                self._cohesion_boundary_mass,
            )),
            "boundary_limited": bool(self._boundary_limited),
            "joint_projection_failures": int(
                self._joint_projection_failures
            ),
            "duplicate_likelihood_block_count": int(
                self._duplicate_likelihood_block_count
            ),
            "duplicate_likelihood_update_count": int(
                self._duplicate_likelihood_update_count
            ),
            "likelihood_evaluations": int(self._likelihood_evaluations),
            "likelihood_block_count": int(
                self._last_likelihood_block_count
            ),
            "likelihood_residual_count": int(
                self._last_likelihood_residual_count
            ),
            # Median absolute residual at the favoured hypothesis, raw m/s^2.
            # For a Student-t4 the median absolute value is 0.7407 scales, so a
            # data-derived noise scale is mad / 0.7407; the frozen r_ax/r_ay can
            # be checked against this instead of being asserted.
            "residual_ax_mad": float(
                np.median(self._residual_ax_abs)
            ) if self._residual_ax_abs else float("nan"),
            "residual_ay_mad": float(
                np.median(self._residual_ay_abs)
            ) if self._residual_ay_abs else float("nan"),
            "residual_sample_count": int(len(self._residual_ax_abs)),
            "load_transfer_mode": str(self._load_transfer_mode),
            "effective_front_load": float(
                self._last_effective_front_load
            ),
            "effective_rear_load": float(self._last_effective_rear_load),
            "effective_load_ax": float(self._last_effective_load_ax),
            "effective_load_ay": float(self._last_effective_load_ay),
            "profile_force_gain": float(self._profile_parameters[0]),
            "profile_ax_bias": float(self._profile_parameters[1]),
            "profile_ay_bias": float(self._profile_parameters[2]),
            "profile_bound_hits": int(self._profile_bound_hits),
            "accepted_dynamics_windows": int(self._accepted_windows),
            "rejected_dynamics_windows": int(self._rejected_windows),
            "gate_rejected_updates": int(self._gate_rejected_updates),
            "projection_wall_time_s": float(
                self._last_projection_wall_time_s
            ),
            "profile_wall_time_s": float(self._last_profile_wall_time_s),
            "observability_wall_time_s": float(
                self._last_observability_wall_time_s
            ),
            "posterior_wall_time_s": float(
                self._last_posterior_wall_time_s
            ),
            "publication_wall_time_s": float(publication_wall_time_s),
            # Profiling and observability are sub-stages of posterior time.
            # The non-overlapping accepted-update cost is therefore projection
            # plus posterior plus the final publication/copy step.
            "update_wall_time_s": float(
                self._last_projection_wall_time_s
                + self._last_posterior_wall_time_s
                + publication_wall_time_s
            ),
        }
        self._last_accepted_snapshot = MappingProxyType(snapshot)
        self._snapshot_update_seq = int(self._joint_updates)

    def estimate(self) -> Tuple[Dict[str, float], float]:
        publication_started = time.perf_counter()
        self._posterior_updates = 0
        self._observation_count = 0
        parameters = terrain_params_for_n(
            self._n_smooth, soft_floor=self._manifold_soft_floor,
            soft_mode=self._manifold_soft_mode,
        )
        parameters["phi"] = float(self._phi_smooth)
        if len(self._cohesion_grid) > 1:
            parameters["c"] *= float(self._cohesion_output)
        self._estimated_params = parameters
        if not self._joint_has_estimate:
            self._confidence = 0.0
            return dict(parameters), 0.0
        n_ratio = self.get_n_uncertainty() / max(
            self._weighted_std(self._grid, self._n_prior), 1.0e-12
        )
        phi_ratio = self.get_phi_uncertainty_deg() / max(
            self._weighted_std(self._phi_grid, self._phi_prior), 1.0e-12
        )
        information_factor = 1.0 - math.exp(-self._joint_information_kl)
        boundary_factor = max(
            0.0,
            1.0
            - max(
                self._n_boundary_mass,
                self._phi_boundary_mass,
                self._cohesion_boundary_mass,
            ),
        )
        self._confidence = float(
            min(1.0 / (1.0 + n_ratio**2), 1.0 / (1.0 + phi_ratio**2))
            * information_factor
            * boundary_factor
        )
        if self._joint_updates > self._snapshot_update_seq:
            self._capture_accepted_snapshot(
                parameters,
                self._confidence,
                time.perf_counter() - publication_started,
            )
        return dict(parameters), self._confidence

    def get_last_accepted_snapshot(
        self,
    ) -> Optional[Mapping[str, Any]]:
        """Return the immutable most recently published accepted update."""

        return self._last_accepted_snapshot

    def get_bekker_n(self) -> float:
        return float(self._n_smooth)

    def get_friction_angle_deg(self) -> float:
        return float(self._phi_smooth)

    def get_phi_uncertainty_deg(self) -> float:
        return self._weighted_std(self._phi_grid, self._phi_posterior)

    def get_terrain_mpc_params(self) -> Dict[str, float]:
        parameters = terrain_params_for_n(self.get_bekker_n())
        parameters["phi"] = self.get_friction_angle_deg()
        if len(self._cohesion_grid) > 1:
            parameters["c"] *= float(self._cohesion_output)
        return parameters

    @property
    def mu_estimate(self) -> float:
        return float(math.tan(math.radians(self.get_friction_angle_deg())))

    @property
    def joint_active(self) -> bool:
        return bool(self._joint_active)

    @property
    def joint_has_estimate(self) -> bool:
        return bool(self._joint_has_estimate)

    @property
    def joint_updates(self) -> int:
        return int(self._joint_updates)

    @property
    def joint_projection_failures(self) -> int:
        return int(self._joint_projection_failures)

    @property
    def joint_information_kl(self) -> float:
        return float(self._joint_information_kl)

    @property
    def n_information_kl(self) -> float:
        return float(self._n_information_kl)

    @property
    def phi_information_kl(self) -> float:
        return float(self._phi_information_kl)

    @property
    def cohesion_information_kl(self) -> float:
        return float(self._cohesion_information_kl)

    @property
    def profile_cost_span(self) -> float:
        return float(self._profile_cost_span)

    @property
    def joint_effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self._joint_posterior**2))

    @property
    def n_effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self._posterior**2))

    @property
    def phi_effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self._phi_posterior**2))

    @property
    def cohesion_effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self._cohesion_posterior**2))

    @property
    def n_boundary_mass(self) -> float:
        return float(self._n_boundary_mass)

    @property
    def phi_boundary_mass(self) -> float:
        return float(self._phi_boundary_mass)

    @property
    def cohesion_boundary_mass(self) -> float:
        return float(self._cohesion_boundary_mass)

    @property
    def boundary_limited(self) -> bool:
        return bool(self._boundary_limited)

    @property
    def n_map_at_boundary(self) -> bool:
        return bool(self._n_map_at_boundary)

    @property
    def phi_map_at_boundary(self) -> bool:
        return bool(self._phi_map_at_boundary)

    @property
    def cohesion_map_at_boundary(self) -> bool:
        return bool(self._cohesion_map_at_boundary)

    @property
    def observability_rank(self) -> int:
        return int(self._observability_rank)

    @property
    def observability_min_singular_value(self) -> float:
        return float(self._observability_min_singular_value)

    @property
    def observability_max_singular_value(self) -> float:
        return float(self._observability_max_singular_value)

    @property
    def observability_condition(self) -> float:
        return float(self._observability_condition)

    @property
    def nuisance_projection_rank(self) -> int:
        return int(self._nuisance_projection_rank)

    @property
    def nuisance_projection_condition(self) -> float:
        return float(self._nuisance_projection_condition)

    @property
    def last_likelihood_block_count(self) -> int:
        return int(self._last_likelihood_block_count)

    @property
    def last_likelihood_residual_count(self) -> int:
        return int(self._last_likelihood_residual_count)

    @property
    def duplicate_likelihood_block_count(self) -> int:
        return int(self._duplicate_likelihood_block_count)

    @property
    def duplicate_likelihood_update_count(self) -> int:
        return int(self._duplicate_likelihood_update_count)

    @property
    def likelihood_evaluations(self) -> int:
        return int(self._likelihood_evaluations)

    @property
    def joint_information_rejections(self) -> int:
        return int(self._joint_information_rejections)

    @property
    def n_information_rejections(self) -> int:
        return int(self._n_information_rejections)

    @property
    def phi_information_rejections(self) -> int:
        return int(self._phi_information_rejections)

    @property
    def observability_rejections(self) -> int:
        return int(self._observability_rejections)

    @property
    def gate_rejected_updates(self) -> int:
        return int(self._gate_rejected_updates)

    @property
    def update_acceptance_fraction(self) -> float:
        return float(
            self._joint_updates / self._likelihood_evaluations
            if self._likelihood_evaluations
            else 0.0
        )

    @property
    def last_joint_update_time(self) -> float:
        return float(self._last_joint_update_time)

    @property
    def load_transfer_mode(self) -> str:
        return self._load_transfer_mode

    @property
    def measured_load_blocks(self) -> int:
        return int(self._measured_load_blocks)

    @property
    def static_load_blocks(self) -> int:
        return int(self._static_load_blocks)

    @property
    def lagged_load_blocks(self) -> int:
        return int(self._lagged_load_blocks)

    @property
    def lagged_load_fallback_blocks(self) -> int:
        return int(self._lagged_load_fallback_blocks)

    @property
    def last_effective_front_load(self) -> float:
        return float(self._last_effective_front_load)

    @property
    def last_effective_rear_load(self) -> float:
        return float(self._last_effective_rear_load)

    @property
    def last_effective_load_ax(self) -> float:
        return float(self._last_effective_load_ax)

    @property
    def last_effective_load_ay(self) -> float:
        return float(self._last_effective_load_ay)

    @property
    def cohesion_enabled(self) -> bool:
        return bool(len(self._cohesion_grid) > 1)

    @property
    def cohesion_multiplier_estimate(self) -> float:
        return float(self._cohesion_output)

    @property
    def cohesion_multiplier_uncertainty(self) -> float:
        return self._weighted_std(
            self._cohesion_grid, self._cohesion_posterior
        )
