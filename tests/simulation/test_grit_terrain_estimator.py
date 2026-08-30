"""Contracts for the joint ``n``/``phi`` terrain estimator.

These tests establish the estimator's numerical core against a straightforward
scalar reference, so the vectorized grid evaluation and its box-constrained
nuisance solve are shown to compute the same result and not merely a faster
one. They also fix the rig projection's feature schema, the causality of its
load reconstruction, and the observability diagnostics that must report
reduced rank rather than overstate what the data identifies.
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from simulation.estimators.grit_terrain_estimator import (
    GritTerrainEstimator,
)
from simulation.shared.param_consistency import HMMWV_VEHICLE_PARAMS


def joint_signal(sample, n_grid, phi_grid, cohesion_grid):
    """Synthetic independently excited n/phi/cohesion sensor model."""

    kappa = float(sample["kappa"])
    n = np.asarray(n_grid, dtype=float)[:, None, None]
    phi = np.asarray(phi_grid, dtype=float)[None, :, None]
    cohesion = np.asarray(cohesion_grid, dtype=float)[None, None, :]
    ax = (
        n * (0.7 + 0.55 * kappa)
        + 0.030 * (phi - 20.0) * kappa
        + 0.9 * (cohesion - 1.0) * (kappa**2 - 0.08)
    )
    ay = (
        n * (1.3 - 0.40 * kappa)
        + 0.045 * (phi - 20.0) * (kappa**2 - 0.08)
        + 0.7 * (cohesion - 1.0) * kappa**3
    )
    shape = (len(n_grid), len(phi_grid), len(cohesion_grid))
    return np.broadcast_to(ax, shape).copy(), np.broadcast_to(ay, shape).copy()


def aliased_signal(sample, n_grid, phi_grid, cohesion_grid):
    del sample
    n = np.asarray(n_grid, dtype=float)[:, None, None]
    phi = np.asarray(phi_grid, dtype=float)[None, :, None]
    cohesion = np.asarray(cohesion_grid, dtype=float)[None, None, :]
    latent = n + 0.01 * phi + np.zeros_like(cohesion)
    return latent, 2.0 * latent


def load_sensitive_signal(sample, n_grid, phi_grid, cohesion_grid):
    """Prediction exposing only the prepared load-reconstruction channels."""

    shape = (len(n_grid), len(phi_grid), len(cohesion_grid))
    ax = np.full(
        shape,
        float(sample["Fz_f"]) / 10_000.0 + float(sample["ax"]),
    )
    ay = np.full(
        shape,
        float(sample["Fz_r"]) / 10_000.0 + float(sample["ay"]),
    )
    return ax, ay


def observation(timestamp, *, n_true=0.76, phi_true=24.0, c_true=1.0):
    kappas = (-0.48, -0.32, -0.16, 0.08, 0.24, 0.42, 0.31, -0.27)
    block = min(int((float(timestamp) + 1.0e-8) // 0.5), len(kappas) - 1)
    kappa = kappas[block]
    sample = {"kappa": kappa}
    ax, ay = joint_signal(
        sample,
        np.asarray([n_true]),
        np.asarray([phi_true]),
        np.asarray([c_true]),
    )
    return {
        "kappa": kappa,
        "alpha_f": 0.08,
        "alpha_r": 0.04,
        "u": 5.0,
        "Fz_f": 5_800.0,
        "Fz_r": 5_800.0,
        "sr": 0.0,
        "alpha_rate_r": 0.0,
        "ay_imu": float(ay[0, 0, 0]),
        "omega_dot": 0.0,
        "sim_time": float(timestamp),
        "omega": 0.08,
        "v_lateral": 0.1,
        "ax_imu": float(ax[0, 0, 0]),
        "steering_angle": 0.10,
        "wheel_omegas": (11.0, 11.0, 11.0, 11.0),
    }


def common_kwargs(**extra):
    values = {
        "initial_terrain": {"n": 0.7, "phi": 29.0},
        "vehicle_params": dict(HMMWV_VEHICLE_PARAMS),
        "joint_prediction": joint_signal,
        "update_interval": 1,
        "block_dt": 0.5,
        "horizon": 6.0,
        "min_windows": 4,
        "min_window_samples": 4,
        "r_ax": 0.025,
        "r_ay": 0.025,
        "min_information": 0.01,
        "min_joint_information": 0.0,
        # These tests exercise the likelihood mechanics on synthetic data, so
        # the observability gate is opened to keep the runtime admission
        # policy from masking the behaviour under test.
        "min_observability_rank": 0,
        "min_observability_singular_value": 0.0,
        "min_yaw_rate_rms": 0.0,
        "smoothing_alpha": 1.0,
        "grid_size": 21,
        "phi_grid_size": 17,
        "enforce_feature_envelope": False,
        "force_gain_std": 0.04,
        "ax_bias_std": 0.04,
        "ay_bias_std": 0.04,
    }
    values.update(extra)
    return values


def feed(estimator, *, n_true=0.76, phi_true=24.0, c_true=1.0):
    for timestamp in np.arange(0.0, 4.01, 0.1):
        accepted = estimator.observe(
            **observation(
                timestamp, n_true=n_true, phi_true=phi_true, c_true=c_true
            )
        )
        if not accepted:
            raise AssertionError("synthetic conventional observation rejected")


def scalar_profile_reference(estimator, measurements, predictions):
    """Evaluate the profile cost one hypothesis at a time.

    This straightforward loop is the reference the vectorized implementation
    is checked against, so an optimization that changes results is detected
    rather than assumed correct.
    """

    values = np.asarray(predictions, dtype=float)
    scales = np.asarray([estimator._r_ax, estimator._r_ay], dtype=float)
    prior_mean = np.asarray([1.0, 0.0, 0.0], dtype=float)
    prior_precision = 1.0 / estimator._profile_prior_std**2
    target = np.asarray(measurements, dtype=float).reshape(-1)
    row_scale = np.tile(scales, len(measurements))
    normalized_target = target / row_scale
    costs = np.empty(values.shape[1], dtype=float)
    parameters = np.empty((values.shape[1], 3), dtype=float)
    for index in range(values.shape[1]):
        prediction = values[:, index, :]
        design = np.empty((2 * len(measurements), 3), dtype=float)
        design[0::2] = np.column_stack(
            (
                prediction[:, 0],
                np.ones(len(measurements)),
                np.zeros(len(measurements)),
            )
        )
        design[1::2] = np.column_stack(
            (
                prediction[:, 1],
                np.zeros(len(measurements)),
                np.ones(len(measurements)),
            )
        )
        normalized_design = design / row_scale[:, None]
        weights = np.ones(len(target), dtype=float)
        beta = prior_mean.copy()
        for _iteration in range(estimator._profile_iterations):
            hessian = (
                normalized_design.T
                @ (weights[:, None] * normalized_design)
                + np.diag(prior_precision)
            )
            rhs = (
                normalized_design.T @ (weights * normalized_target)
                + prior_precision * prior_mean
            )
            beta = estimator._bounded_quadratic_solution(
                hessian,
                rhs,
                estimator._profile_lower,
                estimator._profile_upper,
            )
            residual = normalized_target - normalized_design @ beta
            weights = (estimator._student_dof + 1.0) / (
                estimator._student_dof + residual**2
            )
        residual = normalized_target - normalized_design @ beta
        costs[index] = (
            (estimator._student_dof + 1.0)
            * float(np.sum(np.log1p(
                residual**2 / estimator._student_dof
            )))
            + float(np.sum(
                ((beta - prior_mean) / estimator._profile_prior_std) ** 2
            ))
        )
        parameters[index] = beta
    return costs, parameters


class EstimatorJointProfileTest(unittest.TestCase):
    def test_vectorized_profile_matches_scalar_reference_including_bounds(self):
        estimator = GritTerrainEstimator(
            **common_kwargs(
                profile_iterations=8,
                force_gain_std=1.0,
                ax_bias_std=1.0,
                ay_bias_std=1.0,
            )
        )
        generator = np.random.default_rng(19423)
        windows = 9
        hypotheses = 23
        predictions = generator.normal(
            loc=0.0, scale=1.2, size=(windows, hypotheses, 2)
        )
        # An intentionally incompatible target drives several nuisance
        # solutions onto box faces and exercises the active-set fallback.
        measurements = np.column_stack(
            (
                np.linspace(20.0, 45.0, windows),
                np.linspace(-40.0, -15.0, windows),
            )
        )
        expected_costs, expected_parameters = scalar_profile_reference(
            estimator, measurements, predictions
        )
        actual_costs, actual_parameters = estimator._profile_joint_cost(
            measurements, predictions
        )
        bound_hits = (
            (expected_parameters <= estimator._profile_lower + 1.0e-12)
            | (expected_parameters >= estimator._profile_upper - 1.0e-12)
        )
        self.assertTrue(np.any(bound_hits))
        np.testing.assert_allclose(
            actual_parameters,
            expected_parameters,
            rtol=2.0e-13,
            atol=2.0e-13,
        )
        np.testing.assert_allclose(
            actual_costs,
            expected_costs,
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_batched_active_set_matches_scalar_and_is_faster(self):
        estimator = GritTerrainEstimator(**common_kwargs())
        generator = np.random.default_rng(62017)
        hypotheses = 257
        factors = generator.normal(size=(hypotheses, 3, 3))
        hessians = (
            np.matmul(np.swapaxes(factors, 1, 2), factors)
            + 0.2 * np.eye(3)[None, :, :]
        )
        rhs = generator.normal(scale=10.0, size=(hypotheses, 3))

        actual = estimator._batched_bounded_quadratic_solution(
            hessians, rhs
        )
        expected = np.vstack([
            estimator._bounded_quadratic_solution(
                hessians[index],
                rhs[index],
                estimator._profile_lower,
                estimator._profile_upper,
            )
            for index in range(hypotheses)
        ])
        np.testing.assert_allclose(
            actual, expected, rtol=2.0e-13, atol=2.0e-13
        )
        self.assertTrue(np.all(
            np.any(
                (actual <= estimator._profile_lower + 1.0e-12)
                | (actual >= estimator._profile_upper - 1.0e-12),
                axis=1,
            )
        ))

        start = time.perf_counter()
        estimator._batched_bounded_quadratic_solution(hessians, rhs)
        batched_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for index in range(hypotheses):
            estimator._bounded_quadratic_solution(
                hessians[index],
                rhs[index],
                estimator._profile_lower,
                estimator._profile_upper,
            )
        scalar_seconds = time.perf_counter() - start
        self.assertLess(batched_seconds, 0.5 * scalar_seconds)

    def test_vectorized_profile_full_grid_performance_smoke(self):
        estimator = GritTerrainEstimator(
            **common_kwargs(profile_iterations=8)
        )
        generator = np.random.default_rng(77231)
        windows = 16
        hypotheses = 41 * 17
        base_prediction = generator.normal(
            loc=0.0, scale=0.8, size=(windows, 1, 2)
        )
        hypothesis_scale = np.linspace(
            0.92, 1.08, hypotheses
        )[None, :, None]
        predictions = base_prediction * hypothesis_scale
        measurements = (
            1.02 * base_prediction[:, 0, :]
            + np.asarray([0.03, -0.02])
        )

        # Warm lazy numerical-library paths before comparing the same workload.
        estimator._profile_joint_cost(measurements, predictions)
        scalar_profile_reference(
            estimator, measurements, predictions[:, :3, :]
        )

        start = time.perf_counter()
        estimator._profile_joint_cost(measurements, predictions)
        vectorized_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scalar_profile_reference(estimator, measurements, predictions)
        scalar_seconds = time.perf_counter() - start

        # The vectorized evaluation is expected to win by more than an order
        # of magnitude.  Asserting a factor of two keeps the test portable
        # across machines rather than turning it into a hardware benchmark,
        # while still failing if the grid evaluation reverts to a Python loop.
        self.assertLess(vectorized_seconds, 0.5 * scalar_seconds)

    def test_rate_rig_projection_uses_fourteen_feature_schema(self):
        captured = []

        class RateModel:
            rate_augmented = True
            input_dim = 14

            @staticmethod
            def phi_feature_value(phi):
                return float(np.degrees(phi))

            @staticmethod
            def predict_feature_rows(features):
                captured.append(np.asarray(features, dtype=float).copy())
                return np.zeros((len(features), 2), dtype=float)

        estimator = GritTerrainEstimator(
            **common_kwargs(joint_prediction=None, rate_mode="signed")
        )
        estimator._force_projector._model = RateModel()
        sample = {
            "u": 5.0,
            "v": 0.1,
            "omega": 0.05,
            "delta": 0.08,
            "kappa": 0.04,
            "wheel_omegas": (11.0, 11.0, 11.0, 11.0),
            "sr": 0.20,
            "sr_r": -0.10,
            "Fz_f": 5_800.0,
            "Fz_r": 5_800.0,
            "ax": 0.0,
            "ay": 0.0,
            "dkappa": 0.12,
            "du": -0.35,
        }
        ax, ay = estimator._predict_joint_grid(
            sample,
            estimator._grid,
            estimator._phi_grid,
            estimator._cohesion_grid,
        )
        expected_shape = (
            len(estimator._grid),
            len(estimator._phi_grid),
            len(estimator._cohesion_grid),
        )
        self.assertEqual(ax.shape, expected_shape)
        self.assertEqual(ay.shape, expected_shape)
        self.assertEqual(len(captured), 1)
        features = captured[0]
        self.assertEqual(features.shape[1], 14)
        self.assertTrue(np.all(features[:, 5] == 0.12))
        self.assertTrue(np.all(features[:, 7] == -0.35))
        self.assertTrue(np.any(features[:, 6] != 0.0))

    def test_static_load_projection_is_invariant_to_measured_load_state(self):
        estimator = GritTerrainEstimator(
            **common_kwargs(
                load_transfer_mode="static",
                joint_prediction=load_sensitive_signal,
            )
        )
        first = {
            "Fz_f": 4_000.0, "Fz_r": 7_000.0, "ax": -1.2, "ay": 2.3,
        }
        second = {
            "Fz_f": 7_000.0, "Fz_r": 4_000.0, "ax": 1.8, "ay": -3.1,
        }
        prepared_first = estimator._prepare_projection_sample(first)
        prepared_second = estimator._prepare_projection_sample(second)
        prediction_first = estimator._joint_prediction(
            prepared_first,
            estimator._grid,
            estimator._phi_grid,
            estimator._cohesion_grid,
        )
        prediction_second = estimator._joint_prediction(
            prepared_second,
            estimator._grid,
            estimator._phi_grid,
            estimator._cohesion_grid,
        )
        for first_channel, second_channel in zip(
            prediction_first, prediction_second
        ):
            self.assertTrue(np.array_equal(first_channel, second_channel))
        self.assertEqual(prepared_first["ax"], 0.0)
        self.assertEqual(prepared_first["ay"], 0.0)

    def test_lagged_load_projection_is_causal(self):
        estimator = GritTerrainEstimator(
            **common_kwargs(load_transfer_mode="lagged")
        )
        first_raw = {"Fz_f": 4_100.0, "Fz_r": 6_900.0, "ax": 0.4, "ay": 1.2}
        first = estimator._prepare_projection_sample(first_raw)
        self.assertEqual(first["_joint_load_source"], "lagged_static_fallback")
        estimator._commit_projection_loads(first, first_raw)
        second_raw = {"Fz_f": 6_800.0, "Fz_r": 4_200.0, "ax": -0.8, "ay": -2.0}
        second = estimator._prepare_projection_sample(second_raw)
        self.assertEqual(second["_joint_load_source"], "lagged")
        self.assertEqual(second["Fz_f"], first_raw["Fz_f"])
        self.assertEqual(second["Fz_r"], first_raw["Fz_r"])
        self.assertEqual(second["ax"], first_raw["ax"])
        self.assertEqual(second["ay"], first_raw["ay"])
        self.assertEqual(estimator.lagged_load_fallback_blocks, 1)

    def test_prior_is_broad_and_independent_not_manifold_centered(self):
        estimator = GritTerrainEstimator(**common_kwargs())
        conditional_phi = estimator._joint_prior[:, :, 0]
        conditional_phi /= np.sum(conditional_phi, axis=1, keepdims=True)
        expected = np.full_like(conditional_phi, 1.0 / conditional_phi.shape[1])
        self.assertTrue(np.allclose(conditional_phi, expected, rtol=0.0, atol=1e-15))
        self.assertTrue(np.allclose(
            estimator._posterior,
            np.full_like(estimator._posterior, 1.0 / len(estimator._posterior)),
            rtol=0.0,
            atol=1e-15,
        ))

    def test_one_prediction_and_one_residual_pair_per_accepted_block(self):
        calls = []

        def counted(sample, n_grid, phi_grid, cohesion_grid):
            calls.append(float(sample["kappa"]))
            return joint_signal(sample, n_grid, phi_grid, cohesion_grid)

        estimator = GritTerrainEstimator(
            **common_kwargs(joint_prediction=counted)
        )
        feed(estimator)
        self.assertEqual(len(calls), estimator.accepted_dynamics_windows)
        self.assertEqual(estimator.duplicate_likelihood_block_count, 0)
        self.assertEqual(
            estimator.last_likelihood_block_count, estimator.dynamics_windows
        )
        self.assertEqual(
            estimator.last_likelihood_residual_count,
            2 * estimator.last_likelihood_block_count,
        )
        self.assertGreater(estimator.likelihood_evaluations, 0)

    def test_joint_profile_recovers_independent_n_and_phi(self):
        estimator = GritTerrainEstimator(**common_kwargs())
        feed(estimator, n_true=0.76, phi_true=24.0)
        self.assertTrue(estimator.joint_active)
        self.assertTrue(estimator.joint_has_estimate)
        self.assertGreater(estimator.joint_updates, 0)
        self.assertAlmostEqual(estimator.get_bekker_n(), 0.76, delta=0.04)
        self.assertAlmostEqual(
            estimator.get_friction_angle_deg(), 24.0, delta=2.1
        )
        # A free shared gain removes one local scale direction in this
        # deliberately affine synthetic force model.  Recovery still succeeds
        # because the gain prior is finite, but the parameters are genuinely
        # less observable, and the diagnostic must report that reduced
        # structural rank rather than overstate observability.
        self.assertEqual(estimator.observability_rank, 1)
        self.assertTrue(np.isinf(estimator.observability_condition))
        self.assertGreater(estimator.joint_information_kl, 0.0)

    def test_last_accepted_snapshot_is_immutable_and_rejection_stable(self):
        estimator = GritTerrainEstimator(**common_kwargs())
        self.assertIsNone(estimator.get_last_accepted_snapshot())
        feed(estimator, n_true=0.76, phi_true=24.0)
        parameters, confidence = estimator.estimate()
        snapshot = estimator.get_last_accepted_snapshot()
        self.assertIsNotNone(snapshot)
        self.assertEqual(
            snapshot["snapshot_version"],
            "grit_accepted",
        )
        self.assertEqual(snapshot["update_seq"], estimator.joint_updates)
        self.assertEqual(
            snapshot["likelihood_residual_count"],
            2 * snapshot["likelihood_block_count"],
        )
        self.assertEqual(snapshot["confidence"], confidence)
        self.assertEqual(dict(snapshot["terrain_params"]), parameters)
        self.assertEqual(
            snapshot["max_boundary_mass"],
            max(
                snapshot["n_boundary_mass"],
                snapshot["phi_boundary_mass"],
                snapshot["cohesion_boundary_mass"],
            ),
        )
        self.assertAlmostEqual(snapshot["n"], estimator.get_bekker_n())
        self.assertAlmostEqual(
            snapshot["phi_deg"], estimator.get_friction_angle_deg()
        )
        for key in (
            "projection_wall_time_s",
            "profile_wall_time_s",
            "observability_wall_time_s",
            "posterior_wall_time_s",
            "publication_wall_time_s",
            "update_wall_time_s",
        ):
            self.assertTrue(np.isfinite(snapshot[key]))
            self.assertGreaterEqual(snapshot[key], 0.0)
        self.assertAlmostEqual(
            snapshot["update_wall_time_s"],
            snapshot["projection_wall_time_s"]
            + snapshot["posterior_wall_time_s"]
            + snapshot["publication_wall_time_s"],
        )
        with self.assertRaises(TypeError):
            snapshot["n"] = 0.5
        with self.assertRaises(TypeError):
            snapshot["terrain_params"]["n"] = 0.5

        frozen_values = dict(snapshot)
        duplicate = {
            "time": estimator._history[-1]["time"],
            "measurement": estimator._history[-1]["measurement"].copy(),
            "prediction": estimator._history[-1]["prediction"].copy(),
        }
        estimator._history.append(duplicate)
        self.assertFalse(estimator._update_posterior())
        estimator.estimate()
        self.assertIs(estimator.get_last_accepted_snapshot(), snapshot)
        self.assertEqual(dict(snapshot), frozen_values)

    def test_optional_cohesion_is_in_same_joint_likelihood(self):
        estimator = GritTerrainEstimator(
            **common_kwargs(
                cohesion_grid_size=5,
                cohesion_multiplier_bounds=(0.7, 1.3),
                cohesion_prior_std=0.3,
            )
        )
        feed(estimator, n_true=0.76, phi_true=24.0, c_true=1.15)
        self.assertTrue(estimator.cohesion_enabled)
        self.assertTrue(estimator.joint_has_estimate)
        self.assertGreater(estimator.cohesion_information_kl, 0.0)
        self.assertAlmostEqual(
            estimator.cohesion_multiplier_estimate, 1.15, delta=0.16
        )
        self.assertEqual(
            estimator.last_likelihood_residual_count,
            2 * estimator.last_likelihood_block_count,
        )

    def test_observability_reports_aliased_n_phi(self):
        estimator = GritTerrainEstimator(
            **common_kwargs(
                joint_prediction=aliased_signal,
                min_information=0.0,
                min_observability_rank=2,
            )
        )
        # The measurements need only be finite; this test exercises the local
        # sensitivity after gain/bias nuisance projection.
        feed(estimator)
        self.assertLess(estimator.observability_rank, 2)
        self.assertFalse(estimator.joint_active)

    def test_rank_deficient_nuisance_projection_uses_numerical_rank(self):
        estimator = GritTerrainEstimator(**common_kwargs())
        windows = 8
        tensor = np.empty(
            (
                windows,
                len(estimator._grid),
                len(estimator._phi_grid),
                1,
                2,
            ),
            dtype=float,
        )
        n_center = len(estimator._grid) // 2
        phi_center = len(estimator._phi_grid) // 2
        n_offset = estimator._grid - estimator._grid[n_center]
        phi_offset = estimator._phi_grid - estimator._phi_grid[phi_center]
        phase = np.linspace(-1.0, 1.0, windows)
        for window, value in enumerate(phase):
            tensor[window, ..., 0] = (
                1.0
                + n_offset[:, None, None] * value
                + phi_offset[None, :, None] * (value**2 - 0.4)
            )
            tensor[window, ..., 1] = (
                2.0
                + n_offset[:, None, None] * (value**3 - 0.2)
                + phi_offset[None, :, None] * np.sin(value)
            )
        estimator._update_observability(
            tensor, (n_center, phi_center, 0)
        )
        # At the map the prediction is a constant [1, 2], so its gain
        # tangent is exactly a linear combination of the two bias columns.
        self.assertEqual(estimator.nuisance_projection_rank, 2)
        self.assertEqual(estimator.observability_rank, 2)
        self.assertTrue(np.isfinite(estimator.nuisance_projection_condition))

    def test_separate_n_and_phi_information_gates_reject_independently(self):
        n_gated = GritTerrainEstimator(
            **common_kwargs(min_n_information=1.0e6)
        )
        phi_gated = GritTerrainEstimator(
            **common_kwargs(min_phi_information=1.0e6)
        )
        feed(n_gated)
        feed(phi_gated)
        self.assertFalse(n_gated.joint_has_estimate)
        self.assertFalse(phi_gated.joint_has_estimate)
        self.assertGreater(n_gated.n_information_rejections, 0)
        self.assertEqual(n_gated.phi_information_rejections, 0)
        self.assertGreater(phi_gated.phi_information_rejections, 0)
        self.assertEqual(phi_gated.n_information_rejections, 0)
        self.assertEqual(
            n_gated.gate_rejected_updates, n_gated.likelihood_evaluations
        )

    def test_duplicate_block_diagnostics_are_cumulative(self):
        estimator = GritTerrainEstimator(**common_kwargs())
        feed(estimator)
        duplicate = {
            "time": estimator._history[-1]["time"],
            "measurement": estimator._history[-1]["measurement"].copy(),
            "prediction": estimator._history[-1]["prediction"].copy(),
        }
        estimator._history.append(duplicate)
        before = estimator.duplicate_likelihood_block_count
        self.assertFalse(estimator._update_posterior())
        after_first = estimator.duplicate_likelihood_block_count
        self.assertFalse(estimator._update_posterior())
        self.assertGreater(after_first, before)
        self.assertGreater(
            estimator.duplicate_likelihood_block_count, after_first
        )
        self.assertEqual(estimator.duplicate_likelihood_update_count, 2)

    def test_boundary_mass_and_nuisance_diagnostics_are_finite(self):
        estimator = GritTerrainEstimator(
            **common_kwargs(boundary_warning_mass=0.05)
        )
        feed(estimator, n_true=0.50, phi_true=6.0)
        self.assertGreaterEqual(estimator.n_boundary_mass, 0.0)
        self.assertGreaterEqual(estimator.phi_boundary_mass, 0.0)
        self.assertLessEqual(estimator.n_boundary_mass, 1.0)
        self.assertLessEqual(estimator.phi_boundary_mass, 1.0)
        self.assertTrue(estimator.boundary_limited)
        self.assertTrue(np.isfinite(estimator.profile_force_gain))
        self.assertTrue(np.isfinite(estimator.profile_ax_bias))
        self.assertTrue(np.isfinite(estimator.profile_ay_bias))

    def test_posterior_temperature_is_pinned_to_the_frozen_contract(self):
        # The log-posterior tempering factor (0.5, i.e. exp[-(J - J_min)/2])
        # is part of the promoted estimator's frozen design: changing it
        # reshapes every posterior, confidence, and boundary-mass value while
        # all interface tests keep passing. Pin the exact expression in
        # source, the same convention used to pin the solver batch width.
        import inspect
        from simulation.estimators import grit_terrain_estimator as module
        source = inspect.getsource(module)
        self.assertIn(
            "- 0.5 * (cost_tensor - float(np.min(cost_tensor)))",
            source,
            "the joint posterior temperature departed from the frozen "
            "contract exp[-(J - J_min)/2]; this requires a new "
            "preregistration, not a silent edit",
        )

    def test_forbidden_constructor_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbidden"):
            GritTerrainEstimator(
                **common_kwargs(), ground_datum_z=0.0
            )
        with self.assertRaisesRegex(ValueError, "forbidden"):
            GritTerrainEstimator(
                **common_kwargs(), plant_soil_truth={"n": 0.7}
            )


if __name__ == "__main__":
    unittest.main()
