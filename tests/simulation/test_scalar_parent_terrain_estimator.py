"""Contracts for the scalar dynamics estimator used as the comparison arm.

This estimator infers a single soil coordinate from vehicle dynamics with no
ground datum, and supplies the comparison arm in the terrain-estimator table.
These tests establish that its four-wheel force projection is correct, that
its block accumulation is causal, and that the replay factory constructs it
identically to the live controller, so the comparison measures the estimator
rather than a difference in how the two arms were built.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from simulation.estimators.scalar_parent_terrain_estimator import (
    ScalarParentTerrainEstimator,
)
from simulation.estimators.terrain_parameterization import terrain_params_for_n
from simulation.tire_models.four_wheel_projection import (
    FourWheelProjector,
    ProjectionVehicle,
    four_wheel_body_wrench,
)
from benchmarking.terrain_estimator_replay import (
    ReplayConfig,
    make_estimator as make_replay_estimator,
)


ROOT = Path(__file__).resolve().parents[2]


def hypothesis_prediction(sample, n_grid):
    """Deterministic dynamics with a unique terrain hypothesis."""

    del sample
    grid = np.asarray(n_grid, dtype=float)
    return grid, 2.0 * grid, 0.5 * grid


def make_estimator(prediction=hypothesis_prediction, **overrides):
    parameters = {
        "dynamics_prediction": prediction,
        "initial_terrain": {"n": 0.7},
        "update_interval": 1,
        "block_dt": 0.5,
        "horizon": 6.0,
        "min_windows": 2,
        "min_window_samples": 4,
        "evidence_weight": 1.0,
        "r_ax": 0.02,
        "r_ay": 0.02,
        "r_yaw_accel": 0.02,
        "min_information": 0.01,
        "arrival_std": 0.3,
        "posterior_summary": "map",
        "smoothing_alpha": 1.0,
        "grid_size": 31,
    }
    parameters.update(overrides)
    return ScalarParentTerrainEstimator(**parameters)


def observation(timestamp, *, ax=0.9, ay=1.8, omega=None, **extras):
    if omega is None:
        omega = 0.45 * float(timestamp)
    result = {
        "kappa": 0.03,
        "alpha_f": 0.05,
        "alpha_r": 0.03,
        "u": 5.0,
        "Fz_f": 5_800.0,
        "Fz_r": 5_800.0,
        "sr": 0.0,
        "alpha_rate_r": 0.0,
        "ay_imu": float(ay),
        "omega_dot": 0.0,
        "sim_time": float(timestamp),
        "omega": float(omega),
        "v_lateral": 0.1,
        "ax_imu": float(ax),
        "steering_angle": 0.08,
        "wheel_omegas": (11.0, 11.0, 11.0, 11.0),
    }
    result.update(extras)
    return result


class EstimatorBayesTest(unittest.TestCase):
    def test_replay_ablation_alias_forces_only_the_yaw_gate_off(self):
        config = ReplayConfig(dynamics_min_yaw_rate_rms=0.015)
        gated = make_replay_estimator("scalar_parent", config)
        ungated = make_replay_estimator("grit_ungated", config)
        self.assertAlmostEqual(gated._min_yaw_rate_rms, 0.015)
        self.assertEqual(ungated._min_yaw_rate_rms, 0.0)
        self.assertEqual(gated._horizon, ungated._horizon)
        self.assertEqual(gated._min_windows, ungated._min_windows)

    def test_constructing_estimator_does_not_import_benchmark_module(self):
        code = """
import sys
from simulation.estimators.scalar_parent_terrain_estimator import (
    ScalarParentTerrainEstimator,
)
estimator = ScalarParentTerrainEstimator(
    initial_terrain={"n": 0.7},
    dynamics_prediction=lambda sample, grid: (grid, grid, grid),
)
assert estimator.get_bekker_n() == 0.7
assert "ukf_reference_models" not in sys.modules
assert "matplotlib.pyplot" not in sys.modules
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_projection_matches_historical_nn_wrench(self):
        from simulation.estimators.ukf_reference_models import (
            Vehicle,
            _4wheel_body_wrench,
            manifold_soil_from_n,
        )

        vehicle_values = {
            "M": 2573.0,
            "Izz": 3570.0,
            "Lf": 1.593,
            "Lr": 1.709,
            "T": 1.8194,
            "h_cg": 0.65,
        }
        production_vehicle = ProjectionVehicle.from_mapping(vehicle_values)
        legacy_vehicle = Vehicle(
            m=vehicle_values["M"],
            Iz=vehicle_values["Izz"],
            Lf=vehicle_values["Lf"],
            Lr=vehicle_values["Lr"],
            track=vehicle_values["T"],
            h_cg=vehicle_values["h_cg"],
        )
        model_dir = ROOT / "nn_models" / "tire_force_static_parent"
        cases = (
            (
                np.asarray([0.0, 0.0, 0.0, 5.2, 0.25, 0.12, 0.83]),
                0.17,
                0.35,
                {
                    "ay_in": 0.72,
                    "kappa_in": 0.13,
                    "steering_rate_in": -0.21,
                    "rear_steering_rate_in": 0.08,
                    "Fz_front_mean": 5600.0,
                    "Fz_rear_mean": 6100.0,
                },
            ),
            (
                np.asarray([0.0, 0.0, 0.0, 6.1, -0.18, -0.09, 0.61]),
                -0.12,
                -0.28,
                {
                    "ay_in": -0.45,
                    "wheel_omegas": (14.2, 13.7, 14.5, 13.9),
                    "tire_radius": 0.47,
                    "Fz_front_mean": 5900.0,
                    "Fz_rear_mean": 5850.0,
                },
            ),
            (
                np.asarray([0.0, 0.0, 0.0, 4.4, 0.31, 0.16, 1.02]),
                0.21,
                0.41,
                {"kappa_in": 0.05},
            ),
        )
        for state, delta, ax_in, options in cases:
            n_value = float(state[6])
            terrain = terrain_params_for_n(n_value)
            projector = FourWheelProjector(model_dir, terrain)
            expected = _4wheel_body_wrench(
                state,
                delta,
                ax_in,
                manifold_soil_from_n(n_value),
                legacy_vehicle,
                backend="nn",
                tire_model_dir=str(model_dir),
                **options,
            )
            actual = projector.body_wrench(
                state,
                delta,
                ax_in,
                terrain,
                production_vehicle,
                **options,
            )
            np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-8)

    def test_projection_keeps_left_right_longitudinal_yaw_terms(self):
        class SequencedModel:
            def __init__(self):
                self.outputs = iter(((10.0, 1.0), (20.0, 2.0),
                                     (30.0, 3.0), (40.0, 4.0)))

            def predict_numeric(self, **_kwargs):
                return next(self.outputs)

        vehicle = ProjectionVehicle(
            m=1000.0,
            Iz=1000.0,
            Lf=1.5,
            Lr=1.0,
            track=2.0,
            h_cg=0.5,
        )
        force_x, force_y, moment = four_wheel_body_wrench(
            [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.7],
            0.0,
            0.0,
            terrain_params_for_n(0.7),
            vehicle,
            SequencedModel(),
            ay_in=0.0,
            Fz_front_mean=5000.0,
            Fz_rear_mean=5000.0,
        )
        self.assertAlmostEqual(force_x, 100.0)
        self.assertAlmostEqual(force_y, 10.0)
        self.assertAlmostEqual(moment, 17.5)

    def test_numpy_wheel_speed_array_is_accepted(self):
        estimator = make_estimator(min_windows=99)
        sample = observation(0.0)
        sample["wheel_omegas"] = np.asarray([11.0, 11.1, 11.2, 11.3])
        self.assertTrue(estimator.observe(**sample))
        self.assertEqual(
            estimator._raw_block[0]["wheel_omegas"],
            (11.0, 11.1, 11.2, 11.3),
        )

    def test_block_timespan_gate_scales_with_block_duration(self):
        estimator = make_estimator(
            block_dt=1.0,
            min_windows=99,
            min_window_samples=4,
        )
        for timestamp in (0.0, 0.1, 0.2, 0.3, 1.0):
            self.assertTrue(estimator.observe(**observation(timestamp)))
        self.assertEqual(estimator.rejected_dynamics_windows, 1)
        self.assertEqual(estimator.accepted_dynamics_windows, 0)

        for timestamp in (1.2, 1.4, 1.6, 2.0):
            self.assertTrue(estimator.observe(**observation(timestamp)))
        self.assertEqual(estimator.rejected_dynamics_windows, 1)
        self.assertEqual(estimator.accepted_dynamics_windows, 1)

    def test_rejected_blocks_deactivate_expired_short_history(self):
        estimator = make_estimator(horizon=1.0, min_windows=2)
        for timestamp in np.arange(0.0, 1.01, 0.1):
            self.assertTrue(estimator.observe(**observation(timestamp)))
        self.assertTrue(estimator.dynamics_active)
        self.assertEqual(estimator.dynamics_windows, 2)

        for timestamp in (1.1, 1.2, 1.3, 1.4, 1.5):
            self.assertTrue(estimator.observe(**observation(timestamp, u=11.0)))
        self.assertEqual(estimator.rejected_dynamics_windows, 1)
        self.assertEqual(estimator.dynamics_windows, 1)
        self.assertFalse(estimator.dynamics_active)

    def test_boundary_sample_starts_the_next_causal_block(self):
        projected_samples = []

        def capture_prediction(sample, n_grid):
            projected_samples.append(dict(sample))
            grid = np.asarray(n_grid, dtype=float)
            return grid, np.zeros_like(grid), np.zeros_like(grid)

        estimator = make_estimator(
            capture_prediction,
            min_windows=99,
            r_ax=1.0,
        )
        for timestamp in (0.0, 0.1, 0.2, 0.3, 0.4):
            self.assertTrue(estimator.observe(**observation(timestamp, ax=1.0)))
        self.assertEqual(projected_samples, [])

        # The 0.5 s sample triggers finalization before it is appended.  It
        # must therefore be absent from the [0.0, 0.5) block.
        self.assertTrue(estimator.observe(**observation(0.5, ax=2.0)))
        self.assertEqual(len(projected_samples), 1)
        self.assertEqual(projected_samples[0]["ax"], 1.0)
        self.assertEqual(estimator.accepted_dynamics_windows, 1)
        self.assertEqual(len(estimator._raw_block), 1)
        self.assertEqual(estimator._raw_block[0]["time"], 0.5)

        for timestamp in (0.6, 0.7, 0.8, 0.9):
            self.assertTrue(estimator.observe(**observation(timestamp, ax=2.0)))
        self.assertTrue(estimator.observe(**observation(1.0, ax=3.0)))
        self.assertEqual([sample["ax"] for sample in projected_samples], [1.0, 2.0])
        self.assertEqual(estimator.accepted_dynamics_windows, 2)
        self.assertEqual(len(estimator._raw_block), 1)
        self.assertEqual(estimator._raw_block[0]["time"], 1.0)

    def test_height_and_oracle_extras_do_not_affect_evidence(self):
        reference_samples = []
        contaminated_samples = []

        def predictor(captured):
            def predict(sample, n_grid):
                captured.append(dict(sample))
                return hypothesis_prediction(sample, n_grid)

            return predict

        reference = make_estimator(predictor(reference_samples))
        contaminated = make_estimator(predictor(contaminated_samples))
        for timestamp in np.arange(0.0, 1.01, 0.1):
            self.assertTrue(reference.observe(**observation(timestamp)))
            self.assertTrue(contaminated.observe(**observation(
                timestamp,
                wheel_center_heights=(1.0e9, -1.0e9, np.nan, np.inf),
                ground_datum_z=-1.0e12,
                z_cg=1.0e12,
                n_true=0.5,
                terrain_truth={"n": 1.1, "phi": 37.8},
                soil={"n": 0.5},
                tire_forces=(1.0e20,) * 4,
            )))

        np.testing.assert_allclose(reference._posterior, contaminated._posterior)
        self.assertEqual(reference.get_bekker_n(), contaminated.get_bekker_n())
        self.assertEqual(len(reference_samples), len(contaminated_samples))
        forbidden = {
            "wheel_center_heights",
            "ground_datum_z",
            "z_cg",
            "n_true",
            "terrain_truth",
            "soil",
            "tire_forces",
        }
        for sample in contaminated_samples:
            self.assertTrue(forbidden.isdisjoint(sample))

    def test_out_of_rig_envelope_block_is_counted_and_rejected(self):
        estimator = make_estimator(min_windows=1)
        for timestamp in np.arange(0.0, 0.51, 0.1):
            self.assertTrue(estimator.observe(**observation(
                timestamp, u=11.0
            )))

        self.assertEqual(estimator.accepted_dynamics_windows, 0)
        self.assertEqual(estimator.rejected_dynamics_windows, 1)
        self.assertEqual(estimator.feature_envelope_excursions, 1)
        self.assertFalse(estimator.should_update())

    def test_gyro_excitation_gate_rejects_straight_block_without_projection(self):
        projected_samples = []

        def capture_prediction(sample, n_grid):
            projected_samples.append(dict(sample))
            return hypothesis_prediction(sample, n_grid)

        estimator = make_estimator(
            capture_prediction,
            min_windows=1,
            min_yaw_rate_rms=0.02,
        )
        for timestamp in np.arange(0.0, 0.51, 0.1):
            self.assertTrue(estimator.observe(**observation(
                timestamp, omega=0.01
            )))

        self.assertEqual(projected_samples, [])
        self.assertEqual(estimator.accepted_dynamics_windows, 0)
        self.assertEqual(estimator.rejected_dynamics_windows, 1)
        self.assertEqual(estimator.kinematic_excitation_rejections, 1)
        self.assertTrue(np.isnan(estimator.last_informative_time))
        self.assertFalse(estimator.should_update())

    def test_gyro_excitation_gate_accepts_turning_block_and_records_time(self):
        estimator = make_estimator(min_windows=1, min_yaw_rate_rms=0.02)
        for timestamp in np.arange(0.0, 0.51, 0.1):
            self.assertTrue(estimator.observe(**observation(
                timestamp, omega=0.03
            )))

        self.assertEqual(estimator.accepted_dynamics_windows, 1)
        self.assertEqual(estimator.kinematic_excitation_rejections, 0)
        self.assertAlmostEqual(estimator.last_informative_time, 0.2)
        self.assertTrue(estimator.should_update())

    def test_readiness_holds_prior_until_twelve_accepted_blocks(self):
        estimator = make_estimator(min_windows=12)
        for timestamp in np.arange(0.0, 6.0, 0.1):
            self.assertTrue(estimator.observe(**observation(timestamp, omega=0.1)))

        self.assertEqual(estimator.accepted_dynamics_windows, 11)
        self.assertFalse(estimator.dynamics_active)
        self.assertFalse(estimator.should_update())
        self.assertAlmostEqual(estimator.get_bekker_n(), 0.7)

        self.assertTrue(estimator.observe(**observation(6.0, omega=0.1)))
        self.assertEqual(estimator.accepted_dynamics_windows, 12)
        self.assertTrue(estimator.dynamics_active)
        self.assertTrue(estimator.should_update())

    def test_mocked_dynamics_identify_the_correct_grid_hypothesis(self):
        estimator = make_estimator()
        true_n = 0.9
        for timestamp in np.arange(0.0, 1.01, 0.1):
            self.assertTrue(estimator.observe(**observation(
                timestamp,
                ax=true_n,
                ay=2.0 * true_n,
                omega=0.5 * true_n * timestamp,
            )))

        self.assertTrue(estimator.dynamics_active)
        self.assertEqual(estimator.dynamics_windows, 2)
        self.assertTrue(estimator.should_update())
        self.assertAlmostEqual(
            estimator._grid[int(np.argmax(estimator._posterior))], true_n
        )
        # Quadratic sub-grid interpolation can move a fraction of one grid
        # interval even when the discrete MAP lands exactly on true_n.
        self.assertAlmostEqual(estimator.get_bekker_n(), true_n, places=3)
        parameters, _confidence = estimator.estimate()
        self.assertAlmostEqual(parameters["n"], true_n, places=3)

    def test_uncertainty_uses_weighted_mean_and_piecewise_phi_map(self):
        estimator = make_estimator(min_windows=99)
        estimator._posterior[:] = 0.0
        low_index = int(np.argmin(abs(estimator._grid - 0.55)))
        high_index = int(np.argmin(abs(estimator._grid - 0.95)))
        estimator._posterior[low_index] = 0.25
        estimator._posterior[high_index] = 0.75
        estimator._n_posterior = float(np.sum(
            estimator._grid * estimator._posterior
        ))
        estimator._n_smooth = float(estimator._grid[low_index])

        expected_n_sigma = float(np.sqrt(np.sum(
            estimator._posterior
            * (estimator._grid - estimator._n_posterior) ** 2
        )))
        phi_grid = np.asarray([
            terrain_params_for_n(float(n_value))["phi"]
            for n_value in estimator._grid
        ])
        phi_mean = float(np.sum(estimator._posterior * phi_grid))
        expected_phi_sigma = float(np.sqrt(np.sum(
            estimator._posterior * (phi_grid - phi_mean) ** 2
        )))

        self.assertAlmostEqual(estimator.get_n_uncertainty(), expected_n_sigma)
        self.assertAlmostEqual(
            estimator.get_phi_uncertainty_deg(), expected_phi_sigma
        )
        self.assertNotAlmostEqual(
            estimator.get_phi_uncertainty_deg(), 7.0 * expected_n_sigma
        )


if __name__ == "__main__":
    unittest.main()
