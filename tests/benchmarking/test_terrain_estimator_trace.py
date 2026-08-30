#!/usr/bin/env python3
"""Contracts for sensor-only trace collection, replay, and scoring.

The estimator evidence rests on every method seeing the same recorded sensor
stream and no plant truth, so these tests establish that the trace schema
admits nothing else: oracle columns are rejected in both traces and replay
manifests, unknown fields are refused rather than dropped, non-monotonic time
is rejected, and the exact logger's header matches the schema the sanitizer
enforces.
"""

from __future__ import annotations

import ast
import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
HERE = ROOT_DIR / "benchmarking"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collect_terrain_estimator_traces import (
    INDEPENDENT_COHESION_MULTIPLIER_HI,
    INDEPENDENT_COHESION_MULTIPLIER_LO,
    INDEPENDENT_N_HI,
    INDEPENDENT_N_LO,
    INDEPENDENT_PHI_HI_DEG,
    INDEPENDENT_PHI_LO_DEG,
    TraceTask,
    _audit_probe_maneuver,
    _base_result,
    _initialize_collection,
    _load_recovery,
    _portable_collection_path,
    _write_attempt_tables,
    fixed_controller_extra_args,
    generate_soils,
    manifold_yaml_from_n,
    parse_args as collector_parse_args,
)
from terrain_estimator_replay import (
    BACKEND_LABELS,
    ReplayConfig,
    parse_args as parse_replay_args,
    replay_trace,
    resolve_manifest_trace_path,
)
from terrain_estimator_trace import (
    TRACE_COLUMNS,
    TraceValidationError,
    load_sensor_trace,
    reject_oracle_columns,
    sanitize_approximate_diagnostic,
    sanitize_exact_observations,
    sha256_file,
    validate_sensor_trace_frame,
)


def valid_trace_frame(rows: int = 5) -> pd.DataFrame:
    values = {column: np.zeros(rows, dtype=float) for column in TRACE_COLUMNS}
    values["seq"] = np.arange(rows, dtype=float)
    values["sim_time"] = 0.1 * np.arange(rows, dtype=float)
    values["quat_e0"] = np.ones(rows, dtype=float)
    values["u_raw"] = np.full(rows, 4.0)
    values["u"] = np.full(rows, 4.0)
    values["v_lateral"] = np.linspace(0.0, 0.1, rows)
    values["omega"] = np.linspace(0.0, 0.05, rows)
    values["omega_raw"] = np.linspace(0.001, 0.051, rows)
    values["wheel_omega_fl"] = np.full(rows, 8.5)
    values["wheel_omega_fr"] = np.full(rows, 8.6)
    values["wheel_omega_rl"] = np.full(rows, 8.7)
    values["wheel_omega_rr"] = np.full(rows, 8.8)
    values["Fz_f"] = np.full(rows, 6500.0)
    values["Fz_r"] = np.full(rows, 6100.0)
    return pd.DataFrame(values, columns=TRACE_COLUMNS)


class SensorTraceContractTest(unittest.TestCase):
    def test_controller_exact_logger_header_matches_schema(self):
        controller = HERE.parent / "simulation" / "control" / "acados_mpc_controller_node.py"
        tree = ast.parse(controller.read_text(encoding="utf-8"))
        headers = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "writerow" or not isinstance(node.func.value, ast.Name):
                continue
            if node.func.value.id != "terrain_obs_writer" or not node.args:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.List) and all(
                isinstance(element, ast.Constant) and isinstance(element.value, str)
                for element in argument.elts
            ):
                headers.append(tuple(element.value for element in argument.elts))
        self.assertEqual(headers, [TRACE_COLUMNS])

    def test_exact_sanitizer_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "terrain_observations.csv"
            destination = root / "sensor_trace.csv"
            frame = valid_trace_frame()
            frame["logger_version"] = 1  # harmless source-only bookkeeping
            frame.to_csv(source, index=False, float_format="%.17g")
            metadata = sanitize_exact_observations(source, destination)
            loaded = load_sensor_trace(destination)
            self.assertEqual(list(loaded.columns), list(TRACE_COLUMNS))
            self.assertEqual(metadata["trace_rows"], len(frame))
            self.assertEqual(metadata["trace_sha256"], sha256_file(destination))

    def test_oracle_column_is_rejected(self):
        frame = valid_trace_frame()
        frame["n_true"] = 0.8
        with self.assertRaisesRegex(TraceValidationError, "oracle"):
            validate_sensor_trace_frame(frame)

    def test_replay_manifest_oracle_column_is_rejected(self):
        with self.assertRaisesRegex(TraceValidationError, "oracle"):
            reject_oracle_columns(
                ["trace_id", "trace_path", "n_true"], context="trace manifest"
            )

    def test_exact_source_rejects_unknown_extra_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "terrain_observations.csv"
            frame = valid_trace_frame()
            frame["mystery_label"] = 123.0
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(TraceValidationError, "non-schema"):
                sanitize_exact_observations(source, root / "sensor_trace.csv")

    def test_nonmonotonic_time_is_rejected(self):
        frame = valid_trace_frame()
        frame.loc[3, "sim_time"] = frame.loc[2, "sim_time"]
        with self.assertRaisesRegex(TraceValidationError, "strictly increasing"):
            validate_sensor_trace_frame(frame)

    def test_nonfinite_sensor_value_is_rejected(self):
        frame = valid_trace_frame()
        frame.loc[2, "ay_imu"] = np.nan
        with self.assertRaisesRegex(TraceValidationError, "non-finite"):
            validate_sensor_trace_frame(frame)

    def test_approximate_conversion_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "diag.csv"
            output = Path(directory) / "sensor_trace.csv"
            with self.assertRaisesRegex(TraceValidationError, "disabled"):
                sanitize_approximate_diagnostic(source, output)

    def test_approximate_conversion_whitelists_diagnostic_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "diag.csv"
            output = root / "sensor_trace.csv"
            rows = 4
            diagnostic = pd.DataFrame(
                {
                    "seq": np.arange(rows),
                    "sim_time": 0.1 * np.arange(rows),
                    "x_fa_meas": np.linspace(1.6, 2.8, rows),
                    "y_fa_meas": np.zeros(rows),
                    "psi_meas": np.zeros(rows),
                    "u_meas": np.full(rows, 4.0),
                    "v_meas": np.zeros(rows),
                    "omega_meas": np.zeros(rows),
                    "ax_imu": np.zeros(rows),
                    "ay_imu": np.zeros(rows),
                    "z_cg": np.ones(rows),
                    "quat_e0": np.ones(rows),
                    "quat_e1": np.zeros(rows),
                    "quat_e2": np.zeros(rows),
                    "quat_e3": np.zeros(rows),
                    "az_imu": np.full(rows, 9.81),
                    "omega_x_imu": np.zeros(rows),
                    "omega_y_imu": np.zeros(rows),
                    "wheel_omega_fl": np.full(rows, 8.5),
                    "wheel_omega_fr": np.full(rows, 8.5),
                    "wheel_omega_rl": np.full(rows, 8.5),
                    "wheel_omega_rr": np.full(rows, 8.5),
                    "steering_angle_sensor": np.zeros(rows),
                    "est_alpha_f": np.zeros(rows),
                    "est_alpha_r": np.zeros(rows),
                    "est_Fz_f_mean": np.full(rows, 6500.0),
                    "est_Fz_r_mean": np.full(rows, 6100.0),
                    "est_kappa": np.zeros(rows),
                    "est_alpha_rate_f": np.zeros(rows),
                    "est_alpha_rate_r": np.zeros(rows),
                    "est_u_safe": np.full(rows, 4.0),
                    "n_true": np.full(rows, 0.8),
                    "actual_Fy_front": np.full(rows, 1234.0),
                }
            )
            diagnostic.to_csv(source, index=False)
            metadata = sanitize_approximate_diagnostic(
                source, output, allow_approximate=True
            )
            loaded = load_sensor_trace(output)
            self.assertNotIn("n_true", loaded.columns)
            self.assertNotIn("actual_Fy_front", loaded.columns)
            self.assertEqual(metadata["trace_quality"], "approximate_rounded_diagnostic")


class FixedCollectionContractTest(unittest.TestCase):
    @staticmethod
    def _probe_diagnostic(phases, reasons=None):
        count = len(phases)
        return pd.DataFrame(
            {
                "sim_time": 0.1 * np.arange(count),
                "terrain_probe_phase": phases,
                "terrain_probe_reason": reasons or [""] * count,
                "est_alpha_f": np.linspace(0.13, -0.13, count),
                "est_alpha_rate_f": np.linspace(0.2, -0.2, count),
                "crosstrack_err": np.linspace(0.0, 0.2, count),
                "ay_imu": np.linspace(0.0, 1.0, count),
            }
        )

    def test_probe_audit_requires_both_signs_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diag.csv"
            self._probe_diagnostic(
                [
                    "idle",
                    "arming",
                    "ramp_positive",
                    "hold_positive",
                    "ramp_negative",
                    "hold_negative",
                    "recovery",
                    "complete",
                ]
            ).to_csv(path, index=False)
            audit = _audit_probe_maneuver(path)
            self.assertTrue(audit["probe_audit_ok"])
            self.assertEqual(audit["probe_final_phase"], "complete")
            self.assertEqual(audit["probe_positive_hold_samples"], 1)
            self.assertEqual(audit["probe_negative_hold_samples"], 1)

    def test_probe_audit_rejects_normal_run_with_aborted_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diag.csv"
            self._probe_diagnostic(
                [
                    "idle",
                    "arming",
                    "ramp_positive",
                    "hold_positive",
                    "aborting",
                    "aborted",
                ],
                ["", "", "", "", "cross_track_limit", "cross_track_limit"],
            ).to_csv(path, index=False)
            audit = _audit_probe_maneuver(path)
            self.assertFalse(audit["probe_audit_ok"])
            self.assertEqual(audit["probe_abort_reason"], "cross_track_limit")
            self.assertIn("hold_negative", audit["probe_audit_failure"])

    def test_fixed_collection_has_no_estimator_argument(self):
        task = TraceTask(
            trace_id="trace_0000",
            yaml_path="soil.yaml",
            run_dir="run",
            sim_port=18000,
            ctrl_port=18001,
            sim_seed=720,
            path="sinusoidal",
            speed=5.0,
            sim_time=20.0,
            lead_in=5.0,
            allow_approx_diag=False,
        )
        arguments = fixed_controller_extra_args(task)
        self.assertIn("--controller-prior-terrain", arguments)
        self.assertIn("dirt", arguments)
        self.assertNotIn("--terrain-estimator", arguments)
        self.assertNotIn("--terrain-estimator-backend", arguments)
        self.assertNotIn("--terrain-id-probe", arguments)

    def test_probe_collection_forwards_sensor_gated_maneuver(self):
        task = TraceTask(
            trace_id="trace_0000",
            yaml_path="soil.yaml",
            run_dir="run",
            sim_port=18000,
            ctrl_port=18001,
            sim_seed=720,
            path="sinusoidal",
            speed=5.0,
            sim_time=30.0,
            lead_in=30.0,
            allow_approx_diag=False,
            maneuver_label="terrain_id_probe",
            terrain_id_probe=True,
            probe_target_alpha=0.19,
            probe_slew_rate=0.28,
            probe_signed_dwell=0.8,
            probe_clearance=40.0,
            probe_max_latency=0.25,
        )
        arguments = fixed_controller_extra_args(task)
        self.assertIn("--terrain-id-probe", arguments)
        self.assertIn("--terrain-id-probe-target-alpha", arguments)
        self.assertIn("0.19", arguments)
        self.assertIn("--terrain-id-probe-max-latency", arguments)
        self.assertNotIn("--excitation-steer-amp", arguments)

    def test_independent_n_phi_soils_are_stratified_and_reproducible(self):
        count = 12
        first, first_seed = generate_soils(
            count,
            mode="independent_n_phi",
            seed=31415,
            jitter_fraction=0.99,
        )
        second, second_seed = generate_soils(
            count,
            mode="independent_n_phi",
            seed=31415,
            jitter_fraction=0.0,
        )
        self.assertEqual(first_seed, second_seed)
        self.assertEqual(first, second)

        n_edges = np.linspace(INDEPENDENT_N_LO, INDEPENDENT_N_HI, count + 1)
        phi_edges = np.linspace(
            INDEPENDENT_PHI_LO_DEG, INDEPENDENT_PHI_HI_DEG, count + 1
        )
        n_values = np.asarray([n_true for n_true, _ in first])
        phi_values = np.sort([
            float(soil["friction_angle"]) for _, soil in first
        ])
        for index in range(count):
            self.assertGreaterEqual(n_values[index], n_edges[index])
            self.assertLessEqual(n_values[index], n_edges[index + 1])
            self.assertGreaterEqual(phi_values[index], phi_edges[index])
            self.assertLessEqual(phi_values[index], phi_edges[index + 1])

        for n_true, soil in first:
            manifold = manifold_yaml_from_n(n_true)
            for key in ("Kphi", "Kc", "janosi_shear"):
                self.assertEqual(soil[key], manifold[key])
            multiplier = float(soil["cohesion"] / manifold["cohesion"])
            self.assertGreaterEqual(
                multiplier, INDEPENDENT_COHESION_MULTIPLIER_LO
            )
            self.assertLessEqual(
                multiplier, INDEPENDENT_COHESION_MULTIPLIER_HI
            )

    def test_independent_n_phi_truth_matches_written_soil_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._recovery_args(root)
            args.n = 5
            args.mode = "independent_n_phi"
            _initialize_collection(root, args)
            truth = pd.read_csv(
                root / "truth.csv", float_precision="round_trip"
            ).set_index("trace_id")
            self.assertEqual(set(truth["soil_mode"]), {"independent_n_phi"})
            self.assertTrue((truth["nuisance_jitter_fraction"] == 0.0).all())
            self.assertTrue(
                (truth["cohesion_jitter_fraction"] == 0.15).all()
            )
            for trace_id, row in truth.iterrows():
                index = int(trace_id.rsplit("_", 1)[1])
                soil = yaml.safe_load(
                    (
                        root
                        / "truth_soils"
                        / f"case_{index:04d}.yaml"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(row["n_true"], soil["n"])
                self.assertEqual(row["Kphi_true"], soil["Kphi"])
                self.assertEqual(row["Kc_true"], soil["Kc"])
                self.assertEqual(row["c_true"], soil["cohesion"])
                self.assertEqual(
                    row["phi_true_deg"], soil["friction_angle"]
                )
                self.assertEqual(row["k_true"], soil["janosi_shear"])
                manifold = manifold_yaml_from_n(float(row["n_true"]))
                self.assertAlmostEqual(
                    row["cohesion_multiplier_true"],
                    soil["cohesion"] / manifold["cohesion"],
                )

    @staticmethod
    def _recovery_args(root: Path) -> argparse.Namespace:
        # Start from the collector's own parser defaults so the fixture can
        # never drift when the collector grows an argument, then pin only the
        # fields this contract test exercises.
        args = collector_parse_args([])
        args.n = 2
        args.workers = 1
        args.base_port = 18000
        args.seed = 2468
        args.sim_seed = 720
        args.mode = "manifold"
        args.jitter = 0.10
        args.path = "sinusoidal"
        args.speed = 5.0
        args.time = 14.0
        args.lead_in = 5.0
        args.wheel_center_noise_std = 0.01
        args.wheel_center_calibration_bias_std = 0.003
        args.output_dir = root
        args.quick = False
        args.recover = False
        return args

    def test_recovery_keeps_seed_ports_soil_and_successful_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._recovery_args(root)
            tasks = _initialize_collection(root, args)
            rows = []
            for task, status in zip(tasks, ("ok", "run_failed")):
                row = _base_result(task)
                row["status"] = status
                row["failure"] = "" if status == "ok" else "infrastructure"
                rows.append(row)
            latest = {str(row["trace_id"]): row for row in rows}
            attempts = [
                {**row, "recorded_at": "now", "recovery": False}
                for row in rows
            ]
            _write_attempt_tables(root, latest, attempts)

            args.recover = True
            recovered_latest, recovered_attempts, recovery_tasks = (
                _load_recovery(root, args)
            )
            self.assertEqual(set(recovered_latest), {"trace_0000", "trace_0001"})
            self.assertEqual(len(recovered_attempts), 2)
            self.assertEqual(len(recovery_tasks), 1)
            task = recovery_tasks[0]
            self.assertEqual(task.trace_id, "trace_0001")
            self.assertEqual(task.attempt, 2)
            self.assertTrue(task.recovery)
            self.assertEqual(task.sim_seed, 721)
            self.assertEqual((task.sim_port, task.ctrl_port), (18002, 18003))
            self.assertEqual(
                Path(task.yaml_path),
                (root / "truth_soils" / "case_0001.yaml").resolve(),
            )

            latest["trace_0001"]["status"] = "protocol_violation"
            _write_attempt_tables(root, latest, attempts)
            with self.assertRaisesRegex(SystemExit, "not recoverable"):
                _load_recovery(root, args)


class FakeEstimator:
    def __init__(self):
        self.n = 0.7
        self.phi_deg = 29.0
        self.samples = 0
        self.omega_inputs: list[float] = []
        self.observations: list[dict] = []

    def estimate_omega_dot(self, omega, timestamp):
        self.omega_inputs.append(float(omega))
        return 0.0

    def observe(self, **kwargs):
        self.observations.append(kwargs)
        self.samples += 1
        self.n += 0.01
        self.phi_deg += 0.1
        return True

    def should_update(self):
        return self.samples > 0 and self.samples % 2 == 0

    def estimate(self):
        return {"n": self.n, "phi": self.phi_deg}, 1.0

    def get_bekker_n(self):
        return self.n

    def get_n_uncertainty(self):
        return 0.05

    def get_friction_angle_deg(self):
        return self.phi_deg

    def get_phi_uncertainty_deg(self):
        return 1.5


class ReplayContractTest(unittest.TestCase):
    def test_relative_trace_path_is_based_at_manifest_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "collection" / "trace_manifest.csv"
            expected = root / "collection" / "traces" / "trace_0000.csv"
            self.assertEqual(
                resolve_manifest_trace_path("traces/trace_0000.csv", manifest),
                expected.resolve(),
            )

    def test_collector_makes_in_collection_paths_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            trace = root / "raw" / "trace_0000" / "sensor_trace.csv"
            self.assertEqual(
                _portable_collection_path(str(trace), root),
                "raw/trace_0000/sensor_trace.csv",
            )

    def test_replay_uses_raw_gyro_and_preprocessed_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor_trace.csv"
            frame = valid_trace_frame()
            frame.to_csv(path, index=False, float_format="%.17g")
            fake = FakeEstimator()
            summary, time_series = replay_trace(
                path,
                "scalar_parent",
                ReplayConfig(tail_start=0.2),
                trace_id="trace_0000",
                expected_sha256=sha256_file(path),
                estimator=fake,
            )
            np.testing.assert_allclose(fake.omega_inputs, frame["omega_raw"])
            self.assertAlmostEqual(
                fake.observations[-1]["v_lateral"], frame["v_lateral"].iloc[-1]
            )
            self.assertEqual(summary["trace_sha256"], sha256_file(path))
            self.assertEqual(len(time_series), len(frame))
            self.assertGreater(summary["est_n"], 0.7)
            self.assertGreater(summary["est_phi_deg"], 29.0)
            self.assertIn("phi_internal_deg", time_series)
            self.assertIn("phi_published_deg", time_series)
            self.assertIn("phi_sigma_deg", time_series)
            np.testing.assert_allclose(time_series["x_pos"], frame["x_cg"])
            np.testing.assert_allclose(time_series["y_pos"], frame["y_cg"])
            np.testing.assert_allclose(time_series["psi"], frame["psi"])

    def test_replay_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor_trace.csv"
            valid_trace_frame().to_csv(path, index=False)
            with self.assertRaisesRegex(TraceValidationError, "hash mismatch"):
                replay_trace(
                    path,
                    "scalar_parent",
                    ReplayConfig(tail_start=0.1),
                    expected_sha256="0" * 64,
                    estimator=FakeEstimator(),
                )


if __name__ == "__main__":
    unittest.main()
