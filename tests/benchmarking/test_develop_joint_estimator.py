#!/usr/bin/env python3
"""Contracts for the truth-blind joint-estimator replay driver.

The driver's isolation from truth is what makes its output admissible, so
these tests establish that isolation directly: its command line exposes no
truth or scoring argument, its manifests reject any trace carrying oracle
columns, its force model must have single-tire rig provenance, and only a
complete replay from a clean worktree is marked eligible for promotion
scoring.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import pandas as pd

from benchmarking.develop_joint_estimator import (
    BACKEND,
    CandidateConfig,
    DevelopmentTask,
    _accepted_snapshot,
    _run_one,
    _validate_rig_model,
    _write_estimates_atomic,
    _write_manifest,
    load_trace_manifest,
    parse_args,
)
from benchmarking.terrain_estimator_replay import ReplayConfig
from benchmarking.terrain_estimator_trace import TRACE_SCHEMA_VERSION, sha256_file


class _ValidCandidate:
    joint_active = True
    joint_has_estimate = True
    joint_updates = 3
    joint_projection_failures = 0
    joint_information_kl = 1.2
    n_information_kl = 0.5
    phi_information_kl = 0.7
    cohesion_information_kl = 0.0
    profile_cost_span = 9.0
    joint_effective_sample_size = 12.0
    n_effective_sample_size = 4.0
    phi_effective_sample_size = 5.0
    cohesion_effective_sample_size = 1.0
    n_boundary_mass = 0.02
    phi_boundary_mass = 0.03
    cohesion_boundary_mass = 0.0
    boundary_limited = False
    observability_rank = 2
    observability_min_singular_value = 0.4
    observability_max_singular_value = 5.0
    observability_condition = 12.5
    nuisance_projection_rank = 3
    nuisance_projection_condition = 4.0
    last_likelihood_block_count = 12
    last_likelihood_residual_count = 24
    duplicate_likelihood_block_count = 0
    duplicate_likelihood_update_count = 0
    likelihood_evaluations = 7
    joint_information_rejections = 0
    n_information_rejections = 0
    phi_information_rejections = 0
    observability_rejections = 0
    gate_rejected_updates = 0
    update_acceptance_fraction = 3.0 / 7.0
    last_joint_update_time = 7.5
    load_transfer_mode = "static"
    measured_load_blocks = 0
    static_load_blocks = 16
    lagged_load_blocks = 0
    lagged_load_fallback_blocks = 0
    last_effective_front_load = 5_700.0
    last_effective_rear_load = 5_900.0
    last_effective_load_ax = 0.0
    last_effective_load_ay = 0.0
    cohesion_enabled = False
    cohesion_multiplier_estimate = 1.0
    cohesion_multiplier_uncertainty = 0.0
    snapshot_confidence = 0.8

    def get_last_accepted_snapshot(self):
        terrain = MappingProxyType(
            {
                "Kphi": 2.0e6,
                "Kc": 1.0e4,
                "n": 0.8,
                "c": 2.0e3,
                "phi": 24.0,
                "k": 0.02,
            }
        )
        boundary_masses = (
            float(self.n_boundary_mass),
            float(self.phi_boundary_mass),
            float(self.cohesion_boundary_mass),
        )
        return MappingProxyType(
            {
                "snapshot_version": "grit_accepted",
                "update_seq": int(self.joint_updates),
                "evidence_time_s": float(self.last_joint_update_time),
                "n": 0.8,
                "phi_deg": 24.0,
                "terrain_params": terrain,
                "confidence": float(self.snapshot_confidence),
                "n_sigma": 0.05,
                "phi_sigma_deg": 1.0,
                "joint_information_kl": float(self.joint_information_kl),
                "n_information_kl": float(self.n_information_kl),
                "phi_information_kl": float(self.phi_information_kl),
                "cohesion_information_kl": float(
                    self.cohesion_information_kl
                ),
                "observability_rank": int(self.observability_rank),
                "observability_min_singular_value": float(
                    self.observability_min_singular_value
                ),
                "n_boundary_mass": boundary_masses[0],
                "phi_boundary_mass": boundary_masses[1],
                "cohesion_boundary_mass": boundary_masses[2],
                "max_boundary_mass": max(boundary_masses),
                "boundary_limited": bool(self.boundary_limited),
                "joint_projection_failures": int(
                    self.joint_projection_failures
                ),
                "duplicate_likelihood_block_count": int(
                    self.duplicate_likelihood_block_count
                ),
                "duplicate_likelihood_update_count": int(
                    self.duplicate_likelihood_update_count
                ),
                "likelihood_evaluations": int(self.likelihood_evaluations),
                "likelihood_block_count": int(
                    self.last_likelihood_block_count
                ),
                "likelihood_residual_count": int(
                    self.last_likelihood_residual_count
                ),
                "load_transfer_mode": str(self.load_transfer_mode),
                "effective_front_load": float(
                    self.last_effective_front_load
                ),
                "effective_rear_load": float(
                    self.last_effective_rear_load
                ),
                "effective_load_ax": float(self.last_effective_load_ax),
                "effective_load_ay": float(self.last_effective_load_ay),
                "projection_wall_time_s": 0.01,
                "profile_wall_time_s": 0.02,
                "observability_wall_time_s": 0.003,
                "posterior_wall_time_s": 0.004,
                "publication_wall_time_s": 0.001,
                "update_wall_time_s": 0.015,
            }
        )


def _joint_timeline(
    candidate: _ValidCandidate, *, final_time: float
) -> pd.DataFrame:
    snapshot = candidate.get_last_accepted_snapshot()
    published = float(snapshot["confidence"]) >= 0.20
    return pd.DataFrame(
        {
            "sim_time": [0.0, final_time],
            "n_published": [0.7, snapshot["n"] if published else 0.7],
            "phi_published_deg": [
                29.0,
                snapshot["phi_deg"] if published else 29.0,
            ],
            "confidence": [0.0, snapshot["confidence"]],
            "published_update": [0, int(published)],
            "joint_snapshot_advanced": [0, 1],
            "joint_snapshot_seq": ["", snapshot["update_seq"]],
            "joint_snapshot_evidence_time_s": [
                "",
                snapshot["evidence_time_s"],
            ],
            "joint_snapshot_n": ["", snapshot["n"]],
            "joint_snapshot_phi_deg": ["", snapshot["phi_deg"]],
            "joint_snapshot_n_sigma": ["", snapshot["n_sigma"]],
            "joint_snapshot_phi_sigma_deg": [
                "",
                snapshot["phi_sigma_deg"],
            ],
            "joint_snapshot_confidence": ["", snapshot["confidence"]],
            "joint_snapshot_information_kl": [
                "",
                snapshot["joint_information_kl"],
            ],
            "joint_snapshot_observability_rank": [
                "",
                snapshot["observability_rank"],
            ],
            "joint_snapshot_observability_min_singular_value": [
                "",
                snapshot["observability_min_singular_value"],
            ],
            "joint_snapshot_n_boundary_mass": [
                "",
                snapshot["n_boundary_mass"],
            ],
            "joint_snapshot_phi_boundary_mass": [
                "",
                snapshot["phi_boundary_mass"],
            ],
            "joint_snapshot_max_boundary_mass": [
                "",
                snapshot["max_boundary_mass"],
            ],
            "joint_snapshot_boundary_limited": [
                "",
                snapshot["boundary_limited"],
            ],
            "joint_snapshot_projection_wall_time_s": [
                "",
                snapshot["projection_wall_time_s"],
            ],
            "joint_snapshot_profile_wall_time_s": [
                "",
                snapshot["profile_wall_time_s"],
            ],
            "joint_snapshot_observability_wall_time_s": [
                "",
                snapshot["observability_wall_time_s"],
            ],
            "joint_snapshot_posterior_wall_time_s": [
                "",
                snapshot["posterior_wall_time_s"],
            ],
            "joint_snapshot_publication_wall_time_s": [
                "",
                snapshot["publication_wall_time_s"],
            ],
            "joint_snapshot_update_wall_time_s": [
                "",
                snapshot["update_wall_time_s"],
            ],
        }
    )


class DevelopmentJointReplayContractTest(unittest.TestCase):
    def _manifest(self, root: Path, **extra) -> Path:
        trace = root / "sensor_trace.csv"
        trace.write_text("opaque sensor bytes\n", encoding="utf-8")
        row = {
            "trace_id": "trace_0000",
            "status": "ok",
            "trace_path": trace.name,
            "trace_sha256": sha256_file(trace),
            "trace_quality": "exact_runtime_observations",
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            **extra,
        }
        path = root / "trace_manifest.csv"
        pd.DataFrame([row]).to_csv(path, index=False)
        return path

    def test_cli_is_truth_blind_and_exposes_joint_contract(self):
        args = parse_args(
            [
                "--trace-manifest", "traces.csv",
                "--output-dir", "output",
                "--min-joint-information", "0.3",
                "--min-n-information", "0.1",
                "--min-phi-information", "0.2",
                "--min-observability-rank", "2",
                "--cohesion-grid-size", "5",
                "--load-transfer-mode", "static",
                "--max-final-update-age-s", "2.5",
                "--block-alpha-rate",
            ]
        )
        self.assertAlmostEqual(args.min_joint_information, 0.3)
        self.assertAlmostEqual(args.min_n_information, 0.1)
        self.assertAlmostEqual(args.min_phi_information, 0.2)
        self.assertEqual(args.min_observability_rank, 2)
        self.assertEqual(args.cohesion_grid_size, 5)
        self.assertEqual(args.load_transfer_mode, "static")
        self.assertAlmostEqual(args.max_final_update_age_s, 2.5)
        self.assertTrue(args.block_alpha_rate)
        self.assertFalse(hasattr(args, "truth"))
        self.assertFalse(hasattr(args, "score"))
        self.assertFalse(args.promotion_run)

    def test_cli_defaults_are_the_frozen_candidate(self):
        args = parse_args(
            [
                "--trace-manifest", "traces.csv",
                "--output-dir", "output",
            ]
        )
        self.assertEqual(args.model_dir, "nn_models/tire_force_rate")
        self.assertEqual(args.dynamics_min_windows, 8)
        self.assertEqual(args.dynamics_rate_mode, "zero")
        self.assertEqual(args.load_transfer_mode, "static")
        self.assertEqual(args.posterior_summary, "mean")
        self.assertAlmostEqual(args.tail_start, 7.0)
        self.assertAlmostEqual(args.te_min_confidence, 0.20)
        self.assertEqual(args.min_observability_rank, 2)
        self.assertAlmostEqual(
            args.min_observability_singular_value, 0.10
        )
        self.assertAlmostEqual(args.max_final_update_age_s, 3.5)

    def test_promotion_flag_is_explicit(self):
        args = parse_args(
            [
                "--trace-manifest", "traces.csv",
                "--output-dir", "output",
                "--promotion-run",
            ]
        )
        self.assertTrue(args.promotion_run)

    def test_manifest_marks_only_clean_complete_promotion_run_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_manifest = root / "trace_manifest.csv"
            trace_manifest.write_text(
                "trace_id,status\ntrace_0000,ok\n", encoding="utf-8"
            )
            args = parse_args(
                [
                    "--trace-manifest", str(trace_manifest),
                    "--output-dir", str(root / "output"),
                    "--promotion-run",
                ]
            )
            manifest_path = root / "replay_manifest.csv"
            _write_manifest(
                manifest_path,
                args,
                CandidateConfig(model_dir="unused"),
                pd.DataFrame({"trace_id": ["trace_0000"]}),
                {"git_dirty": "false"},
            )
            manifest = dict(
                pd.read_csv(manifest_path, dtype=str).itertuples(
                    index=False, name=None
                )
            )
            self.assertEqual(manifest["development_only"], "false")
            self.assertEqual(manifest["paper_evidence_eligible"], "true")
            self.assertEqual(
                manifest["inference_semantics_version"],
                "joint_final_snapshot",
            )
            self.assertEqual(
                manifest["accepted_snapshot_version"],
                "grit_accepted",
            )

            args.max_traces = 1
            _write_manifest(
                manifest_path,
                args,
                CandidateConfig(model_dir="unused"),
                pd.DataFrame({"trace_id": ["trace_0000"]}),
                {"git_dirty": "false"},
            )
            partial_manifest = dict(
                pd.read_csv(manifest_path, dtype=str).itertuples(
                    index=False, name=None
                )
            )
            self.assertEqual(
                partial_manifest["paper_evidence_eligible"], "false"
            )

            args.max_traces = None
            args.te_min_confidence = 0.19
            _write_manifest(
                manifest_path,
                args,
                CandidateConfig(model_dir="unused"),
                pd.DataFrame({"trace_id": ["trace_0000"]}),
                {"git_dirty": "false"},
            )
            relaxed_manifest = dict(
                pd.read_csv(manifest_path, dtype=str).itertuples(
                    index=False, name=None
                )
            )
            self.assertEqual(
                relaxed_manifest["paper_evidence_eligible"], "false"
            )

    def test_final_update_age_contract_requires_finite_positive_value(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                CandidateConfig(
                    model_dir="unused", max_final_update_age_s=value
                )

    def test_manifest_rejects_oracle_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._manifest(Path(directory), phi_true_deg=24.0)
            with self.assertRaisesRegex(ValueError, "oracle|forbidden"):
                load_trace_manifest(path)

    def test_manifest_resolves_and_hashes_exact_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._manifest(Path(directory))
            traces = load_trace_manifest(path)
            self.assertEqual(len(traces), 1)
            self.assertTrue(Path(traces.loc[0, "_resolved_trace_path"]).is_file())

    def test_manifest_requires_estimator_disabled_fixed_controller(self):
        invalid = (
            {"terrain_estimator_enabled": True},
            {"controller_prior": "sand"},
            {"controller_prior_n": 0.9},
        )
        for metadata in invalid:
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as directory:
                path = self._manifest(Path(directory), **metadata)
                with self.assertRaisesRegex(
                    ValueError, "estimator-disabled|fixed dirt|prior n=0.7"
                ):
                    load_trace_manifest(path)

    def test_extra_kwargs_cannot_add_oracle_evidence(self):
        for payload in (
            '{"ground_datum_z": 0.0}',
            '{"wheel_center_height": 0.5}',
            '{"contact_force_truth": 1.0}',
        ):
            with self.subTest(payload=payload):
                config = CandidateConfig(
                    model_dir="unused", extra_kwargs_json=payload
                )
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    config.constructor_kwargs()

    def test_snapshot_api_is_required_and_fail_closed(self):
        with self.assertRaisesRegex(
            ValueError, "get_last_accepted_snapshot"
        ):
            _accepted_snapshot(object())

    def test_model_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "repack_manifest.json").write_text(
                json.dumps(
                    {
                        "metadata_injected": {
                            "training_source": "vehicle_trace",
                            "checkpoint_format": "tire_force_static_mlp",
                        },
                        "training_csv_verified": True,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "single-tire"):
                _validate_rig_model(root)

    def test_worker_writes_observability_and_single_use_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_replay(
                _path, backend, _config, *, trace_id, expected_sha256, estimator
            ):
                self.assertIsInstance(estimator, _ValidCandidate)
                return (
                    {
                        "trace_id": trace_id,
                        "backend": backend,
                        "status": "ok",
                        "trace_sha256": expected_sha256,
                    },
                    _joint_timeline(estimator, final_time=8.0),
                )

            task = DevelopmentTask(
                trace_id="trace_0000",
                trace_path="unused.csv",
                trace_sha256="a" * 64,
                trace_quality="exact_runtime_observations",
                output_path=str(root / "trace_0000.csv"),
                replay_config=ReplayConfig(tail_start=0.0),
                candidate_config=CandidateConfig(model_dir="unused"),
            )
            with patch(
                "benchmarking.develop_joint_estimator.build_estimator",
                return_value=_ValidCandidate(),
            ), patch(
                "benchmarking.develop_joint_estimator.replay_trace",
                side_effect=fake_replay,
            ):
                result = _run_one(task)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["backend"], BACKEND)
            self.assertEqual(result["final_observability_rank"], 2)
            self.assertEqual(result["final_last_likelihood_residual_count"], 24)
            self.assertEqual(result["final_load_transfer_mode"], "static")
            self.assertEqual(result["final_accuracy_valid"], 1)
            self.assertEqual(result["final_fresh"], 1)
            self.assertEqual(result["final_confident"], 1)
            self.assertEqual(result["final_publication_ready"], 1)
            written = pd.read_csv(root / "trace_0000.csv")
            self.assertIn("replay_final_joint_information_kl", written.columns)
            self.assertIn("replay_final_n_boundary_mass", written.columns)
            self.assertIn(
                "replay_final_duplicate_likelihood_block_count", written.columns
            )

    def test_worker_marks_double_consumption_contract_failed(self):
        candidate = _ValidCandidate()
        candidate.duplicate_likelihood_block_count = 1
        candidate.last_likelihood_residual_count = 26
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = DevelopmentTask(
                trace_id="trace_0000",
                trace_path="unused.csv",
                trace_sha256="a" * 64,
                trace_quality="exact_runtime_observations",
                output_path=str(root / "trace_0000.csv"),
                replay_config=ReplayConfig(tail_start=0.0),
                candidate_config=CandidateConfig(model_dir="unused"),
            )
            with patch(
                "benchmarking.develop_joint_estimator.build_estimator",
                return_value=candidate,
            ), patch(
                "benchmarking.develop_joint_estimator.replay_trace",
                return_value=(
                    {"trace_id": "trace_0000", "backend": BACKEND, "status": "ok"},
                    _joint_timeline(candidate, final_time=8.0),
                ),
            ):
                result = _run_one(task)
        self.assertEqual(result["status"], "fail")
        self.assertRegex(str(result["failure"]), "accuracy-valid")

    def test_worker_rejects_stale_tail_update(self):
        candidate = _ValidCandidate()
        candidate.last_joint_update_time = 2.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = DevelopmentTask(
                trace_id="trace_0000",
                trace_path="unused.csv",
                trace_sha256="a" * 64,
                trace_quality="exact_runtime_observations",
                output_path=str(root / "trace_0000.csv"),
                replay_config=ReplayConfig(tail_start=7.0),
                candidate_config=CandidateConfig(model_dir="unused", block_dt=0.5),
            )
            with patch(
                "benchmarking.develop_joint_estimator.build_estimator",
                return_value=candidate,
            ), patch(
                "benchmarking.develop_joint_estimator.replay_trace",
                return_value=(
                    {"trace_id": "trace_0000", "backend": BACKEND, "status": "ok"},
                    _joint_timeline(candidate, final_time=10.0),
                ),
            ):
                result = _run_one(task)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["final_accuracy_valid"], 0)
        self.assertEqual(result["final_fresh"], 0)
        self.assertRegex(str(result["failure"]), "accuracy-valid")

    def test_default_freshness_allows_normal_final_rejected_blocks(self):
        candidate = _ValidCandidate()
        candidate.last_joint_update_time = 8.27
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = DevelopmentTask(
                trace_id="trace_0000",
                trace_path="unused.csv",
                trace_sha256="a" * 64,
                trace_quality="exact_runtime_observations",
                output_path=str(root / "trace_0000.csv"),
                replay_config=ReplayConfig(tail_start=7.0),
                candidate_config=CandidateConfig(model_dir="unused"),
            )
            with patch(
                "benchmarking.develop_joint_estimator.build_estimator",
                return_value=candidate,
            ), patch(
                "benchmarking.develop_joint_estimator.replay_trace",
                return_value=(
                    {"trace_id": "trace_0000", "backend": BACKEND, "status": "ok"},
                    _joint_timeline(candidate, final_time=10.0),
                ),
            ):
                result = _run_one(task)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["final_final_update_age_s"], 1.73)
        self.assertAlmostEqual(result["final_final_update_max_age_s"], 3.5)
        self.assertEqual(result["final_accuracy_valid"], 1)
        self.assertEqual(result["final_fresh"], 1)

    def test_stale_accuracy_valid_snapshot_remains_in_scoring_matrix(self):
        candidate = _ValidCandidate()
        candidate.last_joint_update_time = 8.27
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = DevelopmentTask(
                trace_id="trace_0000",
                trace_path="unused.csv",
                trace_sha256="a" * 64,
                trace_quality="exact_runtime_observations",
                output_path=str(root / "trace_0000.csv"),
                replay_config=ReplayConfig(tail_start=7.0),
                candidate_config=CandidateConfig(
                    model_dir="unused", max_final_update_age_s=1.5
                ),
            )
            with patch(
                "benchmarking.develop_joint_estimator.build_estimator",
                return_value=candidate,
            ), patch(
                "benchmarking.develop_joint_estimator.replay_trace",
                return_value=(
                    {"trace_id": "trace_0000", "backend": BACKEND, "status": "ok"},
                    _joint_timeline(candidate, final_time=10.0),
                ),
            ):
                result = _run_one(task)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["final_accuracy_valid"], 1)
        self.assertEqual(result["final_fresh"], 0)
        self.assertEqual(result["final_publication_ready"], 0)

    def test_low_confidence_snapshot_remains_accuracy_valid(self):
        candidate = _ValidCandidate()
        candidate.snapshot_confidence = 0.19
        candidate.last_joint_update_time = 8.27
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = DevelopmentTask(
                trace_id="trace_0000",
                trace_path="unused.csv",
                trace_sha256="a" * 64,
                trace_quality="exact_runtime_observations",
                output_path=str(root / "trace_0000.csv"),
                replay_config=ReplayConfig(
                    tail_start=7.0, min_confidence=0.20
                ),
                candidate_config=CandidateConfig(model_dir="unused"),
            )
            with patch(
                "benchmarking.develop_joint_estimator.build_estimator",
                return_value=candidate,
            ), patch(
                "benchmarking.develop_joint_estimator.replay_trace",
                return_value=(
                    {"trace_id": "trace_0000", "backend": BACKEND, "status": "ok"},
                    _joint_timeline(candidate, final_time=10.0),
                ),
            ):
                result = _run_one(task)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["final_est_n"], 0.8)
        self.assertEqual(result["final_accuracy_valid"], 1)
        self.assertEqual(result["final_confident"], 0)
        self.assertEqual(result["final_snapshot_was_published"], 0)
        self.assertEqual(result["final_publication_ready"], 0)

    def test_boundary_limited_snapshot_remains_accuracy_valid(self):
        candidate = _ValidCandidate()
        candidate.boundary_limited = True
        candidate.n_boundary_mass = 0.30
        candidate.last_joint_update_time = 8.27
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = DevelopmentTask(
                trace_id="trace_0000",
                trace_path="unused.csv",
                trace_sha256="a" * 64,
                trace_quality="exact_runtime_observations",
                output_path=str(root / "trace_0000.csv"),
                replay_config=ReplayConfig(
                    tail_start=7.0, min_confidence=0.20
                ),
                candidate_config=CandidateConfig(model_dir="unused"),
            )
            with patch(
                "benchmarking.develop_joint_estimator.build_estimator",
                return_value=candidate,
            ), patch(
                "benchmarking.develop_joint_estimator.replay_trace",
                return_value=(
                    {"trace_id": "trace_0000", "backend": BACKEND, "status": "ok"},
                    _joint_timeline(candidate, final_time=10.0),
                ),
            ):
                result = _run_one(task)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["final_accuracy_valid"], 1)
        self.assertEqual(result["final_fresh"], 1)
        self.assertEqual(result["final_confident"], 1)
        self.assertEqual(result["final_publication_ready"], 0)

    def test_atomic_estimate_checkpoint_sorts_trace_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "estimates.csv"
            frame = _write_estimates_atomic(
                path,
                [
                    {"trace_id": "trace_0002", "status": "ok"},
                    {"trace_id": "trace_0001", "status": "ok"},
                ],
            )
            self.assertEqual(frame["trace_id"].tolist(), ["trace_0001", "trace_0002"])
            self.assertEqual(pd.read_csv(path)["trace_id"].tolist(), frame["trace_id"].tolist())


if __name__ == "__main__":
    unittest.main()
