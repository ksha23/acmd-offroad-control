#!/usr/bin/env python3
"""Fail-closed contracts for joint-estimator promotion scoring.

The scorer is the gate between a replay matrix and a published number, so
these tests establish that it refuses every way a matrix can be incomplete or
altered: a mutated source trace, a wrong or non-contiguous trace count, an
unregistered port or simulator seed, a replay marked development-only, a
configuration differing from the frozen one, and a preregistration whose
digest does not match. They also fix the scoring semantics, including how
stale, boundary-limited, and low-confidence snapshots are counted.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ConstantInputWarning

from benchmarking.collect_terrain_estimator_traces import (
    generate_soils,
    manifold_yaml_from_n,
)
from benchmarking.terrain_estimator_trace import TRACE_COLUMNS
from benchmarking.score_joint_estimator import (
    ROOT,
    _FROZEN_COLLECTION_SETTINGS,
    _FROZEN_JOINT_SETTINGS,
    _FROZEN_PARENT_SETTINGS,
    _count_at_least_fraction,
    _count_at_most_fraction,
    _safe_spearman,
    _sha256_file,
    main,
    score,
)


class JointScoreContractTest(unittest.TestCase):
    COUNT = 8
    SOIL_SEED = 123
    BASE_PORT = 59000
    SIM_SEED = 7000

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.collection = self.root / "collection"
        self.collection.mkdir()
        self.soils = self.collection / "truth_soils"
        self.soils.mkdir()
        self.trace_ids = [
            f"trace_{index:04d}" for index in range(self.COUNT)
        ]

        config = dict(_FROZEN_COLLECTION_SETTINGS)
        config.update(
            {
                "n": self.COUNT,
                "soil_seed": self.SOIL_SEED,
                "base_port": self.BASE_PORT,
                "sim_seed_first": self.SIM_SEED,
            }
        )
        self.collection_config = self.collection / "collection_config.json"
        self.collection_config.write_text(
            __import__("json").dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )

        trace_rows = []
        hashes = []
        for index, trace_id in enumerate(self.trace_ids):
            relative = Path("raw") / trace_id / "sensor_trace.csv"
            trace_path = self.collection / relative
            trace_path.parent.mkdir(parents=True)
            trace = {
                column: np.zeros(2, dtype=float)
                for column in TRACE_COLUMNS
            }
            trace["seq"] = np.asarray([0, 1], dtype=int)
            trace["sim_time"] = np.asarray([0.0, 13.7])
            trace["x_cg"] = np.asarray([float(index), float(index) + 0.01])
            trace["quat_e0"] = np.ones(2)
            trace["u_raw"] = np.full(2, 5.0)
            trace["u"] = np.full(2, 5.0)
            trace["Fz_f"] = np.full(2, 6000.0)
            trace["Fz_r"] = np.full(2, 6000.0)
            pd.DataFrame(trace, columns=TRACE_COLUMNS).to_csv(
                trace_path, index=False, float_format="%.17g"
            )
            digest = _sha256_file(trace_path)
            hashes.append(digest)
            trace_rows.append(
                {
                    "trace_id": trace_id,
                    "status": "ok",
                    "trace_path": str(relative),
                    "trace_sha256": digest,
                    "trace_rows": 2,
                    "trace_schema_version": 3,
                    "trace_quality": "exact_runtime_observations",
                    "controller_prior": "dirt",
                    "controller_prior_n": 0.7,
                    "terrain_estimator_enabled": False,
                    "sim_seed": self.SIM_SEED + index,
                    "path": "sinusoidal",
                    "speed_mps": 5.0,
                    "sim_time_s": 14.0,
                    "lead_in_m": 5.0,
                    "maneuver_label": "passive",
                    "terrain_id_probe": False,
                }
            )
        self.trace_manifest = self.collection / "trace_manifest.csv"
        pd.DataFrame(trace_rows).to_csv(self.trace_manifest, index=False)

        generated, _nuisance_seed = generate_soils(
            self.COUNT,
            mode="independent_n_phi",
            seed=self.SOIL_SEED,
            jitter_fraction=0.1,
        )
        permutation = np.random.default_rng(
            self.SOIL_SEED + 2_000_003
        ).permutation(self.COUNT)
        generated = [generated[int(index)] for index in permutation]
        n_true = np.asarray([case[0] for case in generated])
        phi_true = np.asarray(
            [case[1]["friction_angle"] for case in generated]
        )
        truth_rows = []
        for index, trace_id in enumerate(self.trace_ids):
            soil = dict(generated[index][1])
            manifold = manifold_yaml_from_n(float(n_true[index]))
            cohesion_multiplier = (
                float(soil["cohesion"]) / float(manifold["cohesion"])
            )
            (self.soils / f"case_{index:04d}.yaml").write_text(
                yaml.safe_dump(soil), encoding="utf-8"
            )
            truth_rows.append(
                {
                    "trace_id": trace_id,
                    "n_true": soil["n"],
                    "Kphi_true": soil["Kphi"],
                    "Kc_true": soil["Kc"],
                    "c_true": soil["cohesion"],
                    "phi_true_deg": soil["friction_angle"],
                    "k_true": soil["janosi_shear"],
                    "soil_draw_seed": self.SOIL_SEED,
                    "nuisance_seed": self.SOIL_SEED + 1_000_003,
                    "case_order_seed": self.SOIL_SEED + 2_000_003,
                    "nuisance_jitter_fraction": 0.0,
                    "soil_mode": "independent_n_phi",
                    "phi_draw_seed": self.SOIL_SEED + 1_000_003,
                    "cohesion_draw_seed": self.SOIL_SEED + 1_000_004,
                    "cohesion_jitter_fraction": 0.15,
                    "cohesion_multiplier_lo": 0.85,
                    "cohesion_multiplier_hi": 1.15,
                    "cohesion_multiplier_true": cohesion_multiplier,
                }
            )
        self.truth = self.collection / "truth.csv"
        pd.DataFrame(truth_rows).to_csv(
            self.truth, index=False, float_format="%.17g"
        )

        common = {
            "trace_id": self.trace_ids,
            "status": ["ok"] * self.COUNT,
            "trace_sha256": hashes,
            "trace_quality": ["exact_runtime_observations"] * self.COUNT,
        }
        self.joint = self.root / "joint.csv"
        joint_frame = pd.DataFrame(
            {
                **common,
                "backend": ["grit"] * self.COUNT,
                "final_est_n": n_true + 0.01,
                "final_est_phi_deg": phi_true + 0.5,
                "final_snapshot_n": n_true + 0.01,
                "final_snapshot_phi_deg": phi_true + 0.5,
                "final_accepted_snapshot_version": [
                    "grit_accepted"
                ] * self.COUNT,
                "final_joint_active": [1] * self.COUNT,
                "final_joint_has_estimate": [1] * self.COUNT,
                "final_joint_updates": [4] * self.COUNT,
                "final_joint_projection_failures": [0] * self.COUNT,
                "final_observability_rank": [2] * self.COUNT,
                "final_observability_min_singular_value": [0.2] * self.COUNT,
                "final_last_likelihood_block_count": [8] * self.COUNT,
                "final_last_likelihood_residual_count": [16] * self.COUNT,
                "final_duplicate_likelihood_block_count": [0] * self.COUNT,
                "final_duplicate_likelihood_update_count": [0] * self.COUNT,
                "final_last_joint_update_time": [13.0] * self.COUNT,
                "final_final_trace_time": [13.7] * self.COUNT,
                "final_final_update_age_s": [0.7] * self.COUNT,
                "final_final_update_max_age_s": [3.5] * self.COUNT,
                "final_accuracy_valid": [1] * self.COUNT,
                "final_fresh": [1] * self.COUNT,
                "final_confident": [1] * self.COUNT,
                "final_control_envelope_valid": [1] * self.COUNT,
                "final_publication_ready": [1] * self.COUNT,
                "final_snapshot_was_published": [1] * self.COUNT,
                "final_publication_confidence": [0.8] * self.COUNT,
                "final_n_sigma": [0.05] * self.COUNT,
                "final_phi_sigma_deg": [1.0] * self.COUNT,
                "final_load_transfer_mode": ["static"] * self.COUNT,
                "final_joint_information_kl": [0.3] * self.COUNT,
                "final_n_information_kl": [0.2] * self.COUNT,
                "final_phi_information_kl": [1.0] * self.COUNT,
                "final_cohesion_information_kl": [0.0] * self.COUNT,
                "final_n_boundary_mass": [0.02] * self.COUNT,
                "final_phi_boundary_mass": [0.03] * self.COUNT,
                "final_cohesion_boundary_mass": [0.0] * self.COUNT,
                "final_max_boundary_mass": [0.03] * self.COUNT,
                "final_boundary_limited": [0] * self.COUNT,
                "final_likelihood_evaluations": [7] * self.COUNT,
                "final_last_effective_front_load": [5700.0] * self.COUNT,
                "final_last_effective_rear_load": [5900.0] * self.COUNT,
                "final_last_effective_load_ax": [0.0] * self.COUNT,
                "final_last_effective_load_ay": [0.0] * self.COUNT,
                "final_projection_wall_time_s": [0.01] * self.COUNT,
                "final_profile_wall_time_s": [0.02] * self.COUNT,
                "final_observability_wall_time_s": [0.003] * self.COUNT,
                "final_posterior_wall_time_s": [0.004] * self.COUNT,
                "final_publication_wall_time_s": [0.001] * self.COUNT,
                "final_update_wall_time_s": [0.015] * self.COUNT,
            }
        )
        joint_timeseries_dir = self.root / "joint_timeseries"
        joint_timeseries_dir.mkdir()
        joint_paths = []
        for index, row in joint_frame.iterrows():
            relative = (
                Path("joint_timeseries")
                / f"{row['trace_id']}_grit.csv"
            )
            timeline = pd.DataFrame(
                {
                    "trace_id": [row["trace_id"]] * 2,
                    "backend": ["grit"] * 2,
                    "sim_time": [0.0, 13.7],
                    "n_published": [0.7, row["final_est_n"]],
                    "phi_published_deg": [
                        29.0,
                        row["final_est_phi_deg"],
                    ],
                    "confidence": [
                        0.0,
                        row["final_publication_confidence"],
                    ],
                    "published_update": [0, 1],
                    "joint_snapshot_advanced": [0, 1],
                    "joint_snapshot_seq": [
                        "",
                        row["final_joint_updates"],
                    ],
                    "joint_snapshot_evidence_time_s": [
                        "",
                        row["final_last_joint_update_time"],
                    ],
                    "joint_snapshot_n": ["", row["final_snapshot_n"]],
                    "joint_snapshot_phi_deg": [
                        "",
                        row["final_snapshot_phi_deg"],
                    ],
                    "joint_snapshot_n_sigma": [
                        "",
                        row["final_n_sigma"],
                    ],
                    "joint_snapshot_phi_sigma_deg": [
                        "",
                        row["final_phi_sigma_deg"],
                    ],
                    "joint_snapshot_confidence": [
                        "",
                        row["final_publication_confidence"],
                    ],
                    "joint_snapshot_information_kl": [
                        "",
                        row["final_joint_information_kl"],
                    ],
                    "joint_snapshot_observability_rank": [
                        "",
                        row["final_observability_rank"],
                    ],
                    "joint_snapshot_observability_min_singular_value": [
                        "",
                        row[
                            "final_observability_min_singular_value"
                        ],
                    ],
                    "joint_snapshot_n_boundary_mass": [
                        "",
                        row["final_n_boundary_mass"],
                    ],
                    "joint_snapshot_phi_boundary_mass": [
                        "",
                        row["final_phi_boundary_mass"],
                    ],
                    "joint_snapshot_max_boundary_mass": [
                        "",
                        row["final_max_boundary_mass"],
                    ],
                    "joint_snapshot_boundary_limited": [
                        "",
                        row["final_boundary_limited"],
                    ],
                    "joint_snapshot_projection_wall_time_s": [
                        "",
                        row["final_projection_wall_time_s"],
                    ],
                    "joint_snapshot_profile_wall_time_s": [
                        "",
                        row["final_profile_wall_time_s"],
                    ],
                    "joint_snapshot_observability_wall_time_s": [
                        "",
                        row["final_observability_wall_time_s"],
                    ],
                    "joint_snapshot_posterior_wall_time_s": [
                        "",
                        row["final_posterior_wall_time_s"],
                    ],
                    "joint_snapshot_publication_wall_time_s": [
                        "",
                        row["final_publication_wall_time_s"],
                    ],
                    "joint_snapshot_update_wall_time_s": [
                        "",
                        row["final_update_wall_time_s"],
                    ],
                }
            )
            for column in joint_frame.columns:
                if (
                    column.startswith("final_")
                    and column
                    not in {
                        "final_est_n",
                        "final_est_phi_deg",
                        "final_load_transfer_mode",
                    }
                ):
                    timeline[f"replay_{column}"] = row[column]
            timeline["replay_final_load_transfer_mode"] = row[
                "final_load_transfer_mode"
            ]
            timeline.to_csv(
                self.root / relative,
                index=False,
                float_format="%.17g",
            )
            joint_paths.append(relative.as_posix())
        joint_frame["timeseries_path"] = joint_paths
        joint_frame.to_csv(
            self.joint, index=False, float_format="%.17g"
        )
        self.parent = self.root / "parent.csv"
        parent_frame = pd.DataFrame(
            {
                **common,
                "backend": ["scalar_parent"] * self.COUNT,
                "final_est_n": [0.7] * self.COUNT,
            }
        )
        parent_timeseries_dir = self.root / "parent_timeseries"
        parent_timeseries_dir.mkdir()
        parent_paths = []
        for _, row in parent_frame.iterrows():
            relative = (
                Path("parent_timeseries")
                / f"{row['trace_id']}_scalar_parent.csv"
            )
            pd.DataFrame(
                {
                    "trace_id": [row["trace_id"]] * 2,
                    "backend": ["scalar_parent"] * 2,
                    "sim_time": [0.0, 13.7],
                    "n_published": [0.7, row["final_est_n"]],
                }
            ).to_csv(
                self.root / relative,
                index=False,
                float_format="%.17g",
            )
            parent_paths.append(relative.as_posix())
        parent_frame["timeseries_path"] = parent_paths
        parent_frame.to_csv(self.parent, index=False)

        trace_digest = _sha256_file(self.trace_manifest)
        joint_manifest = dict(_FROZEN_JOINT_SETTINGS)
        joint_manifest.update(
            {
                "truth_inputs": "none",
                "trace_manifest_sha256": trace_digest,
                "backend": "grit",
                "selected_trace_count": self.COUNT,
                "selected_trace_ids": self.trace_ids,
                "config.model_dir": str(
                    ROOT / "nn_models/tire_force_rate"
                ),
                "config.min_observability_singular_value": 0.1,
                "config.max_final_update_age_s": 3.5,
                "provenance.git_head": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "provenance.git_dirty": False,
                "provenance.driver_source_sha256": _sha256_file(
                    ROOT / "benchmarking/develop_joint_estimator.py"
                ),
                "provenance.candidate_source_sha256": _sha256_file(
                    ROOT
                    / "simulation/estimators/"
                    "grit_terrain_estimator.py"
                ),
                "provenance.parent_estimator_source_sha256": _sha256_file(
                    ROOT
                    / "simulation/estimators/"
                    "scalar_parent_terrain_estimator.py"
                ),
                "provenance.force_checkpoint_sha256": _sha256_file(
                    ROOT / "nn_models/tire_force_rate/best_terrain_nn.pt"
                ),
                "provenance.force_scalers_sha256": _sha256_file(
                    ROOT / "nn_models/tire_force_rate/scalers.pkl"
                ),
            }
        )
        parent_manifest = dict(_FROZEN_PARENT_SETTINGS)
        parent_manifest.update(
            {
                "truth_inputs": "none",
                "trace_manifest_sha256": trace_digest,
                "config.model_dir": "nn_models/tire_force_rate",
            }
        )
        self.joint_replay_manifest = self.root / "joint_manifest.csv"
        self.parent_replay_manifest = self.root / "parent_manifest.csv"
        self._write_manifest(self.joint_replay_manifest, joint_manifest)
        self._write_manifest(self.parent_replay_manifest, parent_manifest)
        self.preregistration = self.root / "preregistration.md"
        self.preregistration.write_text(
            "# Frozen synthetic contract\n", encoding="utf-8"
        )

    @staticmethod
    def _write_manifest(path: Path, mapping: dict[str, object]) -> None:
        pd.DataFrame(
            [(key, repr(value)) for key, value in mapping.items()],
            columns=["key", "value"],
        ).to_csv(path, index=False)

    def _score_kwargs(self) -> dict[str, object]:
        return {
            "truth_path": self.truth,
            "truth_soil_dir": self.soils,
            "collection_config_path": self.collection_config,
            "trace_manifest_path": self.trace_manifest,
            "joint_path": self.joint,
            "joint_replay_manifest_path": self.joint_replay_manifest,
            "parent_path": self.parent,
            "parent_replay_manifest_path": self.parent_replay_manifest,
            "expected_count": self.COUNT,
            "expected_soil_seed": self.SOIL_SEED,
            "expected_base_port": self.BASE_PORT,
            "expected_sim_seed": self.SIM_SEED,
            "bootstrap_seed": 123,
            "bootstrap_resamples": 500,
        }

    def _mutate_manifest(self, path: Path, key: str, value: object) -> None:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.loc[frame["key"] == key, "value"] = repr(value)
        frame.to_csv(path, index=False)

    def _mutate_joint_trace(
        self,
        index: int,
        *,
        summary_updates: dict[str, object],
        snapshot_timeline_updates: dict[str, object],
        final_timeline_updates: dict[str, object] | None = None,
    ) -> None:
        frame = pd.read_csv(self.joint, float_precision="round_trip")
        for column, value in summary_updates.items():
            if (
                pd.api.types.is_integer_dtype(frame[column])
                and isinstance(value, (float, np.floating))
                and not float(value).is_integer()
            ):
                frame[column] = frame[column].astype(float)
            frame.loc[index, column] = value
        timeline_path = self.root / str(
            frame.loc[index, "timeseries_path"]
        )
        timeline = pd.read_csv(
            timeline_path, float_precision="round_trip"
        )
        for column, value in summary_updates.items():
            replay_column = f"replay_{column}"
            if replay_column in timeline.columns:
                timeline[replay_column] = value
        advanced = (
            pd.to_numeric(
                timeline["joint_snapshot_advanced"], errors="coerce"
            )
            == 1
        )
        self.assertTrue(advanced.any())
        advanced_index = timeline.index[advanced][-1]
        for column, value in snapshot_timeline_updates.items():
            timeline.loc[advanced_index, column] = value
        for column, value in (final_timeline_updates or {}).items():
            timeline.loc[timeline.index[-1], column] = value
        timeline.to_csv(
            timeline_path, index=False, float_format="%.17g"
        )
        frame.to_csv(self.joint, index=False, float_format="%.17g")

    def _cli_arguments(self, output: Path) -> list[str]:
        return [
            "--truth",
            str(self.truth),
            "--truth-soil-dir",
            str(self.soils),
            "--collection-config",
            str(self.collection_config),
            "--trace-manifest",
            str(self.trace_manifest),
            "--joint-estimates",
            str(self.joint),
            "--joint-replay-manifest",
            str(self.joint_replay_manifest),
            "--parent-estimates",
            str(self.parent),
            "--parent-replay-manifest",
            str(self.parent_replay_manifest),
            "--preregistration",
            str(self.preregistration),
            "--expected-preregistration-sha256",
            _sha256_file(self.preregistration),
            "--expected-count",
            str(self.COUNT),
            "--expected-soil-seed",
            str(self.SOIL_SEED),
            "--expected-base-port",
            str(self.BASE_PORT),
            "--expected-sim-seed",
            str(self.SIM_SEED),
            "--bootstrap-resamples",
            "50",
            "--output-dir",
            str(output),
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def test_scores_bound_final_values_without_constant_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runs, summary, bootstrap, decision = score(
                **self._score_kwargs()
            )
        self.assertFalse(
            any(
                issubclass(item.category, ConstantInputWarning)
                for item in caught
            )
        )
        self.assertTrue(np.isnan(_safe_spearman(np.ones(8), np.arange(8))))
        self.assertEqual(len(runs), self.COUNT)
        self.assertEqual(
            set(summary["method"]),
            {"joint", "scalar_parent", "uniform_prior"},
        )
        self.assertEqual(set(bootstrap["n_units"]), {self.COUNT})
        self.assertEqual(set(bootstrap["bootstrap_seed"]), {123})
        self.assertTrue(decision["uses_final_estimates_not_tail_means"])
        self.assertTrue(decision["trace_matrix_complete_and_hash_identical"])

    def test_exact_population_gate_edges(self):
        self.assertFalse(_count_at_least_fraction(68, 72, 0.95))
        self.assertTrue(_count_at_least_fraction(69, 72, 0.95))
        self.assertFalse(_count_at_least_fraction(61, 72, 0.85))
        self.assertTrue(_count_at_least_fraction(62, 72, 0.85))
        self.assertTrue(_count_at_most_fraction(7, 72, 0.10))
        self.assertFalse(_count_at_most_fraction(8, 72, 0.10))

    def test_scores_stale_boundary_and_low_confidence_snapshots(self):
        self._mutate_joint_trace(
            0,
            summary_updates={
                "final_last_joint_update_time": 9.7,
                "final_final_update_age_s": 4.0,
                "final_fresh": 0,
                "final_publication_ready": 0,
            },
            snapshot_timeline_updates={
                "joint_snapshot_evidence_time_s": 9.7,
            },
        )
        self._mutate_joint_trace(
            1,
            summary_updates={
                "final_n_boundary_mass": 0.30,
                "final_max_boundary_mass": 0.30,
                "final_boundary_limited": 1,
                "final_publication_ready": 0,
            },
            snapshot_timeline_updates={
                "joint_snapshot_n_boundary_mass": 0.30,
                "joint_snapshot_max_boundary_mass": 0.30,
                "joint_snapshot_boundary_limited": 1,
            },
        )
        self._mutate_joint_trace(
            2,
            summary_updates={
                "final_publication_confidence": 0.19,
                "final_confident": 0,
                "final_publication_ready": 0,
                "final_snapshot_was_published": 0,
            },
            snapshot_timeline_updates={
                "joint_snapshot_confidence": 0.19,
            },
            final_timeline_updates={
                "confidence": 0.19,
                "published_update": 0,
                "n_published": 0.7,
                "phi_published_deg": 29.0,
            },
        )
        runs, summary, _bootstrap, decision = score(
            **self._score_kwargs()
        )
        self.assertEqual(len(runs), self.COUNT)
        self.assertEqual(
            int(summary.set_index("method").loc["joint", "n"]),
            self.COUNT,
        )
        self.assertEqual(decision["fresh_estimate_count"], 7)
        self.assertEqual(decision["material_boundary_limited_count"], 1)
        self.assertEqual(decision["confident_estimate_count"], 7)
        self.assertEqual(decision["publication_ready_count"], 5)
        self.assertFalse(decision["fresh_estimates_at_least_95_pct"])
        self.assertFalse(
            decision["material_boundary_limited_at_most_10_pct"]
        )
        self.assertFalse(decision["publication_ready_at_least_85_pct"])

    def test_rejects_mutated_source_trace(self):
        source = self.collection / "raw/trace_0000/sensor_trace.csv"
        source.write_text("seq,value\n0,99\n1,100\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source sensor trace hash"):
            score(**self._score_kwargs())

    def test_rejects_wrong_count_or_noncontiguous_ids(self):
        with self.assertRaisesRegex(ValueError, "expected 7|contiguous"):
            score(**{**self._score_kwargs(), "expected_count": 7})
        joint = pd.read_csv(self.joint)
        joint.loc[0, "trace_id"] = "trace_9999"
        joint.to_csv(self.joint, index=False)
        with self.assertRaisesRegex(
            ValueError, "contiguous|sets differ|trace ID mismatch"
        ):
            score(**self._score_kwargs())

    def test_rejects_unregistered_port_or_sim_seed(self):
        with self.assertRaisesRegex(ValueError, "base_port"):
            score(
                **{
                    **self._score_kwargs(),
                    "expected_base_port": self.BASE_PORT + 2,
                }
            )
        with self.assertRaisesRegex(ValueError, "sim_seed_first"):
            score(
                **{
                    **self._score_kwargs(),
                    "expected_sim_seed": self.SIM_SEED + 1,
                }
            )

    def test_rejects_fractional_simulator_seed_without_truncation(self):
        traces = pd.read_csv(self.trace_manifest)
        traces["sim_seed"] = traces["sim_seed"].astype(float)
        traces.loc[0, "sim_seed"] = self.SIM_SEED + 0.9
        traces.to_csv(self.trace_manifest, index=False)
        digest = _sha256_file(self.trace_manifest)
        self._mutate_manifest(
            self.joint_replay_manifest,
            "trace_manifest_sha256",
            digest,
        )
        self._mutate_manifest(
            self.parent_replay_manifest,
            "trace_manifest_sha256",
            digest,
        )
        with self.assertRaisesRegex(ValueError, "non-integer simulator seeds"):
            score(**self._score_kwargs())

    def test_rejects_mismatched_replay_manifest_setting(self):
        self._mutate_manifest(
            self.parent_replay_manifest,
            "config.dynamics_horizon",
            6.0,
        )
        with self.assertRaisesRegex(ValueError, "dynamics_horizon"):
            score(**self._score_kwargs())

    def test_rejects_relaxed_publication_confidence_manifest(self):
        self._mutate_manifest(
            self.joint_replay_manifest,
            "replay.minimum_confidence",
            0.19,
        )
        with self.assertRaisesRegex(ValueError, "minimum_confidence"):
            score(**self._score_kwargs())

    def test_rejects_replay_manifest_trace_hash_mismatch(self):
        self._mutate_manifest(
            self.parent_replay_manifest,
            "trace_manifest_sha256",
            "f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "trace_manifest_sha256"):
            score(**self._score_kwargs())

    def test_rejects_development_only_replay(self):
        self._mutate_manifest(
            self.joint_replay_manifest, "development_only", True
        )
        self._mutate_manifest(
            self.joint_replay_manifest, "paper_evidence_eligible", False
        )
        with self.assertRaisesRegex(ValueError, "development_only"):
            score(**self._score_kwargs())

    def test_rejects_low_singular_value_or_inconsistent_age(self):
        joint = pd.read_csv(self.joint)
        joint.loc[0, "final_observability_min_singular_value"] = 0.09
        joint.to_csv(self.joint, index=False)
        with self.assertRaisesRegex(ValueError, "singular-value"):
            score(**self._score_kwargs())

        joint.loc[0, "final_observability_min_singular_value"] = 0.2
        joint.loc[0, "final_final_update_age_s"] = 0.8
        joint.to_csv(self.joint, index=False)
        with self.assertRaisesRegex(ValueError, "inconsistent final-update ages"):
            score(**self._score_kwargs())

    def test_rejects_boundary_flag_inconsistent_with_snapshot_mass(self):
        joint = pd.read_csv(self.joint)
        joint.loc[0, "final_n_boundary_mass"] = 0.30
        joint.loc[0, "final_max_boundary_mass"] = 0.30
        joint.loc[0, "final_boundary_limited"] = 0
        joint.to_csv(self.joint, index=False)
        with self.assertRaisesRegex(ValueError, "boundary-limited"):
            score(**self._score_kwargs())

    def test_rejects_control_envelope_flag_inconsistent_with_estimate(self):
        joint = pd.read_csv(self.joint)
        joint.loc[0, "final_control_envelope_valid"] = 0
        joint.to_csv(self.joint, index=False)
        with self.assertRaisesRegex(ValueError, "control-envelope"):
            score(**self._score_kwargs())

    def test_rejects_relaxed_observability_contract(self):
        with self.assertRaisesRegex(ValueError, "differs from the frozen"):
            score(
                **{
                    **self._score_kwargs(),
                    "min_observability_singular_value": 0.01,
                }
            )

    def test_rejects_final_estimate_not_bound_to_replay_timeline(self):
        joint = pd.read_csv(self.joint)
        joint.loc[0, "final_est_n"] += 0.05
        joint.to_csv(self.joint, index=False, float_format="%.17g")
        with self.assertRaisesRegex(
            ValueError, "final estimate does not match accepted snapshot"
        ):
            score(**self._score_kwargs())

    def test_rejects_truth_seed_or_soil_yaml_mismatch(self):
        truth = pd.read_csv(self.truth)
        truth.loc[0, "soil_draw_seed"] = self.SOIL_SEED + 1
        truth.to_csv(self.truth, index=False)
        with self.assertRaisesRegex(ValueError, "soil_draw_seed"):
            score(**self._score_kwargs())

        truth.loc[0, "soil_draw_seed"] = self.SOIL_SEED
        truth.to_csv(self.truth, index=False, float_format="%.17g")
        soil_path = self.soils / "case_0000.yaml"
        soil = yaml.safe_load(soil_path.read_text(encoding="utf-8"))
        soil["n"] += 0.1
        soil_path.write_text(yaml.safe_dump(soil), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "YAML mismatch"):
            score(**self._score_kwargs())

    def test_rejects_truth_field_in_inference_before_truth_bundle(self):
        joint = pd.read_csv(self.joint)
        joint["n_true"] = 0.7
        joint.to_csv(self.joint, index=False)
        (self.soils / "case_0000.yaml").unlink()
        with self.assertRaisesRegex(ValueError, "truth"):
            score(**self._score_kwargs())

    def test_cli_writes_hash_bound_provenance_manifest(self):
        output = self.root / "score"
        clean_state = {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_dirty": "false",
            "git_status_porcelain": "[]",
            "git_state_sha256": "a" * 64,
        }
        with patch(
            "benchmarking.score_joint_estimator._git_state",
            return_value=clean_state,
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self._cli_arguments(output)), 0)
        manifest = pd.read_csv(
            output / "provenance_manifest.csv",
            dtype=str,
            keep_default_na=False,
        ).set_index("key")["value"]
        self.assertEqual(
            manifest["input.trace_manifest.sha256"],
            _sha256_file(self.trace_manifest),
        )
        self.assertEqual(
            manifest["input.truth.sha256"], _sha256_file(self.truth)
        )
        self.assertEqual(
            manifest["input.preregistration.sha256"],
            _sha256_file(self.preregistration),
        )
        self.assertEqual(
            manifest["expected_preregistration_sha256"],
            _sha256_file(self.preregistration),
        )
        self.assertEqual(
            manifest["expected_base_port"], str(self.BASE_PORT)
        )
        self.assertEqual(
            manifest["expected_sim_seed"], str(self.SIM_SEED)
        )
        self.assertEqual(
            manifest["publication_confidence_floor"], "0.2"
        )
        self.assertEqual(
            manifest["inference_semantics_version"],
            "joint_final_snapshot",
        )
        self.assertEqual(
            manifest["input.truth_soil.count"], str(self.COUNT)
        )
        joint_timeline = (
            self.root
            / "joint_timeseries/trace_0000_grit.csv"
        )
        self.assertEqual(
            manifest["input.joint_timeseries.trace_0000.sha256"],
            _sha256_file(joint_timeline),
        )
        source_trace = (
            self.collection / "raw/trace_0000/sensor_trace.csv"
        )
        self.assertEqual(
            manifest["input.source_trace.trace_0000.sha256"],
            _sha256_file(source_trace),
        )
        self.assertEqual(
            manifest["scorer_source_sha256"],
            _sha256_file(ROOT / "benchmarking/score_joint_estimator.py"),
        )
        self.assertEqual(
            manifest["output.decision.sha256"],
            _sha256_file(output / "decision.json"),
        )
        self.assertRegex(manifest["git_state_sha256"], r"^[0-9a-f]{64}$")

    def test_cli_rejects_dirty_scoring_worktree(self):
        dirty_state = {
            "git_head": "b" * 40,
            "git_dirty": "true",
            "git_status_porcelain": '[" M scorer.py"]',
            "git_state_sha256": "c" * 64,
        }
        with patch(
            "benchmarking.score_joint_estimator._git_state",
            return_value=dirty_state,
        ), self.assertRaisesRegex(SystemExit, "clean current Git worktree"):
            main(self._cli_arguments(self.root / "dirty_score"))

    def test_cli_rejects_unbound_preregistration(self):
        arguments = self._cli_arguments(self.root / "bad_prereg_score")
        index = arguments.index("--expected-preregistration-sha256") + 1
        arguments[index] = "f" * 64
        with self.assertRaisesRegex(SystemExit, "preregistration SHA-256"):
            main(arguments)


if __name__ == "__main__":
    unittest.main()
