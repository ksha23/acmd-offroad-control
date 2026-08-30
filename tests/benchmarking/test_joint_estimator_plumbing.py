"""Contracts binding the joint estimator's live and replay configurations.

Replay evidence only transfers to the deployed system if both paths construct
the same estimator, so these tests establish that the live controller and the
replayer share one constructor contract, that the launcher and controller
defaults select the active backend, and that the replay factory ignores
tuning belonging to other backends. They also confirm the observation filter
withholds the wheel-centre elevation channel, which would act as a ground
datum.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarking.common import (
    RIG_ACTIVE_ESTIMATOR_BACKEND,
    GRIT_ESTIMATOR_BACKEND,
    GRIT_ESTIMATOR_CONTRACT,
    PARENT_ESTIMATOR_BACKEND,
    estimator_contract,
    estimator_runtime_args,
    estimator_artifact_hashes,
)
from benchmarking.terrain_estimator_replay import (
    BACKEND_LABELS,
    ReplayConfig,
    _grit_constructor_kwargs as replay_joint_kwargs,
    _terrain_estimator_observation_for_backend as replay_observation,
    _write_manifest,
    make_estimator,
)
from simulation.control.acados_mpc_controller_node import (
    RIG_ACTIVE_ESTIMATOR_BACKEND as CONTROLLER_ACTIVE_BACKEND,
    RIG_JOINT_BOUNDARY_MASS_LIMIT,
    RIG_JOINT_CONTROL_MIN_PHI_DEG,
    RIG_JOINT_MAX_EVIDENCE_AGE_S,
    RIG_JOINT_MIN_OBSERVABILITY_RANK,
    RIG_JOINT_MIN_OBSERVABILITY_SINGULAR_VALUE,
    RIG_JOINT_MIN_PUBLICATION_CONFIDENCE,
    RIG_JOINT_FALLBACK_N,
    RIG_JOINT_FALLBACK_PHI_DEG,
    TERRAIN_ESTIMATOR_BACKENDS as CONTROLLER_BACKENDS,
    _grit_constructor_kwargs as controller_joint_kwargs,
    _terrain_estimator_observation_for_backend as controller_observation,
    main as controller_main,
)
from simulation.runtime.launch_decoupled import (
    RIG_ACTIVE_ESTIMATOR_BACKEND as LAUNCHER_ACTIVE_BACKEND,
    TERRAIN_ESTIMATOR_BACKENDS as LAUNCHER_BACKENDS,
    main as launcher_main,
)


class JointRuntimePlumbingTest(unittest.TestCase):
    def test_joint_estimator_supports_flat_controller_import(self):
        estimator_dir = (
            Path(__file__).resolve().parents[2]
            / "simulation"
            / "estimators"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(estimator_dir),
                environment.get("PYTHONPATH", ""),
            ]
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from grit_terrain_estimator import "
                    "GritTerrainEstimator; "
                    "assert GritTerrainEstimator.__name__"
                ),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_joint_is_active_without_relabeling_historical_scalar_parent(self):
        self.assertEqual(PARENT_ESTIMATOR_BACKEND, "scalar_parent")
        self.assertEqual(GRIT_ESTIMATOR_BACKEND, "grit")
        self.assertEqual(
            RIG_ACTIVE_ESTIMATOR_BACKEND,
            GRIT_ESTIMATOR_BACKEND,
        )
        self.assertEqual(CONTROLLER_ACTIVE_BACKEND, RIG_ACTIVE_ESTIMATOR_BACKEND)
        self.assertEqual(LAUNCHER_ACTIVE_BACKEND, RIG_ACTIVE_ESTIMATOR_BACKEND)
        self.assertIn(GRIT_ESTIMATOR_BACKEND, CONTROLLER_BACKENDS)
        self.assertIn(GRIT_ESTIMATOR_BACKEND, LAUNCHER_BACKENDS)
        self.assertIn(GRIT_ESTIMATOR_BACKEND, BACKEND_LABELS)
        self.assertEqual(
            GRIT_ESTIMATOR_CONTRACT["promotion_status"],
            "active_paper_backend",
        )

    def test_controller_and_launcher_cli_defaults_select_active_alias(self):
        class ParserCaptured(RuntimeError):
            pass

        def capture_default(entrypoint):
            captured = {}

            def parse_args(parser, *args, **kwargs):
                action = next(
                    item
                    for item in parser._actions
                    if item.dest == "terrain_estimator_backend"
                )
                captured["default"] = action.default
                raise ParserCaptured

            with patch("argparse.ArgumentParser.parse_args", new=parse_args):
                with self.assertRaises(ParserCaptured):
                    entrypoint()
            return captured["default"]

        self.assertEqual(
            capture_default(controller_main),
            RIG_ACTIVE_ESTIMATOR_BACKEND,
        )
        self.assertEqual(
            capture_default(launcher_main),
            RIG_ACTIVE_ESTIMATOR_BACKEND,
        )

    def test_live_and_replay_constructor_contracts_are_identical(self):
        initial = {"n": 0.7, "phi": 29.0}
        vehicle = {"mass": 1_000.0}
        live = controller_joint_kwargs(initial, vehicle)
        replay = replay_joint_kwargs(initial, vehicle)
        self.assertEqual(live, replay)

        self.assertEqual(Path(live["model_dir"]).name, "tire_force_rate")
        self.assertEqual(live["grid_size"], 41)
        self.assertEqual(live["phi_grid_size"], 17)
        self.assertEqual(live["n_bounds"], (0.40, 1.10))
        self.assertEqual(live["manifold_soft_floor"], 0.40)
        self.assertEqual(live["manifold_soft_mode"], "hold")
        self.assertEqual(live["r_ay"], 0.45)
        self.assertEqual(live["r_ax"], 0.35)
        self.assertEqual(live["phi_bounds_deg"], (6.0, 37.8))
        self.assertEqual(live["min_windows"], 8)
        self.assertEqual(live["horizon"], 8.0)
        self.assertEqual(live["load_transfer_mode"], "static")
        self.assertEqual(live["posterior_summary"], "mean")
        self.assertEqual(live["rate_mode"], "zero")
        self.assertEqual(live["min_observability_rank"], 2)
        self.assertEqual(live["min_observability_singular_value"], 0.10)
        self.assertEqual(live["cohesion_grid_size"], 1)
        contract = GRIT_ESTIMATOR_CONTRACT
        self.assertEqual(
            contract["accepted_snapshot_version"],
            "grit_accepted",
        )
        self.assertEqual(
            RIG_JOINT_MIN_PUBLICATION_CONFIDENCE,
            contract["controller_min_confidence"],
        )
        self.assertEqual(
            RIG_JOINT_MAX_EVIDENCE_AGE_S,
            contract["publication_max_evidence_age_s"],
        )
        self.assertEqual(
            RIG_JOINT_MIN_OBSERVABILITY_RANK,
            contract["publication_min_observability_rank"],
        )
        self.assertEqual(
            RIG_JOINT_MIN_OBSERVABILITY_SINGULAR_VALUE,
            contract["publication_min_observability_singular_value"],
        )
        self.assertEqual(
            RIG_JOINT_BOUNDARY_MASS_LIMIT,
            contract["publication_boundary_mass_limit"],
        )
        self.assertEqual(RIG_JOINT_FALLBACK_N, contract["fallback_n"])
        self.assertEqual(
            RIG_JOINT_FALLBACK_PHI_DEG,
            contract["fallback_phi_deg"],
        )
        self.assertEqual(
            RIG_JOINT_CONTROL_MIN_PHI_DEG,
            contract["control_min_phi_deg"],
        )

        constructor_keys = " ".join(live).lower()
        for forbidden in (
            "truth",
            "oracle",
            "ground",
            "datum",
            "height",
            "sinkage",
            "tire_force",
            "contact_force",
        ):
            self.assertNotIn(forbidden, constructor_keys)

    def test_joint_observation_filter_excludes_datum_and_unused_channels(self):
        observation = {
            "kappa": 0.1,
            "alpha_f": 0.05,
            "alpha_r": 0.03,
            "u": 5.0,
            "Fz_f": 5_500.0,
            "Fz_r": 5_700.0,
            "sr": 0.0,
            "alpha_rate_r": 0.0,
            "ay_imu": 0.4,
            "omega_dot": 0.0,
            "omega": 0.1,
            "v_lateral": 0.2,
            "sim_time": 1.0,
            "steering_angle": 0.08,
            "wheel_omegas": (11.0, 11.0, 11.0, 11.0),
            "ax_imu": 0.2,
            "wheel_center_heights": (0.5, 0.5, 0.5, 0.5),
            "x_pos": 2.0,
            "drive_torques": (1.0, 1.0, 1.0, 1.0),
        }
        for filter_observation in (controller_observation, replay_observation):
            filtered = filter_observation(
                GRIT_ESTIMATOR_BACKEND,
                observation,
            )
            self.assertIn("wheel_omegas", filtered)
            self.assertIn("ax_imu", filtered)
            self.assertNotIn("wheel_center_heights", filtered)
            self.assertNotIn("x_pos", filtered)
            self.assertNotIn("drive_torques", filtered)
            self.assertEqual(
                filter_observation("scalar_parent", observation),
                observation,
            )

    def test_replay_factory_ignores_generic_model_and_parent_tuning(self):
        class FakeJointEstimator:
            output_names = ("n", "phi")

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_module = SimpleNamespace(
            GritTerrainEstimator=FakeJointEstimator
        )
        config = ReplayConfig(
            model_dir="/tmp/not_the_joint_model",
            dynamics_min_windows=99,
            dynamics_min_yaw_rate_rms=0.0,
        )
        with patch.dict(
            sys.modules,
            {"grit_terrain_estimator": fake_module},
        ):
            estimator = make_estimator(GRIT_ESTIMATOR_BACKEND, config)
        self.assertEqual(estimator.output_names, ("n", "phi"))
        self.assertEqual(
            Path(estimator.kwargs["model_dir"]).name, "tire_force_rate"
        )
        self.assertEqual(estimator.kwargs["min_windows"], 8)
        self.assertEqual(estimator.kwargs["min_yaw_rate_rms"], 0.015)

    def test_active_contract_and_manifest_are_explicit(self):
        contract = estimator_contract(GRIT_ESTIMATOR_BACKEND)
        self.assertEqual(contract["output_names"], ["n", "phi"])
        self.assertFalse(contract["requires_ground_datum"])
        self.assertEqual(contract["truth_inputs"], "none")
        self.assertEqual(contract["max_final_update_age_s"], 3.5)
        self.assertEqual(
            estimator_runtime_args(GRIT_ESTIMATOR_BACKEND),
            [
                "--terrain-estimator-prior",
                "dirt",
                "--te-min-confidence",
                "0.2",
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_manifest = root / "trace_manifest.csv"
            trace_manifest.write_text("trace_id\ntrace_0000\n", encoding="utf-8")
            output = root / "replay_manifest.csv"
            args = SimpleNamespace(
                trace_manifest=trace_manifest,
                backends=[GRIT_ESTIMATOR_BACKEND],
            )
            _write_manifest(output, args, ReplayConfig())
            text = output.read_text(encoding="utf-8")
        self.assertIn(
            "joint_contract.contract_version,"
            "'independent_n_phi_joint_profile'",
            text,
        )
        self.assertIn("joint_contract.force_model_dir", text)

    def test_active_joint_hash_contract_excludes_scalar_static_map(self):
        hashes = estimator_artifact_hashes(RIG_ACTIVE_ESTIMATOR_BACKEND)
        self.assertEqual(
            set(hashes),
            {"rate_checkpoint_sha256", "rate_scalers_sha256"},
        )
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))


if __name__ == "__main__":
    unittest.main()
