"""Fail-closed and resume contracts for the paper benchmark campaign.

A campaign runs for hours across parallel workers and may be resumed after an
interruption, so these tests establish that resumption cannot quietly change
what was measured: the design is frozen across a resume, every successful row
belongs to the requested matrix, duplicates and misrouted rows are refused,
and worker batching never lets two concurrent runs share a DDS domain.
"""

from __future__ import annotations

import argparse
import copy
import unittest
from argparse import Namespace

import pandas as pd

from benchmarking.common import (
    RIG_ACTIVE_ESTIMATOR_BACKEND,
    PARENT_ESTIMATOR_BACKEND,
    bounded_ros_workers,
    require_active_joint_estimator,
    estimator_contract,
)
from benchmarking.tire_model_with_estimator_ablation import (
    _Task,
    _conditioning_arm_roles,
    _domain_safe_batches,
    _resume_manifest_mismatches,
    _validated_successful_resume_rows,
)


def _task(
    *,
    idx: int = 0,
    sim_port: int = 22000,
    variant: str = "nn_estimator",
) -> _Task:
    return _Task(
        idx=idx,
        variant=variant,
        mpc_model="nn",
        nn_model="tire_force_static",
        extra=(),
        terrain="clay",
        path="sinusoidal",
        speed=5.0,
        bumpiness=0,
        seed=400,
        run_dir_str=f"/tmp/v5/raw/{idx:04d}_{variant}",
        sim_port=sim_port,
        ctrl_port=sim_port + 1,
        sim_time=20.0,
        timeout=500.0,
        lead_in=5.0,
        metric_start=8.0,
        estimator_backend=RIG_ACTIVE_ESTIMATOR_BACKEND,
        estimator_enabled=True,
        conditioning_role="promoted independent-n-phi online estimator",
        controller_prior="dirt",
    )


def _successful_record(task: _Task) -> dict[str, object]:
    return {
        "experiment": "tire_model_with_estimator_ablation",
        "variant": task.variant,
        "controller_mode": "standard",
        "mpc_model": task.mpc_model,
        "nn_model": task.nn_model,
        "terrain": task.terrain,
        "path": task.path,
        "speed_mps": task.speed,
        "bumpiness": task.bumpiness,
        "seed": task.seed,
        "run_dir": task.run_dir_str,
        "status": "ok",
        "extra_launch_identity_match": True,
        "extra_observed_path": task.path,
        "extra_observed_speed_mps": task.speed,
        "extra_observed_sim_seed": task.seed,
        "extra_observed_ros_domain_id": task.sim_port % 101,
        "extra_observed_sim_ports": repr([task.sim_port]),
        "extra_observed_ctrl_ports": repr([task.ctrl_port]),
        "extra_estimator_backend": task.estimator_backend,
        "extra_estimator_contract_version": estimator_contract(
            task.estimator_backend
        )["contract_version"],
        "extra_conditioning_role": task.conditioning_role,
        "extra_controller_prior_terrain": task.controller_prior,
        "extra_reference_policy": "shared_worst_case_phi13_curvature_v1",
        "extra_truth_free_controller_packet": True,
        "extra_controller_tire_force_truth_rows": 0,
    }


class PaperCampaignRobustnessTest(unittest.TestCase):
    def test_worker_count_is_bounded_by_dds_domain_pool(self):
        self.assertEqual(bounded_ros_workers("1"), 1)
        self.assertEqual(bounded_ros_workers("101"), 101)
        for value in ("0", "102", "1.5", "not-an-int"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    bounded_ros_workers(value)

    def test_v5_backend_accepts_only_active_joint(self):
        self.assertEqual(
            require_active_joint_estimator(RIG_ACTIVE_ESTIMATOR_BACKEND),
            RIG_ACTIVE_ESTIMATOR_BACKEND,
        )
        with self.assertRaises(RuntimeError):
            require_active_joint_estimator(PARENT_ESTIMATOR_BACKEND)

    def test_conditioning_manifest_uses_only_requested_arms(self):
        roles = _conditioning_arm_roles(
            ["nn_parent_estimator", "nn_fixed_fallback"]
        )
        self.assertEqual(
            list(roles),
            ["nn_parent_estimator", "nn_fixed_fallback"],
        )
        self.assertNotIn("nn_estimator", roles)
        self.assertNotIn("nn_static", roles)

    def test_resume_manifest_freezes_design_but_not_execution_tuning(self):
        args = Namespace(
            design_version="conditioning",
            variants=["nn_static", "nn_estimator"],
            source_sha256={"source.py": "a" * 64},
            code_git_head="b" * 40,
            base_port=22000,
            workers=8,
            timeout=500.0,
            resume_dir="/tmp/v5",
        )
        manifest = {key: repr(value) for key, value in vars(args).items()}
        self.assertEqual(_resume_manifest_mismatches(manifest, args), [])

        tuned = copy.deepcopy(args)
        tuned.workers = 4
        tuned.timeout = 900.0
        self.assertEqual(_resume_manifest_mismatches(manifest, tuned), [])

        changed = copy.deepcopy(args)
        changed.variants = ["nn_estimator"]
        self.assertTrue(_resume_manifest_mismatches(manifest, changed))
        changed = copy.deepcopy(args)
        changed.source_sha256["source.py"] = "c" * 64
        self.assertTrue(_resume_manifest_mismatches(manifest, changed))

    def test_resume_rejects_unknown_duplicate_and_misrouted_successes(self):
        task = _task()
        record = _successful_record(task)
        kept = _validated_successful_resume_rows(
            pd.DataFrame([record]), [task]
        )
        self.assertEqual(len(kept), 1)

        outside = dict(record)
        outside["seed"] = 999
        with self.assertRaisesRegex(RuntimeError, "out-of-matrix"):
            _validated_successful_resume_rows(
                pd.DataFrame([outside]), [task]
            )

        with self.assertRaisesRegex(RuntimeError, "duplicate successful"):
            _validated_successful_resume_rows(
                pd.DataFrame([record, record]), [task]
            )

        misrouted = dict(record)
        misrouted["extra_observed_ros_domain_id"] = (
            int(record["extra_observed_ros_domain_id"]) + 1
        ) % 101
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            _validated_successful_resume_rows(
                pd.DataFrame([misrouted]), [task]
            )

    def test_sparse_resume_batches_never_repeat_a_domain(self):
        tasks = [
            _task(idx=0, sim_port=10),
            _task(idx=1, sim_port=11),
            _task(idx=2, sim_port=111),
            _task(idx=3, sim_port=12),
        ]
        batches = _domain_safe_batches(tasks, width=4)
        self.assertEqual(
            [task for batch in batches for task in batch],
            tasks,
        )
        self.assertGreater(len(batches), 1)
        for batch in batches:
            domains = [task.sim_port % 101 for task in batch]
            self.assertEqual(len(domains), len(set(domains)))


if __name__ == "__main__":
    unittest.main()
