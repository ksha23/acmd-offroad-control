"""Contracts for the truth-free live estimator diagnostics.

Benchmarks report how often the estimator was available to control, so these
tests establish that availability is read from the controller's complete
readiness decision rather than inferred. A diagnostic that disagrees with its
own snapshot, or claims an applied update without readiness, must fail closed
instead of being scored as a successful estimate.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from benchmarking.active_estimator_diagnostics import (
    live_estimator_diagnostics,
)
from benchmarking.common import GRIT_ESTIMATOR_BACKEND


def _diag(*, ready: bool = True) -> pd.DataFrame:
    readiness = [0, int(ready), int(ready)]
    reasons = ["no_snapshot", "none" if ready else "confidence",
               "none" if ready else "confidence"]
    return pd.DataFrame({
        "sim_time": [5.0, 6.0, 7.0],
        "terrain_update_applied": [0, int(ready), 0],
        "terrain_dynamics_windows": [0, 8, 9],
        "terrain_accepted_dynamics_windows": [0, 8, 10],
        "terrain_rejected_dynamics_windows": [0, 1, 2],
        "terrain_profile_force_gain": [1.0, 1.01, 1.02],
        "terrain_profile_ax_bias": [0.0, 0.01, 0.02],
        "terrain_profile_ay_bias": [0.0, -0.01, -0.02],
        "terrain_profile_bound_hits": [0, 0, 1],
        "terrain_feature_envelope_excursions": [0, 1, 2],
        "terrain_joint_snapshot_seq": [0, 1, 1],
        "terrain_joint_evidence_age_s": [np.nan, 0.1, 1.1],
        "terrain_joint_publication_ready": readiness,
        "terrain_joint_fallback_reason": reasons,
        "terrain_joint_snapshot_confidence": [np.nan, 0.3, 0.3],
        "terrain_joint_n_boundary_mass": [np.nan, 0.1, 0.1],
        "terrain_joint_phi_boundary_mass": [np.nan, 0.08, 0.08],
        "terrain_joint_max_boundary_mass": [np.nan, 0.1, 0.1],
        "terrain_joint_boundary_limited": [np.nan, 0, 0],
        "terrain_joint_observability_rank": [np.nan, 2, 2],
        "terrain_joint_observability_min_singular_value": [np.nan, 0.2, 0.2],
        "terrain_joint_projection_wall_ms": [np.nan, 2.0, 2.0],
        "terrain_joint_profile_wall_ms": [np.nan, 1.2, 1.2],
        "terrain_joint_observability_wall_ms": [np.nan, 0.3, 0.3],
        "terrain_joint_posterior_wall_ms": [np.nan, 5.0, 5.0],
        "terrain_joint_publication_wall_ms": [np.nan, 0.1, 0.1],
        "terrain_joint_update_wall_ms": [np.nan, 7.1, 7.1],
        "actual_Fx_front": [np.nan, np.nan, np.nan],
        "actual_Fx_rear": [np.nan, np.nan, np.nan],
        "actual_Fy_front": [np.nan, np.nan, np.nan],
        "actual_Fy_rear": [np.nan, np.nan, np.nan],
    })


class ActiveEstimatorDiagnosticsTest(unittest.TestCase):
    def test_joint_ready_snapshot_uses_eight_block_contract(self):
        result = live_estimator_diagnostics(
            _diag(),
            backend=GRIT_ESTIMATOR_BACKEND,
            enabled=True,
        )
        self.assertTrue(result["profile_estimator_diagnostics_complete"])
        self.assertTrue(result["profile_estimator_publication_ready"])
        self.assertTrue(result["profile_estimator_publication_applied"])
        self.assertEqual(
            result["profile_estimator_required_concurrent_windows"], 8
        )
        self.assertEqual(result["profile_estimator_final_snapshot_seq"], 1)
        self.assertEqual(result["profile_estimator_snapshot_rows"], 2)
        self.assertEqual(result["profile_estimator_unique_snapshot_count"], 1)
        self.assertEqual(result["profile_estimator_ready_snapshot_count"], 1)
        self.assertEqual(result["profile_estimator_applied_snapshot_count"], 1)
        self.assertAlmostEqual(
            result["profile_estimator_update_wall_ms_max"], 7.1
        )

    def test_timing_is_weighted_once_per_immutable_snapshot(self):
        seed = _diag().iloc[[1]].copy()
        no_snapshot = _diag().iloc[[0]].copy()
        rows = [no_snapshot]
        for offset in range(4):
            row = seed.copy()
            row["sim_time"] = 6.0 + offset
            row["terrain_update_applied"] = int(offset == 0)
            row["terrain_joint_evidence_age_s"] = 0.1 + offset
            rows.append(row)
        second = seed.copy()
        second["sim_time"] = 10.0
        second["terrain_update_applied"] = 1
        second["terrain_joint_snapshot_seq"] = 2
        second["terrain_joint_evidence_age_s"] = 0.1
        second["terrain_joint_projection_wall_ms"] = 4.0
        second["terrain_joint_posterior_wall_ms"] = 10.0
        second["terrain_joint_publication_wall_ms"] = 1.0
        second["terrain_joint_update_wall_ms"] = 15.0
        rows.append(second)
        frame = pd.concat(rows, ignore_index=True)

        result = live_estimator_diagnostics(
            frame,
            backend=GRIT_ESTIMATOR_BACKEND,
            enabled=True,
        )

        self.assertTrue(result["profile_estimator_diagnostics_complete"])
        self.assertEqual(result["profile_estimator_snapshot_rows"], 5)
        self.assertEqual(result["profile_estimator_unique_snapshot_count"], 2)
        self.assertEqual(result["profile_estimator_ready_snapshot_count"], 2)
        self.assertEqual(result["profile_estimator_applied_snapshot_count"], 2)
        self.assertAlmostEqual(
            result["profile_estimator_update_wall_ms_median"], 11.05
        )
        self.assertAlmostEqual(
            result["profile_estimator_update_wall_ms_p95"], 14.605
        )
        self.assertAlmostEqual(
            result["profile_estimator_update_wall_ms_max"], 15.0
        )

    def test_joint_abstention_is_complete_when_snapshot_gate_explains_it(self):
        result = live_estimator_diagnostics(
            _diag(ready=False),
            backend=GRIT_ESTIMATOR_BACKEND,
            enabled=True,
        )
        self.assertTrue(result["profile_estimator_diagnostics_complete"])
        self.assertFalse(result["profile_estimator_publication_ready"])
        self.assertTrue(result["profile_estimator_abstained"])

    def test_boundary_flag_mismatch_fails_closed(self):
        frame = _diag()
        frame.loc[1:, "terrain_joint_max_boundary_mass"] = 0.3
        result = live_estimator_diagnostics(
            frame,
            backend=GRIT_ESTIMATOR_BACKEND,
            enabled=True,
        )
        self.assertFalse(result["profile_estimator_diagnostics_complete"])
        self.assertEqual(
            result["profile_estimator_diagnostics_error"],
            "joint_snapshot_readiness_inconsistent",
        )

    def test_update_without_readiness_fails_closed(self):
        frame = _diag(ready=False)
        frame.loc[1, "terrain_update_applied"] = 1
        result = live_estimator_diagnostics(
            frame,
            backend=GRIT_ESTIMATOR_BACKEND,
            enabled=True,
        )
        self.assertFalse(result["profile_estimator_readiness_consistent"])

    def test_static_joint_arm_is_not_an_abstention(self):
        result = live_estimator_diagnostics(
            None,
            backend=GRIT_ESTIMATOR_BACKEND,
            enabled=False,
        )
        self.assertFalse(result["profile_estimator_diagnostics_applicable"])
        self.assertTrue(result["profile_estimator_diagnostics_complete"])
        self.assertFalse(result["profile_estimator_abstained"])


if __name__ == "__main__":
    unittest.main()
