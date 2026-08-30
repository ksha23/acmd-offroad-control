"""Contracts on the two modules that publish the Table 2 evidence.

`score_joint_evidence.py` computes the paired-bootstrap improvement intervals
the paper quotes, and `joint_n_phi_evidence.py` is the frozen-artifact hash
gate protecting the locked confirmation. Until the 2026-08-26 audit neither
had a single test: flipping the bootstrap's sign convention and disabling the
hash-mismatch raise each survived the full suite. These tests pin both.
"""
from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np

from benchmarking.joint_n_phi_evidence import (
    ARTIFACTS, AUTHORITATIVE_JOINT_N_PHI_RESULT, EXPECTED_RESULT_NAME,
    validate_joint_n_phi_evidence,
)
from benchmarking.score_joint_evidence import _paired_bootstrap

ROOT = Path(__file__).resolve().parents[2]


class PairedBootstrapSignTest(unittest.TestCase):
    def test_positive_delta_means_the_joint_estimator_is_better(self):
        # The paper reads the interval as "GRIT's improvement over the
        # parent": when the joint estimator's errors are uniformly smaller,
        # the point estimate and the whole interval must be positive. A sign
        # flip here inverts every published CI while nothing else fails.
        rng = np.random.default_rng(0)
        joint = np.full(40, 1.0)
        parent = np.full(40, 3.0)
        point, low, high = _paired_bootstrap(rng, joint, parent)
        self.assertAlmostEqual(point, 2.0, places=9)
        self.assertAlmostEqual(low, 2.0, places=9)
        self.assertAlmostEqual(high, 2.0, places=9)

    def test_seeded_reproducibility(self):
        joint = np.linspace(0.5, 1.5, 30)
        parent = np.linspace(1.0, 3.0, 30)
        a = _paired_bootstrap(np.random.default_rng(7), joint, parent)
        b = _paired_bootstrap(np.random.default_rng(7), joint, parent)
        self.assertEqual(a, b)


class EvidenceHashGateTest(unittest.TestCase):
    def setUp(self):
        if not AUTHORITATIVE_JOINT_N_PHI_RESULT.is_dir():
            self.skipTest("locked evidence directory not present")

    def test_authoritative_directory_validates(self):
        record = validate_joint_n_phi_evidence()
        self.assertEqual(len(record["artifacts"]), len(ARTIFACTS))

    def test_single_tampered_byte_is_rejected(self):
        # Build a shadow copy inside the repo (the validator resolves paths
        # relative to ROOT): symlink every locked artifact except one, which
        # is copied and flipped by one byte. The gate must raise, not pass.
        base = ROOT / "benchmarking" / "results" / "_evidence_gate_test"
        shadow = base / EXPECTED_RESULT_NAME
        if base.exists():
            shutil.rmtree(base)
        try:
            for spec in ARTIFACTS.values():
                rel = spec["relative_path"]
                src = AUTHORITATIVE_JOINT_N_PHI_RESULT / rel
                dst = shadow / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src)
            victim = shadow / ARTIFACTS["scored_runs"]["relative_path"]
            data = bytearray(victim.resolve().read_bytes())
            victim.unlink()
            data[len(data) // 2] ^= 0x01
            victim.write_bytes(bytes(data))
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_joint_n_phi_evidence(shadow)
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
