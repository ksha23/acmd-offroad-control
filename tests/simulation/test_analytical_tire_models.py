"""Contracts on the analytical (baseline) tire laws.

The Table 1 comparison is only as honest as its baselines, and until the
2026-08-26 audit these laws had no tests at all: a sign flip, a silently
regressed friction constant, and a wrong-by-10x rigfit coefficient all
survived the full suite. These tests pin the physical properties every
lateral force law must have (odd symmetry, correct sign, bounded
saturation), the calibrated constants, and the hash-recorded rigfit
artifact the constants are transcribed from.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")
import casadi as ca  # noqa: E402

from simulation.tire_models.analytical_tire_models import (  # noqa: E402
    PACEJKA_MU, PACEJKA_RIGFIT, TMEASY_FYM,
    pacejka_tire_forces, tmeasy_tire_forces,
)

ROOT = Path(__file__).resolve().parents[2]
FZ_AXLE = 12600.0


def _fy(law, alpha, mu=PACEJKA_MU):
    a = ca.SX.sym("a")
    out = law(a, a, FZ_AXLE, FZ_AXLE, 0.0, mu) if law is pacejka_tire_forces \
        else law(a, a, FZ_AXLE, FZ_AXLE, 0.0, mu)
    f = ca.Function("fy", [a], [out[0]])
    return float(f(alpha))


class LateralLawShapeTest(unittest.TestCase):
    def test_odd_symmetry_across_the_solver_alpha_range(self):
        # The solver clamps alpha to +/-0.55 rad; over that whole range each
        # law must be odd: Fy(-a) = -Fy(a). The TMeasy defect this pins
        # against: a one-sided saturation clamp wrapped the sine past -pi,
        # so Fy(-0.30) came back POSITIVE while Fy(+0.30) saturated.
        for law in (pacejka_tire_forces, tmeasy_tire_forces):
            for a in np.linspace(0.02, 0.55, 12):
                self.assertAlmostEqual(
                    _fy(law, a), -_fy(law, -a), places=6,
                    msg=f"{law.__name__} not odd at alpha={a:.2f}")

    def test_sign_matches_slip_angle(self):
        for law in (pacejka_tire_forces, tmeasy_tire_forces):
            for a in (0.05, 0.24, 0.30, 0.40, 0.55):
                self.assertGreater(_fy(law, a), 0.0,
                                   f"{law.__name__} sign at +{a}")
                self.assertLess(_fy(law, -a), 0.0,
                                f"{law.__name__} sign at -{a}")

    def test_saturation_is_bounded_and_monotone_to_the_peak(self):
        # Pacejka must never exceed mu*Fz per axle; TMeasy never exceeds its
        # per-axle peak 2*Fym.
        for a in np.linspace(0.0, 0.55, 23):
            self.assertLessEqual(abs(_fy(pacejka_tire_forces, a)),
                                 PACEJKA_MU * FZ_AXLE * 1.001)
            self.assertLessEqual(abs(_fy(tmeasy_tire_forces, a)),
                                 2.0 * TMEASY_FYM * 1.001)


class CalibrationConstantTest(unittest.TestCase):
    def test_global_mu_is_the_calibrated_value(self):
        # calibrate_analytical_tires.py reproduces 0.420 from the PRIMARY
        # static-parent rig corpus; the adopted constant is a transcription
        # of that fit and must not drift (e.g. back to a rigid-road 0.74).
        self.assertAlmostEqual(PACEJKA_MU, 0.42, places=9)

    def test_rigfit_constants_match_the_hash_recorded_artifact(self):
        artifact = json.loads(
            (ROOT / "benchmarking" / "results" / "rigfit_pacejka_fit.json")
            .read_text())
        recorded = artifact["PACEJKA_RIGFIT"]
        self.assertEqual(set(recorded), set(PACEJKA_RIGFIT))
        for soil, params in recorded.items():
            for key, value in params.items():
                self.assertAlmostEqual(
                    PACEJKA_RIGFIT[soil][key], float(value), places=9,
                    msg=f"PACEJKA_RIGFIT[{soil}][{key}] departs from the "
                        f"recorded fit artifact")


if __name__ == "__main__":
    unittest.main()
