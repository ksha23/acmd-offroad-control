#!/usr/bin/env python3
"""The vehicle constants every stage shares must not drift apart.

``param_consistency.HMMWV_VEHICLE_PARAMS`` is the declared single source of
truth for the controller, the estimators and the plant. Nothing enforced that,
so a divergence would have been silent: the estimator would keep publishing
confident soil parameters computed from a vehicle that is not the one being
driven. These tests pin the constants, pin the static wheel loads they imply,
and refuse a partial vehicle mapping, which is how a divergence would enter.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "simulation"))
sys.path.insert(0, os.path.join(ROOT, "simulation", "shared"))
sys.path.insert(0, os.path.join(ROOT, "simulation", "tire_models"))

from param_consistency import (  # noqa: E402
    HMMWV_VEHICLE_PARAMS,
    get_static_fz_per_wheel,
)
from four_wheel_projection import ProjectionVehicle  # noqa: E402

# Measured from the Chrono HMMWV this work drives:
#   veh.HMMWV_Full() -> GetVehicle().GetMass() == 2573.14 kg
CHRONO_HMMWV_TOTAL_MASS_KG = 2573.14

# The per-wheel static loads recorded in the frozen joint (n, phi) evidence.
# Every accepted cell of the published confirmation carries exactly these,
# because the deployed estimator runs at a static load split.
FROZEN_FRONT_FZ_N = 6531.9641384009701
FROZEN_REAR_FZ_N = 6088.6008615990313


class VehicleParameterConsistencyTest(unittest.TestCase):
    def test_mass_matches_the_chrono_plant(self):
        self.assertAlmostEqual(
            float(HMMWV_VEHICLE_PARAMS["M"]), CHRONO_HMMWV_TOTAL_MASS_KG,
            delta=0.5,
            msg="the shared vehicle mass no longer matches the Chrono HMMWV",
        )

    def test_wheelbase_is_internally_consistent(self):
        self.assertAlmostEqual(
            float(HMMWV_VEHICLE_PARAMS["L"]),
            float(HMMWV_VEHICLE_PARAMS["Lf"]) + float(HMMWV_VEHICLE_PARAMS["Lr"]),
            places=9,
        )

    def test_static_loads_match_the_frozen_evidence(self):
        front, rear = get_static_fz_per_wheel()
        self.assertAlmostEqual(front, FROZEN_FRONT_FZ_N, places=6)
        self.assertAlmostEqual(rear, FROZEN_REAR_FZ_N, places=6)

    def test_projection_defaults_track_the_source_of_truth(self):
        """A default that drifts from the shared struct is a silent hazard."""
        default = ProjectionVehicle()
        for field, key in (
            ("m", "M"), ("Iz", "Izz"), ("Lf", "Lf"), ("Lr", "Lr"),
            ("track", "T"), ("h_cg", "h_cg"),
        ):
            self.assertAlmostEqual(
                getattr(default, field), float(HMMWV_VEHICLE_PARAMS[key]),
                places=9, msg=f"ProjectionVehicle.{field} drifted from {key}",
            )

    def test_from_mapping_refuses_a_partial_vehicle(self):
        """Silently substituting a default is how the two sets diverged."""
        partial = dict(HMMWV_VEHICLE_PARAMS)
        partial.pop("h_cg")
        with self.assertRaises(KeyError):
            ProjectionVehicle.from_mapping(partial)

    def test_from_mapping_accepts_the_canonical_mapping(self):
        vehicle = ProjectionVehicle.from_mapping(dict(HMMWV_VEHICLE_PARAMS))
        self.assertAlmostEqual(vehicle.m, float(HMMWV_VEHICLE_PARAMS["M"]))
        self.assertAlmostEqual(vehicle.h_cg, float(HMMWV_VEHICLE_PARAMS["h_cg"]))


if __name__ == "__main__":
    unittest.main()
