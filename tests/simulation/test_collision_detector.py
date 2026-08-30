"""Contracts fixing collision truth to native Chrono body contact.

Every collision count the manuscript reports is native ego-body to
obstacle-body contact, so these tests establish that a geometric proximity
never becomes a collision, that repeated contact with one obstacle counts
once, and that traffic contacts are attributed to the correct vehicle.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pychrono as chrono

from simulation.shared.collision_detector import CollisionLogger


def _box(system, name: str, size: float, x: float, fixed: bool = False):
    material = chrono.ChContactMaterialSMC()
    body = chrono.ChBodyEasyBox(size, size, size, 1000.0, True, True, material)
    body.SetName(name)
    body.SetPos(chrono.ChVector3d(x, 0, 0))
    body.SetFixed(fixed)
    system.Add(body)
    return body


class CollisionLoggerTest(unittest.TestCase):
    def _system(self):
        system = chrono.ChSystemSMC()
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, 0))
        return system

    def test_proxy_overlap_is_not_a_collision(self):
        system = self._system()
        ego = _box(system, "ego", 0.2, 0.0)
        rock = _box(system, "rock", 0.2, 1.0, fixed=True)
        with tempfile.TemporaryDirectory() as tmp:
            logger = CollisionLogger(
                system=system,
                ego_body_ids={ego.GetIdentifier()},
                rocks=[{"body": rock, "x": 1.0, "y": 0.0, "size": 0.2}],
                run_dir=tmp,
            )
            system.DoStepDynamics(0.001)
            result = logger.check(system.GetChTime(), 0.0, 0.0)
            summary = logger.close()
        # The bodies are 0.8 m apart at their surfaces, so Chrono reports no
        # contact. A proximity threshold on centre distance would classify
        # this as a collision, which is why collision truth is native contact
        # and clearance is reported separately as a near miss.
        self.assertFalse(result["any_collision"])
        self.assertTrue(result["any_near_miss"])
        self.assertEqual(summary["total_collisions"], 0)

    def test_body_contact_counts_one_unique_obstacle(self):
        system = self._system()
        ego = _box(system, "ego", 2.0, 0.0)
        rock = _box(system, "rock", 2.0, 1.5, fixed=True)
        with tempfile.TemporaryDirectory() as tmp:
            logger = CollisionLogger(
                system=system,
                ego_body_ids={ego.GetIdentifier()},
                rocks=[{"body": rock, "x": 1.5, "y": 0.0, "size": 2.0}],
                run_dir=tmp,
            )
            system.DoStepDynamics(0.001)
            first = logger.check(system.GetChTime(), 0.0, 0.0)
            system.DoStepDynamics(0.001)
            second = logger.check(system.GetChTime(), 0.0, 0.0)
            summary = logger.close()
            with (Path(tmp) / "collision_log.csv").open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
        self.assertTrue(first["any_collision"])
        self.assertEqual(first["new_collision_ids"], [0])
        self.assertEqual(second["new_collision_ids"], [])
        self.assertEqual(summary["total_collisions"], 1)
        contact_rows = [row for row in rows if row["is_collision"] == "1"]
        self.assertEqual(len(contact_rows), 1)
        self.assertEqual(contact_rows[0]["collision_source"],
                         "chrono_body_contact")
        self.assertGreater(int(contact_rows[0]["contact_count"]), 0)

    def test_hit_obstacle_is_not_also_a_near_miss(self):
        # An obstacle passes through the near band on every approach, so if
        # near misses were not made disjoint from hits, every collision would
        # also register one near miss and the two KPIs would double-count.
        system = self._system()
        ego = _box(system, "ego", 2.0, 0.0)
        rock = _box(system, "rock", 2.0, 3.4, fixed=True)
        with tempfile.TemporaryDirectory() as tmp:
            logger = CollisionLogger(
                system=system,
                ego_body_ids={ego.GetIdentifier()},
                rocks=[{"body": rock, "x": 3.4, "y": 0.0, "size": 2.0}],
                run_dir=tmp,
            )
            # Step 1: near but not touching (surfaces 1.4 m apart; proxy
            # clearance 3.4 - 1.0 - 1.5 = 0.9 m, inside the 1.0 m near band).
            system.DoStepDynamics(0.001)
            first = logger.check(system.GetChTime(), 0.0, 0.0)
            # Step 2: drive the ego into contact with the same rock.
            ego.SetPos(chrono.ChVector3d(2.0, 0, 0))
            system.DoStepDynamics(0.001)
            second = logger.check(system.GetChTime(), 2.0, 0.0)
            summary = logger.close()
        self.assertFalse(first["any_collision"])
        self.assertTrue(first["any_near_miss"])
        self.assertTrue(second["any_collision"])
        self.assertEqual(summary["total_collisions"], 1)
        self.assertEqual(summary["total_near_misses"], 0)

    def test_traffic_body_contact_uses_logical_vehicle_id(self):
        system = self._system()
        ego = _box(system, "ego", 2.0, 0.0)
        traffic = _box(system, "traffic", 2.0, 1.5, fixed=True)
        with tempfile.TemporaryDirectory() as tmp:
            logger = CollisionLogger(
                system=system,
                ego_body_ids={ego.GetIdentifier()},
                rocks=[],
                traffic_body_map={traffic.GetIdentifier(): 1000},
                run_dir=tmp,
            )
            system.DoStepDynamics(0.001)
            result = logger.check(
                system.GetChTime(), 0.0, 0.0,
                extra_obstacles=[(1.5, 0.0, 1.0)],
            )
            summary = logger.close()
        self.assertEqual(result["new_collision_ids"], [1000])
        self.assertEqual(summary["total_collisions"], 1)


if __name__ == "__main__":
    unittest.main()
