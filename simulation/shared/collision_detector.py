#!/usr/bin/env python3
"""Chrono body-contact collision logging for the HMMWV benchmarks.

Collision truth comes exclusively from Chrono's contact container: a collision
is recorded only where one of the ego vehicle's collision-enabled bodies is in
a reported contact with a registered rock or traffic-vehicle body. A 1.5 m
chassis-centre threshold serves as a clearance proxy for the continuous
clearance and near-miss diagnostics, and never sets ``is_collision``. Keeping
the two separate means no geometric heuristic can manufacture a collision that
the physics did not produce.

The reported metric is the number of distinct logical obstacles contacted
during a run. Multiple simultaneous contact points, contact persisting across
physics steps, and a later re-contact with the same obstacle each count once,
so the metric measures how many obstacles were hit rather than how long
contact lasted. The CSV records the first step of every contact episode
together with the Chrono contact count and peak contact force as evidence.
"""

import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401

import csv
import math
import os
from typing import Optional

import numpy as np
import pychrono as chrono


# Circular ground-plane proxy, used only for clearance and near-miss
# reporting. It is not a collision radius: collision truth is the Chrono
# body-contact query below.
VEHICLE_CLEARANCE_RADIUS = 1.5  # metres
NEAR_MISS_MARGIN = 1.0          # metres beyond the clearance proxy
COLLISION_SOURCE = "chrono_body_contact"


class _EgoObstacleContactReporter(chrono.ReportContactCallback):
    """Collect contacts between registered ego and obstacle body identifiers."""

    def __init__(self, ego_body_ids: set[int], obstacle_body_map: dict[int, int]):
        super().__init__()
        self.ego_body_ids = set(int(v) for v in ego_body_ids)
        self.obstacle_body_map = {int(k): int(v) for k, v in obstacle_body_map.items()}
        self.contacts: dict[int, dict] = {}

    def reset(self) -> None:
        self.contacts.clear()

    @staticmethod
    def _body(contactable):
        try:
            body = chrono.CastToChBody(contactable)
            # A failed SWIG shared_ptr cast can survive as a proxy until a
            # method is called, so validate it here.
            body.GetIdentifier()
            return body
        except Exception:
            return None

    def OnReportContact(self, p_a, p_b, plane_coord, distance, eff_radius,
                        react_forces, react_torques, contact_a, contact_b,
                        constraint_offset):
        body_a = self._body(contact_a)
        body_b = self._body(contact_b)
        if body_a is None or body_b is None:
            return True

        id_a = int(body_a.GetIdentifier())
        id_b = int(body_b.GetIdentifier())
        if id_a in self.ego_body_ids and id_b in self.obstacle_body_map:
            obstacle_id = self.obstacle_body_map[id_b]
            ego_body, obstacle_body = body_a, body_b
        elif id_b in self.ego_body_ids and id_a in self.obstacle_body_map:
            obstacle_id = self.obstacle_body_map[id_a]
            ego_body, obstacle_body = body_b, body_a
        else:
            return True

        try:
            force_world = plane_coord * react_forces
            force_n = math.sqrt(force_world.x ** 2 + force_world.y ** 2
                                + force_world.z ** 2)
        except Exception:
            force_n = math.nan
        rec = self.contacts.setdefault(obstacle_id, {
            "contact_count": 0,
            "max_contact_force_n": 0.0,
            "ego_body": str(ego_body.GetName()),
            "obstacle_body": str(obstacle_body.GetName()),
        })
        rec["contact_count"] += 1
        if math.isfinite(force_n):
            rec["max_contact_force_n"] = max(rec["max_contact_force_n"], force_n)
        return True


class CollisionLogger:
    """Log ego-vs-obstacle Chrono contacts plus geometric proximity diagnostics.

    Args:
        system: The shared Chrono system whose contact container is authoritative.
        ego_body_ids: Identifiers of every body belonging to the ego HMMWV.
        rocks: Metadata returned by ``add_rock_obstacles``.
        traffic_body_map: Mapping ``Chrono body identifier -> logical obstacle
            id`` for traffic vehicles. Traffic logical ids start at 1000 and
            align with the ``extra_obstacles`` passed to ``check``, so a
            contact and a clearance measurement refer to the same obstacle.
        clearance_radius: Circular vehicle proxy used only for clearance and
            near-miss distances.
        near_miss_margin: Extra proxy clearance counted as a near miss.
        log_all: Write every obstacle on every step instead of event rows only.
    """

    def __init__(self, *, system, ego_body_ids, rocks: list,
                 traffic_body_map: Optional[dict[int, int]] = None,
                 run_dir: str = "logs/",
                 clearance_radius: float = VEHICLE_CLEARANCE_RADIUS,
                 near_miss_margin: float = NEAR_MISS_MARGIN,
                 log_all: bool = False):
        self.system = system
        self.ego_body_ids = {int(v) for v in ego_body_ids}
        self.clearance_radius = float(clearance_radius)
        self.near_miss_margin = float(near_miss_margin)
        self.log_all = bool(log_all)

        if not self.ego_body_ids:
            raise ValueError("ego_body_ids must contain the ego vehicle bodies")

        if rocks:
            self._rock_x = np.array([r["x"] for r in rocks], dtype=float)
            self._rock_y = np.array([r["y"] for r in rocks], dtype=float)
            self._rock_r = np.array([r["size"] * 0.5 for r in rocks], dtype=float)
        else:
            self._rock_x = np.zeros(0)
            self._rock_y = np.zeros(0)
            self._rock_r = np.zeros(0)
        self.n_rocks = len(self._rock_r)

        obstacle_body_map: dict[int, int] = {}
        for rock_id, rock in enumerate(rocks or []):
            obstacle_body_map[int(rock["body"].GetIdentifier())] = rock_id
        obstacle_body_map.update({int(k): int(v)
                                  for k, v in (traffic_body_map or {}).items()})
        self._reporter = _EgoObstacleContactReporter(
            self.ego_body_ids, obstacle_body_map)

        self._hit_ids: set[int] = set()
        self._near_ids: set[int] = set()
        self._active_contact_ids: set[int] = set()
        self.total_collisions = 0
        self.total_near_misses = 0
        self.first_collision_time: Optional[float] = None
        self._steps = 0

        os.makedirs(run_dir, exist_ok=True)
        csv_path = os.path.join(run_dir, "collision_log.csv")
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "time", "veh_x", "veh_y", "v_veh",
            "rock_id", "rock_x", "rock_y", "rock_r",
            "dist_2d", "clearance_m", "near_margin",
            "is_collision", "is_near_miss", "contact_begin",
            "contact_count", "max_contact_force_n", "ego_body",
            "obstacle_body", "collision_source",
        ])
        print(f"  [COLLISION] Chrono body-contact logger active: "
              f"{len(obstacle_body_map)} obstacle bodies, "
              f"{len(self.ego_body_ids)} ego bodies")
        print(f"  [COLLISION] Clearance proxy only: vehicle_r="
              f"{self.clearance_radius:.1f}m, near_miss_margin="
              f"{self.near_miss_margin:.1f}m")
        print(f"  [COLLISION] Log: {csv_path}")

    def _proximity(self, veh_x: float, veh_y: float,
                   extra_obstacles=None) -> dict[int, tuple[float, float, float, float]]:
        """Logical obstacle id -> (x, y, radius, centre distance)."""
        out: dict[int, tuple[float, float, float, float]] = {}
        for i in range(self.n_rocks):
            d = float(math.hypot(self._rock_x[i] - veh_x,
                                 self._rock_y[i] - veh_y))
            out[i] = (float(self._rock_x[i]), float(self._rock_y[i]),
                      float(self._rock_r[i]), d)
        for j, obs in enumerate(extra_obstacles or []):
            ox, oy, radius = float(obs[0]), float(obs[1]), float(obs[2])
            out[1000 + j] = (ox, oy, radius,
                             float(math.hypot(ox - veh_x, oy - veh_y)))
        return out

    def check(self, sim_time: float, veh_x: float, veh_y: float,
              veh_speed: float = 0.0, extra_obstacles=None) -> dict:
        """Query Chrono contacts for this physics step and update unique KPIs."""
        self._steps += 1
        proximity = self._proximity(veh_x, veh_y, extra_obstacles)

        self._reporter.reset()
        self.system.GetContactContainer().ReportAllContacts(self._reporter)
        contacts = dict(self._reporter.contacts)
        current_ids = set(contacts)
        contact_begin_ids = current_ids - self._active_contact_ids
        new_hit_ids = current_ids - self._hit_ids
        self._active_contact_ids = current_ids
        self._hit_ids.update(current_ids)
        self.total_collisions = len(self._hit_ids)

        if current_ids and self.first_collision_time is None:
            self.first_collision_time = float(sim_time)

        nearest_id = None
        min_dist = float("inf")
        for obstacle_id, (_, _, _, dist) in proximity.items():
            if dist < min_dist:
                min_dist = dist
                nearest_id = obstacle_id

        near_now: set[int] = set()
        for obstacle_id, (_, _, radius, dist) in proximity.items():
            clearance = dist - radius - self.clearance_radius
            # An obstacle that was or is actually struck is a collision, never
            # also a near miss: every hit passes through the near band on
            # approach, so counting both would inflate near misses by roughly
            # one per contact and the two KPIs would stop being disjoint.
            if (obstacle_id not in current_ids
                    and obstacle_id not in self._hit_ids
                    and clearance < self.near_miss_margin):
                near_now.add(obstacle_id)
        self._near_ids.update(near_now)
        self._near_ids -= self._hit_ids
        self.total_near_misses = len(self._near_ids)

        for obstacle_id in sorted(set(proximity) | current_ids):
            ox, oy, radius, dist = proximity.get(
                obstacle_id, (math.nan, math.nan, math.nan, math.nan))
            clearance = (dist - radius - self.clearance_radius
                         if math.isfinite(dist) and math.isfinite(radius)
                         else math.nan)
            is_collision = obstacle_id in current_ids
            is_near = obstacle_id in near_now
            contact_begin = obstacle_id in contact_begin_ids
            if not (contact_begin or is_near or self.log_all):
                continue
            evidence = contacts.get(obstacle_id, {})
            self._csv_writer.writerow([
                f"{sim_time:.4f}", f"{veh_x:.4f}", f"{veh_y:.4f}",
                f"{veh_speed:.3f}", obstacle_id,
                f"{ox:.3f}" if math.isfinite(ox) else "",
                f"{oy:.3f}" if math.isfinite(oy) else "",
                f"{radius:.3f}" if math.isfinite(radius) else "",
                f"{dist:.4f}" if math.isfinite(dist) else "",
                f"{clearance:.4f}" if math.isfinite(clearance) else "",
                f"{self.near_miss_margin:.3f}",
                int(is_collision), int(is_near), int(contact_begin),
                int(evidence.get("contact_count", 0)),
                f"{float(evidence.get('max_contact_force_n', 0.0)):.3f}",
                evidence.get("ego_body", ""), evidence.get("obstacle_body", ""),
                COLLISION_SOURCE,
            ])

        if contact_begin_ids:
            # Downstream tools poll this log while the simulator is still
            # running, so the impact edge is made durable immediately rather
            # than waiting for the periodic flush, which could otherwise report
            # a collision seconds after it occurred.
            self._csv_file.flush()
        elif self._steps % 1000 == 0:
            self._csv_file.flush()

        for obstacle_id in sorted(new_hit_ids):
            ev = contacts[obstacle_id]
            print(f"\n  !!! CHRONO BODY CONTACT !!! t={sim_time:.3f}s "
                  f"obstacle_id={obstacle_id} contacts={ev['contact_count']} "
                  f"max_force={ev['max_contact_force_n']:.1f}N "
                  f"ego={ev['ego_body']} obstacle={ev['obstacle_body']} "
                  f"v={veh_speed:.1f}m/s")

        return {
            "any_collision": bool(current_ids),
            "any_near_miss": bool(near_now),
            "min_dist": min_dist,
            "n_collisions": len(current_ids),
            "n_near_misses": len(near_now),
            "closest_rock_id": nearest_id,
            "new_collision_ids": sorted(new_hit_ids),
            "collision_source": COLLISION_SOURCE,
        }

    def close(self) -> dict:
        """Close the event log and return unique-obstacle summary metrics."""
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
        summary = {
            "total_collisions": self.total_collisions,
            "total_near_misses": self.total_near_misses,
            "first_collision_time": self.first_collision_time,
            "collision_free": self.total_collisions == 0,
            "collision_source": COLLISION_SOURCE,
        }
        status = ("COLLISION-FREE" if summary["collision_free"]
                  else "*** COLLISIONS DETECTED ***")
        print(f"\n  [COLLISION] Final summary: {status}")
        print(f"  [COLLISION] Chrono body-contact collisions: "
              f"{self.total_collisions}  Near misses: {self.total_near_misses}")
        if self.first_collision_time is not None:
            print(f"  [COLLISION] First body contact at "
                  f"t={self.first_collision_time:.3f}s")
        return summary
