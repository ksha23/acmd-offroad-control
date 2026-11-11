"""Contracts of the terminal-set MPSF and the DOB-CBF braking barrier.

The MPSF tests build the real solver (skipped where acados cannot load). They
pin the two properties the reformulation exists for: the braking decision does
not depend on the horizon length, and with a finite sensing range the speed the
filter accepts is the one it can stop inside that range. The DOB-CBF tests need
no solver: the braking rows reduce to an upper bound on alpha, which must move
with the soil's braking authority.
"""
import numpy as np
import pytest

pytest.importorskip("casadi")

from simulation.safety.dob_cbf import CBFSafetyFilter  # noqa: E402
from simulation.safety.mpsf import MPSFSafetyFilter  # noqa: E402

VP = {"M": 2573.0, "Izz": 3570.0, "Lf": 1.593, "Lr": 1.709,
      "h_cg": 0.65, "T": 1.8194}

# A wall of rocks across the lane, 45 m down the x axis.
WALL = [(45.0, float(y), 0.6) for y in np.arange(-7.2, 7.3, 1.2)]
CLAY_BRAKE_GRIP = 2.74 / 6.0   # the surrogate's clay braking authority


def _state(x, v):
    return {"x": x, "y": 0.0, "psi": 0.0, "u": v}


def _terminal_filter(**kw):
    try:
        f = MPSFSafetyFilter(vehicle_params=VP, terminal_set=True, **kw)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"acados solver unavailable in this environment: {e}")
    f._brake_grip = CLAY_BRAKE_GRIP
    return f


@pytest.fixture(scope="module")
def short_and_long():
    short = _terminal_filter(sense_range=0)
    long_ = _terminal_filter(sense_range=0, time_steps=[0.1] * 10 + [0.3] * 10)
    return short, long_


def test_braking_decision_does_not_depend_on_horizon(short_and_long):
    # At 8 m/s on clay the margined stop needs about 13.7 m plus the 3.45 m
    # planning radius, so the command passes at 22 m and is braked at 14 m,
    # with either horizon.
    for f in short_and_long:
        far = f.filter(0.0, 0.5, 0.0, _state(45.0 - 22.0, 8.0), WALL)
        near = f.filter(0.0, 0.5, 0.0, _state(45.0 - 14.0, 8.0), WALL)
        assert not far.was_modified, f"horizon {sum(f.time_steps)} s braked at 22 m"
        assert near.braking > 0.5, f"horizon {sum(f.time_steps)} s did not brake at 14 m"


def test_sensing_range_sets_the_accepted_speed():
    # R = 15 m on clay: the stoppable speed inside the sensed region is about
    # 7.2 m/s. Below it the command passes, above it the filter brakes.
    f = _terminal_filter(sense_range=15.0)
    slow = f.filter(0.0, 0.5, 0.0, _state(0.0, 5.0), [])
    fast = f.filter(0.0, 0.5, 0.0, _state(0.0, 9.0), [])
    assert not slow.was_modified
    assert fast.braking > 0.05


def test_published_formulation_is_unchanged_by_default():
    f = MPSFSafetyFilter(vehicle_params=VP, build_solver=False)
    assert not f.terminal_set
    assert f.N == 20 and f.time_steps == (0.1,) * 20


def _cbf(a_soil):
    c = CBFSafetyFilter(VP, braking_barrier=True, sense_range=30.0)
    c._a_soil = a_soil
    c._alpha = -1.0   # the rate floor then allows full braking
    return c


def test_braking_bound_moves_with_soil():
    # Same gap and speed: the soil-blind 6 m/s^2 still allows throttle at a
    # 16 m gap, the clay authority demands braking there.
    blind, clay = _cbf(6.0), _cbf(2.74)
    x = 45.0 - 16.0
    ub_blind = blind._braking_alpha_bound(x, 0.0, 0.0, 8.0, 0.0, WALL, 0.25, 3.0)
    ub_clay = clay._braking_alpha_bound(x, 0.0, 0.0, 8.0, 0.0, WALL, 0.25, 3.0)
    assert ub_blind > 0.0
    assert ub_clay < 0.0


def test_sensing_edge_limits_speed_without_obstacles():
    c = _cbf(2.74)
    assert c._braking_alpha_bound(0.0, 0.0, 0.0, 6.0, 0.0, [], 0.25, 3.0) > 0.0
    assert c._braking_alpha_bound(0.0, 0.0, 0.0, 12.0, 0.0, [], 0.25, 3.0) < 0.0


def test_sim_clock_drives_the_command_age():
    c = CBFSafetyFilter(VP, teleop_delay=0.3)
    now = [100.0]
    c.set_clock(lambda: now[0])
    c.update_command_age(99.7)
    assert abs(c._cmd_age - 0.3) < 1e-12
