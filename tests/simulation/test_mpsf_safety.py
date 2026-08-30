"""Behavioral contract of the MPSF predictive safety filter.

Most tests drive ``MPSFSafetyFilter`` with ``build_solver=False`` and a fake
acados solver, which makes every guarded branch deterministic without a C
compile: the stale-command brake, the fail-closed path, the predicted-plan
fail-stop (the "unavoidable collision => maximal braking" claim), the
executed-channel lateral-authority clamp, the stoppable-speed funnel actually
sent to the solver, and the hedged soil-authority derivation. Two integration
tests build the real solver (skipped where acados cannot load) and assert the
properties that live inside the OCP itself -- stage-0 lateral rows binding the
executed input, and full braking with zero throttle in unavoidable geometry,
which is the regression test for the throttle-injection defect this suite was
written against.
"""
import math
import time
from pathlib import Path

import numpy as np
import pytest

# The filter builds its OCP symbolically at import time; environments without
# casadi (the system interpreter) skip this module rather than erroring
# collection. The benchmark smoke tier runs it under the scm-terrain
# environment, where a skip cannot happen silently.
pytest.importorskip("casadi")

from simulation.safety.mpsf import MPSFSafetyFilter  # noqa: E402

VP = {"M": 2573.0, "Izz": 3570.0, "Lf": 1.593, "Lr": 1.709,
      "h_cg": 0.65, "T": 1.8194}
L = VP["Lf"] + VP["Lr"]

CLAY = {"Kphi": 282090.0, "Kc": 13190.0, "n": 0.5, "c": 13790.0,
        "phi": 13.0, "k": 0.01}


class _FakeSolver:
    """Records everything the filter sends and returns a scripted solution."""

    def __init__(self, u0=(0.0, 0.0), plan=None, status=0,
                 raise_on_solve=False):
        self.u0 = np.asarray(u0, dtype=float)
        self.plan = plan            # list of N+1 state vectors, or None
        self.status = status
        self.raise_on_solve = raise_on_solve
        self.p = {}
        self.ubx = {}
        self.warm_u = {}
        self.solve_calls = 0

    def set(self, k, field, val):
        if field == "p":
            self.p[k] = np.array(val, dtype=float)
        elif field == "u":
            self.warm_u[k] = np.array(val, dtype=float)
        # lbx/ubx at stage 0 pin the state; nothing to record

    def constraints_set(self, k, field, val):
        if field == "ubx":
            self.ubx[k] = float(np.atleast_1d(val)[0])

    def solve(self):
        self.solve_calls += 1
        if self.raise_on_solve:
            raise RuntimeError("scripted solver failure")
        return self.status

    def get(self, k, field):
        if field == "u":
            return self.u0.copy()
        if field == "x":
            if self.plan is None:
                return np.array([1e6, 1e6, 0.0, 0.0])
            return np.asarray(self.plan[k], dtype=float)
        raise KeyError(field)


class _StubNN:
    """Tire surrogate stand-in whose force scales with phi and n, so hedged
    (weaker-soil) queries verifiably yield lower authority."""

    def __init__(self):
        self.calls = 0

    def predict_numeric(self, slip, fz, v, kappa=0.0, n_terrain=0.7,
                        terrain_params=None, rates=None):
        self.calls += 1
        phi = float(terrain_params["phi"])
        n = float(terrain_params["n"])
        mag = 3000.0 * (phi / 13.0) * (n / 0.5)
        return -mag * abs(float(kappa)), mag * min(abs(float(slip)), 0.3)


class _RaisingNN:
    def predict_numeric(self, *a, **k):
        raise RuntimeError("scripted surrogate failure")


def _make(fake=None, **kw):
    filt = MPSFSafetyFilter(vehicle_params=VP, build_solver=False, **kw)
    filt._solver = fake if fake is not None else _FakeSolver()
    return filt


def _state(v=5.0, x=0.0, y=0.0, psi=0.0):
    return {"x": x, "y": y, "psi": psi, "u": v}


def _steer_limit(filt, v):
    return math.atan(filt._lat_accel_max * L / max(v, 0.5) ** 2) / filt.max_steer


# -- command handling ---------------------------------------------------------

def test_clear_road_passthrough():
    fake = _FakeSolver(u0=(0.2, 0.6))
    filt = _make(fake)
    r = filt.filter(0.2, 0.6, 0.0, _state(v=5.0), [])
    assert r.steering == pytest.approx(0.2)
    assert r.throttle == pytest.approx(0.6)
    assert r.braking == 0.0
    assert not r.was_modified
    assert fake.solve_calls == 1


def test_padded_obstacle_slots_are_inactive():
    fake = _FakeSolver(u0=(0.0, 0.3))
    filt = _make(fake)
    filt.filter(0.0, 0.3, 0.0, _state(), [])
    p = fake.p[0]
    # all three obstacle slots pushed far away with tiny radius
    for j in range(filt.n_obs):
        assert abs(p[5 + 3 * j]) > 1e3
        assert p[7 + 3 * j] <= 0.5


# -- fail-closed and fail-stop ------------------------------------------------

def test_solver_exception_fails_closed_braking():
    filt = _make(_FakeSolver(raise_on_solve=True))
    r = filt.filter(0.1, 0.9, 0.0, _state(v=6.0), [(20.0, 0.0, 0.6)])
    assert r.braking == 1.0
    assert r.throttle == 0.0
    assert r.was_modified


def test_bad_status_fails_closed_braking():
    filt = _make(_FakeSolver(u0=(0.0, 1.0), status=4))
    r = filt.filter(0.0, 1.0, 0.0, _state(v=6.0), [(20.0, 0.0, 0.6)])
    assert r.braking == 1.0
    assert r.throttle == 0.0


def test_nonfinite_solution_fails_closed_braking():
    filt = _make(_FakeSolver(u0=(np.nan, np.nan)))
    r = filt.filter(0.0, 1.0, 0.0, _state(v=6.0), [(20.0, 0.0, 0.6)])
    assert r.braking == 1.0
    assert r.throttle == 0.0


def test_unsafe_plan_overrides_to_maximal_braking():
    # The accepted solution predicts driving through the obstacle envelope
    # while its first input is full throttle -- the shape of the defect this
    # suite regresses: slack bought a collision and the penalty gradient
    # floored the throttle. The filter must execute maximal braking instead.
    plan = [np.array([0.35 * k, 0.0, 0.0, 7.0]) for k in range(21)]
    fake = _FakeSolver(u0=(-1.0, 1.0), plan=plan, status=0)
    filt = _make(fake)
    r = filt.filter(0.0, 0.0, 0.0, _state(v=7.0), [(3.0, 0.0, 0.6)])
    assert r.braking == 1.0
    assert r.throttle == 0.0
    assert r.was_modified
    assert filt.get_diagnostics()["fail_stops"] == 1


def test_unsafe_plan_under_status_2_is_also_rejected():
    plan = [np.array([0.35 * k, 0.0, 0.0, 7.0]) for k in range(21)]
    fake = _FakeSolver(u0=(0.0, 1.0), plan=plan, status=2)
    filt = _make(fake)
    r = filt.filter(0.0, 0.0, 0.0, _state(v=7.0), [(3.0, 0.0, 0.6)])
    assert r.braking == 1.0
    assert r.throttle == 0.0


def test_safe_plan_is_executed():
    # Plan passes 6 m abeam of the obstacle: outside the envelope, no override.
    plan = [np.array([0.3 * k, 6.0, 0.0, 6.0]) for k in range(21)]
    fake = _FakeSolver(u0=(0.1, 0.4), plan=plan, status=0)
    filt = _make(fake)
    r = filt.filter(0.1, 0.4, 0.0, _state(v=6.0, y=6.0), [(3.0, 0.0, 0.6)])
    assert r.throttle == pytest.approx(0.4)
    assert r.braking == 0.0
    assert filt.get_diagnostics()["fail_stops"] == 0


# -- stale-command brake ------------------------------------------------------

def test_stale_command_brakes_without_solving():
    fake = _FakeSolver(u0=(0.0, 0.8))
    filt = _make(fake, stale_cmd_timeout=2.0, teleop_delay=0.2)
    filt._last_cmd_wall = time.time() - 10.0
    r = filt.filter(0.0, 0.8, 0.0, _state(v=6.0), [])
    assert r.braking == 1.0
    assert r.throttle == 0.0
    assert r.was_modified
    assert fake.solve_calls == 0
    assert filt.get_diagnostics()["stale_brakes"] == 1


def test_fresh_command_does_not_stale_brake():
    filt = _make(_FakeSolver(u0=(0.0, 0.8)), stale_cmd_timeout=2.0,
                 teleop_delay=0.2)
    filt.update_command_age(time.time())
    r = filt.filter(0.0, 0.8, 0.0, _state(v=6.0), [])
    assert r.braking == 0.0
    assert r.throttle == pytest.approx(0.8)


def test_no_stale_brake_outside_teleop():
    # Without a teleop delay the plant is commanding locally each step; the
    # staleness rule is teleop-only, matching CBFSafetyFilter.
    filt = _make(_FakeSolver(u0=(0.0, 0.8)), stale_cmd_timeout=2.0)
    filt._last_cmd_wall = time.time() - 10.0
    r = filt.filter(0.0, 0.8, 0.0, _state(v=6.0), [])
    assert r.braking == 0.0


# -- executed-channel lateral authority ---------------------------------------

def test_executed_steer_clamped_to_soil_authority():
    fake = _FakeSolver(u0=(1.0, 0.0))
    filt = _make(fake)
    filt._lat_accel_max = 2.0
    v = 8.0
    r = filt.filter(1.0, 0.0, 0.0, _state(v=v), [])
    lim = _steer_limit(filt, v)
    assert abs(r.steering) <= lim + 1e-9
    implied = v * v * math.tan(filt.max_steer * r.steering) / L
    assert abs(implied) <= filt._lat_accel_max + 1e-6
    assert r.was_modified
    assert filt.get_diagnostics()["lat_clamps"] >= 1


def test_fail_closed_steer_is_also_clamped():
    filt = _make(_FakeSolver(raise_on_solve=True))
    filt._lat_accel_max = 2.0
    r = filt.filter(1.0, 0.0, 0.0, _state(v=8.0), [(20.0, 0.0, 0.6)])
    assert abs(r.steering) <= _steer_limit(filt, 8.0) + 1e-9
    assert r.braking == 1.0


def test_low_speed_steer_unclamped():
    fake = _FakeSolver(u0=(0.9, 0.0))
    filt = _make(fake)
    filt._lat_accel_max = 2.0
    r = filt.filter(0.9, 0.0, 0.0, _state(v=0.3), [])
    assert r.steering == pytest.approx(0.9)


# -- stoppable-speed funnel ---------------------------------------------------

def test_speed_funnel_descends_to_stoppable_speed():
    fake = _FakeSolver(u0=(0.0, 0.0))
    filt = _make(fake)
    filt._brake_grip = 0.4          # 2.4 m/s^2 achievable braking
    a_brk = 0.4 * filt.max_decel
    v0 = 8.0
    filt.filter(0.0, 0.0, 0.0, _state(v=v0), [])
    v_stop = filt.N * filt.dt * a_brk
    assert fake.ubx[1] == pytest.approx(max(v_stop, v0 - filt.dt * a_brk))
    assert fake.ubx[filt.N] == pytest.approx(v_stop)
    caps = [fake.ubx[k] for k in range(1, filt.N + 1)]
    assert all(a >= b - 1e-9 for a, b in zip(caps, caps[1:]))  # non-increasing
    # feasible from the entry state: never demands more than full braking
    assert fake.ubx[1] >= v0 - filt.dt * a_brk - 1e-9


def test_speed_funnel_inactive_at_nominal_grip():
    fake = _FakeSolver(u0=(0.0, 0.0))
    filt = _make(fake)                     # brake_grip 1.0 -> v_stop 12 m/s
    filt.filter(0.0, 0.0, 0.0, _state(v=8.0), [])
    assert all(v == pytest.approx(filt.max_speed) for v in fake.ubx.values())


# -- soil authority and hedging -----------------------------------------------

def test_hedged_authority_is_lower_than_mean_authority():
    filt_mean = _make()
    filt_mean._nn = _StubNN()
    filt_mean.update_terrain(dict(CLAY), phi_uncertainty_deg=4.0,
                             n_sigma=0.1, hedge_k=0.0)
    filt_hedge = _make()
    filt_hedge._nn = _StubNN()
    filt_hedge.update_terrain(dict(CLAY), phi_uncertainty_deg=4.0,
                              n_sigma=0.1, hedge_k=2.0)
    assert filt_hedge._brake_grip < filt_mean._brake_grip
    assert filt_hedge._lat_accel_max < filt_mean._lat_accel_max
    assert filt_hedge._accel_grip < filt_mean._accel_grip


def test_use_terrain_nn_false_keeps_soil_blind_authority():
    filt = _make()
    stub = _StubNN()
    filt._nn = stub
    filt.update_terrain(dict(CLAY), use_terrain_nn=False)
    assert stub.calls == 0
    assert filt._brake_grip == 1.0
    assert filt._lat_accel_max == filt.lat_accel_max_default


def test_surrogate_failure_is_loud_and_flagged(capsys):
    filt = _make()
    filt._nn = _RaisingNN()
    filt.update_terrain(dict(CLAY))
    assert not filt.get_diagnostics()["nn_authority_active"]
    assert "[MPSF]" in capsys.readouterr().out
    # conservative fallback: authorities untouched, not zeroed or inflated
    assert filt._brake_grip == 1.0


def test_model_dir_resolves_against_repo_root():
    filt = _make(nn_model_dir="nn_models/tire_force_static")
    p = Path(filt.nn_model_dir)
    assert p.is_absolute()
    assert p == Path(__file__).resolve().parents[2] / "nn_models" / "tire_force_static"


# -- real-solver integration --------------------------------------------------

@pytest.fixture(scope="module")
def real_filter():
    try:
        filt = MPSFSafetyFilter(vehicle_params=VP)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"acados solver unavailable in this environment: {e}")
    return filt


def _reset(filt):
    filt._brake_grip = 1.0
    filt._accel_grip = 1.0
    filt._lat_accel_max = filt.lat_accel_max_default
    return filt


def test_real_unavoidable_geometry_brakes_fully(real_filter):
    # Regression for the throttle-injection defect: obstacle inside the
    # standoff at speed, on low-grip authority. Pre-fix output was full
    # throttle; contract is zero throttle and full braking.
    filt = _reset(real_filter)
    filt._brake_grip = 0.55
    filt._lat_accel_max = 2.7
    for gap in (3.0, 5.0):
        r = filt.filter(0.0, 0.0, 0.0, _state(v=7.0), [(gap, 0.0, 0.6)])
        assert r.throttle == 0.0, f"throttle injected at gap {gap}"
        assert r.braking == 1.0, f"no maximal braking at gap {gap}"


def test_real_stage0_lateral_row_binds_raw_solver_output(real_filter):
    # The stage-0 hard rows must bind the *solver's own* first input, before
    # any post-solve clamp: read the raw u0 back out of the solve.
    filt = _reset(real_filter)
    filt._lat_accel_max = 2.0
    v = 8.0
    filt.filter(1.0, 0.0, 0.0, _state(v=v), [])
    raw = filt._solver.get(0, "u")
    implied = v * v * math.tan(filt.max_steer * float(raw[0])) / L
    assert abs(implied) <= filt._lat_accel_max + 0.15  # small solver tolerance
