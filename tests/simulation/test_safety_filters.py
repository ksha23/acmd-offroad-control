"""Contracts of the safety filters (simulation/safety): the plant spec, the vehicle model, the DOB-CBF and the MPSF.

The spec, footprint and QP tests need nothing. The model and filter tests need the rig surrogate and CasADi. They
pin that the footprint distances are exact, that the powertrain tables match the Chrono engine data, that the
numpy model (the rollouts) and the CasADi model (the OCP) are one model, that the soil's resistance slows a
coasting vehicle and full brakes hold on a slope they can hold, and the plain filter contracts: a clear road
passes, a wall too close is braked, a stopped car may start behind a standing vehicle, a downhill lengthens the
stop, the MPSF check does not pass a stop it cannot make, a stale teleop link brakes (on the filter's clock), and
an MPSF without a checked plan brakes.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from simulation.safety import vehicle_spec as spec

ROOT = Path(__file__).resolve().parents[2]
IDENT = json.loads((ROOT / "simulation/safety/ident/plant_bounds_chrono9_box.json").read_text())
VP = {"M": 2573.0, "Izz": 3570.0, "Lf": 1.593, "Lr": 1.709, "h_cg": 0.65, "T": 1.8194}


def test_box_distances_match_brute_force():
    from simulation.safety import footprint as fp
    x_r, x_f, w = fp.body_box()
    rng = np.random.default_rng(3)
    gx, gy = np.meshgrid(np.linspace(x_r, x_f, 120), np.linspace(-w, w, 60))
    for _ in range(40):
        a = rng.uniform(-6, 6, 2)
        b = a + rng.uniform(-5, 5, 2)
        t = np.linspace(0, 1, 400)[:, None]
        pts = a + t * (b - a)
        brute = np.min(np.hypot(pts[:, 0][:, None, None] - gx, pts[:, 1][:, None, None] - gy))
        d = float(fp.segment_box(a[0], a[1], b[0], b[1], x_r, x_f, w))
        assert d == pytest.approx(brute, abs=0.06)
        assert float(fp.point_box(a[0], a[1], x_r, x_f, w)) == pytest.approx(
            np.min(np.hypot(a[0] - gx, a[1] - gy)), abs=0.06)


def test_qp_matches_brute_force():
    from simulation.safety.dob_cbf import solve_qp_2d
    rng = np.random.default_rng(1)
    grid = np.stack(np.meshgrid(np.linspace(-3, 3, 601), np.linspace(-3, 3, 601)), -1).reshape(-1, 2)
    for _ in range(20):
        A = rng.normal(size=(4, 2))
        b = rng.uniform(0.2, 1.5, size=4)
        z0 = rng.uniform(-2.5, 2.5, size=2)
        z = solve_qp_2d(z0, A, b)
        feas = grid[np.all(grid @ A.T <= b, axis=1)]
        if len(feas) == 0:
            continue
        best = feas[np.argmin(np.sum((feas - z0) ** 2, axis=1))]
        assert z is not None
        assert np.sum((z - z0) ** 2) <= np.sum((best - z0) ** 2) + 1e-2


def test_simple_map_force_matches_the_engine_data():
    pt = spec.make_powertrain("simple_map")
    first_gear = 330.0 / (0.1708 * spec.CONICAL_RATIO) / spec.WHEEL_RADIUS
    assert pt.wheel_force(1.0, 1.0) == pytest.approx(first_gear)
    assert pt.wheel_force(1.0, 0.5) == pytest.approx(0.5 * first_gear)
    assert pt.wheel_force(5.0, 0.0) == 0.0


def test_shafts_force_falls_with_speed_and_vanishes_at_the_top():
    pt = spec.make_powertrain("shafts")
    f = [pt.wheel_force(v, 1.0) for v in (0.5, 4.0, 10.0, 18.0)]
    assert f[0] > f[1] > f[2] > f[3] > 0.0
    assert pt.wheel_force(22.0, 1.0) <= 0.0


def test_unknown_flavor_is_refused():
    from simulation.safety import make_safety_filter
    with pytest.raises(ValueError):
        make_safety_filter("mppi", VP, ident=IDENT)


@pytest.fixture(scope="module")
def clay():
    pytest.importorskip("casadi")
    pytest.importorskip("torch")
    from simulation.shared.param_consistency import get_terrain_preset, terrain_preset_to_internal
    from simulation.tire_models.nn_tire_model import load_nn_tire_model
    t = terrain_preset_to_internal(get_terrain_preset("clay"))
    try:
        nn = load_nn_tire_model(str(ROOT / "nn_models/tire_force_static"), t)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"surrogate unavailable: {e}")
    return t, nn


@pytest.fixture(scope="module")
def clay_bike(clay):
    from simulation.safety.vehicle_model import SurrogateBicycle
    t, nn = clay
    b = SurrogateBicycle(nn, VP, "shafts")
    b.set_terrain(t)
    return b


def test_numpy_and_casadi_models_agree(clay_bike):
    import casadi as ca
    from simulation.safety.vehicle_model import N_CURVE
    b = clay_bike
    x = ca.SX.sym("x", 7)
    pdot = ca.SX.sym("pd")
    delta = ca.SX.sym("d")
    terrain = ca.SX.sym("t", 6)
    curve = ca.SX.sym("c", N_CURVE)
    grav = ca.SX.sym("g")
    f = ca.Function("f", [x, delta, pdot, terrain, curve, grav],
                    [b.f_expl_sym(x, delta, pdot, terrain, curve, g_along=grav)])
    rng = np.random.default_rng(0)
    for _ in range(25):
        z = np.array([1.0, -2.0, 0.3, rng.uniform(2.0, 12.0), rng.uniform(-0.5, 0.5),
                      rng.uniform(-0.4, 0.4), rng.uniform(-1.0, 1.0)])
        pd_ = float(rng.uniform(-8.0, 8.0))
        d = float(rng.uniform(-0.45, 0.45))
        g = float(rng.uniform(-1.5, 1.5))
        xdot_n = b.deriv(z[None, :], np.array([d]), np.array([pd_]), g_along=g)[0]
        xdot_s = np.array(f(z, d, pd_, b.terrain_vec, b.curve, g)).ravel()
        assert np.allclose(xdot_s, xdot_n, rtol=1e-6, atol=1e-6)


def test_coasting_slows_by_the_soil_resistance(clay_bike):
    b = clay_bike
    nd = 8
    resistance = -4.0 * float(b.curve[-nd]) / b.M
    assert resistance > 0.0
    assert b.settled_accel(0.0, 5.0) == pytest.approx(-resistance)
    assert b.settled_accel(-1.0, 5.0) == pytest.approx(-b.peak_decel())
    # Full throttle at low speed: the wheels spin, the drive curve's top caps the force.
    assert b.settled_accel(1.0, 2.0) == pytest.approx(4.0 * float(b.curve[-1]) / b.M)


def test_brakes_hold_on_a_slope_they_can_hold(clay_bike):
    # Stopped, full brake, a 15% downhill: clay's locked wheels (about 2.4 m/s^2) beat gravity along the slope
    # (1.45 m/s^2), so the vehicle stays put.
    assert clay_bike.peak_decel() > 1.45
    z = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    xdot = clay_bike.deriv(z, np.zeros(1), np.zeros(1), g_along=1.45)
    assert xdot[0, 3] == pytest.approx(0.0, abs=1e-9)


@pytest.fixture()
def dob(clay):
    from simulation.safety.dob_cbf import CBFSafetyFilter
    t, nn = clay
    f = CBFSafetyFilter(VP, IDENT, powertrain="shafts", nn=nn, sense_range=0.0, odd_obstacle_radius=1.0)
    f.update_terrain(t)
    return f


def test_dob_cbf_clear_road_passes_unchanged(dob):
    state = dict(x=0.0, y=0.0, psi=0.0, u=5.0)
    for _ in range(5):
        r = dob.filter(0.1, 0.6, 0.0, state, obstacles=[])
    assert not r.was_modified
    assert r.throttle == pytest.approx(0.6)


def test_dob_cbf_brakes_for_a_wall_too_close(dob):
    state = dict(x=0.0, y=0.0, psi=0.0, u=7.0)
    wall = [(9.0, y, 0.6, False) for y in np.linspace(-4.0, 4.0, 9)]
    r = dob.filter(0.0, 1.0, 0.0, state, obstacles=wall)
    assert r.was_modified and r.braking > 0.0 and r.throttle == 0.0


def test_dob_cbf_does_not_throttle_into_a_vehicle_it_stopped_behind(dob):
    """Standing with its front 0.6 m from a stalled vehicle: the operator's throttle would start the car before
    the brake could ramp in, so it must not pass."""
    from simulation.safety import footprint as fp
    x_r, x_f, w = fp.body_box()
    t_xr, _, _ = fp.traffic_box()
    car_x = x_f + 0.6 - t_xr
    state = dict(x=0.0, y=0.0, psi=0.0, u=0.0)
    r = dob.filter(0.0, 0.6, 0.0, state, obstacles=[(car_x, 0.0, 2.2, True, 0.0, 0.0, 0.0)])
    assert r.throttle < 0.6 and r.was_modified


def test_dob_cbf_starts_behind_a_standing_vehicle(dob):
    state = dict(x=0.0, y=0.0, psi=0.0, u=0.0)
    car = [(16.0, 0.0, 2.2, True, 0.0, 0.0)]
    r = dob.filter(0.0, 0.5, 0.0, state, obstacles=car)
    assert r.throttle > 0.0


def test_dob_cbf_downhill_lengthens_the_backup(dob):
    from simulation.safety.grade_preview import G
    flat, _ = dob._stop(np.array([6.0]), np.array([0.0]), 0.0)
    g = dob._gravity({"x": 0.0, "y": 0.0, "psi": 0.0, "grade_now": 0.148}, 6.0, 0.0)
    assert g == pytest.approx(G * 0.148)
    down, _ = dob._stop(np.array([6.0]), np.array([0.0]), g)
    assert down[0] > 1.5 * flat[0]


def test_dob_cbf_stale_link_brakes_on_its_clock(clay):
    from simulation.safety.dob_cbf import CBFSafetyFilter
    t, nn = clay
    f = CBFSafetyFilter(VP, IDENT, powertrain="shafts", nn=nn, sense_range=0.0, teleop_delay=0.2,
                        stale_cmd_timeout=2.0)
    f.update_terrain(t)
    now = [100.0]
    f.set_clock(lambda: now[0])
    f.update_command_age()
    now[0] = 101.5
    state = dict(x=0.0, y=0.0, psi=0.0, u=5.0)
    assert f.filter(0.0, 0.6, 0.0, state, obstacles=[]).braking == 0.0
    now[0] = 102.5
    r = f.filter(0.0, 0.6, 0.0, state, obstacles=[])
    assert r.braking == 1.0 and r.was_modified
    assert f.get_diagnostics()["stale_brakes"] == 1


def _mpsf(clay, **kw):
    pytest.importorskip("acados_template")
    from simulation.safety.mpsf import MPSFSafetyFilter
    t, nn = clay
    kw.setdefault("odd_obstacle_radius", 1.0)
    kw.setdefault("sense_range", 0.0)
    f = MPSFSafetyFilter(VP, IDENT, powertrain="shafts", nn=nn, build_solver=False, **kw)
    f.update_terrain(t)
    f._last_sense = (0.0, 0.0)
    return f


def test_mpsf_check_needs_the_stop(clay):
    f = _mpsf(clay)
    z0 = np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    brake_now = [(0.0, -1.0)] * f.N
    stop = f._bike.rollout_rk4(z0, np.array(brake_now), np.full(f.N, f.time_steps[0]), f.cert_dt,
                               t_tail=20.0)[-1, 0]
    # Clearance the front edge needs past the predicted stop: the overshoot of the braking distance, the rock,
    # the tube over the extended path.
    ext = f.stop_s0 + (f.stop_s1 - 1.0) * stop
    need = f.box[1] + ext + 1.0 + f.obstacle_buffer + f.tube_c1 * (stop + ext)
    near = stop + need - 0.3
    far = stop + need + 1.0
    assert not f._check(z0, brake_now, [(near, 0.0, 1.0)])
    assert f._check(z0, brake_now, [(far, 0.0, 1.0)])


def test_mpsf_check_knows_a_downhill_stop_is_longer(clay):
    from simulation.safety.grade_preview import gravity_lookup
    f = _mpsf(clay)
    z0 = np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    brake_now = [(0.0, -1.0)] * f.N
    stop = f._bike.rollout_rk4(z0, np.array(brake_now), np.full(f.N, f.time_steps[0]), f.cert_dt,
                               t_tail=20.0)[-1, 0]
    ext = f.stop_s0 + (f.stop_s1 - 1.0) * stop
    far = stop + f.box[1] + ext + 1.0 + f.obstacle_buffer + f.tube_c1 * (stop + ext) + 1.0
    wall = [(far, float(y), 1.0) for y in np.arange(-6.0, 6.1, 1.2)]
    assert f._check(z0, brake_now, wall)
    downhill = gravity_lookup({"x": 0.0, "y": 0.0, "psi": 0.0, "grade_now": 0.148}, f._bike.L, 0.15)
    assert not f._check(z0, brake_now, wall, downhill)


def test_mpsf_starts_behind_a_stalled_vehicle(clay):
    """A stalled vehicle 16 m ahead whose measured speed jitters above zero: its set may run away from the ego,
    which must not block a short start-and-stop."""
    f = _mpsf(clay, odd_obstacle_radius=2.2)
    z0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plan = [(0.0, 0.75)] + [(0.0, -1.0)] * (f.N - 1)
    car = [(16.0, 0.0, 2.2, True, 0.02, 0.0, 0.0)]
    assert f._check(z0, plan, car)


def test_mpsf_oncoming_vehicle_in_the_next_lane_leaves_room(clay):
    """An oncoming vehicle 3.5 m over is its rectangle, half-width 1.28 m, so braking in our own lane certifies."""
    f = _mpsf(clay, odd_obstacle_radius=2.77)
    z0 = np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    brake_now = [(0.0, -1.0)] * f.N
    oncoming = [(12.0, 3.5, 2.2, True, -4.0, 0.0, np.pi)]
    assert f._check(z0, brake_now, oncoming)
    # The same vehicle in our lane blocks it.
    assert not f._check(z0, brake_now, [(12.0, 0.0, 2.2, True, -4.0, 0.0, np.pi)])


def test_mpsf_stale_link_brakes_and_a_failed_solve_fails_closed(clay):
    f = _mpsf(clay, teleop_delay=0.2, stale_cmd_timeout=2.0)
    now = [100.0]
    f.set_clock(lambda: now[0])
    f.update_command_age()
    now[0] = 102.5
    state = dict(x=0.0, y=0.0, psi=0.0, u=5.0)
    r = f.filter(0.0, 0.6, 0.0, state, obstacles=[])
    assert r.braking == 1.0 and r.was_modified
    assert f.get_diagnostics()["stale_brakes"] == 1
    # A fresh command over the link: no stale brake. This filter has no solver, so the solve fails, and with no
    # checked plan yet the filter brakes straight.
    f.update_command_age()
    r = f.filter(0.0, 0.6, 0.0, state, obstacles=[])
    d = f.get_diagnostics()
    assert d["stale_brakes"] == 1 and d["solve_fails"] == 1 and d["fail_stops"] == 1
    assert r.braking == 1.0 and r.steering == 0.0
