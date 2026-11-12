"""Contracts of the derived filters (--safety-derived): the plant spec, the derived model, the derived DOB-CBF and
the derived MPSF check.

The spec tests need nothing. The model and filter tests need the rig surrogate and CasADi. They pin that the
footprint covers the body outline, that the powertrain tables match the Chrono engine data, that the numpy model
(the check) and the CasADi model (the OCP) are one model, that the soil's resistance slows a coasting vehicle,
and the plain filter contracts: a clear road passes, a wall too close is braked, a stopped car may start behind a
standing vehicle, and the MPSF check does not pass a stop it cannot make.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from simulation.safety import vehicle_spec as spec

ROOT = Path(__file__).resolve().parents[2]
IDENT = json.loads((ROOT / "simulation/safety/ident/plant_bounds_chrono9.json").read_text())
VP = {"M": 2573.0, "Izz": 3570.0, "Lf": 1.593, "Lr": 1.709, "h_cg": 0.65, "T": 1.8194}


def test_footprint_discs_cover_the_body_outline():
    centres, r = spec.footprint_discs()
    x0, x1 = spec.BODY_X_REF[0] - spec.COM_X_REF, spec.BODY_X_REF[1] - spec.COM_X_REF
    for x in np.linspace(x0, x1, 200):
        for y in (-spec.BODY_HALF_WIDTH, 0.0, spec.BODY_HALF_WIDTH):
            assert min(np.hypot(x - c, y) for c in centres) <= r + 1e-9


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
    from simulation.safety.derived_model import DerivedBicycle
    t, nn = clay
    b = DerivedBicycle(nn, VP, "shafts")
    b.set_terrain(t)
    return b


def test_numpy_and_casadi_derived_models_agree(clay_bike):
    import casadi as ca
    from simulation.safety.surrogate_dynamics import N_CURVE
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


@pytest.fixture()
def dob(clay):
    from simulation.safety.dob_cbf_derived import DerivedCBFSafetyFilter
    t, nn = clay
    f = DerivedCBFSafetyFilter(VP, IDENT, powertrain="shafts", nn_casadi=nn, sense_range=0.0,
                               odd_obstacle_radius=1.0)
    f.update_terrain(t)
    return f


def test_derived_cbf_clear_road_passes_unchanged(dob):
    state = dict(x=0.0, y=0.0, psi=0.0, u=5.0)
    for _ in range(5):
        r = dob.filter(0.1, 0.6, 0.0, state, obstacles=[])
    assert not r.was_modified
    assert r.throttle == pytest.approx(0.6)


def test_derived_cbf_brakes_for_a_wall_too_close(dob):
    state = dict(x=0.0, y=0.0, psi=0.0, u=7.0)
    wall = [(9.0, y, 0.6, False) for y in np.linspace(-4.0, 4.0, 9)]
    r = dob.filter(0.0, 1.0, 0.0, state, obstacles=wall)
    assert r.was_modified and r.braking > 0.0 and r.throttle == 0.0


def test_derived_cbf_starts_behind_a_standing_vehicle(dob):
    state = dict(x=0.0, y=0.0, psi=0.0, u=0.0)
    car = [(16.0, 0.0, 2.2, True, 0.0, 0.0)]
    r = dob.filter(0.0, 0.5, 0.0, state, obstacles=car)
    assert r.throttle > 0.0


def test_derived_mpsf_check_needs_the_stop(clay):
    pytest.importorskip("acados_template")
    from simulation.safety.mpsf import MPSFSafetyFilter
    t, nn = clay
    f = MPSFSafetyFilter(VP, dynamic_model=True, derived=True, ident=IDENT, powertrain="shafts",
                         odd_obstacle_radius=1.0, sense_range=0.0, build_solver=False)
    f.update_terrain(t, use_terrain_nn=True)
    f._last_sense = (0.0, 0.0, 1.0e4, f._bike.peak_decel())
    z0 = np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    brake_now = [(0.0, -1.0)] * f.N
    stop = f._bike.rollout_rk4(z0, np.array(brake_now), np.full(f.N, f.dt), f.cert_dt, t_tail=20.0)[-1, 0]
    # Clearance the front disc needs at the stop: rock, disc, the tube's constant and its growth over the stop.
    need = max(f.disc_offsets) + 1.0 + f.disc_radius + f.obstacle_buffer + f.tube_c1 * stop
    near = stop + need - 0.3
    far = stop + need + 1.0
    assert not f._check_derived(z0, brake_now, [(near, 0.0, 1.0)])
    assert f._check_derived(z0, brake_now, [(far, 0.0, 1.0)])
