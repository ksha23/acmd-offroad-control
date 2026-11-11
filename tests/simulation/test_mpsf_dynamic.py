"""Contracts of the surrogate dynamic model and the dynamic-model MPSF.

The model tests need the rig surrogate and CasADi. They pin that the numpy
model (the check of each plan) and the CasADi model (the OCP) are one model,
and that braking locks the wheels where the plant does. The filter tests build
the acados solver (skipped where it cannot load) and pin the plain contracts of
a predictive safety filter: a safe command passes unchanged, and a command
that cannot be made safe by any plan of the model is not passed.
"""
import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("torch")

import casadi as ca  # noqa: E402

from simulation.safety.mpsf import MPSFSafetyFilter  # noqa: E402
from simulation.safety.surrogate_dynamics import N_CURVE, SurrogateBicycle  # noqa: E402
from simulation.shared.param_consistency import get_terrain_preset, terrain_preset_to_internal  # noqa: E402
from simulation.tire_models.nn_tire_model import load_nn_tire_model  # noqa: E402

VP = {"M": 2573.0, "Izz": 3570.0, "Lf": 1.593, "Lr": 1.709, "h_cg": 0.65, "T": 1.8194}
MODEL_DIR = "nn_models/tire_force_static"


def _soil(name):
    return terrain_preset_to_internal(get_terrain_preset(name))


@pytest.fixture(scope="module")
def clay_bike():
    t = _soil("clay")
    try:
        nn = load_nn_tire_model(MODEL_DIR, t)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"surrogate unavailable: {e}")
    bike = SurrogateBicycle(nn, VP)
    bike.set_terrain(t)
    return bike


def test_numpy_and_casadi_models_agree(clay_bike):
    b = clay_bike
    x = ca.SX.sym("x", 7)
    pedal = ca.SX.sym("a")
    delta = ca.SX.sym("d")
    terrain = ca.SX.sym("t", 6)
    curve = ca.SX.sym("c", N_CURVE)
    f = ca.Function("f", [x, delta, pedal, terrain, curve],
                    [b.f_expl_sym(x, delta, pedal, terrain, curve, 2.0, 0.85)])
    rng = np.random.default_rng(0)
    for _ in range(25):
        z = np.array([1.0, -2.0, 0.3, rng.uniform(2.0, 10.0), rng.uniform(-0.5, 0.5),
                      rng.uniform(-0.4, 0.4), rng.uniform(-2.0, 1.0)])
        p = float(rng.uniform(-1.0, 1.0))
        d = float(rng.uniform(-0.45, 0.45))
        xdot_n = b.deriv(z[None, :], np.array([d]), np.array([p]), 2.0, 0.85)[0]
        xdot_s = np.array(f(z, d, p, b.terrain_vec, b.curve)).ravel()
        assert np.allclose(xdot_s, xdot_n, rtol=1e-6, atol=1e-6)


def test_clay_brakes_lock_above_a_light_pedal(clay_bike):
    # The plant locks its wheels on clay at a 0.3 pedal (model_check.py).
    cf, cr = clay_bike.lock_force()
    threshold = max(cf, cr) / clay_bike.f_brake_max
    assert 0.18 < threshold < 0.26
    _, k_f, _ = clay_bike.longitudinal(np.array([-0.3, -0.1]), 2.0, 0.85)
    assert k_f[0] == pytest.approx(-0.95)
    assert -0.5 < k_f[1] < -0.01


def test_braking_takes_away_cornering(clay_bike):
    u = np.array([6.0, 6.0])
    zero = np.zeros(2)
    fyf_free, _ = clay_bike.lateral_forces(u, zero, zero, np.full(2, 0.2), zero, zero, zero)
    fyf_lock, _ = clay_bike.lateral_forces(u, zero, zero, np.full(2, 0.2), np.full(2, -2.0),
                                           np.full(2, -0.95), np.full(2, -0.95))
    assert fyf_lock[0] < 0.2 * fyf_free[0]


@pytest.fixture(scope="module")
def clay_filter():
    try:
        f = MPSFSafetyFilter(vehicle_params=VP, dynamic_model=True, nn_model_dir=MODEL_DIR,
                             sense_range=0)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"acados solver or surrogate unavailable: {e}")
    f.update_terrain(_soil("clay"), use_terrain_nn=True)
    return f


def _state(x, u):
    return {"x": x, "y": 0.0, "psi": 0.0, "u": u, "v": 0.0, "omega": 0.0}


def test_clear_road_passes_unchanged(clay_filter):
    clay_filter._sol = None
    r = clay_filter.filter(0.1, 0.4, 0.0, _state(0.0, 6.0), [])
    # The executed input is the checked solver input, within the solver's
    # 0.01 tolerance of the command.
    assert not r.was_modified
    assert r.steering == pytest.approx(0.1, abs=0.01)
    assert r.throttle == pytest.approx(0.4, abs=0.01)


def test_wall_too_close_to_pass(clay_filter):
    # A wall across the lane 12 m ahead at 8 m/s on clay: stopping takes about
    # 13.6 m, and no swerve gets around a wall, so full throttle cannot pass.
    wall = [(45.0, float(y), 0.6) for y in np.arange(-12.0, 12.1, 1.2)]
    clay_filter._sol = None
    clay_filter._backup = None
    r = clay_filter.filter(0.0, 0.8, 0.0, _state(45.0 - 12.0, 8.0), wall)
    assert r.was_modified
    assert r.braking > 0.5


def test_check_predicts_oncoming_traffic():
    # Coasting at 4 m/s for the planned 2 s, then braking: clear of a vehicle
    # standing 30 m ahead, not of the same vehicle driving at us at 8 m/s.
    try:
        f = MPSFSafetyFilter(vehicle_params=VP, dynamic_model=True, nn_model_dir=MODEL_DIR,
                             sense_range=0, build_solver=False)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"surrogate unavailable: {e}")
    f.update_terrain(_soil("dirt"), use_terrain_nn=True)
    f._last_sense = (0.0, 0.0, 1.0e4, 2.0)
    z0 = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    plan = [(0.0, 0.0)] * f.N
    assert f._check(z0, plan, [(30.0, 0.0, 1.2, True)])
    assert not f._check(z0, plan, [(30.0, 0.0, 1.2, True, -8.0, 0.0)])
