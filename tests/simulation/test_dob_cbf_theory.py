"""Contracts of the textbook DOB-CBF (simulation/safety/dob_cbf_theory.py).

The QP solver is checked against brute force, the barrier derivatives against
finite differences along the model, and the filter against the two plain
contracts: a safe command passes unchanged, and a command into a wall too close
to stop for is braked.
"""
import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("torch")

from simulation.safety.dob_cbf_theory import TheoryCBFSafetyFilter, solve_qp_2d  # noqa: E402
from simulation.shared.param_consistency import get_terrain_preset, terrain_preset_to_internal  # noqa: E402

VP = {"M": 2573.0, "Izz": 3570.0, "Lf": 1.593, "Lr": 1.709, "h_cg": 0.65, "T": 1.8194}


def test_qp_matches_brute_force():
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


@pytest.fixture(scope="module")
def clay_filter():
    try:
        f = TheoryCBFSafetyFilter(VP, control_dt=0.1, sense_range=0)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"surrogate unavailable: {e}")
    f.update_terrain(terrain_preset_to_internal(get_terrain_preset("clay")))
    return f


def test_hocbf_derivative_matches_finite_difference(clay_filter):
    f = clay_filter
    # Ego state, then the obstacle's position; the obstacle moves at (-1.5, 0.5).
    x = np.array([2.0, 1.0, 0.2, 6.0, 0.1, 15.0, 2.0])
    o = [2.0, -1.5, 0.5]
    h, h_dot = (float(t) for t in f._hocbf(x, o, f.p1)[:2])
    eps = 1e-5
    xn = x.copy()
    xn[0] += eps * x[3] * np.cos(x[2])
    xn[1] += eps * x[3] * np.sin(x[2])
    xn[2] += eps * x[3] * np.tan(x[4]) / f.L
    xn[5] += eps * o[1]
    xn[6] += eps * o[2]
    h2 = float(f._hocbf(xn, o, f.p1)[0])
    assert (h2 - h) / eps == pytest.approx(h_dot, rel=1e-3)


def test_clay_braking_authority_from_surrogate(clay_filter):
    # The plant brakes at 2.47 to 2.55 m/s^2 on clay. The margined barrier
    # deceleration must sit below that.
    assert 2.0 < clay_filter._a_h < 2.47


def _state(x, u):
    return {"x": x, "y": 0.0, "psi": 0.0, "u": u, "v": 0.0, "omega": 0.0}


def test_clear_road_passes_unchanged(clay_filter):
    r = clay_filter.filter(0.0, 0.4, 0.0, _state(0.0, 6.0), [])
    assert not r.was_modified
    assert r.throttle == pytest.approx(0.4)


def test_wall_too_close_is_braked():
    f = TheoryCBFSafetyFilter(VP, control_dt=0.1, sense_range=0)
    f.update_terrain(terrain_preset_to_internal(get_terrain_preset("clay")))
    wall = [(45.0, float(y), 0.6) for y in np.arange(-12.0, 12.1, 1.2)]
    r = f.filter(0.0, 0.8, 0.0, _state(45.0 - 14.0, 8.0), wall)
    assert r.was_modified
    assert r.braking > 0.0
