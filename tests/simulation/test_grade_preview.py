"""The grade preview the safety filters read (simulation/safety/grade_preview.py)."""
import numpy as np
import pytest

from simulation.safety.grade_preview import G, downgrades, gravity_lookup, unknown_sine


def test_downgrades_over_a_wheelbase():
    s = np.arange(0.0, 20.01, 0.5)
    h = -0.1 * np.maximum(s - 10.0, 0.0)            # flat to 10 m, then 10% down
    ds, down = downgrades({"terrain_ahead": (0.5, h)}, 3.0, 0.15)
    assert ds == 0.5
    assert down[0] == pytest.approx(0.0)
    assert down[-1] == pytest.approx(0.1 / np.sqrt(1.01))
    assert downgrades({}, 3.0, 0.15) is None
    assert downgrades({"grade_now": 0.05}, 3.0, 0.15) == (None, 0.05)


def test_unknown_ground_is_the_steepest_admissible():
    h = np.where(np.arange(41) > 10, np.nan, 0.0)
    _, down = downgrades({"terrain_ahead": (0.5, h)}, 3.0, 0.15)
    assert down[0] == pytest.approx(0.0)
    assert down[-1] == pytest.approx(unknown_sine(0.15))


def test_gravity_lookup_follows_the_heading():
    s = np.arange(0.0, 40.01, 0.5)
    h = -0.1 * np.maximum(s - 10.0, 0.0)
    state = {"x": 5.0, "y": 0.0, "psi": 0.0, "terrain_ahead": (0.5, h)}
    g = gravity_lookup(state, 3.0, 0.15)
    assert g(5.0, 0.0) == pytest.approx(0.0)                       # on the flat
    assert g(25.0, 0.0) == pytest.approx(G * 0.1 / np.sqrt(1.01))  # on the slope
    assert g(80.0, 0.0) == pytest.approx(G * unknown_sine(0.15))   # beyond the preview
    assert gravity_lookup({"x": 0.0, "y": 0.0, "psi": 0.0}, 3.0, 0.15) is None
