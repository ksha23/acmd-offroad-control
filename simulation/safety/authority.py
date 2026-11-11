"""Soil authority from the rig tire surrogate, shared by both safety filters.

Both filters ask the same two questions of the surrogate: how hard can the
vehicle brake on this soil, and how hard can it corner. The answers are the
peak four-wheel force over a slip sweep at the static axle loads, divided by
the vehicle mass. Keeping the sweep in one place means the two filters can
never disagree about the soil.
"""
from __future__ import annotations

import numpy as np


def peak_braking_decel(nn, terrain: dict, fz_f: float, fz_r: float,
                       mass: float) -> float:
    """Peak braking deceleration (m/s^2) over a longitudinal slip sweep.

    Returns 0.0 when no query in the sweep is finite.
    """
    n = float(terrain["n"])
    best = 0.0
    for kappa in np.linspace(-0.9, -0.02, 30):
        fx_f, _ = nn.predict_numeric(0.0, fz_f, 5.0, kappa=float(kappa),
                                     n_terrain=n, terrain_params=terrain,
                                     rates=np.zeros(3))
        fx_r, _ = nn.predict_numeric(0.0, fz_r, 5.0, kappa=float(kappa),
                                     n_terrain=n, terrain_params=terrain,
                                     rates=np.zeros(3))
        if not (np.isfinite(fx_f) and np.isfinite(fx_r)):
            continue
        decel = (2.0 * abs(fx_f) + 2.0 * abs(fx_r)) / mass
        best = max(best, decel)
    return float(best)


def peak_lateral_accel(nn, terrain: dict, fz_f: float, fz_r: float,
                       mass: float) -> float:
    """Peak lateral acceleration (m/s^2) over a slip-angle sweep.

    Returns 0.0 when no query in the sweep is finite.
    """
    n = float(terrain["n"])
    best = 0.0
    for slip in np.linspace(0.02, 0.5, 30):
        fy_f = nn.predict_numeric(float(slip), fz_f, 5.0, kappa=0.0,
                                  n_terrain=n, terrain_params=terrain,
                                  rates=np.zeros(3))[1]
        fy_r = nn.predict_numeric(float(slip), fz_r, 5.0, kappa=0.0,
                                  n_terrain=n, terrain_params=terrain,
                                  rates=np.zeros(3))[1]
        if not (np.isfinite(fy_f) and np.isfinite(fy_r)):
            continue
        lat = (2.0 * abs(fy_f) + 2.0 * abs(fy_r)) / mass
        best = max(best, lat)
    return float(best)


def static_axle_loads():
    """Static per-wheel normal loads (front, rear) in N and the vehicle mass."""
    try:
        try:
            import simulation.shared.param_consistency as pc
        except ModuleNotFoundError:
            import shared.param_consistency as pc
        fz_f, fz_r = pc.get_static_fz_per_wheel()
        fz_f, fz_r = float(fz_f), float(fz_r)
        return fz_f, fz_r, 2.0 * (fz_f + fz_r) / 9.81
    except Exception:  # noqa: BLE001
        return 6532.0, 6089.0, 2573.0
