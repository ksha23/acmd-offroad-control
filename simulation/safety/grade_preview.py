"""What the safety filters know of the grade ahead (the terrain-preview study).

The simulator may put one of two keys in the vehicle state it hands a filter:

* ``grade_now``: the sine of the chassis's nose-down pitch (an IMU), taken to
  hold everywhere ahead.
* ``terrain_ahead``: ``(ds, heights)``, ground heights every ``ds`` metres along
  the heading from the vehicle, nan where the sensor has not seen the ground.

A downgrade is measured over one wheelbase, so one cell does not set it, and
ground the map has not seen takes the steepest admissible downgrade. Gravity
along the heading, positive downhill, is ``G * sin(downgrade)``.
"""
from __future__ import annotations

import numpy as np

G = 9.81


def downgrades(vehicle_state: dict, wheelbase: float, unknown_downgrade: float):
    """The downgrades ahead as ``(ds, sines)``, a sample every ``ds`` metres,
    or ``(None, sine)`` for one grade everywhere, or ``None`` without a preview.

    Sample i is the downgrade from i ds to i ds + wheelbase ahead. Sines are
    clipped to [-0.5, 0.5]."""
    if vehicle_state.get("terrain_ahead") is not None:
        ds, h = vehicle_state["terrain_ahead"]
        h = np.asarray(h, dtype=float)
        k = max(1, int(round(wheelbase / ds)))
        if len(h) <= k:
            return None
        tan = (h[:-k] - h[k:]) / (k * ds)
        tan = np.where(np.isnan(tan), unknown_downgrade, tan)
        return float(ds), np.clip(tan / np.sqrt(1.0 + tan * tan), -0.5, 0.5)
    if vehicle_state.get("grade_now") is not None:
        return None, float(np.clip(vehicle_state["grade_now"], -0.5, 0.5))
    return None


def unknown_sine(unknown_downgrade: float) -> float:
    return unknown_downgrade / np.sqrt(1.0 + unknown_downgrade ** 2)


def gravity_lookup(vehicle_state: dict, wheelbase: float, unknown_downgrade: float):
    """Gravity along the heading (m/s^2, positive downhill) at a planar position,
    for a model that moves through the previewed ground, or None without a preview.

    The vehicle's centre of gravity at distance s along the heading from where
    the preview was taken sits on the downgrade sample starting half a
    wheelbase behind it. Beyond the preview the ground is unknown."""
    prof = downgrades(vehicle_state, wheelbase, unknown_downgrade)
    if prof is None:
        return None
    ds, down = prof
    if ds is None:
        g_const = G * down
        return lambda x, y: g_const
    x0, y0 = float(vehicle_state["x"]), float(vehicle_state["y"])
    c, s = np.cos(float(vehicle_state["psi"])), np.sin(float(vehicle_state["psi"]))
    g_unknown = G * unknown_sine(unknown_downgrade)

    def g_at(x, y):
        dist = (x - x0) * c + (y - y0) * s - 0.5 * wheelbase
        i = int(np.floor(max(dist, 0.0) / ds))
        return G * float(down[i]) if i < len(down) else g_unknown
    return g_at
