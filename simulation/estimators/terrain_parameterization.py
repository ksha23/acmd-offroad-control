"""Shared scalar-terrain parameterization used by the online estimators.

The online estimators identify only the Bekker sinkage exponent ``n``.  The
remaining Bekker--Mohr parameters are interpolated along the canonical
clay--dirt--sand preset manifold.  Keeping this mapping in a model-neutral
module prevents the runtime, safety, and warning code from importing a learned
vehicle-level terrain estimator merely to obtain a deterministic lookup.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    from simulation.shared.param_consistency import (
        TERRAIN_PRESETS,
        terrain_preset_to_internal,
    )
except ModuleNotFoundError:  # flat-path subprocess launch
    from param_consistency import TERRAIN_PRESETS, terrain_preset_to_internal


_PRESET_INTERNAL = {
    name: terrain_preset_to_internal(preset)
    for name, preset in TERRAIN_PRESETS.items()
}
_PRESET_SEQUENCE = tuple(
    sorted(_PRESET_INTERNAL.items(), key=lambda item: float(item[1]["n"]))
)
N_BOUNDS = (
    float(_PRESET_SEQUENCE[0][1]["n"]),
    float(_PRESET_SEQUENCE[-1][1]["n"]),
)
_PHI_BOUNDS = (
    min(float(params["phi"]) for params in _PRESET_INTERNAL.values()),
    max(float(params["phi"]) for params in _PRESET_INTERNAL.values()),
)


def terrain_params_for_n(
    n_value: float, *, soft_floor: Optional[float] = None,
    soft_mode: str = "hold",
) -> Dict[str, float]:
    """Interpolate all six SCM parameters along the preset manifold.

    The manifold is defined between the preset anchors, so by default values
    outside them clip to the nearest preset -- which makes the deployed clay
    preset (n = 0.50) the hard lower edge of everything built on this map. A
    bounded-grid posterior mean can therefore never report clay itself: no
    probability mass exists below the edge to balance the mass above it, and
    the softest soil becomes the least estimable one.

    ``soft_floor`` opts into a linear extrapolation of the clay--dirt segment
    down to the given exponent, so an estimator can place grid nodes below the
    deployed clay preset and treat it as an interior point. Slope continuity
    at the clay anchor is exact; the extrapolated soils are hypotheses for the
    estimator, not claims about plant soil. Default None preserves the
    published clipping behaviour bit-for-bit.
    """

    if soft_floor is not None and float(n_value) < float(N_BOUNDS[0]):
        n_value = float(max(float(n_value), float(soft_floor)))
        (_, lower), (_, upper) = _PRESET_SEQUENCE[0], _PRESET_SEQUENCE[1]
        if soft_mode == "linear":
            # Ablation mode. Extrapolating the clay--dirt trend exits the
            # force surrogate's training envelope almost immediately (Kphi
            # below the trained floor by n~0.45) and drives the Janosi length
            # negative by n~0.36, which is why the deployed configuration
            # holds the sub-clay span instead.
            n_lower, n_upper = float(lower["n"]), float(upper["n"])
            ratio = (n_value - n_lower) / (n_upper - n_lower)
            return {
                key: float(lower[key] + ratio * (upper[key] - lower[key]))
                for key in ("Kphi", "Kc", "n", "c", "phi", "k")
            }
        if soft_mode != "hold":
            raise ValueError(f"unknown soft_mode: {soft_mode!r}")
        # Anchored constant extension: below the measured clay anchor, hold
        # every non-exponent parameter at clay and vary only the exponent --
        # the coordinate being estimated. Nothing is invented, every
        # hypothesis stays inside the surrogate's training envelope by
        # construction, and the estimator's profiled nuisances (cohesion
        # multiplier, force gain) absorb deviation from the held values,
        # which is their designed role.
        held = dict(lower)
        held["n"] = n_value
        return {key: float(held[key]) for key in ("Kphi", "Kc", "n", "c", "phi", "k")}
    n_value = float(np.clip(n_value, N_BOUNDS[0], N_BOUNDS[1]))
    for index, (_, upper) in enumerate(_PRESET_SEQUENCE):
        if n_value <= float(upper["n"]):
            if index == 0:
                return dict(upper)
            _, lower = _PRESET_SEQUENCE[index - 1]
            n_lower = float(lower["n"])
            n_upper = float(upper["n"])
            if n_upper <= n_lower:
                return dict(upper)
            ratio = (n_value - n_lower) / (n_upper - n_lower)
            return {
                key: float(lower[key] + ratio * (upper[key] - lower[key]))
                for key in ("Kphi", "Kc", "n", "c", "phi", "k")
            }
    return dict(_PRESET_SEQUENCE[-1][1])


def closest_preset_name(n_value: float, phi_value: Optional[float] = None) -> str:
    """Return the nearest canonical preset for human-readable telemetry."""

    best_name = "unknown"
    best_score = float("inf")
    for name, preset in _PRESET_INTERNAL.items():
        dn = abs(float(preset["n"]) - float(n_value)) / max(
            N_BOUNDS[1] - N_BOUNDS[0], 1.0e-6
        )
        if phi_value is None:
            score = dn
        else:
            dphi = abs(float(preset["phi"]) - float(phi_value)) / max(
                _PHI_BOUNDS[1] - _PHI_BOUNDS[0], 1.0e-6
            )
            score = dn + dphi
        if score < best_score:
            best_name = name
            best_score = score
    return best_name


# Compatibility aliases for older callers while the rig branch is cleaned.
_terrain_params_for_n = terrain_params_for_n
_closest_preset_name = closest_preset_name
_N_BOUNDS = N_BOUNDS
