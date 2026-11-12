"""What both safety filters share: their result type and the loaders for the rig surrogate and the soil presets.

The loaders work whether the repo root (the tests) or ``simulation/`` (the plant node) is on ``sys.path``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyFilterResult:
    """Output of a safety filter."""
    steering: float          # filtered steering input [-1, 1]
    throttle: float          # filtered throttle input [0, 1]
    braking: float           # filtered braking input [0, 1]
    was_modified: bool       # True if the filter changed the inputs
    active_constraints: int  # constraint rows (DOB-CBF) or obstacles in the OCP (MPSF)
    solve_time_ms: float     # time spent in the filter call
    v_max_terrain: float     # speed bound the filter imposes (nan or a large number: none)
    safety_margin: float     # smallest barrier value (DOB-CBF) or nearest obstacle centre (MPSF)
    dob_norm: float = 0.0    # magnitude of the disturbance estimate (DOB-CBF)


def nn_loader():
    """``load_nn_tire_model`` from the rig surrogate's module."""
    try:
        from simulation.tire_models.nn_tire_model import load_nn_tire_model
    except ModuleNotFoundError:
        from tire_models.nn_tire_model import load_nn_tire_model
    return load_nn_tire_model


def presets():
    """The soil presets module (``param_consistency``)."""
    try:
        import simulation.shared.param_consistency as pc
    except ModuleNotFoundError:
        import shared.param_consistency as pc
    return pc


def nominal_terrain(soil: str) -> dict:
    """A preset soil in the surrogate's internal convention."""
    pc = presets()
    return pc.terrain_preset_to_internal(pc.get_terrain_preset(soil))
