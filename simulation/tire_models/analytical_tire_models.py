#!/usr/bin/env python3
"""
Analytical Tire Models (CasADi symbolic)
=========================================

CasADi-symbolic tire force laws that provide the analytical comparison arms of
the ACADOS optimal control problem.

Each function takes CasADi symbolic slip angles and normal forces and returns
``(Fyf, Fyr, Fx_traction)`` as CasADi expressions suitable for embedding in
an NLP or OCP.

Supported models:
  - Pacejka Magic Formula (simplified single-parameter set from HMMWV_Pac02Tire.tir)
  - TMeasy degressive model (smooth sin-based approximation)

A linear cornering-stiffness law is not offered here: it is valid only at
small slip angles and therefore adds nothing over Pacejka or TMeasy on the SCM
deformable-terrain scenarios this codebase targets.  ``StatePredictor`` in
``mpc_helpers.py`` does carry an internal linear bicycle model, but it serves
delay compensation and is not a selectable tire model.
"""

import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401
import casadi as ca

# ============================================================================
# Default parameters: one global, SCM-calibrated friction coefficient.
# The peak friction coefficient mu dominates the terrain dependence of these
# laws, so it is calibrated to SCM rather than to a rigid road: mu = 0.42 is
# the mean lateral force coefficient |Fy|/Fz over the whole soil box
# (data/tire_rig_static/train.csv; the SCM peak ranges from about 0.18 on
# clay-like soil to about 0.47 on sand-like soil, see
# benchmarking/calibrate_analytical_tires.py). One global value serves every
# terrain because the controller carries no per-terrain knowledge, which makes
# this the deployment-consistent comparison arm. The magic-formula and TMeasy
# shape factors keep their standard values so that cornering stiffness stays
# realistic; fitting force magnitude alone flattens the initial slope and
# degrades closed-loop tracking, which would understate the arm's true
# capability. A single global mu cannot express terrain-to-terrain force
# variation, and that residual gap is what the neural surrogate closes. The
# per-terrain "oracle" parameters below are a separate,
# information-advantaged reference.
# ============================================================================

# Pacejka Magic Formula (standard shape; SCM-calibrated peak friction)
PACEJKA_B = 8.77    # Stiffness factor (|PKY1|/(PCY1*PDY1)) -- standard
PACEJKA_C = 1.5874  # Shape factor (PCY1) -- standard
PACEJKA_E = 0.376   # Curvature factor (PEY1) -- standard
PACEJKA_MU = 0.42   # Peak friction coefficient calibrated to SCM

# TMeasy (standard lateral shape; mu drives traction, calibrated to SCM)
TMEASY_DFY0 = 40000.0       # Initial slope (N/rad per tire)
TMEASY_FYM = 4000.0         # Peak lateral force per tire (N) ~ 0.40*Fz, matches SCM
TMEASY_ALPHA_M = 0.12       # Slip angle at peak (~7 deg)
TMEASY_ALPHA_SLIDE = 0.25   # Slip angle at full sliding (~14 deg)


# ============================================================================
# Terrain-specific ("oracle") Pacejka parameters
# ============================================================================
# These are physically motivated settings for SCM terrain types. They receive
# ground-truth terrain identity, but they are not data-fitted performance upper
# bounds: Mohr--Coulomb soil friction is not identical to effective tire lateral
# friction, and Pacejka still omits SCM pressure-sinkage and cohesion effects.
#
# Relative to the single global parameterization above:
#   mu   → terrain friction coefficient: tan(mohr_friction_angle).
#           This is the dominant effect: soft soils have much lower peak Fy.
#   B    → cornering stiffness factor: lower on soft soils because deformation
#           absorbs lateral load more gradually before reaching peak.
#
# Even with terrain identity supplied, Pacejka cannot express SCM-specific
# effects (Bekker pressure-sinkage saturation, cohesion-enhanced traction,
# non-linear slip stiffness), which is where the neural surrogate separates
# from it, on clay in particular.
#
# Reference: tan(phi) for clay=13°, sand=30°, dirt=29° (see
# simulation/shared/param_consistency.py)
PACEJKA_ORACLE = {
    'clay': {
        'B': 5.5,     # Lower stiffness — SCM clay deforms under lateral load
        'C': 1.5874,
        'E': 0.376,
        'mu': 0.231,  # tan(13°) — very low traction
    },
    'sand': {
        'B': 7.5,     # Moderate stiffness — granular material, still deformable
        'C': 1.5874,
        'E': 0.376,
        'mu': 0.577,  # tan(30°)
    },
    'dirt': {
        'B': 7.5,     # Similar to sand
        'C': 1.5874,
        'E': 0.376,
        'mu': 0.554,  # tan(29°)
    },
}


# ============================================================================
# Rig-fitted per-terrain Pacejka parameters
# ============================================================================
# All four coefficients are least-squares fits to the single-tire rig corpus
# (data/tire_rig_commanded/train.csv, sha256 17bd30b5...), kernel-weighted
# around each SCM preset in normalized Bekker--Mohr space (bandwidth 0.18 of
# each dimension's range; effective sample count 225-444 per soil). C is
# constrained to the standard lateral shape range [1.2, 2.5]: below about 1
# the B-C-mu ridge is degenerate and mu -- which also sets the Coulomb
# traction budget -- ceases to mean peak friction. Fit script:
# benchmarking/fit_rigfit_pacejka.py; holdout report:
# benchmarking/results/rigfit_pacejka_fit.json, where this parameterization
# attains the highest weighted holdout R^2 of the three on every soil. This
# arm carries the same training data as the neural surrogate, expressed
# through the analytical form the controller deploys.
PACEJKA_RIGFIT = {
    'clay': {'B': 12.785, 'C': 1.2, 'E': 0.61, 'mu': 0.402},
    'dirt': {'B': 15.081, 'C': 1.2, 'E': 0.725, 'mu': 0.607},
    'sand': {'B': 15.556, 'C': 1.2, 'E': 0.789, 'mu': 0.708},
}


def get_rigfit_pacejka_params(terrain_name: str) -> dict:
    """Return rig-corpus-fitted Pacejka parameter dict for *terrain_name*."""
    if terrain_name not in PACEJKA_RIGFIT:
        raise ValueError(
            f"No rig-fitted params for terrain {terrain_name!r}. "
            f"Available: {list(PACEJKA_RIGFIT.keys())}"
        )
    return dict(PACEJKA_RIGFIT[terrain_name])


def get_oracle_pacejka_params(terrain_name: str) -> dict:
    """Return oracle Pacejka parameter dict for *terrain_name*.

    These parameters use the terrain's Mohr–Coulomb friction angle as the
    peak friction coefficient. They are an information-advantaged physical
    reference, not a fitted performance upper bound.

    Args:
        terrain_name: One of 'clay', 'sand', 'dirt'.

    Returns:
        Dict with keys 'B', 'C', 'E', 'mu' suitable for
        ``pacejka_tire_forces(**params)``.
    """
    if terrain_name not in PACEJKA_ORACLE:
        raise ValueError(
            f"No oracle params for terrain {terrain_name!r}. "
            f"Available: {list(PACEJKA_ORACLE.keys())}"
        )
    return dict(PACEJKA_ORACLE[terrain_name])


# ============================================================================
# Combined slip reduction factor
# ============================================================================

def combined_slip_factor(kappa):
    """Lateral force reduction due to combined longitudinal+lateral slip.

    Returns a CasADi expression in [sqrt(0.1), 1.0].
    """
    return ca.sqrt(ca.fmax(1.0 - (kappa / 0.2) ** 2, 0.1))


# ============================================================================
# Pacejka Magic Formula
# ============================================================================

def pacejka_tire_forces(alpha_f, alpha_r, Fz_f_axle, Fz_r_axle, kappa,
                        B=PACEJKA_B, C=PACEJKA_C, E=PACEJKA_E, mu=PACEJKA_MU):
    """Pacejka Magic Formula lateral forces (simplified, per-axle).

    Fy = D * sin(C * atan(B*α − E*(B*α − atan(B*α))))
    where D = μ * Fz_axle * combined_slip_factor

    Returns:
        (Fyf, Fyr, Fx_traction) — CasADi symbolic expressions.
    """
    lat = combined_slip_factor(kappa)

    Df = mu * Fz_f_axle * lat
    Dr = mu * Fz_r_axle * lat

    Baf = B * alpha_f
    Bar = B * alpha_r

    Fyf = Df * ca.sin(C * ca.atan(Baf - E * (Baf - ca.atan(Baf))))
    Fyr = Dr * ca.sin(C * ca.atan(Bar - E * (Bar - ca.atan(Bar))))

    Fx_traction = mu * (Fz_f_axle + Fz_r_axle)
    return Fyf, Fyr, Fx_traction


# ============================================================================
# TMeasy degressive model
# ============================================================================

def tmeasy_tire_forces(alpha_f, alpha_r, Fz_f_axle, Fz_r_axle, kappa,
                       dFy0=TMEASY_DFY0, Fym=TMEASY_FYM,
                       alpha_m=TMEASY_ALPHA_M, alpha_slide=TMEASY_ALPHA_SLIDE,
                       mu=PACEJKA_MU):
    """TMeasy degressive lateral force model (smooth sin-based approx).

    Per-tire force is scaled by (Fz_axle / 2*Fz_nom) to account for load
    transfer, then doubled for the axle total.

    Returns:
        (Fyf, Fyr, Fx_traction) — CasADi symbolic expressions.
    """
    lat = combined_slip_factor(kappa)

    Fz_nom = (Fz_f_axle + Fz_r_axle) / 2.0
    Fz_f_ratio = Fz_f_axle / (2.0 * ca.fmax(Fz_nom, 1.0))
    Fz_r_ratio = Fz_r_axle / (2.0 * ca.fmax(Fz_nom, 1.0))

    half_pi = 1.5707963
    am_safe = ca.fmax(alpha_m, 1e-4)

    # The sine argument must be clamped on BOTH sides: a one-sided fmin let
    # slip angles below -2*alpha_m drive the argument past -pi, where the
    # sine wraps and the lateral force changes sign -- well inside the
    # solver's alpha clamp, and asymmetric between left and right turns
    # (2026-08-26 audit, finding F1).
    def _sat(arg):
        return ca.sin(ca.fmax(ca.fmin(arg, half_pi), -half_pi))

    Fyf_per_tire = Fym * Fz_f_ratio * lat * _sat(half_pi * alpha_f / am_safe)
    Fyr_per_tire = Fym * Fz_r_ratio * lat * _sat(half_pi * alpha_r / am_safe)

    Fyf = 2.0 * Fyf_per_tire
    Fyr = 2.0 * Fyr_per_tire

    Fx_traction = mu * (Fz_f_axle + Fz_r_axle)
    return Fyf, Fyr, Fx_traction


# ============================================================================
# Dispatch helper
# ============================================================================

def get_tire_forces(tire_model, alpha_f, alpha_r, Fz_f_axle, Fz_r_axle, kappa,
                    **params):
    """Dispatch to the appropriate tire model by name.

    Args:
        tire_model: One of 'pacejka', 'tmeasy'.
        alpha_f, alpha_r: CasADi symbolic slip angles.
        Fz_f_axle, Fz_r_axle: CasADi symbolic axle normal forces.
        kappa: CasADi symbolic longitudinal slip ratio.
        **params: Model-specific overrides (e.g. B, C, E, mu for Pacejka).

    Returns:
        (Fyf, Fyr, Fx_traction)
    """
    if tire_model == 'pacejka':
        return pacejka_tire_forces(alpha_f, alpha_r, Fz_f_axle, Fz_r_axle,
                                   kappa, **params)
    elif tire_model == 'tmeasy':
        return tmeasy_tire_forces(alpha_f, alpha_r, Fz_f_axle, Fz_r_axle,
                                  kappa, **params)
    else:
        raise ValueError(f"Unknown tire model: {tire_model!r}. "
                         f"Choose from: pacejka, tmeasy")
