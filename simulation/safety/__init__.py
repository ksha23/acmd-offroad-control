"""Swappable safety filters.

A safety filter sits between the command source -- the tracking NMPC or a human
operator -- and the plant, and admits the command unchanged whenever doing so
is safe. Two filters implement that contract and are selected by name through
:func:`make_safety_filter`:

* ``dob_cbf`` -- a reactive minimum-deviation control-barrier-function QP with
  a disturbance observer and delay compensation (:mod:`.dob_cbf`).
* ``mpsf`` -- a predictive model predictive safety filter, which passes the
  command through while a stoppable trajectory remains feasible over a short
  horizon (:mod:`.mpsf`).

Both return a :class:`SafetyFilterResult` and expose ``filter``,
``update_command_age``, ``set_teleop_delay``, and ``get_diagnostics``, so the
plant node treats them interchangeably. Running with no filter is the third
supported configuration and is the reference against which either filter's
intervention rate and harm-prevention are measured.
"""

from .dob_cbf import (  # noqa: F401
    CBFSafetyFilter,
    DelayCompensator,
    DisturbanceObserver,
    SafetyFilterResult,
)



# ============================================================================
# Safety-filter factory. Registration is by name, so an additional filter is
# added by implementing the shared interface and adding one branch below.
# ============================================================================

SAFETY_FLAVORS = ('dob_cbf', 'mpsf')

# Names that are rejected with a pointer to the supported set rather than a
# bare "unknown flavor", because they read as plausible filter names.
_UNSUPPORTED_FLAVORS = ('mppi', 'mppi_shield', 'nmpc', 'nmpc_shield')


def make_safety_filter(flavor: str,
                       vehicle_params: dict,
                       nn_model=None,
                       terrain_params: dict | None = None,
                       **flavor_kwargs):
    """Construct the safety filter named by ``flavor``.

    Args:
        flavor: one of ``SAFETY_FLAVORS``. ``'dob_cbf'`` selects the reactive
            minimum-deviation barrier QP; ``'mpsf'`` selects the predictive
            filter.
        vehicle_params: dict with ``M, Lf, Lr, Izz, ...``.
        nn_model: a loaded ``NNTireModel`` supplying tire forces. Optional for
            ``dob_cbf``, which falls back to a kinematic steering model and
            fixed longitudinal limits when it is absent.
        terrain_params: accepted for signature uniformity across flavors;
            neither filter reads it at construction, because both take their
            soil belief through ``update_terrain`` once estimates arrive.
        **flavor_kwargs: forwarded verbatim to the selected filter.

    Returns:
        A filter instance exposing ``.filter(...)``,
        ``.update_command_age(...)``, ``.set_teleop_delay(...)``,
        and ``.get_diagnostics()``.
    """
    f = (flavor or '').lower()
    if f in ('dob_cbf', 'cbf', 'legacy', 'dob-cbf'):
        return CBFSafetyFilter(vehicle_params=vehicle_params,
                               nn_casadi=nn_model,
                               **flavor_kwargs)
    if f in ('mpsf', 'predictive', 'mpc_safety'):
        from .mpsf import MPSFSafetyFilter
        return MPSFSafetyFilter(vehicle_params=vehicle_params, **flavor_kwargs)
    if f in _UNSUPPORTED_FLAVORS:
        raise ValueError(
            f"safety flavor {flavor!r} is not implemented. The supported "
            f"filters are {SAFETY_FLAVORS}: 'dob_cbf' for the reactive "
            f"minimum-deviation barrier QP, 'mpsf' for the predictive filter.")
    raise ValueError(f"Unknown safety flavor {flavor!r}; "
                     f"expected one of {SAFETY_FLAVORS}")


__all__ = [
    "CBFSafetyFilter",
    "DelayCompensator",
    "DisturbanceObserver",
    "SafetyFilterResult",
    "SAFETY_FLAVORS",
    "make_safety_filter",
]
