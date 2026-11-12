"""Swappable safety filters.

A safety filter sits between the command source (the tracking NMPC or a human operator) and the plant, and
passes the command unchanged whenever doing so is safe. Two filters implement that contract and are selected by
name through :func:`make_safety_filter`:

* ``dob_cbf``: a minimum-deviation barrier QP over the steering and the pedal, with a disturbance observer
  (:mod:`.dob_cbf`).
* ``mpsf``: a model predictive safety filter, which passes the command while a plan ending in a full stop stays
  clear on the vehicle model (:mod:`.mpsf`).

Both take every number from the plant's specification, the rig surrogate or the bounds identified on separate
plant manoeuvres (``ident/``), return a :class:`SafetyFilterResult` and expose ``filter``, ``update_terrain``,
``update_command_age``, ``set_teleop_delay``, ``set_clock`` and ``get_diagnostics``, so the plant node treats them
interchangeably. Running with no filter is the third configuration, the reference for either filter's
intervention rate and harm prevented.
"""

from .common import SafetyFilterResult  # noqa: F401

SAFETY_FLAVORS = ('dob_cbf', 'mpsf')


def make_safety_filter(flavor: str, vehicle_params: dict, **flavor_kwargs):
    """Construct the safety filter named by ``flavor`` (one of ``SAFETY_FLAVORS``).

    ``vehicle_params`` is a dict with ``M, Lf, Lr, Izz, h_cg, T``. ``flavor_kwargs`` go to the filter's
    constructor (``dob_cbf.CBFSafetyFilter`` or ``mpsf.MPSFSafetyFilter``). Both need ``ident``, the identified
    bounds.
    """
    f = (flavor or '').lower()
    if f == 'dob_cbf':
        from .dob_cbf import CBFSafetyFilter
        return CBFSafetyFilter(vehicle_params=vehicle_params, **flavor_kwargs)
    if f == 'mpsf':
        from .mpsf import MPSFSafetyFilter
        return MPSFSafetyFilter(vehicle_params=vehicle_params, **flavor_kwargs)
    raise ValueError(f"Unknown safety flavor {flavor!r}; expected one of {SAFETY_FLAVORS}")


__all__ = ["SafetyFilterResult", "SAFETY_FLAVORS", "make_safety_filter"]
