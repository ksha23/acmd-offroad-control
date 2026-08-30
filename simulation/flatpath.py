"""Import-path bootstrap for the simulation package.

Places the ``simulation`` directory and each of its subpackage directories on
``sys.path`` so a module may be imported either by its package path
(``simulation.tire_models.nn_tire_model``) or by its bare module name
(``nn_tire_model``). Both spellings appear across the simulation, control, and
benchmarking entry points, and this bootstrap makes them resolve regardless of
the directory a script is launched from. The operation is idempotent:
importing this module, directly or via ``import simulation``, is sufficient.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
# The package root plus every role subpackage directory beneath it.
_dirs = [_ROOT] + [_os.path.join(_ROOT, x) for x in sorted(_os.listdir(_ROOT))
                   if _os.path.isdir(_os.path.join(_ROOT, x)) and not x.startswith(("_", "."))]
for _d in _dirs:
    if _os.path.isdir(_d) and _d not in _sys.path:
        _sys.path.insert(0, _d)
