"""Test package: puts the project packages and their flat-import roots on
sys.path so tests import their subjects exactly as the runtime does."""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "benchmarking"),
           _os.path.join(_ROOT, "nn_training")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import simulation.flatpath  # noqa: E402,F401  (simulation role subdirs)
