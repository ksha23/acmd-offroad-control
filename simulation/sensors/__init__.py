"""
Sensors subpackage
==================

Places the obstacle population a run must avoid, and exposes its geometry to
the controller, the safety filter, and the collision logger.
"""

from .obstacles import add_rock_obstacles, get_rock_positions, get_rock_radii
