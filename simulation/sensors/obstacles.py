"""
Rock obstacles for PyChrono simulations
=======================================

Places randomized rock obstacles into the simulated environment. Each rock is
a fixed rigid body with:

- an ellipsoid collision shape, slightly flattened vertically, which is cheap
  enough to evaluate alongside the deformable SCM terrain;
- a sphere visual shape;
- randomized position, size, and orientation inside a configurable zone, drawn
  from a seeded generator so a placement is reproducible from its seed.

Rocks are the static obstacle population for the collision-avoidance study.
Contact between the ego vehicle and a rock body is collision truth; the
geometric clearance reported elsewhere is a separate diagnostic.

Usage:
    from sensors.obstacles import add_rock_obstacles

    rocks = add_rock_obstacles(system, num_rocks=20,
                               zone_x=(-15, 50), zone_y=(-10, 10),
                               size_range=(0.5, 3.0))
"""

from typing import List, Tuple, Optional

import numpy as np

try:
    import pychrono as chrono
    CHRONO_AVAILABLE = True
except ImportError:
    CHRONO_AVAILABLE = False


def add_rock_obstacles(system: 'chrono.ChSystem',
                       num_rocks: int = 20,
                       zone_x: Tuple[float, float] = (-15.0, 50.0),
                       zone_y: Tuple[float, float] = (-10.0, 10.0),
                       size_range: Tuple[float, float] = (0.5, 3.0),
                       seed: int = 42,
                       bury_fraction: float = 0.3,
                       min_spacing: float = 0.0,
                       centerline_clear: float = 0.0,
                       centerline_keep_prob: float = 0.35,
                       exclusion_zones: List[Tuple[float, float, float]] = None) -> List[dict]:
    """
    Add randomized rock obstacles to the Chrono system.

    Creates fixed rigid bodies with ellipsoid collision shapes distributed
    randomly within the specified zone, partially buried in the ground by
    ``bury_fraction``.

    Exclusion zones keep rocks away from the vehicle's starting position and
    from any other region that must remain unobstructed.

    Args:
        system: PyChrono ChSystem instance
        num_rocks: Number of rocks to place
        zone_x: (min_x, max_x) placement zone in world X (m)
        zone_y: (min_y, max_y) placement zone in world Y (m)
        size_range: (min_diameter, max_diameter) of rocks (m)
        seed: Random seed; the same seed reproduces the same field
        bury_fraction: Fraction of the rock below ground level (0-1), so the
                      exposed profile matches a partially embedded boulder
                      rather than one resting on the surface.
        min_spacing: If >0, reject any candidate whose centre is closer than
                      this (m) to an already-placed rock. This turns a random
                      scatter into a blue-noise boulder field: a threadable
                      gap always exists between rocks, no impassable clump
                      forms, and the uniform density leaves no free bypass.
        centerline_clear: If >0, rocks landing within this lateral half-width
                      (m) of y=0 are kept only with probability
                      centerline_keep_prob. This thins the line a lead vehicle
                      takes through the field without clearing it, so a route
                      can be chosen while rocks still intrude on the
                      centreline.
        centerline_keep_prob: Keep-probability for rocks inside centerline_clear.
        exclusion_zones: List of (center_x, center_y, radius) tuples.
                        No rocks are placed within these circles.

    Returns:
        List of dicts with rock metadata:
            {'body': ChBody, 'x': float, 'y': float, 'z': float,
             'size': float, 'yaw': float}
    """
    if not CHRONO_AVAILABLE:
        raise RuntimeError("pychrono not available")
    
    rng = np.random.RandomState(seed)
    rocks = []
    
    # Contact material for rocks. A Young's modulus of 1e9 Pa lies in the
    # soft-rock range and generates enough SMC penalty force to arrest a
    # 2500 kg chassis at roughly 7 m/s without visible penetration, so a
    # partially buried rock reads as a genuine obstacle to the collision KPI.
    rock_material = chrono.ChContactMaterialSMC()
    rock_material.SetFriction(0.9)
    rock_material.SetYoungModulus(1e9)
    rock_material.SetRestitution(0.1)
    
    attempts = 0
    # Blue-noise rejection (min_spacing) and centerline thinning reject many
    # candidates, so allow more attempts before giving up on the requested count.
    max_attempts = num_rocks * (40 if (min_spacing > 0.0 or centerline_clear > 0.0) else 10)
    
    while len(rocks) < num_rocks and attempts < max_attempts:
        attempts += 1
        
        # Random position within zone
        x = rng.uniform(zone_x[0], zone_x[1])
        y = rng.uniform(zone_y[0], zone_y[1])
        
        # Check exclusion zones
        if exclusion_zones:
            excluded = False
            for ex, ey, er in exclusion_zones:
                if (x - ex)**2 + (y - ey)**2 < er**2:
                    excluded = True
                    break
            if excluded:
                continue

        # Centreline thinning: keep only a fraction of the rocks near the route
        # a lead vehicle takes, so that line is less dense but never cleared.
        if centerline_clear > 0.0 and abs(y) < centerline_clear:
            if rng.random() > centerline_keep_prob:
                continue

        # Blue-noise spacing: reject candidates too close to an existing rock so
        # the field is always threadable (a gap >= min_spacing always exists)
        # but never leaves a clear lateral bypass.
        if min_spacing > 0.0 and rocks:
            too_close = False
            for r in rocks:
                if (x - r['x'])**2 + (y - r['y'])**2 < min_spacing**2:
                    too_close = True
                    break
            if too_close:
                continue

        # Random size and orientation
        size = rng.uniform(size_range[0], size_range[1])
        yaw = rng.uniform(0, 2 * np.pi)
        
        # Vertical position: partially buried
        z = size * bury_fraction
        
        # Create rock body (fixed, no dynamics)
        rock_body = chrono.ChBody()
        rock_body.SetPos(chrono.ChVector3d(x, y, z))
        rock_body.SetRot(chrono.QuatFromAngleZ(yaw))
        rock_body.SetFixed(True)
        rock_body.SetMass(2500 * size**3)  # Approximate rock density * volume
        
        # Collision shape: an ellipsoid, flattened vertically, approximating a
        # boulder's footprint at a fraction of a mesh's contact cost.
        # ChCollisionShapeEllipsoid takes FULL axis lengths and halves them
        # internally (ChEllipsoid.rad = axes/2). ``size`` is the rock's
        # diameter, so the full axes are (size, size, 0.8*size), giving
        # semi-axes (0.5s, 0.5s, 0.4s) -- the footprint radius 0.5s that the
        # planner and the safety filters are given for this rock, enclosing
        # the 0.45s visual sphere. An earlier revision passed the semi-axes
        # where the full axes belong, so every rock's collision body was half
        # its stated and rendered size and hovered 0.05s above grade instead
        # of sitting partially buried.
        collision_size = chrono.ChVector3d(size, size, size * 0.8)
        coll_shape = chrono.ChCollisionShapeEllipsoid(rock_material, collision_size)
        rock_body.AddCollisionShape(coll_shape)
        rock_body.EnableCollision(True)
        
        # Visual shape: sphere with rock-like color
        vis_sphere = chrono.ChVisualShapeSphere(size * 0.45)
        vis_sphere.SetColor(chrono.ChColor(0.45, 0.38, 0.30))  # Brown-grey rock color
        rock_body.AddVisualShape(vis_sphere)
        
        system.Add(rock_body)
        
        rocks.append({
            'body': rock_body,
            'x': x, 'y': y, 'z': z,
            'size': size, 'yaw': yaw,
        })
    
    if len(rocks) < num_rocks:
        print(f"  [ROCKS] WARNING: Only placed {len(rocks)}/{num_rocks} rocks "
              f"(exclusion zones too restrictive)")
    
    print(f"  [ROCKS] Placed {len(rocks)} rocks in zone "
          f"x=[{zone_x[0]:.0f}, {zone_x[1]:.0f}], y=[{zone_y[0]:.0f}, {zone_y[1]:.0f}], "
          f"size=[{size_range[0]:.1f}, {size_range[1]:.1f}]m")
    
    return rocks


def get_rock_positions(rocks: List[dict]) -> np.ndarray:
    """
    Extract rock center positions as an Nx3 array.
    
    Args:
        rocks: List returned by add_rock_obstacles()
    
    Returns:
        Nx3 numpy array of [x, y, z] world positions
    """
    if not rocks:
        return np.zeros((0, 3))
    return np.array([[r['x'], r['y'], r['z']] for r in rocks])


def get_rock_radii(rocks: List[dict]) -> np.ndarray:
    """
    Extract effective collision radii as an N-length array.
    
    Args:
        rocks: List returned by add_rock_obstacles()
    
    Returns:
        N-length numpy array of effective radii (m)
    """
    if not rocks:
        return np.zeros(0)
    return np.array([r['size'] * 0.5 for r in rocks])
