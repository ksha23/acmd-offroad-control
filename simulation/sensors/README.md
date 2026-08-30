# Sensors

Places the obstacle population a run must avoid, and exposes its geometry to
the controller, the safety filter, and the collision logger.

## `obstacles.py` — rock placement

Adds fixed rigid bodies to the Chrono system as the static obstacle population
for the collision-avoidance study.

- Configurable count, size range, and placement zone.
- Ellipsoid collision shapes, cheap enough to evaluate alongside the deformable
  SCM terrain, with sphere visual shapes.
- Rocks are partially buried, so the exposed profile matches an embedded
  boulder rather than one resting on the surface.
- Exclusion zones keep the vehicle spawn and any other protected region clear.
- Placement is drawn from a seeded generator, so a field is reproducible from
  its seed alone.
- Optional blue-noise minimum spacing turns a random scatter into a field that
  is uniformly dense: a threadable gap always exists between rocks, no
  impassable clump forms, and there is no free bypass around the field.
- Optional centreline thinning reduces rock density along the route a lead
  vehicle takes without clearing it, so a line can be chosen through the field
  while rocks still intrude on it.

Contact between an ego body and a rock body is collision truth, recorded by
`simulation/shared/collision_detector.py`. The radii returned by
`get_rock_radii` feed the obstacle geometry used by the controller and the
safety filter; they are planning quantities, not collision truth.

```python
from sensors.obstacles import add_rock_obstacles, get_rock_positions, get_rock_radii

rocks = add_rock_obstacles(system, num_rocks=20,
                           zone_x=(-15, 50), zone_y=(-10, 10),
                           size_range=(0.5, 3.0), seed=42)
positions = get_rock_positions(rocks)   # (N, 3) world coordinates
radii = get_rock_radii(rocks)           # (N,) effective radii in metres
```

## Command-line use

```bash
# Twenty rocks along an autonomous run
python simulation/runtime/launch_decoupled.py --model nn --rocks 20

# A threadable blue-noise field with the spawn kept clear
python simulation/runtime/launch_decoupled.py --model nn --rocks 40 \
    --rock-min-spacing 6 --rock-spawn-clear 12

# Operator driving through a rock field with the safety filter engaged
python simulation/runtime/launch_decoupled.py --manual --rocks 15 --safety-filter
```
