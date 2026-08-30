#!/usr/bin/env python3
"""
Chrono simulation setup
=======================

Constructs the plant a run is evaluated on: the PyChrono HMMWV vehicle, the SCM
deformable terrain and its soil parameters, and the visual trajectory markers.
Keeping this construction in one module ensures every entry point -- launcher,
benchmark, and data collector -- builds an identical plant.
"""

import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401
import numpy as np
import yaml

import pychrono as chrono
import pychrono.vehicle as veh

from param_consistency import TERRAIN_PRESETS, get_bumpiness_params
from terrain_gen import generate_heightmap_bmp


def _viz_type(name: str):
    """Return visualization enum value across Chrono Python API variants."""
    attr = f"VisualizationType_{name}"
    if hasattr(veh, attr):
        return getattr(veh, attr)
    if hasattr(chrono, attr):
        return getattr(chrono, attr)
    raise AttributeError(
        f"PyChrono visualization enum '{attr}' not found in pychrono or pychrono.vehicle"
    )


def _augment_chassis_collision(vehicle):
    """Body-conforming primitive boxes for the chassis envelope -- UNUSED.

    Written to close the straddled-rock gap: the stock HMMWV primitive
    envelope is one 2.0 x 1.0 x 0.2 m mid-body box whose band sits
    0.60--0.80 m above grade (chassis reference plane measured at 0.596 m),
    so a rock whose top stays below 0.60 m passes between the wheel tracks
    with zero native contact. On RIGID terrain these boxes work as intended
    (no self-collision, straddled rocks across the benchmark size range
    arrested, offset rocks untouched). On the paper's DEFORMABLE SCM terrain
    they are a regression: SCMTerrain ray-casts every collision shape inside
    the per-wheel active domains with no family filter, so under cornering
    roll and sinkage on soft soil the underbody and bumper boxes are treated
    as soil-contacting surfaces and plow the ground (clay/right_left neural
    arm: RMS crosstrack 5.46 m with them, 0.28 m without; 2026-08-27 A/B).
    Convex hulls are worse still (self-collision immobilises the vehicle).
    The function is kept, not called, so the geometry and the evidence stay
    with the code; the paper discloses the residual gap: with full-size rock
    collision bodies, rocks below ~0.92 m diameter (top under 0.60 m) can
    pass between the wheel tracks uncounted.
    """
    mat = chrono.ChContactMaterialSMC()
    mat.SetFriction(0.9)
    mat.SetYoungModulus(2e7)
    mat.SetRestitution(0.1)
    boxes = [
        ((0.00, 0.0, -0.121), (2.30, 1.40, 0.25)),  # underbody: 0.35-0.60 m
        ((1.95, 0.0, 0.004), (0.70, 0.90, 0.50)),   # front bumper: 0.35-0.85 m
        ((-1.95, 0.0, 0.004), (0.70, 1.40, 0.50)),  # rear bumper: 0.35-0.85 m
        ((0.00, 0.0, 0.354), (2.10, 2.04, 0.70)),   # cabin sides: 0.60-1.30 m
    ]
    chassis_body = vehicle.GetVehicle().GetChassisBody()
    for (cx, cy, cz), (lx, ly, lz) in boxes:
        shape = chrono.ChCollisionShapeBox(mat, lx, ly, lz)
        chassis_body.AddCollisionShape(
            shape, chrono.ChFramed(chrono.ChVector3d(cx, cy, cz), chrono.QUNIT))


def setup_chrono_vehicle(visualize=True, payload_mass=0.0, simple_powertrain=False):
    """Construct the PyChrono HMMWV vehicle.

    ``payload_mass`` (kg) adds an unmodelled cargo mass to the chassis body
    after initialization. The controller's bicycle model retains the nominal
    empty-vehicle mass, so a non-zero payload creates a persistent plant/model
    mismatch, which is the condition the robustness study measures the
    controller against. The chassis rotational inertia is scaled by the same
    mass ratio, so the added payload is dynamically consistent rather than a
    point mass at the centre of gravity.
    """

    # Set Chrono data path for mesh files
    chrono.SetChronoDataPath(chrono.GetChronoDataPath())
    # The vehicle data-path helper is named SetDataPath in some Chrono releases
    # and SetVehicleDataPath in others; accept either so this module works
    # across the versions the project builds against.
    _set_veh_data = getattr(veh, "SetVehicleDataPath", None) or veh.SetDataPath
    _set_veh_data(chrono.GetChronoDataPath() + 'vehicle/')

    # The vehicle is created before anything else because it constructs its own
    # ChSystem internally, which every later body is added to.
    vehicle = veh.HMMWV_Full()
    vehicle.SetContactMethod(chrono.ChContactMethod_SMC)
    vehicle.SetChassisFixed(False)
    # Primitive chassis collision, so the chassis makes contact with rigid
    # obstacles and a collision registers. Convex hulls self-collide with the
    # wheels and suspension and immobilise the vehicle (re-verified 2026-08-26:
    # with CollisionType_HULLS and no obstacle at all, the vehicle covers
    # 2.4 m in 8 s against 46 m under PRIMITIVES); a collision mesh costs too
    # much per step at this physics rate. The stock primitive set is a single
    # mid-body box (2.0 x 1.0 x 0.2 m -- upstream Chrono marks it
    # "TODO: a more appropriate contact shape"), which leaves the underbody,
    # bumpers, and body sides uncovered, so _augment_chassis_collision below
    # completes the envelope with body-conforming boxes after Initialize().
    vehicle.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
    vehicle.SetInitPosition(chrono.ChCoordsysd(
        chrono.ChVector3d(0, 0, 1.5),
        chrono.ChQuaterniond(1, 0, 0, 0)
    ))
    if simple_powertrain:
        # Near-direct drive: a linear torque engine (Te ~ throttle * T_max, no
        # engine speed map) with a continuously variable transmission (no
        # gear-shift discontinuities), so the map from throttle to wheel torque
        # is close to linear and independent of soil. Soil dependence remains
        # where it belongs, in the tire's longitudinal force Fx(kappa). This
        # makes throttle an effective torque command, which is the actuation
        # map the force-balance NMPC formulation assumes.
        vehicle.SetEngineType(veh.EngineModelType_SIMPLE)
        vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_CVT)
    else:
        vehicle.SetEngineType(veh.EngineModelType_SHAFTS)
        vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
    vehicle.SetDriveType(veh.DrivelineTypeWV_AWD)
    vehicle.SetTireType(veh.TireModelType_RIGID)
    vehicle.Initialize()
    # NOT applied: see _augment_chassis_collision. On deformable SCM terrain
    # the added boxes are ray-hit by the soil model under cornering roll and
    # sinkage and plow the ground (A/B on clay/right_left: RMS crosstrack
    # 5.46 m with the boxes, 0.28 m without), so the stock envelope stands and
    # the residual straddle gap is disclosed instead.

    # Get system FROM vehicle after initialization
    system = vehicle.GetSystem()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    if payload_mass and payload_mass > 0.0:
        chassis = vehicle.GetChassisBody()
        m0 = chassis.GetMass()
        m1 = m0 + float(payload_mass)
        ratio = m1 / m0
        chassis.SetMass(m1)
        inertia = chassis.GetInertiaXX()
        chassis.SetInertiaXX(chrono.ChVector3d(
            inertia.x * ratio, inertia.y * ratio, inertia.z * ratio))
        print(f"  [PAYLOAD] chassis mass {m0:.0f} -> {m1:.0f} kg "
              f"(+{payload_mass:.0f} kg unmodelled cargo)")
    
    if visualize:
        # Meshes where they carry visual detail, primitives elsewhere.
        vehicle.SetChassisVisualizationType(_viz_type("MESH"))
        vehicle.SetSuspensionVisualizationType(_viz_type("PRIMITIVES"))
        vehicle.SetSteeringVisualizationType(_viz_type("PRIMITIVES"))
        vehicle.SetWheelVisualizationType(_viz_type("MESH"))
        vehicle.SetTireVisualizationType(_viz_type("MESH"))
    else:
        vehicle.SetChassisVisualizationType(_viz_type("PRIMITIVES"))
        vehicle.SetWheelVisualizationType(_viz_type("PRIMITIVES"))
        vehicle.SetTireVisualizationType(_viz_type("PRIMITIVES"))
    
    return system, vehicle


def load_terrain_config(config_path):
    """
    Load terrain configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        dict with terrain parameters (numeric values converted to float)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required = ['Kphi', 'Kc', 'n', 'cohesion', 'friction_angle', 'janosi_shear']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required terrain parameter: {field}")
    
    # Convert all values to float (handles scientific notation strings like '2.1e6')
    numeric_fields = ['Kphi', 'Kc', 'n', 'cohesion', 'friction_angle', 'janosi_shear',
                      'elastic_stiffness', 'damping', 'length', 'width', 'mesh_resolution',
                      'bump_amplitude', 'bump_wavelength', 'bump_max_slope']
    for field in numeric_fields:
        if field in config:
            config[field] = float(config[field])
    
    # Integer fields
    int_fields = ['bump_octaves']
    for field in int_fields:
        if field in config:
            config[field] = int(config[field])
    
    return config


def setup_scm_terrain(system, vehicle=None, visualize=True, terrain_preset='sand',
                      terrain_config=None, mesh_resolution=None,
                      bumpiness=0, bump_seed=12345, texture=True,
                      spatial_spec=None, terrain_length=None, terrain_width=None):
    """Setup SCM deformable terrain

    Args:
        system: Chrono system
        vehicle: Chrono vehicle (for moving patch optimization)
        visualize: Enable visualization
        terrain_preset: Preset name ('sand', 'clay', 'dirt')
                       Ignored if terrain_config provided.
        terrain_config: Dict with terrain params from config file (overrides preset)
        mesh_resolution: Override mesh spacing (m). Default: 0.08 for headless, 0.05 for vis.
        bumpiness: Terrain bumpiness level 0-10 (0=flat, 10=extreme).
                   Maps to TOPOLOGY_LEVELS in param_consistency.
        bump_seed: Random seed for reproducibility
        texture: Apply dirt texture to terrain mesh
        spatial_spec: Optional SpatialTransitionSpec. When provided, the soil
                   varies with x (one preset, a short blend, then another) via
                   a per-location SCM callback. The uniform SetSoilParameters
                   call still runs as a fallback; pass
                   terrain_preset=spatial_spec.start_preset so the base soil
                   and the callback agree on the start of the patch.
    """
    import tempfile
    
    # Resolve bumpiness level to Perlin noise parameters
    bp = get_bumpiness_params(bumpiness, seed=bump_seed)
    bump_amplitude = bp['bump_amplitude']
    bump_wavelength = bp['bump_wavelength']
    bump_octaves = bp['bump_octaves']
    bump_max_slope = bp['bump_max_slope']
    if bumpiness > 0:
        print(f"  Bumpiness {bumpiness} ({bp['description']}): "
              f"amp={bump_amplitude:.2f}m, wl={bump_wavelength:.0f}m, "
              f"octaves={bump_octaves}, slope={bump_max_slope*100:.0f}%")
    
    terrain = veh.SCMTerrain(system)
    
    # Load params from config or use presets
    if terrain_config is not None:
        Kphi = terrain_config['Kphi']
        Kc = terrain_config['Kc']
        n = terrain_config['n']
        c = terrain_config['cohesion']
        phi = terrain_config['friction_angle']
        k = terrain_config['janosi_shear']
        elastic_stiffness = terrain_config.get('elastic_stiffness', 2e8)
        damping = terrain_config.get('damping', 3e4)
        terrain_name = terrain_config.get('description', 'Custom config')
        print(f"  Terrain: {terrain_name}")
        print(f"    Kphi={Kphi:.2e}, Kc={Kc:.0f}, n={n:.2f}")
        print(f"    cohesion={c:.0f}, friction={phi:.0f}°, janosi={k:.3f}")
    else:
        # Use preset
        if terrain_preset not in TERRAIN_PRESETS:
            raise ValueError(f"Unknown terrain preset: {terrain_preset}. "
                           f"Available: {list(TERRAIN_PRESETS.keys())}")
        preset = TERRAIN_PRESETS[terrain_preset]
        Kphi = preset['Kphi']
        Kc = preset['Kc']
        n = preset['n']
        c = preset['cohesion']
        phi = preset['friction_angle']
        k = preset['janosi_shear']
        elastic_stiffness = preset.get('elastic_stiffness', 2e8)
        damping = preset.get('damping', 3e4)
        print(f"  Terrain: {terrain_preset} - {preset.get('description', '')}")
    
    # SetSoilParameters takes the friction angle in degrees, not radians.
    terrain.SetSoilParameters(
        Kphi, Kc, n, c, phi, k, elastic_stiffness, damping
    )

    # Spatially varying soil: a per-location callback blends from one preset to
    # another along +x, so the transition is resolved in the SCM physics rather
    # than by switching a global parameter. The uniform SetSoilParameters call
    # above remains the fallback.
    if spatial_spec is not None:
        from spatial_terrain import TransitionSoilCallback
        soil_cb = TransitionSoilCallback(spatial_spec)
        terrain.RegisterSoilParametersCallback(soil_cb)
        # The SWIG director is owned by Python, so a live reference must be
        # held for the terrain's lifetime or the callback is collected mid-run.
        terrain._soil_param_callback = soil_cb
        print(f"  Spatial soil transition: {spatial_spec.start_preset} -> "
              f"{spatial_spec.end_preset} at x={spatial_spec.transition_x:.1f}m "
              f"(blend {spatial_spec.transition_width:.1f}m)")

    # Mesh resolution trades terrain fidelity against step cost: a coarser mesh
    # evaluates fewer SCM nodes per step. The default of 0.08 m is the paper
    # fidelity; interactive runs pass 0.12 m to hold real time.
    if mesh_resolution is not None:
        print(f"  Mesh: custom resolution {mesh_resolution}m")
        delta = mesh_resolution
    else:
        delta = 0.08
        print(f"  Mesh: {delta}m")
    
    # Terrain dimensions. Physics cost is insensitive to patch size because the
    # per-wheel active domain keeps computation local, but the camera ray-traces
    # the entire deformable mesh each frame and rebuilds its bounding-volume
    # hierarchy, so a smaller patch is the effective real-time lever in
    # camera-rendered multi-vehicle scenes. The defaults are 200 x 80 m.
    length = float(terrain_length) if terrain_length else 200.0
    width = float(terrain_width) if terrain_width else 80.0
    
    # Initialize terrain - flat or bumpy
    if bump_amplitude > 0:
        # Generate the Perlin-noise heightmap into a unique temporary file.
        # A fixed filename would let one parallel worker overwrite the bitmap
        # while Chrono reads it in another.
        heightmap_tmp = tempfile.NamedTemporaryFile(
            prefix="scm_heightmap_", suffix=".bmp", delete=False
        )
        heightmap_file = heightmap_tmp.name
        heightmap_tmp.close()
        # Image resolution: ~1 pixel per 0.5m for reasonable detail
        img_width = int(length * 2)
        img_height = int(width * 2)
        
        # Convert wavelength to frequency: 
        # wavelength is in meters, frequency is per-pixel
        # With 2 pixels per meter, freq = 1 / (wavelength * 2)
        pixel_frequency = 1.0 / (bump_wavelength * 2)
        
        generate_heightmap_bmp(heightmap_file, img_width, img_height,
                               amplitude=bump_amplitude, octaves=bump_octaves,
                               frequency=pixel_frequency, seed=bump_seed,
                               max_slope=bump_max_slope)
        # Initialize with heightmap: maps pixel values to height range
        terrain.Initialize(heightmap_file, length, width, 
                          0.0, bump_amplitude, delta)
        print(f"  Bumpy terrain: amplitude={bump_amplitude:.2f}m, "
              f"wavelength={bump_wavelength:.0f}m, max_slope={bump_max_slope*100:.0f}%")
    else:
        terrain.Initialize(length, width, delta)
    
    # Per-wheel active domain: a tighter box leaves fewer SCM nodes to evaluate
    # each step. Chrono Python builds expose this as either AddActiveDomain or
    # AddMovingPatch, so both names are probed.
    if vehicle is not None:
        add_patch = None
        if hasattr(terrain, "AddActiveDomain"):
            add_patch = terrain.AddActiveDomain
        elif hasattr(terrain, "AddMovingPatch"):
            add_patch = terrain.AddMovingPatch
        else:
            print("  WARNING: SCMTerrain moving patch API not found; running without moving patches")

        for ax in vehicle.GetVehicle().GetAxles():
            if add_patch is None:
                break
            add_patch(ax.m_wheels[0].GetSpindle(),
                      chrono.ChVector3d(0, 0, 0),
                      chrono.ChVector3d(1, 0.5, 1))
            add_patch(ax.m_wheels[1].GetSpindle(),
                      chrono.ChVector3d(0, 0, 0),
                      chrono.ChVector3d(1, 0.5, 1))
    
    if visualize:
        terrain.SetPlotType(veh.SCMTerrain.PLOT_SINKAGE, 0, 0.1)
        if texture:
            _veh_data_file = getattr(veh, "GetVehicleDataFile", None) or veh.GetDataFile
            terrain.SetTexture(_veh_data_file("terrain/textures/dirt.jpg"), 10, 10)
    
    print(f"  SCM mesh: {delta}m, terrain: {length}x{width}m"
          + (", moving patch ON" if vehicle else ""))
    
    return terrain, {'Kphi': Kphi, 'Kc': Kc, 'n': n, 'c': c, 'phi': phi, 'k': k}


def add_trajectory_markers(system, path_type='lane_change', marker_z=None,
                           lead_in=0.0, **_kwargs):
    """
    Add visual sphere markers along the reference path loaded from CSV.

    Loads waypoints from ``data/paths/<path_type>.csv`` and places markers at
    regular arc-length intervals along the path. The markers are visual only
    and are never read by the controller or by any metric.

    Args:
        system: Chrono system to add markers to.
        path_type: Name of the CSV file, without extension, in ``data/paths/``.
        marker_z: Z height for markers (default 0.15).
        lead_in: Optional straight lead-in distance prepended to the path.
    """
    from pathlib import Path as _P

    marker_spacing = 4.0   # arc-length metres between markers
    marker_radius = 0.15
    marker_height = marker_z if marker_z is not None else 0.15

    paths_dir = _P(__file__).resolve().parents[2] / "data" / "paths"
    csv_path = paths_dir / f"{path_type}.csv"
    if not csv_path.exists():
        print(f"  WARNING: path CSV not found: {csv_path}, skipping markers")
        return

    data = np.loadtxt(str(csv_path), delimiter=',', skiprows=1)
    if data.shape[1] == 2:
        x_all, y_all = data[:, 0], data[:, 1]
    else:
        x_all, y_all = data[:, 1], data[:, 2]

    # Optionally prepend lead-in straight section
    if lead_in > 0:
        ds = 0.25
        n_lead = max(1, int(lead_in / ds))
        x_lead = np.linspace(0, lead_in, n_lead, endpoint=False)
        y_lead = np.zeros(n_lead)
        x_all = np.concatenate([x_lead, x_all + lead_in])
        y_all = np.concatenate([y_lead, y_all])

    # Compute cumulative arc length
    dx = np.diff(x_all)
    dy = np.diff(y_all)
    ds_arr = np.sqrt(dx ** 2 + dy ** 2)
    s_cum = np.concatenate([[0.0], np.cumsum(ds_arr)])
    s_total = s_cum[-1]

    n_markers = int(s_total / marker_spacing) + 1
    print(f"  Adding {n_markers} trajectory markers for {path_type} ({s_total:.0f}m arc)...")

    # Subsample at regular arc-length intervals
    s_targets = np.linspace(0, s_total, n_markers)

    for i, s_t in enumerate(s_targets):
        idx = int(np.searchsorted(s_cum, s_t, side='right')) - 1
        idx = max(0, min(idx, len(x_all) - 1))
        x = float(x_all[idx])
        y = float(y_all[idx])

        marker = chrono.ChBodyEasySphere(marker_radius, 1000, True, False)
        marker.SetPos(chrono.ChVector3d(x, y, marker_height))
        marker.SetFixed(True)

        # Gradient color: green → yellow → blue along path progress
        t = i / max(n_markers - 1, 1)
        color = chrono.ChColor(0.2 + 0.7 * t, 0.8 - 0.5 * t, 0.2 + 0.6 * t)
        marker.GetVisualShape(0).SetColor(color)
        system.Add(marker)
