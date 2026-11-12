# Safety filters

A safety filter sits between the command source (the tracking NMPC or a human
operator) and the plant. It passes the incoming command unchanged whenever doing
so is safe, and changes it only as much as needed to keep the vehicle clear of
obstacles. So the same module serves autonomous runs and shared control, where
keeping the operator's intent is itself a requirement.

Two filters implement the interface, `dob_cbf` and `mpsf`. Both are selected at
launch with `--safety-filter --safety-flavor <name>` and built by
`make_safety_filter` in `__init__.py`, the only entry point the plant node uses.
Running with no filter is the third configuration, the reference for either
filter's intervention rate and harm prevented.

Neither filter takes a hand-set number. Every number comes from the plant's
specification, the rig surrogate, or bounds identified on plant manoeuvres kept
apart from every evaluation scenario ([Parameters](#parameters)). What remains
are the free design parameters the theory leaves open, which are reported with a
check that the results do not hinge on them.

The boundary is in-process, inside `chrono_sim_node.py`. Every 0.1 s the plant
takes the latest command (after the configured channel latency), calls the
filter with the measured state and the obstacles within `--filter-sense-range`,
and passes the filtered command to the Chrono driver. Between calls it holds the
last filtered command when that command was an intervention. Both filters act on
the measured state on the vehicle, so neither projects the state over a teleop
delay.

| Module | Contents |
|---|---|
| `vehicle_spec.py` | The plant's numbers: body outline, actuator rates, brakes, the quasi-static powertrains |
| `footprint.py` | The body rectangle and exact distances to points, segments and other rectangles |
| `vehicle_model.py` | The vehicle model both filters predict with (numpy and CasADi) |
| `dob_cbf.py` | The DOB-CBF |
| `mpsf.py` | The MPSF |
| `grade_preview.py` | What the filters know of the grade ahead |
| `common.py` | The result type and the surrogate and soil loaders |
| `ident/` | The identified bounds |

## The vehicle model (`vehicle_model.py`)

A dynamic single-track model with state [X, Y, psi, u, v, r, p], p the applied
pedal in [-1, 1] (throttle positive, brake negative). The inputs are the
road-wheel angle and the pedal rate, which the plant limits to 8 per second.

* Lateral: the rig surrogate's lateral forces at each axle's slip angle, normal
  load (longitudinal and lateral transfer, as in the NMPC) and slip ratio. A
  locked wheel gives almost no cornering force. Below 3 m/s the lateral states
  relax toward the kinematic ones (the surrogate's training speeds start near
  2 m/s).
* Longitudinal: drawbar pull = tractive effort - motion resistance (Wong,
  Theory of Ground Vehicles). The surrogate's net force at zero slip is the
  resistance, and a wheel torque adds torque / radius until the curve's end: the
  locked-wheel force braking, the top of the drive curve driving. Drive torque is
  the plant powertrain's quasi-static wheel torque at the current speed, brake
  torque the pedal times 4000 N m per wheel. A net deceleration fades out at a
  standstill, so the vehicle stays stopped on a slope its brakes can hold.
* Gravity along the heading comes from the grade preview, when there is one.

## `dob_cbf`: barrier QP with a disturbance observer

A minimum-deviation QP over the road-wheel angle and the pedal, every period.

* Model: a kinematic bicycle whose speed changes at the vehicle model's rate
  a(p, v) plus a disturbance d. The observer estimates d from the measured speed.
  Its error bound sigma is identified, and tightens every row in the worst
  direction (Alan et al. 2023).
* Backup: straight, the pedal ramped to full brake at its rate limit and held to
  a stop. On the model it travels S_model(v, p). The identified bounds put the
  plant's front edge within s0 + s1 S_model of its start and every corner within
  l0 + l1 S_model of the sides, so the backup sweeps the body rectangle
  lengthened by that stop and widened by that drift.
* Barrier, per obstacle: the distance from the obstacle's set to the swept
  rectangle. A rock's set is its disc. Another vehicle's is its own rectangle
  (the same HMMWV), stretched along its heading by what it can cover during the
  backup's stop, from its speed plus the measurement error, accelerating at the
  soil's drive capability. Under the backup the stop point does not move, so a
  static obstacle's barrier does not decrease and the QP is feasible from h >= 0
  (backup control barrier functions, Gurriet et al. 2018, Chen et al. 2021).
* Sensing edge: the swept rectangle's far corners stay inside the sensing range
  less the largest obstacle the scenario holds.
* Condition: the discrete-time one of a filter whose input is held over its
  period (Agrawal and Sreenath 2017), h at the end of the period >= (1 - gamma
  dt) h now. h is concave in the pedal, so two secants through the input box's
  ends and the held pedal, as two rows, lie below it. The wheel angle enters
  through the end heading, linearly.
* Sampled-data margin: 1/2 L dt^2 off h, with L the largest |h''| over a period
  at the corners of the input box.
* Input box: the steering and pedal rate limits over one period, and the
  steering the soil's peak lateral acceleration allows at the current speed.
* Infeasible: the backup's first period (straight, the pedal ramped down).

## `mpsf`: model predictive safety filter

The standard predictive safety filter (Wabersich and Zeilinger, Automatica 2021)
on the vehicle model, solved with acados.

* Cost: deviation from the operator's command on the first input (the pedal it
  reaches by the end of the period against the operator's pedal), and a 1e-4
  regularization on the later inputs.
* Constraints, at every node: the body rectangle clear of every selected
  obstacle's set by the tube, and inside the sensed region less the tube. The
  tube is c0 + c1 s + eps S (s the path length, S the integral over distance of
  the accumulated heading change), identified sideways against the plant. The
  OCP carries s, the accumulated turn and S as states.
* Terminal set: after the planned stages (20 x 0.1 s by default) come 22 stages
  of 0.2 s with the input fixed to straight braking with the pedal ramped down,
  and the vehicle must be below 0.3 m/s at the end. So safety does not depend on
  the horizon length.
* Check: the plan and the braking tail roll out on the model with the OCP's
  integrator, straight braking continues until the model stops (it must), and
  the prediction is extended past the stop by the identified overshoot of its
  braking part. Every pose is then checked without slack against every obstacle,
  widened by the gap to the next pose (the travel of the rectangle's farthest
  point and the sagitta of its arc). If the plan passes, its first input is
  executed and the plan kept. Otherwise the next input of the last kept plan is
  executed, which ends in straight full braking.
* Other vehicles: their HMMWV rectangle, stretched along the heading between
  where it is when braking as hard as the soil allows and where it is when
  speeding up at the soil's drive capability, from the measured speed give or
  take the error bound. Between two poses the set lies in one rectangle, from its
  slow position at the first to its fast position at the second. The OCP sees a
  vehicle as the capsule round its centreline, the check as the rectangle.
* Obstacle selection: the OCP holds `--mpsf-n-obstacles` obstacles (3), those in
  a corridor along the heading first, soonest reached, then the nearest. The
  check tests every obstacle. The corridor's reach (1.0 + 1.6 m plus c0, and
  1.5 m to the sides) is the published filter's footprint circle. It only orders
  which obstacles the planner sees, never what the check accepts.

`--traffic-speed-noise s` adds a Gaussian error of standard deviation s to each
vehicle's speed as both filters receive it, and sets the speed error bound to
3 s.

## Grade preview (`--filter-grade-preview`)

`grade_preview.py` reads what the simulator puts in the vehicle state:
`grade_now`, the sine of the chassis's nose-down pitch (an IMU), or
`terrain_ahead`, ground heights every 0.5 m along the heading to 40 m (from SCM
`GetHeight`, an oracle map, or as a roof lidar 2.1 m up sees them, with ground
behind a crest unknown). A downgrade is measured over one wheelbase, and unknown
ground takes the steepest admissible downgrade (15%).

* DOB-CBF: the backup's stop runs on the steepest previewed downgrade within it
  (a fixed point over the stop's length).
* MPSF: the vehicle model carries gravity along the heading. The OCP takes it at
  each node's warm-start position, the check at every position of the plan.

Without a preview both filters assume flat ground.

## Soil belief (`--shield-init-terrain-belief`)

Both filters run the vehicle model on a soil belief. `match` is the true soil, a
soil name forces that soil, and `none` keeps the nominal soil (dry sand, the
firmest in the study), which makes the filter soil-blind. Left empty, the filter
starts on the nominal soil and takes the controller's estimate as it arrives with
the commands, so in a run without a controller it stays soil-blind.

## Teleop link

Over a teleop link (a delay above zero) the filter records each command's
arrival on simulation time, and brakes straight once no command has arrived for
`--stale-cmd-timeout` seconds (2 s). Locally there is no stale-command brake.

## Parameters

**Specification** (`vehicle_spec.py`, read from the Chrono 9.0.0 HMMWV model)

| Number | Source |
|---|---|
| Footprint: the rectangle x in [-2.52, 2.16] m, y in [-1.28, 1.28] m about the centre of mass (`footprint.py`) | The collision mesh's outline (`hmmwv_chassis_col.obj`, 4.67 x 2.56 m). The chassis centre of mass sits 0.056 m ahead of the mesh origin. |
| Other vehicles: the same rectangle about their reported chassis frame | Traffic is HMMWV_Reduced, the same chassis |
| Steering 16 per s of full lock, pedal 8 per s, 0.49 rad at full lock | The plant node's actuator limits |
| 4000 N m per wheel at full brake, tyre radius 0.467 m | HMMWV_BrakeSimple / BrakeShafts, HMMWV_RigidTire |
| Drive torque at the wheels | The plant's powertrain, quasi-static: `shafts` (engine torque and loss maps, torque converter, gears 0.2/0.4/0.8, up-shift at 2500 rpm of the turbine), `simple_map` (330 N m, 110 kW, six gears) or `simple_cvt`. The simulator reads which one it built. |

**Rig surrogate** (`--nn-model`, default `tire_force_static`): the tire forces
of the vehicle model, the soil's motion resistance, the locked-wheel braking, the
drive capability and the peak lateral acceleration, all on the believed soil.

**Identified** (`--safety-ident`, default `ident/plant_bounds_chrono9_box.json`,
per powertrain, over the three soils). From plant manoeuvres shaped like a
certificate: a manoeuvre for 1 or 2 s from a cruise of 3 to 11 m/s or from rest,
then straight full braking to a stop (`benchmarking/safety_ident/`). Half the
runs, drawn by a fixed seed within each powertrain, soil and manoeuvre family,
identify. The rest check. Each bound is the tightest of its form that covers
every identification point (a linear program), measured on the rectangle's four
corners.

| Bound | Full powertrain | Simple powertrain | Held out |
|---|---|---|---|
| Stop: front edge past its start <= s0 + s1 S_model | 0.59 + 1.14 S_model | 0.56 + 1.08 S_model | full: 2 of 198 runs above. Simple: 9 of 212, worst by 0.16 m |
| Drift: corners past the sides <= l0 + l1 S_model | 0.71 + 0.073 S_model | 0.65 + 0.063 S_model | full: 0 of 198. Simple: 1 of 212 |
| MPSF tube (sideways), c0 + c1 s + eps S | 0.57 + 0.030 s + 0.35 S | 0.93 + 0.030 s + 0.17 S | full: 8 of 6,883 points outside, worst by 0.14 m. Simple: 14 of 7,772, worst by 0.27 m |
| Observer error (m/s^2) | 6.7 | 5.0 | full: 1 of 6,977 points above. Simple: 1 of 7,891 |

The tube is measured against the MPSF's extended prediction: for each plant
pose, the largest corner-to-corner distance to the one predicted pose that
matches it best. So the whole plant rectangle lies within the tube of that
predicted rectangle. Bounds for `simple_cvt` (newer Chrono) are not identified
yet. `ident/plant_bounds_chrono9.json` is an earlier version (three footprint
discs, one round tube, cruise speeds up to 7 m/s), kept for the record.

**From the scenario**: the sensing range (`--filter-sense-range`, 30 m), the
largest obstacle the scenario holds (the sensing edge leaves room for one just
past the range, and another vehicle reaches 2.77 m from its reported point), and the
speed measurement's error bound (`--traffic-speed-noise`).

**Free design parameters** (reported, with a check that the results do not hinge
on them): the DOB-CBF's decay rate (`--cbf-gamma`, 3 per s) and observer gain
(k dt = 0.5, the value the identification ran it with), and the MPSF's horizon
(`--mpsf-time-grid`, 20 x 0.1 s), braking tail (22 x 0.2 s), cost weights, SQP
iterations (`--mpsf-max-iter`, 20), obstacle capacity (`--mpsf-n-obstacles`, 3)
and OCP margin (0.1 m, which only moves how often the check rejects a plan).
Numerical smoothing: the braking fade at a standstill (0.25 m/s) and the
low-speed kinematic blend (1.5 to 3 m/s).

**What the identification showed** (journal workspace, `experiments/derived/`).
The rig surrogate misses the full vehicle's motion resistance by 0.7 to
0.9 m/s^2 (coasting on clay: 1.25 m/s^2 in the plant, 0.37 at zero slip in the
surrogate), and its spin-axis moment (which the rig records) puts the
free-rolling slip at -0.24 on clay where the plant rolls at -0.03. The observer
error comes from events the model does not have (wheel lock onset, gear shifts
with their 1 s latency, starts from rest), not from a lag: a first-order lag of
0 to 0.3 s leaves it at 4 to 7 m/s^2. The largest short-range sideways errors
come from three regimes the model gets wrong: partial braking while steering on
clay (the model locks the wheels and barely turns, the plant turns), full-lock
starts from rest, and partial braking on sand (the model brakes much harder).

## Tire model supervision

The rig surrogate is supervised only by the controlled single-tire Chrono SCM
rig. No filter consumes Chrono terrain truth, tire-force truth, or any other
quantity unavailable to a physical vehicle.

## Usage

```python
import json
from safety import make_safety_filter

ident = json.load(open("simulation/safety/ident/plant_bounds_chrono9_box.json"))
sf = make_safety_filter("dob_cbf", vehicle_params, ident=ident, powertrain="shafts",
                        odd_obstacle_radius=1.4, sense_range=30.0)
sf.update_terrain(terrain)          # the soil belief, in the surrogate's convention

result = sf.filter(desired_steering, desired_throttle, desired_brake,
                   vehicle_state, obstacles)
```

`vehicle_state` holds `x, y, psi, u` (and `v, omega` for the MPSF), and
optionally `grade_now` or `terrain_ahead`. Each obstacle is `(x, y, radius,
False)` for a rock or `(x, y, radius, True, vx, vy, heading)` for another
vehicle.

```bash
# DOB-CBF
python simulation/runtime/launch_decoupled.py --manual --safety-filter

# MPSF on the true soil
python simulation/runtime/launch_decoupled.py --manual --safety-filter \
    --safety-flavor mpsf --shield-init-terrain-belief match

# Operator driving through a rock field with the filter engaged
python simulation/runtime/launch_decoupled.py --manual --rocks 20 --safety-filter
```

The filters as published in the ACMD 2026 paper, and how they became these, are
in [`HISTORY.md`](HISTORY.md).
