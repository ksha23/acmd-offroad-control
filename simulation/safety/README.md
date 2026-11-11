# Safety filters

A safety filter sits between the command source — the tracking NMPC or a human
operator — and the plant. It admits the incoming command unchanged whenever
doing so is safe, and alters it only as much as is required to keep the vehicle
clear of obstacles. This makes the same module usable for autonomous runs and
for shared control, where preserving operator intent is itself a requirement.

Two filters implement the interface. Both are selected at launch with
`--safety-filter --safety-flavor <name>`, and both are constructed through
`make_safety_filter` in `__init__.py`, which is the only entry point the plant
node uses. Running with no filter is the third supported configuration and is
the reference against which either filter's intervention rate and
harm-prevention are measured.

The boundary is in-process and lives inside `chrono_sim_node.py`: each control
period the plant gathers the most recent command, applies the configured
channel latency, calls the selected filter, and passes the filtered command to
the Chrono driver. The filter is evaluated at 10 Hz, matching the NMPC rate;
between evaluations the plant holds the last filtered command whenever that
command was an intervention.

## `dob_cbf` — reactive barrier QP

`dob_cbf.py` solves one quadratic program per call that minimises deviation
from the incoming command subject to a control barrier function constraint per
obstacle. Because the objective is deviation, the command passes through
untouched whenever the barrier constraints are inactive, and the vehicle is
steered or braked only by the amount the constraints demand.

Three elements shape its behaviour:

1. **Heading-aligned ellipsoidal barrier.** The barrier is
   `h(x) = (p − p_obs)ᵀ P (p − p_obs) − r_safe²` with
   `P = R(ψ)ᵀ diag(w_long, w_lat) R(ψ)`. Setting `w_long < w_lat` elongates the
   safe set along the heading, so lateral escape is nearer in barrier space and
   the QP prefers steering around a static obstacle over stopping in front of
   it. A forward shift of the barrier centre (`forward_bias`) concentrates the
   grip-limited margin ahead of the vehicle, where it is needed. Obstacles are
   tagged as static or vehicle; for a vehicle the longitudinal and lateral
   weights are equalised, because stopping is an equally valid response to
   another vehicle.
2. **Disturbance observer.** A first-order observer estimates the lumped
   unmodelled longitudinal acceleration — compaction drag, sinkage resistance,
   grade — and feeds it into the barrier's second derivative, so the constraint
   accounts for deceleration the nominal model does not predict. When a neural
   tire model is supplied it provides the nominal acceleration, leaving the
   observer to estimate only the genuine residual.
3. **Delay compensation.** A discrete predictor with a command FIFO and
   derivative-plus-proportional feedback estimates the state the vehicle has
   actually reached across the actuation delay, and the obstacle buffer is
   inflated by the distance travelled during the command round trip. A command
   older than `stale_cmd_timeout` triggers a brake, so a dropped teleoperation
   link fails closed.

Terrain awareness is optional and enters through `update_terrain`. Grip falls
on softer soil, so the filter's acceleration and braking authority — and hence
its stopping distance — are scaled by the live soil estimate. Supplying the
estimator's posterior standard deviation together with a risk factor evaluates
that authority at a pessimistic soil quantile, so an uncertain estimate widens
the stopping buffer rather than assuming the posterior mean.

With `--no-safety-nn` the filter runs without the neural tire model, falling
back to kinematic steering authority and fixed longitudinal acceleration
limits. This is the ablation that isolates the tire model's contribution.

### The shared longitudinal model

The observer and the barrier describe the same motion with one model,

    a_x(alpha) = f_drag + g_thrust * alpha + hdv0

and each uses the half that belongs to it. The observer subtracts the whole
nominal at the commanded input, so `hdv0` is a genuine residual; the barrier
adds the control-independent part back as its autonomous term and keeps
`g_thrust` as the coefficient on the throttle it is choosing. Evaluated at the
current command the model reproduces the measured acceleration by
construction, which is the property that makes the observer's residual and the
barrier's disturbance the same quantity.

Each term is chosen so the *direction* of its inevitable error is safe, and
each was previously wrong in a way with a measured consequence:

* **`f_drag` is tapered to zero at rest and signed by the motion.** The
  surrogate is queried at `max(u, 0.5)` m/s to keep the slip-angle arctangents
  well-conditioned, so at rest it reports the resistance of a vehicle rolling
  at the clamp. A stationary vehicle has none — its tires are held by static
  friction — and feeding that resistance to the observer makes the difference
  against a measured zero acceleration read as an unexplained *forward*
  disturbance, which the barrier answers with brake, sustaining the standstill
  that produced it.
* **The observer's brake nominal is the unscaled envelope.** The drive-slip
  secant understates true braking, and the shortfall would be booked as free
  deceleration the barrier then counts on with the brakes released — the
  permissive direction, in the one regime where something is in front of the
  vehicle. The envelope upper-bounds any braking the plant can produce, so the
  residual biases toward a phantom forward push instead, which tightens the
  barrier. Under throttle the drive-slip secant is used and its error was
  measured conservative.
* **The barrier's authority coefficient is a function of state alone** (always
  the drive-slip secant, clipped to `[0.3, max_accel]`). Deriving it from the
  pedal made the safe set change shape with the command and step
  discontinuously across a touch of the brake; the drive secant under-credits
  braking, so the QP brakes harder than strictly needed when it intervenes.
* **The terrain speed cap bounds throttle from above.** Its row was
  sign-inverted — it capped braking and, past the limit, demanded throttle —
  and the bound is floored at full brake so the row can never go infeasible
  and trip the emergency fallback that discards the obstacle constraints.
* **The closing rate is the measured speed.** Reusing the `max(u, 0.1)` guard
  against division by zero as a kinematic approach rate models a stopped
  vehicle as still closing, which alone keeps the barrier constraint active at
  a standstill.

`tests/simulation/test_dob_cbf_longitudinal_model.py` holds these as
contracts; it exercises the braking branch, reads the autonomous term back out
of the logged constraint rows, and probes the speed cap just below the limit,
where a sign reversion is distinguishable from the infeasible-row emergency
brake. Deliberately deferred, documented in the development repository's
design note `DOB_CBF_STANDSTILL_DEADLOCK.md`: the slip-to-pedal calibration behind
the secant, the `0.3` authority floor, the observer's deadbeat tuning at the
production control period, and the lateral channel omitted from
`h_ddot_auto`.

### Remaining limit: the standoff does not shrink with speed

The barrier is a function of position alone, so its standoff is as wide at rest
as at speed — roughly `r_safe/√w_long + forward_bias`, about 7.6 m ahead —
while its preferred escape is steering, whose authority `v/L · sec²β · δ_max`
vanishes as the vehicle stops. A vehicle halted on that boundary may not
accelerate and cannot steer. This is a property of the barrier's form rather
than an implementation fault, and repairing it means making the standoff
speed-dependent. Where a study needs sustained motion, `mpsf` admits the
command until no safe trajectory remains and does not exhibit it.
`benchmarking/audit_stalled_runs.py` reports how often any result set ends
stopped, and why.

### Textbook DOB-CBF (`--cbf-theory`)

`dob_cbf_theory.py` implements the same filter with only the parts the theory
specifies. The published filter above stays the default, unchanged.

* Model: kinematic bicycle with the steering angle as a state, so the inputs
  are the road-wheel rate and the longitudinal acceleration, and the rate
  limits form the input box. The pedal-to-acceleration map and the rolling
  resistance come from the surrogate (braking as the torque balance of
  `surrogate_dynamics.py`, traction capped by its drive force).
* Barriers: the paper's heading-aligned ellipse as a high-order CBF with real
  roots (p1 = p2 = sqrt(cbf_alpha)) and exact derivatives, the braking-distance
  barrier of Ames et al. (TAC 2017) on lane obstacles with the margined
  surrogate deceleration, and the same barrier at the sensing edge. Traffic
  enters the derivatives at its measured velocity.
* Observer: the disturbance observer of Alan et al. (L-CSS 2023) on the speed
  channel, gain k with k dt = 0.5, whose error bound b/k tightens every row.
* QP: weighted deviation from the command (the published threat weights) with
  hard rows, solved exactly in two variables. Infeasible means full braking
  with the steering held.
* Teleop: the stale-command brake is a link watchdog, and a round-trip delay is
  handled by predicting the state over it on the same model.

What it drops from the published filter: the slack, the reactive-steering
backup, the emergency predictor, the rear-approach guard, the speed-dependent
radius inflation, the steering lag and slew applied after the QP, the terrain
speed cap, and the dropping of receding barrier rows. With the published
ellipse (w_long 0.06, bias 3.0) the ellipse is a long standoff, and on clay the
QP turns infeasible once the vehicle has braked inside it, which brakes and
holds: the published standoff behaviour, now explicit instead of absorbed by
slack.

## `mpsf` — predictive filter

`mpsf.py` solves a short-horizon optimal control problem with acados that keeps
the entire predicted trajectory clear of obstacles while deviating as little as
possible from the incoming command:

```
min_U  Σ_k ‖u_k − u_cmd‖²_W
s.t.   kinematic bicycle with grip-limited longitudinal dynamics,
       (X_k − o_j)² + (Y_k − o_j)² ≥ safe_r_j²   for all k, j   (slacked),
       |a_lat| ≤ lat_accel_max                                  (hard),
       |steer|, |alpha| ≤ 1.
```

Safety is a trajectory-level feasibility question here rather than a pointwise
inequality, so the command is admitted until no safe future exists. The
obstacle rows are slacked under a large penalty, which keeps the problem
feasible. The lateral-acceleration row is hard: braking is always a feasible
way to clear an obstacle, so the vehicle is never required to corner harder
than the soil supports.

Safety of the executed command does not rest on the slack penalty's gradient.
Adversarial review (2026-08-26) found two defects in the original
implementation: acados leaves stage 0 unconstrained unless a stage-0
constraint set is declared, so the lateral row never bound the input the
vehicle executes; and in unavoidable geometry the slack gradient could floor
the *throttle* toward the obstacle rather than brake. Four guards now sit
between the solver and the vehicle:

* the lateral rows are imposed at stage 0, where the executed input lives
  (steering zero always satisfies them, so they are hard without risking
  infeasibility);
* an accepted solution whose own predicted trajectory still enters an
  obstacle's buffer-stripped envelope — the unavoidable-collision case — is
  replaced by maximal braking with the solver's evasive steering retained;
* the executed steering is clamped to the soil's cornering authority on every
  exit path, including fail-closed ones;
* a per-stage speed funnel, descending from the current speed at the
  achievable braking rate, keeps the vehicle at speeds it can null within the
  horizon it actually constrains — beyond the horizon, avoidance is deferred
  to stages the OCP does not see.

A stale teleop command stream triggers braking, matching `dob_cbf`, and the
`--shield-terrain-nn` / `--no-safety-nn` switches now govern this filter's
tire-surrogate queries exactly as they govern the barrier filter's (the
original implementation ignored both and conditioned on the live estimate
unconditionally). `tests/simulation/test_mpsf_safety.py` pins each guard;
the smoke tier runs it under the environment that has acados.

The filter solves in its own normalised command space, `[steer, alpha]`, using
the same normalised-command to physical-actuation bridge as `dob_cbf`. It does
not reuse the tracking NMPC's state or control space, so the two remain
independent. Both braking and cornering authority are supplied as solver
parameters derived from the tire surrogate at the current soil, which is what
makes the avoidance mode terrain-dependent: on low-grip soil the achievable
lateral acceleration is small, the swerve is infeasible, and the slacked
obstacle rows drive braking instead; on firm soil the swerve is permitted and,
being cheaper than braking, is preferred.

Because the OCP is sized for a fixed obstacle count, the filter selects the
most threatening obstacles at each step — those ahead and inside the
heading-aligned corridor, ranked by how soon they are reached — and fills any
remaining slots with the nearest of the rest. Clearance is reported over the
whole obstacle field, not the selected subset.

### Terminal-set formulation (`--mpsf-terminal-set`)

The published formulation above stops the vehicle through the speed funnel,
which ties the brake point to the horizon length (a 4 s horizon brakes about
twice as far out as a 2 s one). The terminal-set formulation removes the
funnel. Every executed command is certified by an explicit rollout: one filter
period of the command, then straight braking to a stop at the margined
surrogate deceleration (`--mpsf-brake-margin`, default 0.85), with the plant's
pedal rate limit modelled. The rollout must keep the planning radius from every
obstacle in the field and stay inside the sensed region (`--filter-sense-range`
minus an unseen obstacle's footprint). The rollout runs to the stop, so safety
does not depend on the horizon. The OCP, with its deviation cost on the first
input and a terminal set from which straight braking stops clear, only proposes
corrections when the command fails. Below 0.5 m/s an uncertified command is held
at full straight braking. With the kinematic model a swerve is executed only if
straight braking after it also certifies.

### Predictive safety filter on the surrogate model (`--mpsf-dynamic`)

The standard predictive safety filter (Wabersich and Zeilinger) on the tracking
NMPC's own model, `surrogate_dynamics.py`: a dynamic bicycle whose lateral and
yaw accelerations come from the rig surrogate's tire forces at each axle's slip
angle, load (longitudinal and lateral transfer) and slip ratio. Braking is a
torque balance on the surrogate's longitudinal curve. The pedal asks each wheel
for pedal x 4000 N m / 0.467 m, and past the locked-wheel force the wheel locks
and loses almost all cornering force (on clay above a pedal of about 0.22, as
the plant does). The applied longitudinal acceleration follows the pedal with a
0.1 s lag, which stands for the plant's pedal rate limit and brake build-up.

* Cost: deviation from the operator's command on the first input only, the
  textbook filter (`--mpsf-psf-cost first`, the default). The later inputs carry
  a 1e-4 regularization. `horizon` charges deviation on every planned stage, as
  the published MPSF does.
* Constraints: three footprint discs along the heading (radius 1.4 m, 1.3 m
  behind and 1.2 m ahead of the CG) keep the planning radius plus an error tube
  from every selected obstacle, at every place a vehicle among them can be by
  then (below), and the path stays inside the sensed region less the tube.
* Terminal set: the planned stages are followed by 22 stages of 0.2 s with the
  input fixed to straight full braking, and the vehicle must be below 0.3 m/s at
  the end, clear and inside the sensed region. This replaces the speed funnel,
  so safety does not depend on the horizon length.
* Execution: the plan is rolled out on the same model and every constraint is
  checked without slack against the whole obstacle field. If it holds, the first
  input is executed and the plan kept. Otherwise the next input of the last kept
  plan is executed, which ends in straight full braking.
* No delay advance. The filter runs on the vehicle, so its command acts at once,
  and it solves from the measured state. The published MPSF advances its state
  over the command round trip (up to 0.5 s). A plan checked from that advanced
  state and then run from the real one is not the plan that was checked, and in
  the failure-mode battery that let the vehicle stop against a rock.

The tube is the 0.25 m obstacle buffer plus 0.03 m per metre travelled plus 0.06
x the integral over distance of the accumulated heading change (m rad). The OCP
carries path length, accumulated turn and their integral as states, so it
enforces the same tube. Against the plant (the journal workspace's
`experiments/filter_fixes/model_check.py`, 12 open-loop manoeuvres on three
soils at three speeds) the tube covers every checked point, for the CG and the
discs. There, with the plant's speed imposed, the surrogate model's lateral
error at 2 s on clay is 0.18 m where the kinematic model's is 1.08 m.

Other vehicles are predicted as sets. Each keeps its measured heading, and its
speed can do anything between braking as hard as the soil allows (locked wheels
on the believed soil, which is how the plant's vehicles brake at full pedal) and
speeding up at 1 m/s^2 (`traffic_accel`), starting from the measured speed give
or take `traffic_speed_err`. At each future time the vehicle is then somewhere
on a segment along its heading, and the discs must clear the whole segment, in
the OCP and in the check. A lead vehicle that brakes, an oncoming one and one
cutting in are all inside the set, so the kept plan stays safe while they move.
`--traffic-speed-noise s`
adds a Gaussian error of standard deviation s to each vehicle's speed as both
filters receive it, and sets `traffic_speed_err` to 3 s.

## Tire model supervision

Where either filter queries a learned tire model, that model is supervised only
by the controlled single-tire Chrono SCM rig. No filter consumes Chrono terrain
truth, tire-force truth, or any other quantity unavailable to a physical
vehicle.

## Usage

```python
from safety import make_safety_filter

sf = make_safety_filter(
    'dob_cbf',
    vehicle_params=params,
    nn_model=nn_model,        # optional; enables tire-model traction limits
    cbf_alpha=1.0,            # barrier class-K gain
    obstacle_buffer=0.25,     # extra margin around obstacles (m)
    delay_steps=5,            # actuation delay, in control steps
    control_dt=0.1,           # filter period (s)
)

result = sf.filter(
    desired_steering, desired_throttle, desired_brake,
    vehicle_state, obstacles,
)

steering = result.steering
throttle = result.throttle
braking = result.braking

print(f"modified={result.was_modified} active={result.active_constraints}")
```

## Command-line use

```bash
# Reactive filter with default parameters
python simulation/runtime/launch_decoupled.py --manual --safety-filter

# Predictive filter
python simulation/runtime/launch_decoupled.py --manual --safety-filter \
    --safety-flavor mpsf

# Non-default barrier gain and margin
python simulation/runtime/launch_decoupled.py --manual --safety-filter \
    --cbf-alpha 5.0 --safety-buffer 2.0 --delay-steps 10

# Operator driving through a rock field with the filter engaged
python simulation/runtime/launch_decoupled.py --manual --rocks 20 --safety-filter
```

## Parameters

Shared by both filters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obstacle_buffer` | 0.25 m | Extra margin added to every obstacle radius (`--safety-buffer`) |
| `vehicle_radius` | 1.0 m | Geometric vehicle-footprint radius used by the barrier. Distinct from the 1.5 m clearance proxy in `collision_detector.py`, and never collision truth |
| `teleop_delay` | 0.0 s | One-way command-channel delay; 0 disables delay compensation |
| `stale_cmd_timeout` | 2.0 s | Command age above which the filter brakes |

`dob_cbf` only:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cbf_alpha` | 1.0 | First barrier class-K gain; higher is more conservative |
| `cbf_alpha2` | 0.8 | Second-order barrier gain |
| `w_long` | 0.15 | Barrier weight along the heading; smaller enlarges the safe zone ahead |
| `w_lat` | 0.50 | Barrier weight laterally; larger strengthens the steering signal |
| `forward_bias` | 1.5 m | Forward shift of the barrier centre |
| `dob_bandwidth` | 10.0 | Disturbance-observer bandwidth; higher tracks faster and is noisier |
| `delay_steps` | 5 | Actuation delay, in control steps, for the predictor |
| `max_steering_rate` | 8.0 rad/s | Road-wheel rate limit imposed on the filter output |
| `steer_tau` | 0.12 s | Steering-actuator lag applied to the filter output |
| `max_speed` | 15.0 m/s | Absolute speed cap |
| `control_dt` | 0.1 s | Filter period as constructed by the plant node |

`mpsf` only:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 20 | Prediction steps |
| `dt` | 0.1 s | Prediction step |
| `n_obstacles` | 3 | Obstacle rows the OCP is sized for (`--mpsf-n-obstacles`) |
| `ego_forward_extent` | 1.6 m | Body extent ahead of the CG added to each safe radius |
| `w_steer` | 1.0 | Steering-deviation weight; below 1 makes weaving cheaper than braking |
| `w_progress` | 0.0 | Penalty on crawling below a 2 m/s floor; 0 keeps braking primary |
| `max_decel` | 6.0 m/s² | Nominal braking envelope, scaled by the soil-derived braking authority |
| `lat_accel_max_default` | 4.0 m/s² | Cornering limit used before a soil estimate arrives |
| `terminal_set` | False | Terminal-set formulation (`--mpsf-terminal-set`) |
| `time_steps` | 0.1 x 20 | Nonuniform horizon grid (`--mpsf-time-grid`, e.g. `0.1x10,0.3x10`) |
| `brake_margin` | 0.85 | Fraction of the surrogate braking deceleration the certificate assumes |
| `sense_range` | 30 m | Sensed region every certified path must stay inside (0 = none) |
| `dynamic_model` | False | Predictive safety filter on the surrogate model (`--mpsf-dynamic`) |
| `tube_c1`, `tube_eps` | 0.03, 0.06 | Error tube per metre travelled and per metre x radian turned |
| `traffic_speed_err`, `traffic_accel` | 0, 1.0 | Other vehicles' speed error bound (m/s) and acceleration bound (m/s^2) in the dynamic model's prediction |
| `nominal_soil` | sand | Soil the dynamic model assumes before a belief arrives (soil-blind arm) |
