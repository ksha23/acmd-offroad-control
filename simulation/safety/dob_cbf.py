"""Reactive minimum-deviation safety filter for the HMMWV.

A disturbance-observer-based control barrier function (DOB-CBF) filter. Each
call solves one quadratic program that minimises deviation from the incoming
command subject to a barrier constraint per obstacle, so the command passes
through untouched whenever those constraints are inactive and is altered only
by the amount they demand.

Elements:
1. **Weighted ellipsoidal barrier**: a heading-aligned ellipse makes lateral
   escape nearer in barrier space than braking, so the QP steers around a
   static obstacle instead of stopping in front of it.
2. **Disturbance observer**: a first-order observer estimates the lumped
   unmodelled longitudinal acceleration (compaction drag, grade, sinkage
   resistance) and feeds it into the barrier's second derivative, so the
   constraint accounts for deceleration the nominal model does not predict.
3. **Directional bias**: a forward-shifted ellipse centre concentrates the
   grip-limited margin ahead of the vehicle, where the threat is.
4. **Terrain-aware speed limiting**: the neural tire model's lateral-force
   prediction sets the speed at which an evasive manoeuvre remains available.
   That model is supervised only by the controlled single-tire Chrono SCM rig.
5. **Latency compensation**: a discrete predictor with derivative feedback
   estimates the state reached across the actuation delay.

Mathematical Formulation
------------------------
State: ``x = [x, y, psi, v, beta]`` where beta is the current road-wheel angle.

Barrier function (per obstacle):
    ``h(x) = (p - p_obs)^T P (p - p_obs) - r_safe^2``

where ``P = R(psi)^T diag(w_long, w_lat) R(psi)`` is a heading-aligned
ellipsoidal weight matrix.  Setting ``w_long < w_lat`` (e.g. 1/100 vs 1/9)
elongates the safe zone along the heading -> lateral escape is "closer" in
barrier-space -> the QP naturally prefers steering over braking.

Control inputs: ``u = [dbeta (steering angle rate), alpha (normalized throttle)]``

The QP minimizes deviation from the driver's desired input:
    min  ||u - u_des||^2
    s.t. -psi1_i @ u <= psi0_i   for each obstacle i
         actuator limits

DOB update:
    ``hdv0 = p0v + a_v * v``   (velocity disturbance estimate)
    ``dp0v = -a_v * (f_nom(v, alpha) + hdv0)``

Usage:
    from simulation.safety import CBFSafetyFilter

    cbf = CBFSafetyFilter(vehicle_params, nn_casadi=nn_model)

    # In loop:
    result = cbf.filter(
        desired_steering, desired_throttle, desired_brake,
        vehicle_state, obstacles, terrain_roughness
    )
"""

import os
import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class SafetyFilterResult:
    """Output of the CBF safety filter."""
    steering: float       # Filtered steering input [-1, 1]
    throttle: float       # Filtered throttle input [0, 1]
    braking: float        # Filtered braking input [0, 1]
    was_modified: bool    # True if inputs were changed by safety filter
    active_constraints: int  # Number of active CBF constraints
    solve_time_ms: float  # QP solve time in milliseconds
    v_max_terrain: float  # Terrain-limited max safe speed (m/s)
    safety_margin: float  # Minimum h(x) across all obstacles
    dob_norm: float = 0.0 # Norm of DOB disturbance estimate


class DelayCompensator:
    """
    Discrete predictor for latency compensation.

    Holds a FIFO of issued control commands and runs a discrete predictor with
    derivative and proportional feedback, so that under a known actuation delay
    the filter reasons about the command the vehicle is executing now rather
    than the one just issued.

    Args:
        delay_steps: Number of time steps of actuation delay
        dt: Control loop time step (s)
        k1: Derivative feedback gain (default: 0.6)
        k2: Proportional feedback gain (default: 2.0)
    """

    def __init__(self, delay_steps: int = 5, dt: float = 0.02,
                 k1: float = 0.6, k2: float = 2.0):
        self.td = max(int(delay_steps), 0)
        self.dt = dt
        self.k1 = k1
        self.k2 = k2

        from collections import deque
        maxlen = self.td + 1
        self.throttle_hist = deque([0.0] * maxlen, maxlen=maxlen)
        self.steer_hist = deque([0.0] * maxlen, maxlen=maxlen)
        self.throttle_pred = deque([0.0] * maxlen, maxlen=maxlen)
        self.steer_pred = deque([0.0] * maxlen, maxlen=maxlen)

    def update(self, throttle: float, steering: float) -> Tuple[float, float]:
        """
        Push new command and return delay-compensated estimate.

        Returns:
            (compensated_throttle, compensated_steering)
        """
        self.throttle_hist.append(throttle)
        self.steer_hist.append(steering)

        if self.td == 0 or len(self.throttle_hist) <= self.td:
            self.throttle_pred.append(throttle)
            self.steer_pred.append(steering)
            return throttle, steering

        th = list(self.throttle_hist)
        sh = list(self.steer_hist)
        tp = list(self.throttle_pred)
        sp = list(self.steer_pred)

        v_td = th[-self.td - 1]
        dv_td = (th[-self.td] - v_td) / self.dt
        vp_td = tp[-self.td - 1]
        dvp_td = (tp[-self.td] - vp_td) / self.dt

        vp_now = tp[-1] + self.dt * (
            dv_td + self.k1 * (dv_td - dvp_td) + self.k2 * (v_td - vp_td)
        )
        self.throttle_pred.append(vp_now)

        s_td = sh[-self.td - 1]
        ds_td = (sh[-self.td] - s_td) / self.dt
        sp_td = sp[-self.td - 1]
        dsp_td = (sp[-self.td] - sp_td) / self.dt

        sp_now = sp[-1] + self.dt * (
            ds_td + self.k1 * (ds_td - dsp_td) + self.k2 * (s_td - sp_td)
        )
        self.steer_pred.append(sp_now)

        return float(vp_now), float(sp_now)


class DisturbanceObserver:
    """
    First-order velocity disturbance observer (DOB).

    Estimates the unmodeled longitudinal disturbance force (terrain drag,
    SCM sinkage resistance, grade, wind) as a lumped acceleration term.

    The observer dynamics are:
        hdv0 = p0v + a_v * v          (disturbance estimate)
        dp0v = -a_v * (f_nom + hdv0)  (observer state update)

    where f_nom is the nominal acceleration from the powertrain model.

    The observer is longitudinal only. Extending it to position and heading
    would correct dead-reckoning drift, which this filter does not accumulate:
    obstacle positions arrive in the same frame as the measured vehicle pose,
    so the barrier never integrates a position estimate.

    Args:
        a_v: Observer bandwidth for velocity (higher = faster tracking,
             but more noise sensitivity). Default 10.0.
        max_accel: Maximum powertrain acceleration for nominal model (m/s^2).
        max_decel: Maximum braking deceleration for nominal model (m/s^2).
    """

    def __init__(self, a_v: float = 10.0,
                 max_accel: float = 3.0, max_decel: float = -6.0):
        self.a_v = a_v
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.p0v = 0.0  # Observer internal state
        self._initialized = False

    def update(self, v: float, alpha: float, dt: float,
               f_nom_override: Optional[float] = None) -> float:
        """
        Update the DOB and return the velocity disturbance estimate.

        Args:
            v: Current longitudinal speed (m/s)
            alpha: Current normalized throttle/brake input [-1, 1]
                   (positive = throttle, negative = brake)
            dt: Time step (s)
            f_nom_override: If provided, use this as the nominal longitudinal
                acceleration instead of the internal linear model. This allows
                the NN tire model to supply a physics-based f_nom so the DOB
                only estimates *true* unmodeled disturbances.

        Returns:
            hdv0: Estimated longitudinal disturbance acceleration (m/s^2).
                  Positive = unexplained acceleration, negative = drag/resistance.
        """
        # On first call, initialize p0v so that hdv0 = 0 (no initial disturbance)
        if not self._initialized:
            self.p0v = -self.a_v * v
            self._initialized = True

        hdv0 = self.p0v + self.a_v * v

        # Nominal acceleration model
        if f_nom_override is not None:
            f_nom = f_nom_override
        elif alpha >= 0:
            f_nom = self.max_accel * alpha
        else:
            f_nom = self.max_decel * abs(alpha)

        # Observer dynamics: dp0v = -a_v * (f_nom + hdv0)
        dp0v = -self.a_v * (f_nom + hdv0)
        self.p0v += dp0v * dt

        return hdv0

    @property
    def disturbance_estimate(self) -> float:
        """Current disturbance estimate without updating."""
        return self.p0v

    def reset(self):
        self.p0v = 0.0
        self._initialized = False


class CBFSafetyFilter:
    """
    DOB-CBF safety filter with weighted ellipsoidal barrier.

    Filters manual or MPC control inputs to enforce safety constraints:
    1. Obstacle avoidance via heading-aligned ellipsoidal CBF (prefers steering)
    2. Terrain-aware speed limits (from roughness + NN traction prediction)
    3. Disturbance observer for longitudinal force estimation
    4. Latency compensation (forward prediction of vehicle state)

    The filter solves a QP at each step:
        min  ||u - u_des||^2_W
        s.t. -psi1_i @ u <= psi0_i  (CBF constraints per obstacle)
             actuator limits

    Control inputs: u = [dbeta (steering rate), alpha (throttle)]

    The ellipsoidal barrier uses P = R^T diag(w_long, w_lat) R with
    w_long << w_lat, so lateral avoidance is cheaper than braking in barrier
    space. An isotropic barrier admits no such preference and reduces throttle
    where a steering correction would suffice.

    Args:
        vehicle_params: Dict with M, Lf, Lr, Izz, h_cg, T
        nn_casadi: Optional NNCasADi for tire force prediction
        max_steering_rate: Maximum steering angle rate (rad/s)
        cbf_alpha: First CBF class-K gain (l1cbf). Higher = more conservative.
        cbf_alpha2: Second CBF class-K gain (l2cbf).
        obstacle_buffer: Extra safety margin around obstacles (m)
        vehicle_radius: Effective geometric vehicle-footprint radius (m)
        max_speed: Absolute maximum allowed speed (m/s)
        delay_steps: Actuation delay in control steps for compensator
        control_dt: Safety filter update period (s)
        w_long: Barrier weight in longitudinal (heading) direction.
                Smaller values -> larger "safe zone" ahead -> earlier braking.
        w_lat: Barrier weight in lateral direction.
               Larger values -> tighter lateral clearance -> stronger steer signal.
        forward_bias: Forward shift of barrier center (m). Makes filter more
                      proactive for obstacles ahead. 0 = no bias.
        dob_bandwidth: DOB velocity observer bandwidth. Higher = faster.
        cbf_flavor: 'balance' (equal cost to modify steering/throttle),
                    'steer_priority' (strongly prefers throttle modification),
                    'throttle_priority' (strongly preserves throttle).
        teleop_delay: Estimated one-way network delay for teleoperation (s).
                      0 = no teleop compensation (local control). When > 0,
                      the barrier is inflated by v * RTT to account for operator
                      reaction lag, and stale commands trigger emergency braking.
        stale_cmd_timeout: Maximum command age before auto-brake (s).
                           Only active when teleop_delay > 0.
    """

    # Speed at which the tire surrogate stops being queried at the true state.
    # The query clamp and the resistance taper that repairs it read this one
    # value, so the two cannot drift apart.
    _NN_QUERY_SPEED_FLOOR = 0.5  # m/s

    def __init__(self,
                 vehicle_params: dict,
                 nn_casadi=None,
                 max_steering_rate: float = 8.0,
                 steer_tau: float = 0.12,
                 max_alpha_rate: float = 8.0,
                 cbf_alpha: float = 1.0,
                 cbf_alpha2: float = 0.8,
                 obstacle_buffer: float = 0.25,
                 vehicle_radius: float = 1.0,
                 max_speed: float = 15.0,
                 delay_steps: int = 5,
                 control_dt: float = 0.02,
                 w_long: float = 0.15,
                 w_lat: float = 0.50,
                 forward_bias: float = 1.5,
                 dob_bandwidth: float = 10.0,
                 cbf_flavor: str = 'balance',
                 teleop_delay: float = 0.0,
                 stale_cmd_timeout: float = 2.0,
                 variant: str = 'dob_cbf'):

        # Filter variant label, recorded in diagnostics so a run's logs
        # identify which barrier configuration produced them.
        self.variant = (variant or 'dob_cbf').lower()

        # Vehicle parameters
        self.M = vehicle_params['M']
        self.Lf = vehicle_params['Lf']
        self.Lr = vehicle_params['Lr']
        self.L = self.Lf + self.Lr
        self.Izz = vehicle_params['Izz']
        self.h_cg = vehicle_params.get('h_cg', 0.65)
        self.T = vehicle_params.get('T', 1.8194)

        # NN tire model for traction-aware limits
        self.nn_casadi = nn_casadi

        # CBF parameters
        self.alpha1 = cbf_alpha      # class-K gain on h
        self.alpha2 = cbf_alpha2     # class-K gain on h_dot
        self.obstacle_buffer = obstacle_buffer
        self.vehicle_radius = vehicle_radius
        self.max_speed = max_speed
        self.max_steer_rate = max_steering_rate
        self.steer_tau = max(steer_tau, 1e-3)   # first-order steering-actuator lag (s)
        self.max_alpha_rate = max_alpha_rate    # throttle/brake (alpha) rate limit (1/s)
        self.control_dt = control_dt

        # Ellipsoidal barrier weights; w_long < w_lat elongates the safe set
        # along the heading, making lateral escape the cheaper correction.
        self.w_long = w_long
        self.w_lat = w_lat
        self.forward_bias = forward_bias

        # CBF flavor controls QP cost weighting
        self.cbf_flavor = cbf_flavor

        # Teleop delay compensation
        self._teleop_enabled = teleop_delay > 0.0  # Only activate if explicitly set
        self._teleop_delay = max(teleop_delay, 0.0)
        self._stale_cmd_timeout = stale_cmd_timeout
        self._cmd_age = 0.0        # Latest measured command age (s)
        self._last_cmd_wall = None  # Wall-clock time of last received command
        self._delay_ema = teleop_delay  # EMA-smoothed one-way delay estimate
        self._delay_ema_alpha = 0.15    # EMA smoothing factor

        # Obstacle filtering range — must cover the ellipsoidal barrier's
        # longitudinal reach: reach ~ forward_bias + sqrt(max_safe_r^2 / w_long)
        max_safe_r = vehicle_radius + obstacle_buffer + 5.0
        self.r_precpt = forward_bias + np.sqrt(max_safe_r**2 / w_long) + 10.0

        # Delay compensator
        self.delay_comp = DelayCompensator(
            delay_steps=delay_steps, dt=control_dt
        )

        # Disturbance observer
        self.dob = DisturbanceObserver(
            a_v=dob_bandwidth,
            max_accel=3.0,
            max_decel=-6.0,
        )

        # Steering angle conversion
        self.max_road_steer_angle = 0.49  # rad (HMMWV)

        # Actuator limits
        self.max_accel = 3.0   # m/s^2 throttle
        self.max_decel = -6.0  # m/s^2 braking

        # Live-terrain conditioning of the filter's own tire queries, and the
        # measured grip scale from the controller's force-map adapter. Both
        # are opt-in through update_terrain; when neither is enabled the tire
        # model is queried at its nominal soil and the authority scaling
        # follows the linear map of the Bekker exponent alone.
        self._terrain_n = None
        self._terrain_params = None
        self._use_terrain_nn = False
        self._grip_scale = 1.0
        self._use_grip_scale = False

        # Current steering angle state (integrated from dbeta)
        self._beta = 0.0  # road wheel angle (rad)
        self._alpha = 0.0  # last throttle command

        # Steering-output actuator model. A large call-to-call jump in the
        # commanded road-wheel angle -- the QP switching between two far-apart
        # avoidance solutions -- would impulse the steering rack and front
        # suspension, which no physical rack can execute. The output is
        # therefore rate-limited to the rack's slew rate and passed through a
        # first-order lag, so the filter only ever commands motion the actuator
        # can produce.
        self._last_safe_steering = 0.0   # last (slewed) normalized steering out
        self._reactive_dir = 0.0         # committed reactive-steer side (hysteresis)
        self._last_filter_wall = None    # wall time of previous filter() call
        self._steer_track_ema = 0.0      # EMA of |measured - commanded| road angle
        self._steer_broken = False       # latched once a break is detected

        # Rear-approach avoidance: when a vehicle closes from behind, never let
        # the driver slow down (braking raises rear-end risk), and accelerate to
        # try to escape when the path ahead is clear -- including from a standstill.
        self._rear_threat_dist = 12.0    # m: a closing rear obstacle within this is a threat
        self._rear_half_w = 2.5          # m: lateral half-width counted as "same lane"
        self._prev_rear_dist = float('inf')

        # State for logging
        self._last_result = None
        self._filter_count = 0
        self._modify_count = 0

        # CSV logging for diagnostics
        self._csv_file = None
        self._csv_writer = None
        self._obs_csv_file = None
        self._obs_csv_writer = None
        self._init_csv_logging()

        # Cache for NN tire force queries (avoid redundant calls per step)
        self._nn_cache = {}

    # ------------------------------------------------------------------
    # Teleop delay API
    # ------------------------------------------------------------------

    def set_teleop_delay(self, delay_s: float):
        """Set estimated one-way teleop network delay (seconds)."""
        self._teleop_delay = max(delay_s, 0.0)
        self._delay_ema = max(delay_s, 0.0)
        self._teleop_enabled = self._teleop_delay > 0.0

    def update_terrain(self, terrain_params, phi_uncertainty_deg=None,
                       n_sigma=None, hedge_k=None,
                       use_terrain_nn=None, grip_scale=None,
                       use_grip_scale=None):
        """Re-condition the filter on the live online terrain estimate.

        The filter's longitudinal authority is grip-limited, and grip drops on
        softer soil, so the available accel/brake (which set the CBF's
        deceleration budget and effective stopping buffer) are scaled by the
        firmness of the live soil. A lower Bekker exponent n means softer soil
        and yields a more conservative filter. The plant node calls this only
        while the online terrain estimator is running; otherwise the nominal
        limits stand.

        Belief-robust authority: when the estimator's posterior std ``n_sigma``
        and a risk factor ``hedge_k`` are supplied, the authority is evaluated
        at the pessimistic quantile ``n - hedge_k * n_sigma`` -- the filter
        trusts its brakes only as much as the *worst plausible* soil allows,
        so an uncertain estimate widens the stopping buffer instead of
        silently assuming the mean.
        """
        try:
            n = float(terrain_params["n"] if isinstance(terrain_params, dict)
                      else terrain_params)
        except (TypeError, ValueError, KeyError):
            return
        self._terrain_n = n
        if isinstance(terrain_params, dict):
            self._terrain_params = dict(terrain_params)
        if use_terrain_nn is not None:
            self._use_terrain_nn = bool(use_terrain_nn)
        if use_grip_scale is not None:
            self._use_grip_scale = bool(use_grip_scale)
        if grip_scale is not None:
            self._grip_scale = float(np.clip(grip_scale, 0.4, 1.2))
        n_eff = n
        if n_sigma is not None and hedge_k is not None and float(hedge_k) > 0.0:
            n_eff = float(np.clip(n - float(hedge_k) * float(n_sigma), 0.4, 1.3))
        self._terrain_n_eff = n_eff
        # n in [~0.4 soft, ~1.3 firm] -> grip scale in [0.6, 1.0]
        grip = float(np.clip(0.6 + 0.4 * (n_eff - 0.4) / 0.9, 0.5, 1.0))
        # Measured grip scale from the controller's force-map adapter: the
        # ratio of realised to modelled lateral force. On soils with less
        # grip than their n implies (off-manifold friction/cohesion), the
        # n-map above cannot see the deficit; s can.
        if self._use_grip_scale:
            grip *= self._grip_scale
        self.max_accel = 3.0 * grip
        self.max_decel = -6.0 * grip

    def update_command_age(self, cmd_wall_time: float):
        """
        Update teleop delay estimate from a received command's wall-clock stamp.

        Call this each time a ControlCommand arrives.  Computes one-way
        latency as ``time.time() - cmd_wall_time`` and feeds an EMA filter.
        Also tracks the wall-clock of the most recent command for staleness.

        Args:
            cmd_wall_time: The ``wall_time`` field from the ControlCommand.
        """
        now = time.time()
        one_way = max(now - cmd_wall_time, 0.0)
        self._cmd_age = one_way
        self._last_cmd_wall = now  # arrival time of the most recent command
        # The delay estimate is updated only when teleoperation was enabled at
        # construction (delay > 0). Loopback transport latency of one or two
        # milliseconds must not engage the teleop prediction or the
        # stale-command brake.
        if self._teleop_enabled:
            a = self._delay_ema_alpha
            self._delay_ema = a * one_way + (1.0 - a) * self._delay_ema
            self._teleop_delay = self._delay_ema

    def _effective_obstacle_buffer(self, v: float) -> float:
        """
        Compute obstacle buffer inflated by teleop round-trip delay.

        At speed *v* with one-way delay *tau*, the vehicle travels
        ``v * 2 * tau`` metres during the round trip before the operator can
        react. That distance, scaled by 0.5, is added to the static
        obstacle_buffer.

        Returns the effective buffer in metres.
        """
        if self._teleop_delay <= 0.0:
            return self.obstacle_buffer
        rtt = 2.0 * self._teleop_delay
        return self.obstacle_buffer + v * rtt * 0.5

    def _is_command_stale(self) -> bool:
        """True if no command received within stale_cmd_timeout (teleop only)."""
        if self._teleop_delay <= 0.0 or self._last_cmd_wall is None:
            return False
        age = time.time() - self._last_cmd_wall
        return age > self._stale_cmd_timeout

    def _init_csv_logging(self):
        """Initialize CSV files for safety filter diagnostics."""
        import csv
        import os
        # Parallel sweeps set HIL_RUN_LOG_DIR to a unique per-run directory
        # (see benchmarking/common.py) so concurrent workers never share a log
        # file. Live and operator-driven runs leave it unset and write to the
        # repository-level logs/ directory.
        log_dir = os.environ.get('HIL_RUN_LOG_DIR') or os.path.join(
            os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Main filter log: one row per filter() call
        main_path = os.path.join(log_dir, 'cbf_filter_log.csv')
        self._csv_file = open(main_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'step', 'x', 'y', 'psi_deg', 'v', 'beta_deg', 'hdv0',
            'n_obs', 'n_constraints', 'min_h',
            'steer_in', 'throttle_in', 'brake_in',
            'steer_out', 'throttle_out', 'brake_out',
            'alpha_cmd', 'alpha_out',
            'was_modified', 'active_constraints', 'solve_ms',
            'qp_success', 'v_max_terrain',
        ])

        # Per-obstacle log: one row per obstacle per filter() call
        obs_path = os.path.join(log_dir, 'cbf_obstacle_log.csv')
        self._obs_csv_file = open(obs_path, 'w', newline='')
        self._obs_csv_writer = csv.writer(self._obs_csv_file)
        self._obs_csv_writer.writerow([
            'step', 'obs_x', 'obs_y', 'obs_r', 'dist',
            'd_along', 'd_cross', 'd_along_unbiased',
            'h', 'h_dot', 'h_ddot_auto',
            'A_steer', 'A_alpha', 'psi0',
            'psi1_steer', 'psi1_alpha',
            'skipped_hdot', 'skipped_far',
        ])
        print(f"  [CBF] CSV logging to {log_dir}/cbf_*.csv")

    def _log_csv_obstacle(self, step, obs_x, obs_y, obs_r, dist,
                          d_along, d_cross, d_along_unbiased,
                          h, h_dot, h_ddot_auto,
                          A_steer, A_alpha, psi0, psi1_steer, psi1_alpha,
                          skipped_hdot, skipped_far):
        if self._obs_csv_writer:
            self._obs_csv_writer.writerow([
                step, f'{obs_x:.3f}', f'{obs_y:.3f}', f'{obs_r:.3f}', f'{dist:.2f}',
                f'{d_along:.3f}', f'{d_cross:.3f}', f'{d_along_unbiased:.3f}',
                f'{h:.4f}', f'{h_dot:.4f}', f'{h_ddot_auto:.4f}',
                f'{A_steer:.6f}', f'{A_alpha:.6f}', f'{psi0:.4f}',
                f'{psi1_steer:.6f}', f'{psi1_alpha:.6f}',
                int(skipped_hdot), int(skipped_far),
            ])

    def _log_csv_main(self, step, x, y, psi, v, beta, hdv0,
                      n_obs, n_constraints, min_h,
                      steer_in, throttle_in, brake_in,
                      steer_out, throttle_out, brake_out,
                      alpha_cmd, alpha_out,
                      was_modified, active_constraints, solve_ms,
                      qp_success, v_max_terrain):
        if self._csv_writer:
            self._csv_writer.writerow([
                step, f'{x:.3f}', f'{y:.3f}', f'{np.degrees(psi):.2f}',
                f'{v:.3f}', f'{np.degrees(beta):.2f}', f'{hdv0:.4f}',
                n_obs, n_constraints, f'{min_h:.4f}',
                f'{steer_in:.4f}', f'{throttle_in:.4f}', f'{brake_in:.4f}',
                f'{steer_out:.4f}', f'{throttle_out:.4f}', f'{brake_out:.4f}',
                f'{alpha_cmd:.4f}', f'{alpha_out:.4f}',
                int(was_modified), active_constraints, f'{solve_ms:.2f}',
                int(qp_success), f'{v_max_terrain:.2f}',
            ])
            # Flush periodically so the log is readable mid-run.
            if step % 100 == 0:
                self._csv_file.flush()
                self._obs_csv_file.flush()

    def _resistive_acceleration(self, nn_forces, u_measured: float) -> float:
        """Control-independent longitudinal resistance at the current state.

        The tire surrogate is queried at ``max(u, _NN_QUERY_SPEED_FLOOR)`` to
        keep the slip-angle arctangents well-conditioned, so below that speed
        it reports the resistance of a vehicle rolling at the clamp rather than
        of the one in front of it. At rest that resistance is not acting: the
        tires are held by static friction. Feeding it to the observer would
        make the difference against a measured zero acceleration read as an
        unexplained forward disturbance, which the barrier charges to h_ddot
        and answers with brake -- sustaining the standstill that produced it.
        Scaling by the speed ratio restores the physical limit at ``u = 0`` and
        is exactly unity at and above the clamp, where the surrogate is
        evaluated at the true state.
        """
        if nn_forces is None:
            return 0.0
        taper = min(1.0, abs(float(u_measured)) / self._NN_QUERY_SPEED_FLOOR)
        # Resistance opposes the motion; the surrogate reports it for forward
        # rolling. Rolling backward, the same magnitude acts forward.
        direction = 1.0 if float(u_measured) >= 0.0 else -1.0
        return float(nn_forces['ax_nn']) * taper * direction

    # Representative drive slip ratio at which the surrogate is asked what
    # longitudinal force the soil will actually deliver.
    _AUTHORITY_SLIP_RATIO = 0.15

    # Braking cap used in the observer's nominal when the executed command is
    # a brake. It must upper-bound any deceleration the plant can produce, so
    # it is 1 g -- no tire-soil pairing here brakes harder -- and deliberately
    # NOT the grip-scaled ``max_decel``, whose model reaches 7.2 m/s^2 under a
    # high grip-scale and whose heuristic can undershoot realized braking on
    # firm ground. An undershoot here flips the bias into crediting the
    # barrier with deceleration that stops when the brakes are released;
    # overestimating is the safe direction, booked by the observer as a
    # phantom *forward* disturbance that tightens the barrier.
    _BRAKE_NOMINAL_ENVELOPE = 9.81  # m/s^2

    def _longitudinal_authority(self, u: float, v_lat: float, omega: float,
                                delta: float, baseline) -> float:
        """Acceleration one unit of alpha buys on the soil underneath.

        This is the coefficient the barrier places on its decision variable,
        so it is a function of state alone: deriving it from the driver's
        pedal would make the safe set change shape with the command, and the
        asymmetric drive/brake envelopes would make it discontinuous across a
        touch of the brake. The drive-slip secant deliberately *under*-credits
        braking -- the QP then brakes harder than strictly needed when it
        intervenes, which is the conservative direction. ``baseline`` is the
        zero-slip query already made this step; when it failed there is no
        difference to take, and the actuator envelope is the fallback rather
        than a lone slipped query masquerading as one.
        """
        if self.nn_casadi is None or baseline is None:
            return self.max_accel
        slipped = self._compute_nn_tire_forces(
            u, v_lat, omega, delta, kappa=self._AUTHORITY_SLIP_RATIO)
        if slipped is None:
            return self.max_accel
        delta_ax = float(slipped['ax_nn']) - float(baseline['ax_nn'])
        return float(np.clip(abs(delta_ax), 0.3, self.max_accel))

    def _compute_nn_tire_forces(self, u: float, v_lat: float, omega: float,
                                 delta: float, kappa: float = 0.0
                                 ) -> Optional[Dict[str, float]]:
        """
        Query NN tire model for front/rear Fx, Fy and derived quantities.

        Uses bicycle-model slip angles to query the NN at front and rear axles.
        Returns None if nn_casadi is not available.

        Returns dict with:
            Fx_total: Total longitudinal force (N), both axles, both sides
            Fy_total: Total lateral force (N), both axles, both sides
            Mz_yaw:   Yaw moment from lateral forces (N·m)
            ax_nn:     Longitudinal acceleration (m/s²)
            ay_nn:     Lateral acceleration (m/s²)
            alpha_dot: Yaw angular acceleration (rad/s²)
            Fy_f:      Front axle lateral force per wheel (N)
            Fy_r:      Rear axle lateral force per wheel (N)
            alpha_f:   Front slip angle (rad)
            alpha_r:   Rear slip angle (rad)
        """
        if self.nn_casadi is None:
            return None

        g = 9.81
        u_safe = max(u, self._NN_QUERY_SPEED_FLOOR)  # Avoid division by zero

        # Bicycle model slip angles
        alpha_f = delta - np.arctan2(v_lat + self.Lf * omega, u_safe)
        alpha_r = -np.arctan2(v_lat - self.Lr * omega, u_safe)

        # Clamp slip angles to training-data range
        _alpha_max = 0.55
        alpha_f = float(max(-_alpha_max, min(_alpha_max, alpha_f)))
        alpha_r = float(max(-_alpha_max, min(_alpha_max, alpha_r)))

        # Normal forces (static weight distribution, per wheel)
        Fz_f = self.M * g * self.Lr / self.L / 2.0
        Fz_r = self.M * g * self.Lf / self.L / 2.0

        try:
            _tkw = {}
            if self._use_terrain_nn and self._terrain_n is not None:
                _tkw = dict(n_terrain=self._terrain_n,
                            terrain_params=self._terrain_params)
            Fx_f, Fy_f = self.nn_casadi.predict_numeric(alpha_f, Fz_f, u_safe, kappa, **_tkw)
            Fx_r, Fy_r = self.nn_casadi.predict_numeric(alpha_r, Fz_r, u_safe, kappa, **_tkw)
            if self._use_grip_scale:
                # Measured-over-modelled force ratio from the adapter: the
                # filter's predicted control authority shrinks with it.
                Fx_f *= self._grip_scale; Fy_f *= self._grip_scale
                Fx_r *= self._grip_scale; Fy_r *= self._grip_scale

            # Total forces (×2 for left+right wheels)
            Fx_total = 2.0 * (Fx_f + Fx_r)
            Fy_total = 2.0 * (Fy_f + Fy_r)
            # Yaw moment: front pushes one way, rear the other
            Mz_yaw = 2.0 * (self.Lf * Fy_f - self.Lr * Fy_r)

            return {
                'Fx_total': Fx_total,
                'Fy_total': Fy_total,
                'Mz_yaw': Mz_yaw,
                'ax_nn': Fx_total / self.M,
                'ay_nn': Fy_total / self.M,
                'alpha_dot': Mz_yaw / self.Izz,
                'Fy_f': Fy_f,
                'Fy_r': Fy_r,
                'alpha_f': alpha_f,
                'alpha_r': alpha_r,
            }
        except Exception:
            return None

    def _compute_nn_steering_sensitivity(self, u: float, v_lat: float,
                                          omega: float, delta: float
                                          ) -> Optional[Tuple[float, float]]:
        """
        Estimate dFy/ddelta and dMz/ddelta via finite difference on the NN.

        Returns (dFy_ddelta, dMz_ddelta) or None if NN unavailable.
        These give the barrier the lateral force and yaw moment produced per
        unit of steering, which is the filter's steering control authority.
        """
        if self.nn_casadi is None:
            return None

        eps = 0.005  # ~0.3 deg perturbation
        forces_plus = self._compute_nn_tire_forces(u, v_lat, omega, delta + eps)
        forces_minus = self._compute_nn_tire_forces(u, v_lat, omega, delta - eps)

        if forces_plus is None or forces_minus is None:
            return None

        dFy_ddelta = (forces_plus['Fy_total'] - forces_minus['Fy_total']) / (2 * eps)
        dMz_ddelta = (forces_plus['Mz_yaw'] - forces_minus['Mz_yaw']) / (2 * eps)

        return dFy_ddelta, dMz_ddelta

    def _compute_reactive_steering(self, vehicle_state: dict,
                                    obstacles: List[Tuple[float, float, float]]) -> float:
        """
        Compute a reactive steering avoidance command for nearby obstacles.

        The second-order CBF cannot steer away from head-on obstacles because
        the barrier gradient is orthogonal to the steering effect (geometric
        singularity). This layer handles steering directly: for each nearby
        obstacle ahead, it contributes a steering command away from it,
        proportional to proximity.

        Returns:
            Steering adjustment in [-1, 1] (normalized).
        """
        x = vehicle_state.get('x', 0.0)
        y = vehicle_state.get('y', 0.0)
        psi = vehicle_state.get('psi', 0.0)
        v = vehicle_state.get('u', 0.0)

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Act on the single nearest obstacle that is both close and on a
        # collision course. Summing an avoidance push over every nearby
        # obstacle would, in a dense field, produce a large standing bias that
        # fights the driver's own weaving; restricting the layer to an imminent
        # collision leaves ordinary driving -- steering around obstacles well
        # before they threaten -- entirely to the driver.
        nearest = None
        for (obs_x, obs_y, obs_r) in obstacles:
            dx_world = obs_x - x
            dy_world = obs_y - y
            dx_body = dx_world * cos_psi + dy_world * sin_psi   # forward
            dy_body = -dx_world * sin_psi + dy_world * cos_psi  # left-positive
            if dx_body < 1.0:
                continue
            safe_r = obs_r + self.vehicle_radius + self.obstacle_buffer
            # Only obstacles we'd actually hit on the current heading.
            if abs(dy_body) > safe_r:
                continue
            dist = np.sqrt(dx_body**2 + dy_body**2)
            # Engagement range of roughly two vehicle lengths, growing with
            # speed: wide enough to act in time, narrow enough that the layer
            # stays silent across the rest of the field.
            react_range = safe_r + 3.0 + v * 0.4
            if dist > react_range:
                continue
            if nearest is None or dist < nearest[0]:
                nearest = (dist, dy_body, safe_r, react_range)

        if nearest is None:
            return 0.0
        dist, dy_body, safe_r, react_range = nearest
        proximity = np.clip(1.0 - (dist - safe_r) / max(react_range - safe_r, 0.1), 0.0, 1.0)
        # Steer away from the obstacle (dy_body > 0 places it to the left, so
        # the correction is to the right, i.e. negative). The side decision is
        # hysteretic: once committed, the side is held until the obstacle lies
        # clearly (> 0.8 m) on the other side, so a near-dead-ahead obstacle
        # cannot make the correction alternate between the two escape routes.
        if dy_body > 0.8:
            self._reactive_dir = -1.0
        elif dy_body < -0.8:
            self._reactive_dir = 1.0
        elif self._reactive_dir == 0.0:
            self._reactive_dir = -1.0   # first commit: default to the right
        return float(np.clip(self._reactive_dir * proximity * 0.4, -1.0, 1.0))

    def filter(self,
               desired_steering: float,
               desired_throttle: float,
               desired_brake: float,
               vehicle_state: dict,
               obstacles: List[Tuple[float, float, float]] = None,
               terrain_roughness: float = 0.0) -> SafetyFilterResult:
        """
        Apply DOB-CBF safety filter to desired control inputs.

        Args:
            desired_steering: Desired steering input [-1, 1] (normalized)
            desired_throttle: Desired throttle input [0, 1]
            desired_brake: Desired brake input [0, 1]
            vehicle_state: Dict with keys:
                'x', 'y': world position (m)
                'psi': heading angle (rad)
                'u': longitudinal velocity (m/s)
                'v': lateral velocity (m/s)
                'omega': yaw rate (rad/s)
                'delta': current steering angle (rad)
            obstacles: List of (x, y, radius) tuples
            terrain_roughness: Terrain roughness metric (m)

        Returns:
            SafetyFilterResult with filtered inputs and diagnostics.
        """
        t_start = time.time()
        self._filter_count += 1

        if obstacles is None:
            obstacles = []

        # Normalize obstacles to (x, y, r) plus a parallel is-vehicle flag.
        # Callers may pass a 4th element (True for a moving vehicle); when it
        # is absent the flag is inferred from the radius, since rocks are small
        # (below ~0.7 m) and an HMMWV bounds at ~2.2 m. The flag lets the QP
        # prefer steering around a static rock while weighting braking equally
        # for a vehicle, where stopping is an equally valid response.
        _obs_in = obstacles
        obstacles = []
        obs_is_vehicle = []
        for _o in _obs_in:
            obstacles.append((_o[0], _o[1], _o[2]))
            obs_is_vehicle.append(bool(_o[3]) if len(_o) >= 4 else (_o[2] >= 1.5))

        # Extract vehicle state
        x = vehicle_state.get('x', 0.0)
        y = vehicle_state.get('y', 0.0)
        psi = vehicle_state.get('psi', 0.0)
        v_meas = float(vehicle_state.get('u', 0.5))  # true speed, unclamped
        v = max(v_meas, 0.1)  # Avoid /0
        v_lat = vehicle_state.get('v', 0.0)
        omega = vehicle_state.get('omega', 0.0)
        delta = vehicle_state.get('delta', 0.0)

        # Update internal beta from measured steering angle
        self._beta = delta

        # Detection of a broken steering rack belongs to the plant node, which
        # can read the measured front road-wheel angle. Here `delta` is the
        # commanded angle, which carries no information about whether the
        # actuator followed it.

        # Teleop: stale command detection — emergency brake if no recent cmds
        if self._is_command_stale():
            self._modify_count += 1
            safe_result = SafetyFilterResult(
                steering=desired_steering,  # hold last steering
                throttle=0.0,
                braking=1.0,
                was_modified=True,
                active_constraints=0,
                solve_time_ms=0.0,
                v_max_terrain=0.0,
                safety_margin=0.0,
                dob_norm=0.0,
            )
            self._last_result = safe_result
            if self._filter_count % 20 == 0:
                age = time.time() - self._last_cmd_wall if self._last_cmd_wall else float('inf')
                print(f"  [CBF #{self._filter_count}] STALE COMMAND — age={age:.2f}s > "
                      f"timeout={self._stale_cmd_timeout:.1f}s  ** EMERGENCY BRAKE **")
            return safe_result

        # Teleop: compute delay-inflated obstacle buffer for this step
        effective_buffer = self._effective_obstacle_buffer(v)

        # Apply delay compensation
        comp_throttle, comp_steering = self.delay_comp.update(
            desired_throttle, desired_steering
        )

        # Compute net throttle command for DOB: alpha in [-1, 1]
        if desired_brake > 0.05:
            alpha_cmd = -desired_brake
        else:
            alpha_cmd = desired_throttle

        # Query NN tire model for current state (used by DOB + CBF)
        nn_forces = self._compute_nn_tire_forces(v, v_lat, omega, delta)
        nn_steer_sens = self._compute_nn_steering_sensitivity(v, v_lat, omega, delta)

        # ---- Longitudinal model: a_x(alpha) = f_drag + g_thrust*alpha + hdv0 ----
        # The barrier's autonomous term must never claim deceleration the
        # vehicle would not sustain with the pedals released. Two choices make
        # that a provable inequality for an obstacle ahead rather than a
        # calibration hope.
        #
        # First, the nominal is evaluated at the alpha the plant actually
        # EXECUTED (last step's filtered output), not the driver's desired
        # command. During an intervention the two differ by up to the full
        # pedal range, and a nominal at the desired command books the gap --
        # measured at up to 8 m/s^2 -- as free deceleration, deactivating the
        # constraint the filter is in the middle of enforcing and releasing
        # the driver's throttle back toward the obstacle in a release/re-brake
        # limit cycle.
        #
        # Second, the coefficient is one-sided. Under executed braking the
        # 1 g bound upper-bounds any deceleration this vehicle can produce on
        # any soil, so the residual can only bias toward a phantom forward
        # push, which tightens the barrier. Under executed throttle the
        # coefficient is zero: throttle only ever adds forward acceleration,
        # so the whole measured acceleration is carried as the autonomous
        # term, overstating what alpha = 0 would sustain -- again the
        # conservative direction. (A secant estimate here was measured
        # permissive at speed, where the power-limited plant delivers less
        # than the tire-capacity secant and the shortfall was booked as free
        # deceleration.) The barrier's a_x_auto reduces to
        # a_meas - g_nom*min(alpha_exec, 0), continuous across the pedal
        # transition and independent of the desired command.
        f_drag = self._resistive_acceleration(nn_forces, v_meas)
        g_thrust = self._longitudinal_authority(v, v_lat, omega, delta,
                                                nn_forces)

        alpha_exec = float(np.clip(self._alpha, -1.0, 1.0))
        g_nom = 0.0 if alpha_exec >= 0 else self._BRAKE_NOMINAL_ENVELOPE
        hdv0 = self.dob.update(v, alpha_exec, self.control_dt,
                               f_nom_override=f_drag + g_nom * alpha_exec)

        # Terrain-aware speed limit
        v_max_terrain = self._compute_terrain_speed_limit(v, terrain_roughness, delta)

        # ---- Position-based QP formulation ----
        # QP variables: u = [steer_out, alpha_out]
        #   steer_out: normalized steering position [-1, 1]
        #   alpha_out: throttle/brake [-1, 1]
        # u_desired = [desired_steering, alpha_cmd]
        #
        # The decision variable is the steering position rather than its rate.
        # Optimizing over the rate would place a barrier-tracking feedback loop
        # in series with the QP's own correction; the two then oppose each
        # other on the steering channel, and the QP settles on braking even
        # where a steering correction is available.

        desired_alpha = alpha_cmd
        u_desired = np.array([desired_steering, desired_alpha])

        # Current road-wheel angle for barrier linearization
        # Use last applied steering as best estimate of current vehicle state
        current_beta = self._beta

        # Build heading-aligned weight matrix P
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        R = np.array([[cos_psi, sin_psi],
                       [-sin_psi, cos_psi]])
        Q_barrier = np.diag([self.w_long, self.w_lat])
        P = R.T @ Q_barrier @ R

        # NN-informed dynamics for CBF constraints
        # Autonomous lateral acceleration and yaw moment from NN
        if nn_forces is not None:
            # NN provides actual tire-generated accelerations at current state
            ay_tire = nn_forces['ay_nn']          # lateral accel from Fy (m/s^2)
            alpha_dot_tire = nn_forces['alpha_dot']  # yaw accel from Mz (rad/s^2)
        else:
            ay_tire = 0.0
            alpha_dot_tire = 0.0

        # Steering sensitivity: how much lateral force / yaw per unit delta change
        if nn_steer_sens is not None:
            dFy_ddelta, dMz_ddelta = nn_steer_sens
            # Convert to accelerations per unit dbeta (dbeta = ddelta/dt, so
            # sensitivity is per unit delta; multiply by control_dt handled in QP)
            day_ddelta = dFy_ddelta / self.M
            dalpha_dot_ddelta = dMz_ddelta / self.Izz
        else:
            # Kinematic fallback: d(omega)/dt ~ v/L * ddelta
            day_ddelta = 0.0
            dalpha_dot_ddelta = v / self.L

        # Longitudinal control authority: how much acceleration one unit of
        # alpha buys, from the state-only drive-slip secant. A function of the
        # command here would make the safe set change shape with the pedal.
        accel_gain = g_thrust

        # Build CBF constraints for each nearby obstacle
        A_ineq_list = []
        b_ineq_list = []
        min_h = float('inf')
        threat_is_vehicle = False   # type of the most-threatening (min-h) obstacle
        self._pending_obs_logs = []

        # Expand obstacle filtering range to cover delay-inflated buffer
        delay_inflate = effective_buffer - self.obstacle_buffer
        r_precpt_eff = self.r_precpt + delay_inflate

        for _oi, (obs_x, obs_y, obs_r) in enumerate(obstacles):
            # Distance check -- skip far obstacles
            dd = (x - obs_x)**2 + (y - obs_y)**2
            if dd > r_precpt_eff**2:
                continue

            safe_r = obs_r + self.vehicle_radius + effective_buffer

            # ----------------------------------------------------------
            # Position-based CBF: steering enters through h_dot (1st
            # derivative) via omega = v/L * tan(beta).  Throttle enters
            # through h_ddot (2nd derivative) via acceleration.
            #
            # Constraint (2nd-order CBF):
            #   h_ddot + alpha2 * h_dot(steer_out) + alpha1 * h >= 0
            #
            # h_dot is linearized around current_beta:
            #   h_dot(s) ≈ h_dot_0 + A_steer * (s - s_current)
            # where s is the normalized steering output and
            #   A_steer = dh_dot/ds = dh_dot/d(omega) * d(omega)/d(s)
            # ----------------------------------------------------------

            beta = current_beta
            w1 = self.w_long
            w2 = self.w_lat
            bias = self.forward_bias

            # Body-frame decomposition
            raw_dx = x - obs_x
            raw_dy = y - obs_y
            d_along = cos_psi * raw_dx + sin_psi * raw_dy + bias
            d_cross = -sin_psi * raw_dx + cos_psi * raw_dy

            # Barrier value
            h = w1 * d_along**2 + w2 * d_cross**2 - safe_r**2
            if h < min_h:
                min_h = h
                threat_is_vehicle = obs_is_vehicle[_oi]

            # Euclidean distance from vehicle CG to obstacle center
            dist_eucl = np.sqrt(dd)

            # Kinematic quantities at current beta
            # Use actual measured yaw rate for barrier dynamics — the
            # kinematic bicycle model (omega_kin = v/L*tan(delta)) is
            # inaccurate on low-friction surfaces where tire slip causes
            # the real yaw rate to diverge from the kinematic prediction.
            d_along_unbiased = d_along - bias

            # The measured speed, not the /0 guard. Reusing the guard as a
            # kinematic rate models a stopped vehicle as closing on the
            # obstacle at 0.1 m/s, which alone is enough to keep the barrier
            # constraint active and its argument negative at a standstill.
            d_along_dot = v_meas + omega * d_cross
            d_cross_dot = v_lat - omega * d_along_unbiased

            h_dot = 2 * w1 * d_along * d_along_dot + 2 * w2 * d_cross * d_cross_dot

            # Drop obstacles the vehicle is receding from (h increasing while
            # the barrier is already satisfied).
            if h_dot > 0 and h > 0:
                continue

            # Drop an obstacle behind the vehicle only when it is also receding
            # (h_dot > 0), that is, already passed. One behind but closing
            # (h_dot < 0) stays in the QP, so coverage is omnidirectional on
            # demand while the forward bias continues to concentrate the scarce
            # grip-limited margin in the direction of travel.
            if d_along_unbiased > 1.0 and h_dot > 0:
                continue

            # Drop a receding obstacle even at h < 0, where the vehicle has
            # passed it but the barrier ellipse still overlaps.
            if h_dot > 0 and d_along_unbiased > -0.5:
                continue

            # Strict barrier value. Feasibility when deep inside the barrier is
            # handled by the QP's penalized slack variable (below), not by
            # relaxing the constraint here -- so the CBF never abandons its own
            # forward-invariance guarantee. A mild floor only keeps the
            # constraint magnitude numerically sane at gross violation.
            h_eff = max(h, -safe_r**2)

            # Steering sensitivity: dh_dot / d(steer_normalized)
            # Uses kinematic model for control authority (how steering
            # changes omega), which is a vehicle design property.
            tan_beta = np.tan(beta) if abs(beta) < 1.5 else np.sign(beta) * 1e3
            sec2_beta = 1.0 + tan_beta**2
            d_omega_d_steer = v / self.L * sec2_beta * self.max_road_steer_angle

            # dh_dot/d(omega) = 2*d_cross*(w1*d_along - w2*d_along_unbiased)
            dh_dot_d_omega = 2 * d_cross * (w1 * d_along - w2 * d_along_unbiased)

            A_steer = dh_dot_d_omega * d_omega_d_steer

            # Autonomous h_ddot (second derivative at zero control). The
            # longitudinal part is the control-independent half of the shared
            # model: resistance plus the observer's residual. Omitting f_drag
            # here while the observer subtracts it leaves the nominal counted
            # once with the wrong sign, so hdv0 alone would report a phantom
            # acceleration of -f_drag at every steady speed.
            a_x_auto = f_drag + hdv0
            omega_dot_auto = a_x_auto / self.L * tan_beta if abs(v_meas) > 0.1 else 0.0
            d_along_ddot_auto = (a_x_auto
                                 + d_cross_dot * omega
                                 + d_cross * omega_dot_auto)
            d_cross_ddot_auto = (-d_along_dot * omega
                                 - d_along_unbiased * omega_dot_auto)
            # The velocity-squared (centrifugal) terms are omitted from
            # h_ddot_auto. The full chain-rule expansion of d²h/dt² contains
            # d_along_dot² and d_cross_dot², both non-negative, which inflate
            # h_ddot_auto — d_cross_dot² especially so in a turn — and leave
            # the constraint non-binding: the barrier reads as improving on its
            # own while the vehicle is only rotating its body frame, not gaining
            # clearance. Retaining the position-times-acceleration terms alone
            # keeps the constraint conservative enough to command braking or
            # steering while either still has authority.
            h_ddot_auto = (2 * w1 * d_along * d_along_ddot_auto
                         + 2 * w2 * d_cross * d_cross_ddot_auto)

            # Throttle sensitivity: alpha affects h_ddot through acceleration
            A_alpha = 2 * w1 * d_along * accel_gain

            # Position-based CBF constraint using effective barrier:
            # h_ddot_auto + A_alpha*alpha + alpha2*(h_dot_0 + A_steer*(s - s_cur)) + alpha1*h_eff >= 0
            #
            # Rearrange to: -[alpha2*A_steer, A_alpha] @ [s, alpha] <= psi0
            s_current = self._beta / self.max_road_steer_angle
            psi0 = (h_ddot_auto
                    + self.alpha2 * (h_dot - A_steer * s_current)
                    + self.alpha1 * h_eff)
            psi1 = np.array([self.alpha2 * A_steer, A_alpha])

            # QP variable u = [steer_out, alpha_out, slack]. The CBF constraint
            #   psi1 @ [steer, alpha] + slack >= -psi0
            # is kept for every obstacle; the (heavily penalized) slack only
            # activates when the barrier is already violated and cannot be fully
            # recovered, so the QP always drives toward safety -- braking when
            # steering alone cannot -- instead of dropping the constraint.
            A_ineq_list.append(np.array([-psi1[0], -psi1[1], -1.0]))
            b_ineq_list.append(psi0)

            # Buffer obstacle data for CSV (logged only if intervention occurs)
            self._pending_obs_logs.append((
                self._filter_count, obs_x, obs_y, obs_r, np.sqrt(dd),
                d_along, d_cross, d_along_unbiased,
                h, h_dot, h_ddot_auto,
                A_steer, A_alpha, psi0,
                psi1[0], psi1[1],
            ))

        # Speed limit constraint (terrain-aware). Column 3 is the CBF slack,
        # which does not enter the speed limit. Throttle is what drives the
        # speed up, so the row bounds alpha from *above*; a negated
        # coefficient here turns the cap into a floor that strips the
        # driver's braking and, past the limit, demands throttle. The bound
        # is floored at the most negative alpha the *rate* limits allow this
        # step -- not merely at full brake -- because a row the rate rows
        # cannot satisfy makes the whole QP infeasible and trips the
        # emergency fallback, which discards every obstacle constraint. A
        # step arriving over the cap at high throttle then sheds alpha at the
        # rate limit instead of losing its steering for a step.
        if v > v_max_terrain * 0.9:
            speed_margin = v_max_terrain - v
            accel_for_speed = accel_gain
            _a_cur = float(np.clip(self._alpha, -1.0, 1.0))
            _da = self.max_alpha_rate * self.control_dt
            alpha_floor = max(-1.0, _a_cur - _da)
            A_ineq_list.append(np.array([0.0, accel_for_speed, 0.0]))
            b_ineq_list.append(
                max(self.alpha1 * speed_margin,
                    accel_for_speed * alpha_floor))

        # The barrier constraints are never dropped, not even at near-zero
        # speed inside the barrier. Dropping them there would let a persistent
        # operator throttle creep the vehicle into a head-on obstacle. The
        # penalized slack below supplies feasibility instead, so the optimizer
        # brakes to hold the standoff without the constraint being relaxed.

        # Solve QP
        was_modified = False
        active_constraints = 0
        qp_success = True

        if len(A_ineq_list) > 0:
            A_ineq = np.array(A_ineq_list)   # 3 columns: [steer, alpha, slack]
            b_ineq = np.array(b_ineq_list)

            # Actuator limits on [steer, alpha] plus a physical steering/throttle
            # rate limit baked into the QP so the CBF only plans control it can
            # actually execute. The slack (column 3) is bounded below by 0 only.
            s_cur = float(np.clip(self._beta / self.max_road_steer_angle, -1.0, 1.0))
            dmax = self.max_steer_rate * self.control_dt / self.max_road_steer_angle
            a_cur = float(np.clip(self._alpha, -1.0, 1.0))
            da = self.max_alpha_rate * self.control_dt
            A_limits = np.array([
                [1.0, 0.0, 0.0],   # steer_out <= 1
                [-1.0, 0.0, 0.0],  # -steer_out <= 1
                [0.0, 1.0, 0.0],   # alpha_out <= 1
                [0.0, -1.0, 0.0],  # -alpha_out <= 1
                [1.0, 0.0, 0.0],   # steer rate up
                [-1.0, 0.0, 0.0],  # steer rate down
                [0.0, 1.0, 0.0],   # alpha rate up
                [0.0, -1.0, 0.0],  # alpha rate down
                [0.0, 0.0, -1.0],  # -slack <= 0  (slack >= 0)
            ])
            b_limits = np.array([1.0, 1.0, 1.0, 1.0,
                                 min(1.0, s_cur + dmax),
                                 min(1.0, dmax - s_cur),
                                 min(1.0, a_cur + da),
                                 min(1.0, da - a_cur),
                                 0.0])

            A_all = np.vstack([A_ineq, A_limits])
            b_all = np.hstack([b_ineq, b_limits])

            u_desired3 = np.array([desired_steering, desired_alpha, 0.0])

            # Cost: min ||[steer,alpha] - u_desired||^2_W + rho * slack^2.
            # The steer/alpha weights express a tracking preference only
            # (steering cheaper than braking for a static rock); they do not
            # weaken the barrier, which is a hard constraint softened solely by
            # the heavily penalized slack. Where steering cannot maintain the
            # barrier -- a head-on obstacle on low-grip soil, for instance --
            # the optimizer brakes to minimize the violation, with no special
            # case in the code.
            if self.cbf_flavor == 'steer_priority':
                Wd = [500.0, 1.0]
            elif self.cbf_flavor == 'throttle_priority':
                Wd = [1.0, 500.0]
            elif self.cbf_flavor == 'balance' and not threat_is_vehicle:
                Wd = [1.0, 10.0]
            else:
                Wd = [1.0, 1.0]
            rho_slack = 1.0e4
            W3 = np.diag([Wd[0], Wd[1], rho_slack])
            H = 2.0 * W3
            f = -2.0 * W3 @ u_desired3

            def qp_objective(u_var):
                return 0.5 * u_var @ H @ u_var + f @ u_var

            constraints = [{
                'type': 'ineq',
                'fun': lambda u_var: b_all - A_all @ u_var
            }]

            result = minimize(
                qp_objective, u_desired3,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 60, 'ftol': 1e-8}
            )

            if result.success:
                u_safe = result.x[:2]
                diff = np.linalg.norm(u_safe - u_desired)
                was_modified = diff > 1e-4
                if was_modified:
                    self._modify_count += 1
                    active_constraints = sum(
                        1 for i in range(len(A_ineq))
                        if A_ineq[i] @ result.x > b_ineq[i] - 1e-3
                    )
            else:
                # Solver failure: emergency brake, hold steering.
                u_safe = np.array([desired_steering, -1.0])
                was_modified = True
                qp_success = False
                self._modify_count += 1
                active_constraints = len(A_ineq_list)
        else:
            u_safe = u_desired

        # Extract filtered controls directly (position-based, no integration)
        safe_steering = np.clip(u_safe[0], -1.0, 1.0)
        safe_alpha = u_safe[1]

        # Reactive steering layer: add avoidance commands for nearby obstacles.
        # This handles head-on obstacles where the CBF geometric singularity
        # prevents the QP from choosing steering, and provides guidance during
        # low-speed creeping through obstacle fields.
        if len(obstacles) > 0 and min_h < 2.0:
            reactive_steer = self._compute_reactive_steering(vehicle_state, obstacles)
            if abs(reactive_steer) > 0.01:
                # A driver already steering away from the obstacle is not
                # overridden; the layer engages only when the driver is passive
                # or steering toward it.
                driver_avoiding = (abs(desired_steering) > 0.2
                                   and np.sign(desired_steering) == np.sign(reactive_steer))
                if not driver_avoiding:
                    safe_steering = np.clip(safe_steering + reactive_steer, -1.0, 1.0)
                    if not was_modified:
                        was_modified = True
                        self._modify_count += 1

        # Teleop: forward-predict over delay horizon and emergency-brake
        # if the QP-safe output still leads to a collision within the RTT
        if self._teleop_delay > 0.0 and len(obstacles) > 0:
            pred_horizon = 2.0 * self._teleop_delay + 0.3  # RTT + reaction
            pred_steps = max(int(pred_horizon / 0.1), 1)
            preds = self.predict_state(
                vehicle_state, safe_steering,
                max(safe_alpha, 0.0), dt=0.1, steps=pred_steps)
            collision, t_collide = self.check_predicted_collision(preds, obstacles)
            if collision:
                # Override to emergency brake; keep steering
                safe_alpha = -1.0
                was_modified = True
                if not qp_success or active_constraints == 0:
                    active_constraints = 1

        # Steering-actuator model. A physical steering rack cannot snap between
        # angles, so the commanded steering passes through (1) a hard slew-rate
        # cap and (2) a first-order lag of time constant steer_tau. The lag
        # matters because the QP and the reactive layer can select opposite
        # avoidance solutions on consecutive solves a few milliseconds apart;
        # that alternation is high-frequency and the lag attenuates it into a
        # single steady command, while a sustained operator or avoidance input
        # is low-frequency and passes through with only the actuator's small
        # delay. The wall-clock interval between calls is used, so both the cap
        # and the lag are correct at the rate the filter actually runs.
        dt_call = (t_start - self._last_filter_wall) if self._last_filter_wall else self.control_dt
        self._last_filter_wall = t_start
        dt_call = float(np.clip(dt_call, 0.005, 0.1))
        prev = self._last_safe_steering
        # 1) first-order lag toward the full target -- passes the steady (human /
        #    sustained-avoidance) component, attenuates the high-frequency QP/
        #    reactive flip-flop.
        a = dt_call / (self.steer_tau + dt_call)
        smoothed = prev + a * (safe_steering - prev)
        # 2) hard slew cap as a physical backstop (won't bind on normal input).
        max_dsteer = self.max_steer_rate * dt_call / self.max_road_steer_angle
        smoothed = float(np.clip(smoothed, prev - max_dsteer, prev + max_dsteer))
        if abs(smoothed - safe_steering) > 1e-4:
            was_modified = True
        safe_steering = smoothed
        self._last_safe_steering = safe_steering

        # Track applied steering for next call's linearization point
        self._beta = safe_steering * self.max_road_steer_angle
        self._alpha = safe_alpha

        if safe_alpha >= 0:
            safe_throttle = min(safe_alpha, 1.0)
            safe_brake = 0.0
        else:
            safe_throttle = 0.0
            safe_brake = min(abs(safe_alpha), 1.0)

        # --- Rear-approach avoidance ---
        # A vehicle closing from behind cannot always be avoided, but slowing
        # down raises the rear-end risk rather than lowering it. The nearest
        # obstacle behind the ego and inside its lane is located; if it is
        # closing within the threat distance, deceleration is refused, and when
        # the path ahead is clear the throttle ramps with proximity to open the
        # gap, including from a standstill.
        rear_dist = float('inf')
        fwd_dist = float('inf')
        cps, sps = np.cos(psi), np.sin(psi)
        for (obs_x, obs_y, obs_r) in obstacles:
            rx, ry = obs_x - x, obs_y - y
            lon = rx * cps + ry * sps            # +forward in body frame
            lat = -rx * sps + ry * cps
            d = float(np.hypot(rx, ry)) - obs_r
            if abs(lat) < self._rear_half_w:
                if lon < -1.0:
                    rear_dist = min(rear_dist, d)
                elif lon > 0.0:
                    fwd_dist = min(fwd_dist, d)
        rear_closing = rear_dist < self._prev_rear_dist - 0.02
        self._prev_rear_dist = rear_dist if np.isfinite(rear_dist) else float('inf')
        if rear_dist < self._rear_threat_dist and rear_closing:
            safe_brake = 0.0                     # never brake into a rear approach
            if fwd_dist > self._rear_threat_dist:  # path ahead clear -> accelerate away
                escape = float(np.clip(1.2 - rear_dist / self._rear_threat_dist, 0.4, 1.0))
                safe_throttle = max(safe_throttle, desired_throttle, escape)
            else:                                  # blocked ahead -> at least don't slow
                safe_throttle = max(safe_throttle, desired_throttle)
            was_modified = True

        solve_time = (time.time() - t_start) * 1000

        self._last_result = SafetyFilterResult(
            steering=safe_steering,
            throttle=safe_throttle,
            braking=safe_brake,
            was_modified=was_modified,
            active_constraints=active_constraints,
            solve_time_ms=solve_time,
            v_max_terrain=v_max_terrain,
            safety_margin=min_h if min_h < float('inf') else float('inf'),
            dob_norm=abs(hdv0),
        )

        # CSV log only when filter intervenes
        if was_modified:
            self._log_csv_main(
                self._filter_count, x, y, psi, v, self._beta, hdv0,
                len(obstacles), len(A_ineq_list), min_h,
                desired_steering, desired_throttle, desired_brake,
                safe_steering, safe_throttle, safe_brake,
                alpha_cmd, safe_alpha,
                was_modified, active_constraints, solve_time,
                qp_success, v_max_terrain,
            )
            # Flush buffered obstacle logs for this intervention
            for obs_log in self._pending_obs_logs:
                self._log_csv_obstacle(*obs_log, skipped_hdot=False, skipped_far=False)

        # Diagnostic logging (only when filter intervenes)
        if was_modified and self._filter_count % 20 == 0:
            n_obs = len(obstacles)
            mod_tag = " ** MODIFIED **" if was_modified else ""
            n_constraints = len(A_ineq_list)
            delay_tag = f" delay={self._teleop_delay*1000:.0f}ms buf={effective_buffer:.2f}m" if self._teleop_delay > 0 else ""
            print(f"  [CBF #{self._filter_count}] v={v:.2f} pos=({x:.1f},{y:.1f}) psi={np.degrees(psi):.1f}°"
                  f" | obs={n_obs} constraints={n_constraints} min_h={min_h:.2f} dob={hdv0:+.2f}"
                  f" | IN steer={desired_steering:+.3f} thr={desired_throttle:.3f} brk={desired_brake:.3f}"
                  f" | OUT steer={safe_steering:+.3f} thr={safe_throttle:.3f} brk={safe_brake:.3f}"
                  f" | alpha_cmd={alpha_cmd:+.3f} -> safe_alpha={safe_alpha:+.3f}"
                  f" | v_max_t={v_max_terrain:.1f}{delay_tag}{mod_tag}")

        return self._last_result

    def _compute_terrain_speed_limit(self, speed: float, roughness: float,
                                      delta: float) -> float:
        """
        Compute terrain-aware maximum safe speed.

        Uses terrain roughness and (optionally) NN tire force
        predictions to determine the maximum speed that maintains vehicle
        stability on the current terrain.
        """
        # Base speed limit from terrain roughness
        if roughness > 0.01:
            roughness_factor = 1.0 / (1.0 + 10.0 * roughness)
            v_max_rough = self.max_speed * roughness_factor
        else:
            v_max_rough = self.max_speed

        # NN tire model traction check (if available)
        v_max_traction = self.max_speed
        if self.nn_casadi is not None:
            v_max_traction = self._nn_traction_speed_limit(speed, delta)

        return max(min(v_max_rough, v_max_traction, self.max_speed), 2.0)

    def _nn_traction_speed_limit(self, speed: float, delta: float) -> float:
        """Estimate the maximum speed at which traction remains available.

        The query uses a fixed moderate steering angle representative of an
        evasive manoeuvre rather than the driver's current angle, so the limit
        expresses the traction budget reserved for avoidance rather than the
        demand of the turn presently being driven.
        """
        g = 9.81
        Fz_f = self.M * g * self.Lr / self.L / 2.0
        Fz_r = self.M * g * self.Lf / self.L / 2.0

        alpha_max = 0.15  # ~8.6 deg

        try:
            _tkw = {}
            if self._use_terrain_nn and self._terrain_n is not None:
                _tkw = dict(n_terrain=self._terrain_n,
                            terrain_params=self._terrain_params)
            _, Fy_f = self.nn_casadi.predict_numeric(alpha_max, Fz_f, max(speed, 2.0), 0.0, **_tkw)
            _, Fy_r = self.nn_casadi.predict_numeric(alpha_max, Fz_r, max(speed, 2.0), 0.0, **_tkw)

            Fy_max = 2.0 * (abs(Fy_f) + abs(Fy_r))
            if self._use_grip_scale:
                Fy_max *= self._grip_scale
            ay_max = Fy_max / self.M

            # Use a fixed moderate evasion angle (~5 deg) to set the speed
            # limit.  This avoids penalizing the driver's actual steering.
            evasion_delta = 0.09  # ~5 deg road wheel
            R = self.L / max(np.tan(evasion_delta), 0.01)
            v_max = np.sqrt(ay_max * R)

            return min(v_max, self.max_speed)

        except Exception:
            return self.max_speed

    def predict_state(self, state: dict, steering: float, throttle: float,
                      dt: float = 0.1, steps: int = 5) -> List[dict]:
        """Forward-simulate vehicle state for look-ahead collision checking."""
        x = state.get('x', 0.0)
        y_pos = state.get('y', 0.0)
        psi = state.get('psi', 0.0)
        u = state.get('u', 0.5)

        delta = steering * self.max_road_steer_angle
        ax = 3.0 * throttle

        predictions = []
        for _ in range(steps):
            x += u * np.cos(psi) * dt
            y_pos += u * np.sin(psi) * dt
            psi += (u / self.L) * np.tan(delta) * dt
            u += ax * dt
            u = max(u, 0.0)

            predictions.append({
                'x': x, 'y': y_pos, 'psi': psi, 'u': u
            })

        return predictions

    def check_predicted_collision(self, predictions: List[dict],
                                   obstacles: List[Tuple[float, float, float]]) -> Tuple[bool, float]:
        """Check if predicted trajectory collides with any obstacle."""
        for i, pred in enumerate(predictions):
            for obs_x, obs_y, obs_r in obstacles:
                safe_r = obs_r + self.vehicle_radius + self.obstacle_buffer
                dist_sq = (pred['x'] - obs_x)**2 + (pred['y'] - obs_y)**2
                if dist_sq < safe_r**2:
                    return True, i * 0.1
        return False, float('inf')

    @property
    def intervention_rate(self) -> float:
        """Fraction of filter calls that modified the input (0-1)."""
        if self._filter_count == 0:
            return 0.0
        return self._modify_count / self._filter_count

    def get_diagnostics(self) -> dict:
        """Get diagnostic info for logging."""
        result = self._last_result
        return {
            'filter_calls': self._filter_count,
            'interventions': self._modify_count,
            'intervention_rate': self.intervention_rate,
            'last_solve_ms': result.solve_time_ms if result else 0.0,
            'last_modified': result.was_modified if result else False,
            'last_v_max_terrain': result.v_max_terrain if result else self.max_speed,
            'last_safety_margin': result.safety_margin if result else float('inf'),
            'last_active_constraints': result.active_constraints if result else 0,
            'last_dob_norm': result.dob_norm if result else 0.0,
        }
