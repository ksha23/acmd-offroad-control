#!/usr/bin/env python3
"""
MPC Helper Classes
==================

Support classes for the MPC controller node: transport-delay estimation and
compensation, integration of the solver's rate commands into vehicle inputs,
finite-difference rate features, and tracking analytics.

They live apart from the controller node so that scripts and analyses can
reuse them without importing the node, and so no import cycle is required.
"""

import os as _os, sys as _sys  # flat-import bootstrap (simulation/flatpath.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import flatpath  # noqa: E402,F401
import math

import numpy as np


# =============================================================================
# Transport delay estimator
# =============================================================================

class DelayEstimator:
    """Exponential-moving-average estimator for one-way transport delay.

    Measures the wall-clock difference between the state's wall_time stamp
    and the local receive time.  This gives approximately the one-way
    sim→controller latency.  The full round-trip (state arrives, MPC solves,
    command sent back, sim applies it) is estimated as:

        τ_roundtrip ≈ τ_one_way + t_solve + τ_one_way ≈ 2·τ_one_way + t_solve

    Delay compensation in the MPC requires the interval between the instant a
    state is measured and the instant the resulting command is applied:

        τ_compensation = τ_one_way + t_solve + τ_one_way
                       ≈ 2 · τ_one_way + t_solve_avg

    Solve time is tracked separately and τ_compensation is exposed as a
    property.
    """

    def __init__(self, alpha: float = 0.15, initial_delay: float = 0.02):
        self.alpha = alpha
        self.one_way_delay = initial_delay  # seconds
        self.solve_time = 0.01  # seconds (initial guess)

    def update_transport(self, state_wall_time: float, recv_wall_time: float):
        """Update one-way delay estimate from a received state message."""
        measured = max(recv_wall_time - state_wall_time, 0.0)
        self.one_way_delay += self.alpha * (measured - self.one_way_delay)

    def update_solve(self, solve_seconds: float):
        self.solve_time += self.alpha * (solve_seconds - self.solve_time)

    @property
    def compensation_delay(self) -> float:
        """Total delay to compensate for in MPC (seconds)."""
        return 2.0 * self.one_way_delay + self.solve_time


# =============================================================================
# State predictor (propagates state forward through delay using dynamics)
# =============================================================================

class StatePredictor:
    """Propagates the vehicle state forward through the transport delay.

    Given the measured state z(t_meas) and the buffer of commands already sent
    over the interval [t_meas, t_meas + τ_d], this integrates a linear-tire
    bicycle model to obtain z(t_meas + τ_d), the state the MPC should treat as
    its initial condition.
    """

    def __init__(self, vehicle_params: dict, dt_prop: float = 0.01):
        p = vehicle_params
        self.M = p["M"]
        self.Izz = p["Izz"]
        self.Lf = p["Lf"]
        self.Lr = p["Lr"]
        # Linear cornering stiffness, used exclusively by this short-horizon
        # delay prediction; it is not the tire model the MPC optimizes over.
        self.Cf = 80000.0
        self.Cr = 80000.0
        self.dt = dt_prop

    @staticmethod
    def _sample_command(control_samples, t_query: float) -> tuple[float, float]:
        """Return (delta_dot, Jx) command active at query time."""
        if not control_samples:
            return 0.0, 0.0
        # If query is before the first sample, hold the earliest available.
        if t_query <= float(control_samples[0][0]):
            return float(control_samples[0][1]), float(control_samples[0][2])
        # Otherwise hold the most recent sample at/before query time.
        for t_cmd, d_dot, jx in reversed(control_samples):
            if float(t_cmd) <= t_query:
                return float(d_dot), float(jx)
        return float(control_samples[0][1]), float(control_samples[0][2])

    def propagate(
        self,
        z0: np.ndarray,
        control_buffer,
        delay_s: float,
        sim_time_s: float | None = None,
        command_lag_s: float = 0.0,
        return_last_cmd: bool = False,
    ):
        """Propagate state z0 forward by delay_s seconds.

        Args:
            z0: 8-state vector [x, y, psi, u, v, omega, delta, ax]
            control_buffer: deque of (sim_time, delta_dot, Jx) sorted by time.
            delay_s: Total delay to compensate (seconds).
            sim_time_s: State timestamp for z0 in simulation time.
            command_lag_s: Additional lag between command timestamp and
                application in plant dynamics (typically one-way delay).
            return_last_cmd: If True, also return the final sampled
                (delta_dot, Jx) used during propagation.

        Returns:
            z_pred: Predicted 8-state vector at t + delay_s.
            Optionally, (delta_dot_last, Jx_last) when return_last_cmd=True.
        """
        if delay_s <= 0:
            if return_last_cmd:
                return z0.copy(), 0.0, 0.0
            return z0.copy()

        z = z0.copy()
        n_steps = max(1, int(round(delay_s / self.dt)))
        dt = delay_s / n_steps

        buf_list = list(control_buffer)
        t0 = float(sim_time_s) if sim_time_s is not None else (
            float(buf_list[-1][0]) if buf_list else 0.0
        )
        lag = max(float(command_lag_s), 0.0)
        delta_dot_last = 0.0
        jx_last = 0.0

        for step_i in range(n_steps):
            t_query = t0 + step_i * dt - lag
            delta_dot_cmd, jx_cmd = self._sample_command(buf_list, t_query)
            delta_dot_last, jx_last = delta_dot_cmd, jx_cmd

            z = self._rk4_step(z, delta_dot_cmd, jx_cmd, dt)

        if return_last_cmd:
            return z, float(delta_dot_last), float(jx_last)
        return z

    def _rk4_step(self, z, delta_dot, Jx, dt):
        k1 = self._dynamics(z, delta_dot, Jx)
        k2 = self._dynamics(z + 0.5 * dt * k1, delta_dot, Jx)
        k3 = self._dynamics(z + 0.5 * dt * k2, delta_dot, Jx)
        k4 = self._dynamics(z + dt * k3, delta_dot, Jx)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _dynamics(self, z, delta_dot, Jx):
        _, _, psi, u, v, omega, delta, ax = z
        u_safe = max(abs(u), 0.5)
        Lf, Lr = self.Lf, self.Lr
        M, Izz = self.M, self.Izz

        alpha_f = delta - np.arctan2(v + Lf * omega, u_safe)
        alpha_r = -np.arctan2(v - Lr * omega, u_safe)
        # Clamp slip angles so the linear law cannot produce unbounded lateral
        # force as the speed approaches the u_safe floor.
        _alpha_max = 0.55
        alpha_f = float(max(-_alpha_max, min(_alpha_max, alpha_f)))
        alpha_r = float(max(-_alpha_max, min(_alpha_max, alpha_r)))
        Fyf = self.Cf * alpha_f
        Fyr = self.Cr * alpha_r

        dz = np.zeros(8)
        dz[0] = u * np.cos(psi) - (v + Lf * omega) * np.sin(psi)
        dz[1] = u * np.sin(psi) + (v + Lf * omega) * np.cos(psi)
        dz[2] = omega
        dz[3] = ax
        dz[4] = (Fyf + Fyr) / M - u * omega
        dz[5] = (Fyf * Lf - Fyr * Lr) / Izz
        dz[6] = delta_dot
        dz[7] = Jx
        return dz


# =============================================================================
# Quaternion helper
# =============================================================================

def quat_to_yaw(e0, e1, e2, e3):
    """Extract yaw angle from quaternion (Chrono convention: e0 is scalar)."""
    return np.arctan2(2.0 * (e0 * e3 + e1 * e2), 1.0 - 2.0 * (e2**2 + e3**2))


# =============================================================================
# Control integrator
# =============================================================================

class ControlIntegrator:
    """Integrates MPC rate commands (delta_dot, Jx) into steering/throttle/brake.

    On deformable terrain the predictive longitudinal model u_dot = a_x
    under-commands throttle, because the throttle-to-acceleration gain on soil
    falls well below the rigid-surface value the open-loop map assumes.  An
    asymmetric velocity-error disturbance observer (DOB) on the throttle
    channel closes the resulting steady-state speed error.  It is a classical
    integral compensator with two safety properties:

      (a) *Asymmetric* -- d_hat is clipped to [0, d_max], so the observer only
          ever adds throttle and never opposes MPC-commanded braking.  A
          positive d_hat accumulated during straight-line cruise therefore
          cannot carry into a corner and erode braking authority.

      (b) *Anti-windup* -- d_hat stops growing whenever the resulting throttle
          command saturates at 1.0 and whenever the MPC is braking, and bleeds
          toward zero under sustained braking, so an estimate formed in one
          manoeuvre does not persist into the next.

    The gain ``dob_ki`` (default 0.15 per m/s) is deliberately conservative:
    on sand at a 5 m/s reference, a steady 0.4 m/s speed deficit grows d_hat
    by about 0.06/s and reaches the typical 0.15 throttle compensation in
    roughly 2.5 s.  That is slow enough not to compete with the MPC's own
    transient ramp and fast enough to remove the steady-state error within a
    few seconds of cruise.

    Any MPC object exposing ``delta_max``, ``ax_min``, and ``ax_max`` is
    accepted; the AcadosMPC solver satisfies this interface.
    """

    def __init__(self, mpc, v_target: float = 5.0,
                 dob_ki: float = 0.15, dob_max: float = 0.35,
                 dob_bleed: float = 0.5):
        self.v_target = v_target
        self.steering_angle = 0.0   # δ
        self.acceleration = 0.0     # ax
        self.delta_max = mpc.delta_max
        self.ax_min = mpc.ax_min
        self.ax_max = mpc.ax_max
        self.steering_gain = 1.0 / self.delta_max
        self.throttle_gain = 1.0
        # Known model/plant mismatch, kept deliberately: with the brake-torque
        # ceiling derived as M*|ax_min|*r, a gain of 1.0 would make the pedal
        # mapping exact, and 0.6 under-realizes planned deceleration by ~40%.
        # The constant is a hand-tuned softener that predates every published
        # matrix; it applies identically to all controller arms (comparisons
        # stay paired), the safety filters command their own brake channel and
        # are unaffected, and the throttle DOB never closes brake error. Fixing
        # it is a controller recalibration, not a bug fix, and is out of scope
        # for the audited generation (2026-08-26 review, finding 4).
        self.brake_gain = 0.6
        self._prev_throttle = 0.0
        self._prev_braking = 0.0
        self.dob_ki = float(dob_ki)         # integrator gain [throttle / (m/s) / s]
        self.dob_max = float(dob_max)       # upper clip on d_hat (asymmetric)
        self.dob_bleed = float(dob_bleed)   # exponential bleed rate during braking
        self._d_hat = 0.0                   # estimated throttle bias (integral DOB)
        self._d_ff = 0.0                    # feedforward terrain throttle offset

    def update(self, delta_dot: float, Jx: float, dt: float, u: float,
               v_ref_now: float | None = None,
               desired_ax: float | None = None):
        """Integrate rate commands and produce vehicle inputs.

        Args:
            delta_dot: steering rate command from MPC [rad/s]
            Jx: longitudinal jerk command from MPC [m/s^3]
            dt: control interval [s]
            u: measured forward speed [m/s]
            v_ref_now: current reference speed [m/s].  When provided, enables
                the velocity-error DOB on the throttle channel.
            desired_ax: when provided (force-balance mode, where the MPC
                longitudinal control is slip-rate not jerk), set the integrator
                acceleration directly to this force-balance-planned value instead
                of integrating ``Jx``.

        Returns:
            (steering, throttle, braking) -- normalised Chrono inputs.
        """
        self.steering_angle += delta_dot * dt
        self.steering_angle = np.clip(self.steering_angle,
                                      -self.delta_max, self.delta_max)
        if desired_ax is not None:
            self.acceleration = float(np.clip(desired_ax, self.ax_min, self.ax_max))
        else:
            self.acceleration += Jx * dt
            self.acceleration = np.clip(self.acceleration,
                                        self.ax_min, self.ax_max)

        # Steering → normalised
        steering = np.clip(self.steering_angle * self.steering_gain, -1.0, 1.0)

        dead_band = 0.1  # m/s²
        is_braking = self.acceleration < -dead_band
        if self.acceleration > dead_band:
            throttle = min(self.acceleration / self.ax_max * self.throttle_gain, 1.0)
            braking = 0.0
        elif is_braking:
            throttle = 0.0
            braking = min(-self.acceleration / abs(self.ax_min) * self.brake_gain, 1.0)
        else:
            throttle = 0.0
            braking = 0.0

        # --- Asymmetric velocity-error DOB on the throttle channel ---
        # Closes the steady-state speed gap that the predictive u_dot = a_x
        # model leaves open on deformable terrain.
        if v_ref_now is not None and self.dob_ki > 0.0:
            v_err = float(v_ref_now) - float(u)
            if is_braking:
                # MPC is decelerating: bleed d_hat exponentially toward 0 so a
                # stale drag estimate does not subtract braking authority.
                self._d_hat *= max(0.0, 1.0 - self.dob_bleed * dt)
            else:
                # Pre-saturation candidate: needed to gate anti-windup.
                cand_throttle = throttle + self._d_hat
                will_saturate = cand_throttle >= 1.0 and v_err > 0.0
                if not will_saturate:
                    self._d_hat += self.dob_ki * v_err * dt
                # Asymmetric clip: only positive bias, capped at dob_max.
                self._d_hat = float(np.clip(self._d_hat, 0.0, self.dob_max))
            throttle = float(np.clip(throttle + self._d_hat, 0.0, 1.0))

        # --- Feedforward terrain-aware throttle offset ---
        # The integral observer above converges to a per-terrain throttle
        # offset that the open-loop throttle = a_x / a_x_max map omits on
        # deformable soil. This term applies that offset directly, as a
        # precomputed feedforward d_ff(n_hat) with no integral accumulation.
        # With dob_ki = 0 it constitutes a purely static terrain-aware
        # actuation map; the two mechanisms may also be combined, the
        # feedforward supplying the bulk and a small integral gain the
        # residual.
        # The bias is held whenever the MPC is not braking, mirroring the
        # observer, which gates the growth of _d_hat rather than its
        # application. Gating instead on an instantaneous v_err > 0 would
        # switch the bias off at equilibrium and let the speed sag; the
        # not-braking gate already prevents the term from opposing MPC
        # braking, and any overshoot is absorbed by the MPC's own brake
        # command.
        if self._d_ff > 0.0 and not is_braking:
            throttle = float(np.clip(throttle + self._d_ff, 0.0, 1.0))

        # Low-pass filter: prevent instant throttle↔brake switching.
        alpha = min(4.0 * dt, 1.0)
        throttle = alpha * throttle + (1.0 - alpha) * self._prev_throttle
        braking = alpha * braking + (1.0 - alpha) * self._prev_braking
        self._prev_throttle = throttle
        self._prev_braking = braking

        return steering, throttle, braking


# =============================================================================
# Tracking analytics
# =============================================================================

class TrackingAnalytics:
    """Accumulates path-tracking metrics over the simulation.

    Two lateral-error definitions are recorded. The axis-projected error
    ``y - y_ref(x)`` is the quantity written to the diagnostic CSV column
    ``crosstrack_err``; it is well defined only where the path is a function
    of x, and it flatters a vehicle that is barely moving. The geometric path
    error, taken at the closest point on the reference spline, is the one to
    rank runs by: it gives the Frenet lateral and longitudinal components and
    the heading relative to the path tangent there.

    Speed error, control inputs, and position are recorded alongside them for
    post-run analysis and for judging whether a low error reflects tracking or
    a stopped vehicle.
    """

    def __init__(self, ref_path, v_target: float,
                 rms_time_start: float = 0.0,
                 path_type: str = ''):
        self.ref_path = ref_path
        self.v_target = v_target
        self.rms_time_start = rms_time_start
        self.path_type = path_type

        self.times: list[float] = []
        self.crosstrack_errors: list[float] = []
        self.heading_errors: list[float] = []
        self.speed_errors: list[float] = []
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.us: list[float] = []
        self.v_refs_speed: list[float] = []

        self.steerings: list[float] = []
        self.throttles: list[float] = []
        self.brakings: list[float] = []
        self.deltas: list[float] = []
        self.accelerations: list[float] = []
        self.solve_times_ms: list[float] = []
        self.tau_comp_ms: list[float] = []
        self.ctrl_times: list[float] = []

        self.fy_times: list[float] = []
        self.actual_Fy_front: list[float] = []
        self.actual_Fy_rear: list[float] = []
        self.pred_Fy_front: list[float] = []
        self.pred_Fy_rear: list[float] = []
        self.actual_Fx_front: list[float] = []
        self.actual_Fx_rear: list[float] = []
        self.pred_Fx_front: list[float] = []
        self.pred_Fx_rear: list[float] = []

        self.y_refs: list[float] = []
        self.psi_refs: list[float] = []

        # Geometric path metrics (closest point on ReferencePath spline)
        self.path_pos_errors: list[float] = []
        self.path_lat_errors: list[float] = []
        self.path_lon_errors: list[float] = []
        self.heading_path_errors: list[float] = []  # rad, vs tangent at closest point

        self._window: list[float] = []

    def record(self, t: float, x: float, y: float, psi: float, u: float,
               v_ref_now: float | None = None):
        y_ref, psi_ref = self.ref_path.evaluate_at_x(x, y)

        ct_err = y - y_ref
        hd_err = psi - psi_ref
        hd_err = (hd_err + np.pi) % (2 * np.pi) - np.pi
        if v_ref_now is not None and np.isfinite(v_ref_now):
            v_ref_speed = float(v_ref_now)
        else:
            v_ref_speed = float(self.v_target)
        sp_err = u - v_ref_speed

        self.times.append(t)
        self.crosstrack_errors.append(ct_err)
        self.heading_errors.append(hd_err)
        self.speed_errors.append(sp_err)
        self.xs.append(x)
        self.ys.append(y)
        self.us.append(u)
        self.v_refs_speed.append(v_ref_speed)
        self.y_refs.append(y_ref)
        self.psi_refs.append(psi_ref)
        self._window.append(abs(ct_err))

        cp = self.ref_path.closest_point_on_path(x, y)
        self.path_pos_errors.append(cp["pos_err"])
        self.path_lat_errors.append(cp["e_lat"])
        self.path_lon_errors.append(cp["e_lon"])
        hp = psi - cp["psi_ref"]
        hp = (hp + np.pi) % (2 * np.pi) - np.pi
        self.heading_path_errors.append(float(hp))

    def record_control(self, t: float, steering: float, throttle: float,
                       braking: float, delta: float, acceleration: float,
                       solve_ms: float, tau_comp_ms: float):
        self.ctrl_times.append(t)
        self.steerings.append(steering)
        self.throttles.append(throttle)
        self.brakings.append(braking)
        self.deltas.append(delta)
        self.accelerations.append(acceleration)
        self.solve_times_ms.append(solve_ms)
        self.tau_comp_ms.append(tau_comp_ms)

    def record_tire_forces(self, t: float,
                           actual_Fy_f: float, actual_Fy_r: float,
                           pred_Fy_f: float, pred_Fy_r: float,
                           actual_Fx_f: float = 0.0, actual_Fx_r: float = 0.0,
                           pred_Fx_f: float = 0.0, pred_Fx_r: float = 0.0):
        self.fy_times.append(t)
        self.actual_Fy_front.append(actual_Fy_f)
        self.actual_Fy_rear.append(actual_Fy_r)
        self.pred_Fy_front.append(pred_Fy_f)
        self.pred_Fy_rear.append(pred_Fy_r)
        self.actual_Fx_front.append(actual_Fx_f)
        self.actual_Fx_rear.append(actual_Fx_r)
        self.pred_Fx_front.append(pred_Fx_f)
        self.pred_Fx_rear.append(pred_Fx_r)

    def periodic_summary(self, last_n: int = 20) -> str:
        if not self.crosstrack_errors:
            return ""
        recent_ct = self.crosstrack_errors[-last_n:]
        recent_hd = self.heading_errors[-last_n:]
        recent_sp = self.speed_errors[-last_n:]
        rms_ct = np.sqrt(np.mean(np.square(recent_ct)))
        rms_hd = np.degrees(np.sqrt(np.mean(np.square(recent_hd))))
        mean_sp = np.mean(recent_sp)
        return (f"ct={rms_ct:.3f}m  hd={rms_hd:.1f}°  "
                f"Δv={mean_sp:+.2f}m/s")

    def final_summary(self) -> str:
        if not self.crosstrack_errors:
            return "  No tracking data collected."

        ct = np.array(self.crosstrack_errors)
        hd = np.array(self.heading_errors)
        sp = np.array(self.speed_errors)
        ts = np.array(self.times)

        mask = ts >= self.rms_time_start
        ct_m = ct[mask] if mask.any() else ct
        hd_m = hd[mask] if mask.any() else hd
        sp_m = sp[mask] if mask.any() else sp

        mean_ct  = np.mean(np.abs(ct_m))
        rms_ct   = np.sqrt(np.mean(ct_m ** 2))
        max_ct   = np.max(np.abs(ct_m))
        rms_hd   = np.degrees(np.sqrt(np.mean(hd_m ** 2)))
        mean_sp  = np.mean(sp_m)

        pp = np.array(self.path_pos_errors)
        plat = np.array(self.path_lat_errors)
        plon = np.array(self.path_lon_errors)
        hdp = np.array(self.heading_path_errors)
        pp_m = pp[mask] if mask.any() and len(pp) == len(ts) else pp
        plat_m = plat[mask] if mask.any() and len(plat) == len(ts) else plat
        plon_m = plon[mask] if mask.any() and len(plon) == len(ts) else plon
        hdp_m = hdp[mask] if mask.any() and len(hdp) == len(ts) else hdp
        us = np.array(self.us)
        us_m = us[mask] if mask.any() and len(us) == len(ts) else us
        vrs = np.array(self.v_refs_speed)
        if len(vrs) == len(ts):
            vrs_m = vrs[mask] if mask.any() else vrs
        else:
            vrs_m = np.full_like(us_m, self.v_target, dtype=float)

        avg_path_pos = float(np.mean(pp_m)) if len(pp_m) else float("nan")
        rms_path_pos = float(np.sqrt(np.mean(pp_m ** 2))) if len(pp_m) else float("nan")
        rms_plat = float(np.sqrt(np.mean(plat_m ** 2))) if len(plat_m) else float("nan")
        rms_plon = float(np.sqrt(np.mean(plon_m ** 2))) if len(plon_m) else float("nan")
        rms_hdp_deg = float(np.degrees(np.sqrt(np.mean(hdp_m ** 2)))) if len(hdp_m) else float("nan")
        mean_u = float(np.mean(us_m)) if len(us_m) else float("nan")
        mean_v_ref = float(np.mean(vrs_m)) if len(vrs_m) else float(self.v_target)
        spd_ratio = mean_u / mean_v_ref if mean_v_ref > 1e-6 else float("nan")

        br = np.array(self.brakings) if self.brakings else np.array([])
        if len(br) == len(ts):
            br_m = br[mask] if mask.any() else br
            brake_duty = float(np.mean(br_m > 0.15))
            mean_brake = float(np.mean(br_m))
        else:
            brake_duty = float("nan")
            mean_brake = float("nan")

        # Combined pose-style metric (metres + radians scaled to ~metres)
        pose_scale = 1.0  # 1 rad yaw error counts like 1 m lateral for ranking
        if len(pp_m) and len(hdp_m) == len(pp_m):
            pose_rms = float(
                np.sqrt(np.mean(pp_m ** 2 + (pose_scale * hdp_m) ** 2)))
        else:
            pose_rms = float("nan")

        lines = [
            f"\n  Tracking Summary (t≥{self.rms_time_start:.1f}s):",
            f"    Axis-projected y−y_ref(x) (CSV crosstrack_err):",
            f"      Avg |CTE|:  {mean_ct:.4f} m",
            f"      RMS CTE:    {rms_ct:.4f} m",
            f"      Max |CTE|:  {max_ct:.4f} m",
            f"      RMS heading (y_ref from x-proj): {rms_hd:.2f}°",
            f"    Geometric path (closest point on spline — use for ranking):",
            f"      Avg path pos err: {avg_path_pos:.4f} m  (RMS {rms_path_pos:.4f} m)",
            f"      RMS |Frenet lat|: {rms_plat:.4f} m   RMS |Frenet lon|: {rms_plon:.4f} m",
            f"      RMS heading vs path tangent: {rms_hdp_deg:.2f}°",
            f"      RMS pose (pos + 1·|ψ_err| rad): {pose_rms:.4f}",
            f"    Motion:",
            f"      Mean speed: {mean_u:.3f} m/s  (ratio u/u_ref = {spd_ratio:.2f}, mean u_ref={mean_v_ref:.3f} m/s)",
            f"      Mean Δspeed (u−u_ref): {mean_sp:+.3f} m/s",
            f"      Brake duty (>0.15): {100*brake_duty:.1f}%   mean brake cmd: {mean_brake:.3f}",
        ]
        if np.isfinite(spd_ratio) and spd_ratio < 0.35:
            lines.append(
                "    *** Low speed ratio — the axis-projected CTE can look small while the vehicle is held on the brakes rather than tracking. ***"
            )
        return "\n".join(lines)

    def plot_results(self, plot_dir: str, terrain_name: str = '', model_label: str = ''):
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(plot_dir, exist_ok=True)

        times = np.array(self.times)
        crosstrack = np.array(self.crosstrack_errors)
        heading = np.degrees(np.array(self.heading_errors))
        speed = np.array(self.us)
        speed_err = np.array(self.speed_errors)
        xs = np.array(self.xs)
        ys = np.array(self.ys)

        plt.figure()
        plt.plot(times, crosstrack, label='Cross-track error (m)')
        plt.xlabel('Time (s)')
        plt.ylabel('Cross-track error (m)')
        plt.title(f'Cross-track Error\n{terrain_name} {model_label}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'crosstrack_error.png'))
        plt.close()

        plt.figure()
        plt.plot(times, heading, label='Heading error (deg)')
        plt.xlabel('Time (s)')
        plt.ylabel('Heading error (deg)')
        plt.title(f'Heading Error\n{terrain_name} {model_label}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'heading_error.png'))
        plt.close()

        plt.figure()
        plt.plot(times, speed_err, label='Speed error (m/s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed error (m/s)')
        plt.title(f'Speed Error\n{terrain_name} {model_label}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'speed_error.png'))
        plt.close()

        plt.figure()
        plt.plot(xs, ys, label='Actual trajectory')
        if hasattr(self.ref_path, 'x_pts') and hasattr(self.ref_path, 'y_pts'):
            plt.plot(self.ref_path.x_pts, self.ref_path.y_pts, '--', label='Reference path')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'XY Trajectory\n{terrain_name} {model_label}')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'xy_trajectory.png'))
        plt.close()

        # --- Tire force comparison plots (Fy and Fx, front & rear) ---
        if self.fy_times and self.pred_Fy_front:
            ft = np.array(self.fy_times)

            fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(ft, self.actual_Fy_front, alpha=0.6, label='Chrono (actual)')
            axes[0].plot(ft, self.pred_Fy_front, alpha=0.8, label='Model (predicted)')
            axes[0].set_ylabel('Fy (N)')
            axes[0].set_title(f'Front Axle Lateral Force (Fy)\n{terrain_name} {model_label}')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(ft, self.actual_Fy_rear, alpha=0.6, label='Chrono (actual)')
            axes[1].plot(ft, self.pred_Fy_rear, alpha=0.8, label='Model (predicted)')
            axes[1].set_ylabel('Fy (N)')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_title('Rear Axle Lateral Force (Fy)')
            axes[1].legend()
            axes[1].grid(True)

            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, 'tire_forces_Fy.png'), dpi=150)
            plt.close(fig)

            if any(v != 0 for v in self.pred_Fx_front):
                fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                axes[0].plot(ft, self.actual_Fx_front, alpha=0.6, label='Chrono (actual)')
                axes[0].plot(ft, self.pred_Fx_front, alpha=0.8, label='Model (predicted)')
                axes[0].set_ylabel('Fx (N)')
                axes[0].set_title(f'Front Axle Longitudinal Force (Fx)\n{terrain_name} {model_label}')
                axes[0].legend()
                axes[0].grid(True)

                axes[1].plot(ft, self.actual_Fx_rear, alpha=0.6, label='Chrono (actual)')
                axes[1].plot(ft, self.pred_Fx_rear, alpha=0.8, label='Model (predicted)')
                axes[1].set_ylabel('Fx (N)')
                axes[1].set_xlabel('Time (s)')
                axes[1].set_title('Rear Axle Longitudinal Force (Fx)')
                axes[1].legend()
                axes[1].grid(True)

                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, 'tire_forces_Fx.png'), dpi=150)
                plt.close(fig)

        print(f"Plots saved to {plot_dir}")


class RateTracker:
    """Finite-difference rate features for rate-augmented tire surrogates.

    Rate-augmented checkpoints are trained on differences taken over a fixed
    sampling interval, so the rate features supplied at runtime must use the
    same interval. Differencing at every control step, roughly 2-4 ms, divides
    a very small change by a very small interval and yields large spikes far
    outside the feature range the network was supervised on. This class
    anchors its samples in simulation time and refreshes the rates only once
    at least ``sample_dt`` seconds have elapsed.
    """

    def __init__(self, sample_dt: float = 0.05):
        self.sample_dt = float(sample_dt)
        self._anchor_t = None
        self._anchor_f = None
        self._anchor_r = None
        self._rates_front = np.zeros(3)
        self._rates_rear = np.zeros(3)

    def update(self, t, kappa_f, alpha_f, u_f, kappa_r, alpha_r, u_r):
        cur_f = np.array([kappa_f, alpha_f, u_f], dtype=float)
        cur_r = np.array([kappa_r, alpha_r, u_r], dtype=float)
        if self._anchor_t is None:
            self._anchor_f = cur_f
            self._anchor_r = cur_r
            self._anchor_t = float(t)
            return
        dt_e = float(t) - self._anchor_t
        if dt_e >= self.sample_dt:
            dt_e = max(dt_e, 1e-6)
            self._rates_front = (cur_f - self._anchor_f) / dt_e
            self._rates_rear = (cur_r - self._anchor_r) / dt_e
            self._anchor_f = cur_f
            self._anchor_r = cur_r
            self._anchor_t = float(t)

    @property
    def front(self):
        return self._rates_front.copy()

    @property
    def rear(self):
        return self._rates_rear.copy()
