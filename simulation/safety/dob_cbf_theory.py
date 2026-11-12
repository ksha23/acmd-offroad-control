"""DOB-CBF in its textbook form (``--cbf-theory``).

The published DOB-CBF (``dob_cbf.py``) is kept unchanged for reproduction. This
filter implements the same idea with only the parts the theory specifies:

Model. Kinematic bicycle with the steering angle as a state,
    X' = v cos(psi), Y' = v sin(psi), psi' = v tan(delta) / L,
    v' = u_a + f_drag(v) + d, delta' = u_delta,
with inputs u = (u_delta, u_a), the road-wheel rate and the commanded
longitudinal acceleration. The pedal-to-acceleration map and the resistance
f_drag come from the rig surrogate (surrogate_dynamics.py): braking is the
torque balance on its longitudinal curve, which saturates at wheel lock, and
traction is capped by its drive force. d is what the model leaves out.

Disturbance observer (Chen; Alan, Molnar, Das, Ames, Orosz, IEEE L-CSS 2023).
    d_hat = k (v - xi),  xi' = a_applied + f_drag(v) + d_hat,
with k dt well below 1. Its error obeys |e(t)| <= b/k + (e0 - b/k) exp(-k t)
for |d'| <= b, and that bound sigma tightens every barrier row in the
worst-case direction.

Barriers.
* Each obstacle: the distance barrier h = |p - p_o|^2 - r^2, relative degree
  two, enforced as the high-order CBF condition
  psi1 = h' + p1 h,  psi1' + p2 psi1 >= 0, with real roots p1 = p2 =
  sqrt(cbf_alpha) and the exact derivatives (CasADi). This is the ellipse
  barrier of Zhang et al. (CCTA 2025, eq. 6) with equal semi-axes. The
  published filter's ellipse is about four times longer along the heading
  than across and is shifted forward, so a vehicle 9 m ahead of a stopped car
  is already inside it. Headway comes from the braking-distance barrier.
* Each obstacle in the lane: the braking-distance barrier of Ames et al. (IEEE
  TAC 2017, adaptive cruise control), h_b = gap - v tau - v^2 / (2 a_h), with
  a_h the margined surrogate braking deceleration on the believed soil, as a
  first-order condition h_b' + gamma h_b >= 0. Full braking satisfies it
  whenever h_b >= 0, so with these rows the QP stays feasible from the safe set.
* The sensing edge: the same braking barrier with the gap fixed at the sensed
  range, so the vehicle never outruns what it can see.

QP. min (u - u_cmd)' W (u - u_cmd) over the input box (steering rate and
authority, pedal rate), all rows hard. W is the paper's threat weighting. With
two inputs the QP is solved exactly. If it is infeasible: full braking with the
steering held.

Surrogate parts kept from the published filter: the soil's braking and
traction authority, the resistance, and the steering-angle bound that keeps
the kinematic lateral acceleration below the surrogate's peak cornering. The
stale-command brake is a teleop link watchdog applied before the filter. A
teleop round-trip delay is handled by predicting the state over it with the
same model (predictor feedback). Rocks are taken as static at their
sensed positions, and traffic moves at its measured velocity.
"""
from __future__ import annotations

import time
from pathlib import Path

import casadi as ca
import numpy as np

from .authority import peak_lateral_accel
from .dob_cbf import SafetyFilterResult
from .surrogate_dynamics import SurrogateBicycle


def _nn_loader():
    try:
        from simulation.tire_models.nn_tire_model import load_nn_tire_model
    except ModuleNotFoundError:
        from tire_models.nn_tire_model import load_nn_tire_model
    return load_nn_tire_model


def _presets():
    try:
        import simulation.shared.param_consistency as pc
    except ModuleNotFoundError:
        import shared.param_consistency as pc
    return pc


def solve_qp_2d(z0, A, b):
    """argmin ||z - z0||^2 subject to A z <= b, z in R^2, exactly.

    The minimum is z0 if feasible, else the projection onto one constraint line
    or a vertex of two. Returns None when the set is empty.
    """
    z0 = np.asarray(z0, dtype=float)
    tol = 1e-7
    if len(b) == 0 or np.all(A @ z0 <= b + tol):
        return z0
    best, best_d = None, np.inf
    for i in range(len(b)):
        a = A[i]
        n2 = float(a @ a)
        if n2 < 1e-12:
            continue
        z = z0 - (float(a @ z0) - b[i]) / n2 * a
        if np.all(A @ z <= b + tol):
            d = float(np.sum((z - z0) ** 2))
            if d < best_d:
                best, best_d = z, d
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            M = np.array([A[i], A[j]])
            if abs(np.linalg.det(M)) < 1e-10:
                continue
            z = np.linalg.solve(M, np.array([b[i], b[j]]))
            if np.all(A @ z <= b + tol):
                d = float(np.sum((z - z0) ** 2))
                if d < best_d:
                    best, best_d = z, d
    return best


class TheoryCBFSafetyFilter:
    """The DOB-CBF of the module docstring, with the plant node's filter interface."""

    def __init__(self,
                 vehicle_params: dict,
                 nn_casadi=None,
                 cbf_alpha: float = 1.0,
                 obstacle_buffer: float = 0.25,
                 vehicle_radius: float = 1.0,
                 control_dt: float = 0.1,
                 dob_bandwidth: float = 10.0,
                 cbf_flavor: str = 'balance',
                 teleop_delay: float = 0.0,
                 stale_cmd_timeout: float = 2.0,
                 brake_margin: float = 0.85,
                 gamma_brake: float = 3.0,
                 sense_range: float = 30.0,
                 unseen_obstacle_radius: float = 1.0,
                 front_extent: float = 2.6,
                 corridor_half_width: float = 1.2,
                 reaction_time: float = 0.2,
                 disturbance_rate_bound: float = 2.0,
                 initial_error_bound: float = 3.0,
                 max_road_steer_angle: float = 0.49,
                 max_steer_rate: float = 16.0,
                 max_pedal_rate: float = 8.0,
                 max_accel: float = 3.0,
                 tau_long: float = 0.1,
                 max_speed: float = 15.0,
                 nn_model_dir: str = "nn_models/tire_force_static",
                 nominal_soil: str = "sand",
                 **_ignored):
        self.variant = "dob_cbf_theory"
        self.L = float(vehicle_params.get("Lf", 1.593)) + float(vehicle_params.get("Lr", 1.709))
        self.vehicle_params = dict(vehicle_params)
        self.dt = float(control_dt)
        # Real roots of the second-order condition, the smallest a2 for a1 = cbf_alpha.
        self.p1 = self.p2 = float(np.sqrt(max(float(cbf_alpha), 1e-6)))
        self.obstacle_buffer = float(obstacle_buffer)
        self.vehicle_radius = float(vehicle_radius)
        # Observer gain, k dt <= 0.5 (dt k = 1 makes the estimate a raw difference).
        self.k_obs = min(float(dob_bandwidth), 0.5 / self.dt)
        self.b_d = float(disturbance_rate_bound)
        self.e0 = float(initial_error_bound)
        self.cbf_flavor = str(cbf_flavor)
        self.brake_margin = float(brake_margin)
        self.gamma = float(gamma_brake)
        self.sense_range = float(sense_range) if sense_range else 0.0
        self.unseen_obstacle_radius = float(unseen_obstacle_radius)
        self.front_extent = float(front_extent)
        self.corridor_half_width = float(corridor_half_width)
        self.tau_b = float(reaction_time)
        self.max_steer = float(max_road_steer_angle)
        self.steer_rate = float(max_steer_rate) * self.max_steer      # rad/s at the road wheel
        self.pedal_rate = float(max_pedal_rate)
        self.max_accel = float(max_accel)
        self.tau_long = float(tau_long)
        self.max_speed = float(max_speed)
        self._teleop_delay = max(float(teleop_delay), 0.0)
        self._stale_cmd_timeout = float(stale_cmd_timeout)
        self._clock = time.time
        self._last_cmd_wall = None

        _dir = Path(str(nn_model_dir)).expanduser()
        if not _dir.is_absolute():
            _dir = Path(__file__).resolve().parents[2] / _dir
        pc = _presets()
        nominal = pc.terrain_preset_to_internal(pc.get_terrain_preset(nominal_soil))
        self._nn = nn_casadi if nn_casadi is not None else _nn_loader()(str(_dir), nominal)
        self._bike = SurrogateBicycle(self._nn, self.vehicle_params, max_steer=self.max_steer)
        self._soil = f"nominal:{nominal_soil}"
        self._set_soil(nominal)

        # Executed actuator state, tracked through the plant's rate limits.
        self._delta = 0.0
        self._pedal = 0.0
        self._a_app = 0.0
        # Observer state.
        self._xi = None
        self._d_hat = 0.0
        self._t_obs = 0.0

        self._hocbf = self._build_hocbf()
        self._filter_count = 0
        self._modify_count = 0
        self._infeasible_count = 0
        self._brake_rows_binding = 0
        self._stale_brake_count = 0
        self._solve_ms_ema = 0.0
        self._last_result = None
        self._last_min_h = float("inf")

    # -- interface parity ---------------------------------------------------------
    def set_clock(self, clock):
        self._clock = clock

    def set_teleop_delay(self, delay_s: float):
        self._teleop_delay = max(float(delay_s), 0.0)

    def update_command_age(self, cmd_wall_time: float):
        self._last_cmd_wall = self._clock()

    def _is_command_stale(self) -> bool:
        if self._teleop_delay <= 0.0 or self._last_cmd_wall is None:
            return False
        return (self._clock() - self._last_cmd_wall) > self._stale_cmd_timeout

    def update_terrain(self, terrain_params, phi_uncertainty_deg=None, n_sigma=None,
                       hedge_k=None, **_):
        """Re-tabulate the soil's authority on the (hedged) believed soil."""
        if not isinstance(terrain_params, dict):
            return
        terrain = dict(terrain_params)
        hk = float(hedge_k) if hedge_k is not None else 0.0
        if n_sigma is not None and hk > 0.0:
            terrain["n"] = float(np.clip(float(terrain["n"]) - hk * float(n_sigma), 0.4, 1.3))
            if phi_uncertainty_deg is not None and terrain.get("phi") is not None:
                terrain["phi"] = max(5.0, float(terrain["phi"]) - hk * float(phi_uncertainty_deg))
        try:
            self._set_soil(terrain)
            self._soil = f"belief:n={float(terrain['n']):.2f},phi={float(terrain['phi']):.1f}"
        except Exception as e:  # noqa: BLE001
            print(f"  [CBF-THEORY] soil update failed ({type(e).__name__}: {e})")

    def _set_soil(self, terrain):
        b = self._bike
        b.set_terrain(terrain)
        self._a_peak = b.peak_decel()                       # nominal locked-wheel deceleration
        self._a_h = max(self.brake_margin * self._a_peak, 0.3)
        # Traction cap: four wheels at the top of the surrogate's drive curve.
        self._a_trac = 4.0 * float(b.curve[-1]) / b.M
        fz_f, fz_r = b.fz_f, b.fz_r
        self._a_lat = 0.9 * peak_lateral_accel(self._nn, terrain, fz_f, fz_r, b.M)
        # Pedal to acceleration on this soil, and its inverse.
        grid = np.linspace(-1.0, 1.0, 401)
        a = np.array([self._pedal_to_accel(g) for g in grid])
        a = a + 1e-6 * np.arange(len(a))                     # strictly increasing for the inverse
        self._map = (grid, a)

    def _pedal_to_accel(self, pedal):
        if pedal >= 0.0:
            return min(self.max_accel * pedal, self._a_trac)
        return self._bike.settled_accel(pedal, 0.0, 1.0)

    def _accel_to_pedal(self, a):
        grid, acc = self._map
        if a <= acc[0] + 1e-6:
            return -1.0
        return float(np.interp(a, acc, grid))

    def _drag(self, v):
        """Surrogate rolling resistance at speed v (m/s^2, negative), zero at rest."""
        b = self._bike
        u = max(abs(v), 2.0)
        fx, _ = b._tire(np.zeros(2), np.array([b.fz_f, b.fz_r]), np.full(2, u), np.zeros(2))
        return float(2.0 * (fx[0] + fx[1]) / b.M) * min(1.0, abs(v) / 2.0)

    # -- barriers -------------------------------------------------------------------
    def _build_hocbf(self):
        """Exact derivatives of the distance barrier along the kinematic model.

        The obstacle's position is part of the state and moves at its measured
        constant velocity, so a moving vehicle enters h' and h''.
        """
        x = ca.SX.sym("x", 7)     # X, Y, psi, v, delta, ox, oy
        o = ca.SX.sym("o", 3)     # r_safe, vox, voy
        p1 = ca.SX.sym("p1")
        X, Y, psi, v, delta, ox, oy = (x[i] for i in range(7))
        h = (X - ox) ** 2 + (Y - oy) ** 2 - o[0] ** 2
        f0 = ca.vertcat(v * ca.cos(psi), v * ca.sin(psi), v * ca.tan(delta) / self.L, 0, 0,
                        o[1], o[2])
        h_dot = ca.mtimes(ca.jacobian(h, x), f0)
        psi1 = h_dot + p1 * h
        J = ca.jacobian(psi1, x)
        return ca.Function("hocbf", [x, o, p1],
                           [h, h_dot, psi1, ca.mtimes(J, f0), J[3], J[4]])

    def _rows(self, xs, obstacles, obs_vehicle, f_drag, sigma):
        """Barrier rows c_delta u_delta + c_a u_a <= b, and the nearest threat."""
        X, Y, psi, v, delta = xs
        rows = []
        min_h, threat_vehicle = np.inf, False
        for (ox, oy, orad, vox, voy), is_veh in zip(obstacles, obs_vehicle):
            r_safe = orad + self.vehicle_radius + self.obstacle_buffer
            x_aug = [X, Y, psi, v, delta, ox, oy]
            h, _, psi1, lf, dv, dd = (float(t) for t in self._hocbf(x_aug, [r_safe, vox, voy], self.p1))
            if h < min_h:
                min_h, threat_vehicle = h, is_veh
            # psi1' + p2 psi1 >= 0 with v' = u_a + f_drag + d, d in d_hat +- sigma.
            rhs = lf + dv * (f_drag + self._d_hat) - abs(dv) * sigma + self.p2 * psi1
            rows.append((-dd, -dv, rhs))
        # Braking-distance rows: obstacles in the lane, and the sensing edge.
        c, s = np.cos(psi), np.sin(psi)
        omega = v * np.tan(delta) / self.L
        k_v = self.tau_b + max(v, 0.0) / self._a_h
        for ox, oy, orad, vox, voy in obstacles:
            rx, ry = ox - X, oy - Y
            lon, lat = rx * c + ry * s, -rx * s + ry * c
            if lon <= 0.0 or abs(lat) > orad + self.corridor_half_width + self.obstacle_buffer:
                continue
            gap = lon - (orad + self.front_extent + self.obstacle_buffer)
            # The obstacle is not credited with stopping distance of its own (it
            # may stop at once), but its motion along the heading sets the
            # closing speed.
            h_b = gap - v * self.tau_b - v * v / (2.0 * self._a_h)
            v_c = v - (vox * c + voy * s) - omega * lat
            rows.append((0.0, k_v, self.gamma * h_b - v_c - k_v * (f_drag + self._d_hat + sigma)))
        if self.sense_range > 0.0:
            r_eff = self.sense_range - (self.unseen_obstacle_radius + self.front_extent
                                        + self.obstacle_buffer)
            h_s = r_eff - v * self.tau_b - v * v / (2.0 * self._a_h)
            rows.append((0.0, k_v, self.gamma * h_s - k_v * (f_drag + self._d_hat + sigma)))
        return rows, min_h, threat_vehicle

    # -- filter ---------------------------------------------------------------------
    def _result(self, steer, pedal, modified, n_rows, t0, min_h):
        steer = float(np.clip(steer, -1.0, 1.0))
        pedal = float(np.clip(pedal, -1.0, 1.0))
        # Track the executed actuators through the plant's rate limits.
        self._delta += float(np.clip(steer * self.max_steer - self._delta,
                                     -self.steer_rate * self.dt, self.steer_rate * self.dt))
        self._pedal += float(np.clip(pedal - self._pedal, -self.pedal_rate * self.dt,
                                     self.pedal_rate * self.dt))
        lag = 1.0 - np.exp(-self.dt / self.tau_long)
        self._a_app += lag * (self._pedal_to_accel(self._pedal) - self._a_app)
        if modified:
            self._modify_count += 1
        ms = (time.time() - t0) * 1e3
        self._solve_ms_ema = 0.9 * self._solve_ms_ema + 0.1 * ms
        self._last_result = SafetyFilterResult(steer, max(pedal, 0.0), max(-pedal, 0.0), bool(modified),
                                               n_rows, ms, self.max_speed, float(min_h),
                                               abs(self._d_hat))
        return self._last_result

    def filter(self, desired_steering, desired_throttle, desired_brake, vehicle_state,
               obstacles=None, **_):
        t0 = time.time()
        self._filter_count += 1
        s_op = float(np.clip(desired_steering, -1.0, 1.0))
        a_cmd = float(desired_throttle) - float(desired_brake)
        pedal_op = float(np.clip(a_cmd, -1.0, 1.0))
        obs, obs_veh = [], []
        for o in obstacles or ():
            vel = (float(o[4]), float(o[5])) if len(o) >= 6 else (0.0, 0.0)
            obs.append((float(o[0]), float(o[1]), float(o[2])) + vel)
            obs_veh.append(bool(o[3]) if len(o) >= 4 else float(o[2]) >= 1.5)
        X = float(vehicle_state.get("x", 0.0))
        Y = float(vehicle_state.get("y", 0.0))
        psi = float(vehicle_state.get("psi", 0.0))
        v = float(vehicle_state.get("u", 0.0))

        # Observer: update with the measured speed and the applied acceleration.
        f_drag = self._drag(v)
        if self._xi is None:
            self._xi = v
        self._d_hat = self.k_obs * (v - self._xi)
        self._xi += self.dt * (self._a_app + f_drag + self._d_hat)
        self._t_obs += self.dt
        sigma = self.b_d / self.k_obs + max(0.0, self.e0 - self.b_d / self.k_obs) * np.exp(
            -self.k_obs * self._t_obs)

        if self._is_command_stale():
            self._stale_brake_count += 1
            return self._result(self._delta / self.max_steer, -1.0, True, 0, t0, self._last_min_h)

        # Predictor over the teleop round trip, on the same model.
        if self._teleop_delay > 0.0:
            adv = min(2.0 * self._teleop_delay, 0.5)
            n = max(1, int(round(adv / 0.05)))
            h = adv / n
            for _ in range(n):
                X += h * v * np.cos(psi)
                Y += h * v * np.sin(psi)
                psi += h * v * np.tan(self._delta) / self.L
                v = max(v + h * (self._a_app + self._drag(v) + self._d_hat), 0.0)
            f_drag = self._drag(v)
        xs = np.array([X, Y, psi, max(v, 0.0), self._delta])

        rows, min_h, threat_vehicle = self._rows(xs, obs, obs_veh, f_drag, sigma)
        self._last_min_h = min_h

        # Input box: steering rate and the soil's cornering authority, pedal rate.
        d_lim = self.max_steer
        if v > 0.5:
            d_lim = min(d_lim, float(np.arctan(self._a_lat * self.L / (v * v))))
        ud_lo = max(-self.steer_rate, (-d_lim - self._delta) / self.dt)
        ud_hi = min(self.steer_rate, (d_lim - self._delta) / self.dt)
        if ud_lo > ud_hi:
            # The wheel is outside the authority (the speed rose): back toward
            # the limit as fast as the rate allows.
            back = float(np.clip((np.sign(self._delta) * d_lim - self._delta) / self.dt,
                                 -self.steer_rate, self.steer_rate))
            ud_lo = ud_hi = back
        p_lo = max(-1.0, self._pedal - self.pedal_rate * self.dt)
        p_hi = min(1.0, self._pedal + self.pedal_rate * self.dt)
        ua_lo, ua_hi = self._pedal_to_accel(p_lo), self._pedal_to_accel(p_hi)

        # The command as the actuators would execute it: the plant's rate limits
        # apply to it anyway, so clipping to them is not an intervention. The
        # cornering-authority bound is.
        ud_rate = float(np.clip((s_op * self.max_steer - self._delta) / self.dt,
                                -self.steer_rate, self.steer_rate))
        ud_op = float(np.clip(ud_rate, ud_lo, ud_hi))
        authority_clip = abs(ud_op - ud_rate) > 1e-9
        pedal_eff = float(np.clip(pedal_op, p_lo, p_hi))
        ua_op = self._pedal_to_accel(pedal_eff)

        # Threat weighting, as published: steering cheaper than braking for a rock.
        if self.cbf_flavor == 'steer_priority':
            wd = (500.0, 1.0)
        elif self.cbf_flavor == 'throttle_priority':
            wd = (1.0, 500.0)
        elif self.cbf_flavor == 'balance' and not threat_vehicle:
            wd = (1.0, 10.0)
        else:
            wd = (1.0, 1.0)
        # Scaled variables z = (sqrt(w_s) delta_next / delta_max, sqrt(w_a) u_a / a_ref).
        a_ref = self.max_accel
        k1 = self.max_steer / np.sqrt(wd[0])
        k2 = a_ref / np.sqrt(wd[1])

        def to_z(ud, ua):
            return np.array([(self._delta + ud * self.dt) / k1, ua / k2])

        A, bvec = [], []
        for c_d, c_a, rhs in rows + [(1.0, 0.0, ud_hi), (-1.0, 0.0, -ud_lo),
                                     (0.0, 1.0, ua_hi), (0.0, -1.0, -ua_lo)]:
            # c_d u_delta + c_a u_a <= rhs, with u_delta = (k1 z1 - delta)/dt, u_a = k2 z2.
            A.append([c_d * k1 / self.dt, c_a * k2])
            bvec.append(rhs + c_d * self._delta / self.dt)
        A, bvec = np.array(A), np.array(bvec)
        z_op = to_z(ud_op, ua_op)
        z = solve_qp_2d(z_op, A, bvec)
        n_rows = len(rows)
        if z is None:
            # No input satisfies the rows: full braking, steering held.
            self._infeasible_count += 1
            return self._result(self._delta / self.max_steer, -1.0, True, n_rows, t0, min_h)
        if np.allclose(z, z_op, atol=1e-9):
            # The command (as the actuators execute it) satisfies every row.
            rate_only = abs(ud_rate * self.dt - (s_op * self.max_steer - self._delta)) > 1e-9
            steer = (self._delta + ud_op * self.dt) / self.max_steer if (rate_only or authority_clip) else s_op
            return self._result(steer, pedal_op, authority_clip, n_rows, t0, min_h)
        ud = (k1 * z[0] - self._delta) / self.dt
        ua = k2 * z[1]
        if ua < ua_op - 1e-9:
            self._brake_rows_binding += 1
        steer = (self._delta + ud * self.dt) / self.max_steer
        return self._result(steer, self._accel_to_pedal(ua), True, n_rows, t0, min_h)

    def intervention_rate(self) -> float:
        return self._modify_count / self._filter_count if self._filter_count else 0.0

    def get_diagnostics(self) -> dict:
        return {"filter_calls": self._filter_count,
                "interventions": self._modify_count,
                "intervention_rate": self.intervention_rate(),
                "mean_solve_ms": self._solve_ms_ema,
                "variant": self.variant,
                "soil": self._soil,
                "a_brake_margined": self._a_h,
                "a_lat_max": self._a_lat,
                "d_hat": self._d_hat,
                "infeasible": self._infeasible_count,
                "brake_rows_binding": self._brake_rows_binding,
                "stale_brakes": self._stale_brake_count,
                "hocbf_gains": (self.p1, self.p2)}
