"""DOB-CBF with no hand-set numbers (``--cbf-theory --safety-derived``).

The textbook DOB-CBF (``dob_cbf_theory.py``) stays as it is. This one keeps its structure (a minimum-deviation QP
over two inputs, hard barrier rows, a disturbance observer whose error bound tightens every row) and takes every
number from the plant's specification (``vehicle_spec``), the rig surrogate (``derived_model``) or the bounds
identified on separate plant manoeuvres (``--safety-ident``). The README section "Derived parameters" lists them.

Model. Kinematic bicycle, X' = v cos(psi), Y' = v sin(psi), psi' = v tan(delta) / L, v' = a(p, v) + d, with the
road-wheel angle delta and the pedal p as the inputs over one period: delta jumps to its command, p moves to its
command at the plant's rate limit. a(p, v) is the derived model's speed rate (rig surrogate, the plant's powertrain,
the soil's motion resistance, braking fading out at a standstill, gravity on a previewed grade).

Backup. Straight (delta = 0), the pedal ramped to full brake at the rate limit and held to a stop. On the model it
travels S_model(v, p). The identified bounds put the plant's front edge within s0 + s1 S_model of its starting
place and every corner within l0 + l1 S_model of the sides (``footprint.py``: the body rectangle). The swept region
is the body rectangle lengthened forward by that stop and widened by that drift, in the body frame now.

Barrier, per obstacle: h = distance(obstacle's set, swept rectangle) - sampled-data margin. A rock's set is its disc.
A vehicle's is its own rectangle (the same HMMWV, ``footprint.traffic_box``) stretched along its heading by what it
can cover during the backup's stopping time, up to its speed plus the measurement error and accelerating at the
drive capability of the believed soil. Under the backup the stop point does not move and the swept segment shrinks into
itself, so a static obstacle's h does not decrease: the QP is feasible from h >= 0 (backup control barrier
functions, Gurriet et al. 2018, Chen et al. 2021).

Sensing edge: the swept rectangle's far corners must stay inside the sensed range less the largest obstacle the
scenario can hold.

Condition: h' + gamma h >= 0, with h' linear in tan(delta) and the commanded pedal and the observer's error bound
in the worst direction (Alan et al. 2023). The derivatives of h are taken by central differences.

Sampled-data margin: the input is held over the period, so h can dip below its linear prediction. The margin is
1/2 L dt^2, with L the largest |h''| over the period at the corners of the input box (second differences along
the model).

QP: identity cost on the inputs scaled by their ranges (tan(delta) over tan of full lock, the pedal). Infeasible:
the backup's first period (straight, the pedal ramped down). No state projection over the teleop delay: the filter
runs on the vehicle.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import footprint as fp
from . import vehicle_spec as spec
from .authority import peak_lateral_accel
from .dob_cbf import SafetyFilterResult
from .dob_cbf_theory import _nn_loader, _presets, solve_qp_2d
from .derived_model import DerivedBicycle
from .grade_preview import G, downgrades
from .surrogate_dynamics import BRAKE_FADE_SPEED


class DerivedCBFSafetyFilter:
    """The DOB-CBF of the module docstring, with the plant node's filter interface."""

    def __init__(self, vehicle_params: dict, ident: dict, powertrain: str = "shafts", nn_casadi=None,
                 control_dt: float = 0.1, gamma_brake: float = 3.0, observer_k_dt: float = 0.5,
                 sense_range: float = 30.0, odd_obstacle_radius: float = 1.0, stale_cmd_timeout: float = 2.0,
                 teleop_delay: float = 0.0, traffic_speed_err: float = 0.0, unknown_downgrade: float = 0.15,
                 nn_model_dir: str = "nn_models/tire_force_static", nominal_soil: str = "sand", **_ignored):
        self.variant = "dob_cbf_derived"
        self.vehicle_params = dict(vehicle_params)
        self.dt = float(control_dt)
        self.gamma = float(gamma_brake)
        self.k_obs = float(observer_k_dt) / self.dt
        self.sense_range = float(sense_range) if sense_range else 0.0
        self.odd_radius = float(odd_obstacle_radius)
        self.traffic_speed_err = float(traffic_speed_err)
        self.unknown_downgrade = float(unknown_downgrade)
        if ident.get("footprint") != "box":
            raise ValueError("these bounds were identified for another footprint (need the body rectangle)")
        b = ident["powertrains"][powertrain]
        self.s0, self.s1 = b["stop"]["s0"], b["stop"]["s1"]
        self.l0, self.l1 = b["lateral"]["l0"], b["lateral"]["l1"]
        self.sigma = b["observer"]["sigma"]
        self.e0 = b["observer"]["e0"]
        self.x_rear, self.x_front, self.half_width = fp.body_box()
        self.max_steer = spec.MAX_ROAD_WHEEL
        self.w_max = float(np.tan(self.max_steer))
        self.steer_step = spec.STEER_RATE * self.max_steer * self.dt
        self.pedal_step = spec.PEDAL_RATE * self.dt
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
        self._bike = DerivedBicycle(self._nn, self.vehicle_params, powertrain)
        self.L = self._bike.L
        self._soil = f"nominal:{nominal_soil}"
        self._set_soil(nominal)

        self._delta = 0.0          # applied road-wheel angle, tracked through the rate limit
        self._pedal = 0.0          # applied pedal
        self._xi = None            # observer state
        self._d_hat = 0.0
        self._a_mean = 0.0         # the model's mean du/dt over the last period
        self._t_obs = 0.0
        self._filter_count = 0
        self._modify_count = 0
        self._infeasible_count = 0
        self._stale_brake_count = 0
        self._solve_ms_ema = 0.0
        self._last_result = None
        self._last_min_h = float("inf")
        self._last_margin = 0.0
        self._last_stop = 0.0

    # -- interface parity -----------------------------------------------------------------------
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

    def update_terrain(self, terrain_params, phi_uncertainty_deg=None, n_sigma=None, hedge_k=None, **_):
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
            print(f"  [CBF-DERIVED] soil update failed ({type(e).__name__}: {e})")

    def _set_soil(self, terrain):
        b = self._bike
        b.set_terrain(terrain)
        self._a_lat = peak_lateral_accel(self._nn, terrain, b.fz_f, b.fz_r, b.M)
        self._traffic_decel = b.peak_decel()
        self._traffic_accel = b.drive_capability()

    # -- the backup -------------------------------------------------------------------------------
    def _rate(self, p, v, g_along):
        a = self._bike.longitudinal(p, v)[0] + g_along
        return np.where(a < 0.0, a * np.tanh(np.maximum(v, 0.0) / BRAKE_FADE_SPEED), a)

    def _stop(self, v, p, g_along):
        """Model stopping distance and time of the backup from arrays of (v, p)."""
        v = np.array(v, dtype=float)
        p = np.array(p, dtype=float)
        dist = np.zeros_like(v)
        t_stop = np.zeros_like(v)
        h = 0.02
        for _ in range(int(60.0 / h)):
            moving = v > 0.05
            if not moving.any():
                break
            p = np.maximum(p - spec.PEDAL_RATE * h, -1.0)
            v_new = np.maximum(v + h * self._rate(p, v, g_along), 0.0)
            dist += np.where(moving, 0.5 * (v + v_new) * h, 0.0)
            t_stop += np.where(moving, h, 0.0)
            v = np.where(moving, v_new, v)
        return dist, t_stop

    def _gravity(self, vehicle_state, v, p):
        """Gravity along the heading assumed over the backup: the steepest previewed downgrade within its stop
        (fixed point over the stop length), none without a preview, the steepest admissible downgrade unseen."""
        prof = downgrades(vehicle_state, self.L, self.unknown_downgrade)
        if prof is None:
            return 0.0
        ds, down = prof
        if ds is None:
            return G * float(down)
        g = 0.0
        for _ in range(4):
            s_m, _ = self._stop([v], [p], g)
            stop = self.s0 + self.s1 * float(s_m[0]) + self.x_front
            n = min(len(down), int(np.ceil(stop / ds)) + 1)
            g = G * float(np.max(down[:n]))
        return g

    # -- barriers ---------------------------------------------------------------------------------
    def _h(self, X, Y, psi, s_m, t_stop, obs, dt_obs=0.0):
        """Barrier values for every obstacle (before the sampled-data margin), arrays over states."""
        s_b = self.s0 + self.s1 * s_m
        lat = self.l0 + self.l1 * s_m
        front = self.x_front + s_b
        width = self.half_width + lat
        out = []
        t_xr, t_xf, t_w = fp.traffic_box()
        for ox, oy, orad, vox, voy, hx, hy, speed, vehicle in obs:
            ox, oy = ox + vox * dt_obs, oy + voy * dt_obs
            if vehicle:
                reach = (max(speed, 0.0) + self.traffic_speed_err) * t_stop + 0.5 * self._traffic_accel * t_stop ** 2
                cs = fp.rect_corners(ox, oy, hx, hy, t_xr, t_xf + reach, t_w)
                qx, qy = fp.to_body(cs[..., 0], cs[..., 1], X, Y, psi)
                out.append(fp.box_quad(qx, qy, self.x_rear, front, width))
            else:
                ax, ay = fp.to_body(ox, oy, X, Y, psi)
                out.append(fp.point_box(ax, ay, self.x_rear, front, width) - orad)
        if self.sense_range > 0.0:
            out.append(self.sense_range - (self.odd_radius + np.hypot(front, width)))
        return np.array(out)

    # -- filter -------------------------------------------------------------------------------------
    def _result(self, delta_cmd, pedal, modified, n_rows, t0, min_h):
        steer = float(np.clip(delta_cmd / self.max_steer, -1.0, 1.0))
        pedal = float(np.clip(pedal, -1.0, 1.0))
        self._delta += float(np.clip(steer * self.max_steer - self._delta, -self.steer_step, self.steer_step))
        self._pedal += float(np.clip(pedal - self._pedal, -self.pedal_step, self.pedal_step))
        if modified:
            self._modify_count += 1
        ms = (time.time() - t0) * 1e3
        self._solve_ms_ema = 0.9 * self._solve_ms_ema + 0.1 * ms
        self._last_result = SafetyFilterResult(steer, max(pedal, 0.0), max(-pedal, 0.0), bool(modified),
                                               n_rows, ms, float("nan"), float(min_h), abs(self._d_hat))
        return self._last_result

    def filter(self, desired_steering, desired_throttle, desired_brake, vehicle_state, obstacles=None, **_):
        t0 = time.time()
        self._filter_count += 1
        s_op = float(np.clip(desired_steering, -1.0, 1.0))
        pedal_op = float(np.clip(float(desired_throttle) - float(desired_brake), -1.0, 1.0))
        X = float(vehicle_state.get("x", 0.0))
        Y = float(vehicle_state.get("y", 0.0))
        psi = float(vehicle_state.get("psi", 0.0))
        v = max(float(vehicle_state.get("u", 0.0)), 0.0)
        p = self._pedal

        # Observer: the model's mean du/dt over the last period, the measured speed.
        if self._xi is None:
            self._xi = v
        self._d_hat = self.k_obs * (v - self._xi)
        self._xi += self.dt * (self._a_mean + self._d_hat)
        self._t_obs += self.dt
        sigma = self.sigma + max(0.0, self.e0 - self.sigma) * np.exp(-self.k_obs * self._t_obs)

        if self._is_command_stale():
            self._stale_brake_count += 1
            return self._result(self._delta, -1.0, True, 0, t0, self._last_min_h)

        obs = []
        for o in obstacles or ():
            vox, voy = (float(o[4]), float(o[5])) if len(o) >= 6 else (0.0, 0.0)
            speed = float(np.hypot(vox, voy))
            if len(o) >= 7:
                hx, hy = np.cos(float(o[6])), np.sin(float(o[6]))
                speed = max(vox * hx + voy * hy, 0.0)
            elif speed > 1e-9:
                hx, hy = vox / speed, voy / speed
            else:
                hx, hy = 1.0, 0.0
            vehicle = bool(o[3]) if len(o) >= 4 else False
            obs.append((float(o[0]), float(o[1]), float(o[2]), vox, voy, hx, hy, speed, vehicle))

        g_along = self._gravity(vehicle_state, v, p)
        a_now = float(self._rate(np.array([p]), np.array([v]), g_along)[0])

        # Input box: the steering and pedal rate limits over one period, the soil's cornering authority.
        d_lim = self.max_steer
        if v > 0.5:
            d_lim = min(d_lim, float(np.arctan(self._a_lat * self.L / (v * v))))
        d_lo = max(-d_lim, self._delta - self.steer_step)
        d_hi = min(d_lim, self._delta + self.steer_step)
        if d_lo > d_hi:
            d_lo = d_hi = float(np.clip(np.sign(self._delta) * d_lim, self._delta - self.steer_step,
                                        self._delta + self.steer_step))
        q_lo, q_hi = max(-1.0, p - self.pedal_step), min(1.0, p + self.pedal_step)

        # Stops and barriers at the state and its perturbations (central differences), and along the model
        # at the corners of the input box (second differences, the sampled-data margin).
        ev, ep, ex, epsi, et = 0.05, 0.02, 0.05, 0.005, 0.05
        p_lo_fd, p_hi_fd = max(p - ep, -1.0), min(p + ep, 1.0)
        corners = [(dl, q) for dl in (d_lo, d_hi) for q in (q_lo, q_hi)]
        mid, end = [], []
        for dl, q in corners:
            for frac, store in ((0.5, mid), (1.0, end)):
                hstep = frac * self.dt
                p_c = p + frac * (q - p)
                v_c = max(v + hstep * a_now, 0.0)
                psi_c = psi + hstep * v * np.tan(dl) / self.L
                X_c = X + hstep * v * np.cos(0.5 * (psi + psi_c))
                Y_c = Y + hstep * v * np.sin(0.5 * (psi + psi_c))
                store.append((X_c, Y_c, psi_c, v_c, p_c))
        vs = np.array([v, v + ev, max(v - ev, 0.0), v, v] + [s[3] for s in mid + end])
        ps = np.array([p, p, p, p_hi_fd, p_lo_fd] + [s[4] for s in mid + end])
        s_m, t_stop = self._stop(vs, ps, g_along)
        self._last_stop = float(s_m[0])

        def h_at(i, Xq=X, Yq=Y, psiq=psi, dt_obs=0.0):
            return self._h(np.array(Xq), np.array(Yq), np.array(psiq), s_m[i], t_stop[i], obs, dt_obs)
        h0 = h_at(0)
        h_v = (h_at(1) - h_at(2)) / (vs[1] - vs[2])
        h_p = (h_at(3) - h_at(4)) / max(ps[3] - ps[4], 1e-9)
        h_X = (h_at(0, Xq=X + ex) - h_at(0, Xq=X - ex)) / (2 * ex)
        h_Y = (h_at(0, Yq=Y + ex) - h_at(0, Yq=Y - ex)) / (2 * ex)
        h_psi = (h_at(0, psiq=psi + epsi) - h_at(0, psiq=psi - epsi)) / (2 * epsi)
        h_t = (h_at(0, dt_obs=et) - h0) / et
        L_dd = np.zeros_like(h0)
        n_c = len(corners)
        for k in range(n_c):
            sm, se = mid[k], end[k]
            h_m = self._h(np.array(sm[0]), np.array(sm[1]), np.array(sm[2]), s_m[5 + k], t_stop[5 + k], obs,
                          0.5 * self.dt)
            h_e = self._h(np.array(se[0]), np.array(se[1]), np.array(se[2]), s_m[5 + n_c + k],
                          t_stop[5 + n_c + k], obs, self.dt)
            L_dd = np.maximum(L_dd, np.abs(h_e - 2.0 * h_m + h0) / (0.5 * self.dt) ** 2)
        margin = 0.5 * L_dd * self.dt ** 2
        h = h0 - margin
        self._last_margin = float(margin.max()) if len(margin) else 0.0

        # Rows c_w w + c_q q <= rhs in w = tan(delta), q = the pedal commanded for the end of the period.
        rows = []
        for i in range(len(h)):
            drift = (h_X[i] * v * np.cos(psi) + h_Y[i] * v * np.sin(psi) + h_v[i] * (a_now + self._d_hat)
                     - abs(h_v[i]) * sigma + h_t[i] - h_p[i] * p / self.dt)
            rows.append((-h_psi[i] * v / self.L, -h_p[i] / self.dt, self.gamma * h[i] + drift))
        min_h = float(h.min()) if len(h) else float("inf")
        self._last_min_h = min_h

        # z = (tan(delta) / tan(full lock), q): identity cost on the inputs scaled by their ranges.
        A, bvec = [], []
        for c_w, c_q, rhs in rows:
            A.append([c_w * self.w_max, c_q])
            bvec.append(rhs)
        for row, rhs in (([1.0, 0.0], np.tan(d_hi) / self.w_max), ([-1.0, 0.0], -np.tan(d_lo) / self.w_max),
                         ([0.0, 1.0], q_hi), ([0.0, -1.0], -q_lo)):
            A.append(row)
            bvec.append(rhs)
        A, bvec = np.array(A), np.array(bvec)
        delta_op = float(np.clip(s_op * self.max_steer, d_lo, d_hi))
        authority_clip = abs(delta_op - np.clip(s_op * self.max_steer, self._delta - self.steer_step,
                                                self._delta + self.steer_step)) > 1e-9
        z_op = np.array([np.tan(delta_op) / self.w_max, float(np.clip(pedal_op, q_lo, q_hi))])
        z = solve_qp_2d(z_op, A, bvec)
        n_rows = len(rows)
        if z is None:
            self._infeasible_count += 1
            delta_c, q = float(np.clip(0.0, d_lo, d_hi)), q_lo
            modified = True
        elif np.allclose(z, z_op, atol=1e-9):
            # The command, as the actuators execute it, satisfies every row.
            delta_c, q, modified = s_op * self.max_steer, pedal_op, authority_clip
            if authority_clip:
                delta_c = delta_op
        else:
            delta_c, q, modified = float(np.arctan(z[0] * self.w_max)), float(z[1]), True
        # The model's mean du/dt over this period, for the observer's next step.
        q_eff = float(np.clip(q, q_lo, q_hi))
        sub = np.linspace(0.0, 1.0, 5)
        self._a_mean = float(np.mean(self._rate(p + sub * (q_eff - p), np.full(5, v), g_along)))
        return self._result(delta_c, q, modified, n_rows, t0, min_h)

    def intervention_rate(self) -> float:
        return self._modify_count / self._filter_count if self._filter_count else 0.0

    def get_diagnostics(self) -> dict:
        return {"filter_calls": self._filter_count,
                "interventions": self._modify_count,
                "intervention_rate": self.intervention_rate(),
                "mean_solve_ms": self._solve_ms_ema,
                "variant": self.variant,
                "soil": self._soil,
                "a_lat_max": self._a_lat,
                "d_hat": self._d_hat,
                "infeasible": self._infeasible_count,
                "stale_brakes": self._stale_brake_count,
                "sampled_data_margin": self._last_margin,
                "backup_stop_model": self._last_stop,
                "gamma": self.gamma}
